"""
trainer.py — Main training loop for Marchon.

One iteration:
  1. Self-play   -> play N games, push experiences to replay buffer
  2. Train       -> sample mini-batches, compute losses, back-prop
  3. Checkpoint  -> save model every K iterations
  4. Evaluate    -> pit new vs previous checkpoint, update ELO

Speed additions vs Archon:
  - Mixed precision (AMP) when use_amp=True and device=cuda
  - torch.compile() kernel fusion when use_compile=True
  - MoE load-balance aux loss integrated into training step
"""

import logging
import multiprocessing as mp
import threading
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Optional

import chess
import torch
import torch.nn.functional as F
import torch.optim as optim

from env.chess_env import ChessEnv
from model.network import MarchonNet
from training.self_play import SelfPlay
from utils.replay_buffer import ReplayBuffer
from utils.elo import ELOTracker


def _run_game_worker(args: tuple):
    """
    Subprocess worker: reconstruct the network from a CPU state dict and play
    one complete self-play game.  Spawned per game so all games in an
    iteration execute in parallel.
    """
    state_dict_cpu, config = args

    network = MarchonNet(
        num_planes     = config.num_planes,
        num_res_blocks = config.num_res_blocks,
        channels       = config.channels,
        num_moe_blocks = config.num_moe_blocks,
        num_experts    = config.num_experts,
        top_k          = config.top_k,
    )
    network.load_state_dict(state_dict_cpu)
    network.to(config.device)
    network.eval()

    sp = SelfPlay(network, config, shared_state=None)
    return sp.play_game()


class Trainer:

    def __init__(self, config, shared_state=None):
        self.config  = config
        self.shared  = shared_state
        self.log     = logging.getLogger("Marchon.Trainer")

        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.games_dir.mkdir(parents=True, exist_ok=True)

        # ── Core network ──────────────────────────────────────────────────────
        self.network = MarchonNet(
            num_planes     = config.num_planes,
            num_res_blocks = config.num_res_blocks,
            channels       = config.channels,
            num_moe_blocks = config.num_moe_blocks,
            num_experts    = config.num_experts,
            top_k          = config.top_k,
        ).to(config.device)

        # Optional torch.compile — significant speedup on modern GPUs
        if config.use_compile:
            try:
                self.network = torch.compile(self.network)
                self.log.info("torch.compile() applied to network.")
            except Exception as exc:
                self.log.warning(f"torch.compile() unavailable: {exc}")

        self.replay_buffer = ReplayBuffer(config.replay_buffer_capacity)
        self.elo           = ELOTracker(k_factor=config.elo_k_factor)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr           = config.learning_rate,
            weight_decay = config.weight_decay,
        )

        # Mixed precision scaler (CUDA only)
        self._use_amp = config.use_amp and config.device == "cuda"
        self._scaler  = torch.amp.GradScaler("cuda") if self._use_amp else None

        # ── Anchor network ────────────────────────────────────────────────────
        self._anchor_network = None
        if config.anchor_checkpoint:
            anchor = MarchonNet(
                num_planes     = config.num_planes,
                num_res_blocks = config.num_res_blocks,
                channels       = config.channels,
                num_moe_blocks = config.num_moe_blocks,
                num_experts    = config.num_experts,
                top_k          = config.top_k,
            ).to(config.device)
            ckpt = torch.load(config.anchor_checkpoint, map_location=config.device)
            anchor.load_state_dict(ckpt["model_state"])
            anchor.eval()
            self._anchor_network = anchor
            self.log.info(f"Anchor checkpoint loaded: {config.anchor_checkpoint}")

        # ── Bookkeeping ───────────────────────────────────────────────────────
        self.iteration    = 0
        self.total_games  = 0
        self.policy_losses:   list = []
        self.value_losses:    list = []
        self.combined_losses: list = []
        self._decisive_window: deque = deque(maxlen=50)

        self.log.info(
            f"MarchonNet ready — {self.network.count_parameters():,} parameters "
            f"on {config.device}  AMP={self._use_amp}"
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, num_iterations: int = 1000, stop_event: Optional[threading.Event] = None):
        prev_network = None

        start = self.iteration + 1
        end   = start + num_iterations

        for it in range(start, end):
            if stop_event and stop_event.is_set():
                self.log.info("Stop signal received — exiting training loop.")
                break

            self.iteration = it
            if self.shared:
                self.shared.update_iteration(it, len(self.replay_buffer))
            self.log.info("-" * 50)
            self.log.info(f"Iteration {it}   buffer={len(self.replay_buffer)}")

            self._run_self_play()

            if len(self.replay_buffer) >= self.config.min_buffer_size:
                p_loss, v_loss = self._train_step()
                self.log.info(
                    f"  Loss — policy: {p_loss:.4f}   value: {v_loss:.4f}"
                )

            if it % self.config.checkpoint_every_n_iters == 0:
                self._save_checkpoint(it)

            if it % self.config.eval_every_n_iters == 0:
                if prev_network is not None:
                    self._evaluate_against(prev_network)
            if it % self.config.checkpoint_every_n_iters == 0:
                prev_network = deepcopy(self.network)

            if (self._anchor_network is not None
                    and it % self.config.anchor_eval_every_n_iters == 0):
                self._evaluate_against_anchor()

    # ── Self-play ─────────────────────────────────────────────────────────────

    def _run_self_play(self):
        state_dict_cpu = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}

        n    = self.config.games_per_iteration
        args = [(state_dict_cpu, self.config)] * n

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as pool:
            records = list(pool.map(_run_game_worker, args))

        wins = draws = losses = 0
        for record in records:
            self.total_games += 1
            record.game_id = self.total_games
            self.replay_buffer.push(record.experiences)

            pgn_path = self.config.games_dir / f"game_{self.total_games:05d}.pgn"
            pgn_path.write_text(record.pgn)

            if   record.result == "white": wins   += 1
            elif record.result == "black": losses += 1
            else:                          draws  += 1

            if self.shared:
                self.shared.push_game(record)

        self._decisive_window.append(wins + losses)
        decisive_games  = sum(self._decisive_window)
        total_in_window = len(self._decisive_window) * self.config.games_per_iteration
        decisive_pct    = decisive_games / total_in_window * 100 if total_in_window else 0.0

        self.log.info(
            f"  Self-play: {wins}W / {draws}D / {losses}L   "
            f"total games={self.total_games}   "
            f"decisive(last {len(self._decisive_window)}): {decisive_pct:.1f}%"
        )
        if self.shared:
            self.shared.update_stats(wins, draws, losses)

    # ── Training step ─────────────────────────────────────────────────────────

    def _train_step(self) -> tuple:
        """
        Run config.train_steps_per_iter gradient updates with AMP when enabled.

        Total loss = policy_loss + value_loss + load_balance_coeff * aux_loss
        """
        self.network.train()
        total_p = total_v = 0.0

        for _ in range(self.config.train_steps_per_iter):
            states, policies, values = self.replay_buffer.sample(self.config.batch_size)

            s = torch.FloatTensor(states).to(self.config.device)
            p = torch.FloatTensor(policies).to(self.config.device)
            v = torch.FloatTensor(values).unsqueeze(1).to(self.config.device)

            if self._use_amp:
                with torch.amp.autocast("cuda"):
                    policy_logits, value_pred, aux_loss = self.network(s)
                    log_p  = F.log_softmax(policy_logits, dim=1)
                    p_loss = -(p * log_p).sum(dim=1).mean()
                    v_loss = F.mse_loss(value_pred, v)
                    loss   = p_loss + v_loss + self.config.load_balance_coeff * aux_loss
                self.optimizer.zero_grad()
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                policy_logits, value_pred, aux_loss = self.network(s)
                log_p  = F.log_softmax(policy_logits, dim=1)
                p_loss = -(p * log_p).sum(dim=1).mean()
                v_loss = F.mse_loss(value_pred, v)
                loss   = p_loss + v_loss + self.config.load_balance_coeff * aux_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()

            total_p += p_loss.item()
            total_v += v_loss.item()

        self.network.eval()
        n     = self.config.train_steps_per_iter
        avg_p = total_p / n
        avg_v = total_v / n

        self.policy_losses.append(avg_p)
        self.value_losses.append(avg_v)
        self.combined_losses.append(avg_p + avg_v)

        if self.shared:
            self.shared.update_losses(
                self.policy_losses, self.value_losses, self.combined_losses
            )

        return avg_p, avg_v

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate_against(self, old_network):
        from mcts.mcts import MCTS

        new_mcts = MCTS(self.network,  self.config)
        old_mcts = MCTS(old_network,   self.config)

        wins = draws = losses = 0

        for game_idx in range(self.config.eval_games):
            new_plays_white = (game_idx % 2 == 0)
            white_mcts      = new_mcts if new_plays_white else old_mcts
            black_mcts      = old_mcts if new_plays_white else new_mcts

            env = ChessEnv()
            env.reset()
            ply = 0

            while not env.done and ply < self.config.max_game_length:
                mcts = white_mcts if env.board.turn == chess.WHITE else black_mcts
                move = mcts.select_move(env.board, temperature=0.0, add_noise=False)
                if move is None:
                    break
                env.step(move)
                ply += 1

            result  = env.result or "draw"
            new_won = (result == "white") == new_plays_white

            if   result == "draw": draws  += 1
            elif new_won:          wins   += 1
            else:                  losses += 1

        new_elo = self.elo.update_from_match(
            wins=wins, draws=draws, losses=losses,
            opponent_rating=self.elo.rating,
        )
        self.log.info(
            f"  Eval vs prev: {wins}W/{draws}D/{losses}L   ELO -> {new_elo:.0f}"
        )
        if self.shared:
            self.shared.update_elo(new_elo)

    def _evaluate_against_anchor(self):
        from mcts.mcts import MCTS

        new_mcts    = MCTS(self.network,         self.config)
        anchor_mcts = MCTS(self._anchor_network, self.config)

        wins = draws = losses = 0

        for game_idx in range(self.config.eval_games):
            new_plays_white = (game_idx % 2 == 0)
            white_mcts      = new_mcts    if new_plays_white else anchor_mcts
            black_mcts      = anchor_mcts if new_plays_white else new_mcts

            env = ChessEnv()
            env.reset()
            ply = 0

            while not env.done and ply < self.config.max_game_length:
                mcts = white_mcts if env.board.turn == chess.WHITE else black_mcts
                move = mcts.select_move(env.board, temperature=0.0, add_noise=False)
                if move is None:
                    break
                env.step(move)
                ply += 1

            result  = env.result or "draw"
            new_won = (result == "white") == new_plays_white

            if   result == "draw": draws   += 1
            elif new_won:          wins    += 1
            else:                  losses  += 1

        score = (wins + 0.5 * draws) / self.config.eval_games * 100
        self.log.info(
            f"  Eval vs anchor: {wins}W/{draws}D/{losses}L   score={score:.1f}%"
        )

    # ── Checkpoints ───────────────────────────────────────────────────────────

    def _save_checkpoint(self, iteration: int):
        path = self.config.checkpoint_dir / f"marchon_iter_{iteration:04d}.pt"
        torch.save(
            {
                "iteration":       iteration,
                "model_state":     self.network.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "policy_losses":   self.policy_losses,
                "value_losses":    self.value_losses,
                "elo":             self.elo.rating,
                "total_games":     self.total_games,
            },
            path,
        )
        self.log.info(f"  Checkpoint saved -> {path.name}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.config.device)
        self.network.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.iteration        = ckpt.get("iteration", 0)
        self.policy_losses    = ckpt.get("policy_losses", [])
        self.value_losses     = ckpt.get("value_losses", [])
        self.combined_losses  = [p + v for p, v in zip(self.policy_losses, self.value_losses)]
        self.elo.rating       = ckpt.get("elo", 1000.0)
        self.total_games      = ckpt.get("total_games", 0)

        if self.shared:
            self.shared.update_losses(self.policy_losses, self.value_losses, self.combined_losses)
            self.shared.update_elo(self.elo.rating)
            self.shared.update_iteration(self.iteration, 0)

        self.log.info(f"Loaded checkpoint: iteration {self.iteration}, "
                      f"total_games {self.total_games}, elo {self.elo.rating:.0f}")
