"""
self_play.py — Generate training data by having Archon play against itself.

Each game produces a list of Experience objects:
  (board_state, mcts_policy, game_outcome)

The MCTS policy is the "improved" policy π — it knows more than the raw
neural network output because it has searched N positions ahead.  Training
the network to predict π causes it to improve over time.

The game outcome (value target) is the final result propagated backwards:
  - The player who won gets +1 for every position they were to move from.
  - The player who lost gets -1.
  - Draw gives 0 for both sides.
"""

import chess
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from env.chess_env import ChessEnv
from mcts.mcts import MCTS
from utils.replay_buffer import Experience


@dataclass
class GameRecord:
    """Everything we know about a completed self-play game."""
    game_id:    int
    moves:      List[str]           # SAN notation
    result:     str                 # 'white' | 'black' | 'draw'
    num_moves:  int
    experiences: List[Experience]   # training samples
    pgn:        str = ""


class SelfPlay:
    """
    Runs full self-play games using MCTS + ArchonNet.

    One SelfPlay instance is reused across many games; the MCTS tree is
    rebuilt from scratch for each game (no tree reuse for simplicity).
    """

    def __init__(self, network, config, shared_state=None):
        self.network       = network
        self.config        = config
        self.mcts          = MCTS(network, config)
        self._game_counter = 0
        self.shared        = shared_state

    def play_game(self) -> GameRecord:
        """
        Play one complete game via self-play and return a GameRecord.

        Temperature schedule:
          - First *temp_threshold* plies: temperature=1.0  (diverse exploration)
          - After that: temperature=0  (greedy / strongest play)
        """
        self._game_counter += 1
        env   = ChessEnv()
        state = env.reset()

        # We'll accumulate (state, full_policy_vector) per ply
        positions: List[Tuple[np.ndarray, np.ndarray]] = []

        # Live board mirrors env.board for move-by-move display
        live_board = chess.Board()
        live_san: List[str] = []

        ply = 0
        while not env.done and ply < self.config.max_game_length:
            temperature = (
                self.config.temperature
                if ply < self.config.temp_threshold
                else 0.0
            )
            add_noise = ply < self.config.temp_threshold   # noise throughout the exploratory phase

            policy_dist, legal_moves = self.mcts.search(
                env.board, add_noise=add_noise
            )

            # Build the full ACTION_SIZE policy vector
            full_policy = np.zeros(env.ACTION_SIZE, dtype=np.float32)
            for prob, move in zip(policy_dist, legal_moves):
                action = env.move_to_action(move)
                if action < env.ACTION_SIZE:
                    full_policy[action] = prob

            positions.append((state.copy(), full_policy))

            # Sample or pick the move
            if temperature < 1e-6:
                chosen = legal_moves[int(np.argmax(policy_dist))]
            else:
                probs  = policy_dist ** (1.0 / temperature)
                probs /= probs.sum()
                chosen = np.random.choice(legal_moves, p=probs)  # type: ignore

            try:
                move_san = live_board.san(chosen)
            except Exception:
                move_san = str(chosen)
            live_board.push(chosen)
            live_san.append(move_san)
            if self.shared:
                self.shared.update_live_board(live_board, list(live_san), last_move=chosen)

            state, _, _ = env.step(chosen)
            ply += 1

        result = env.result or "draw"

        # Reconstruct SAN move list for display / PGN
        san_moves = _moves_to_san(env.move_history)

        # Assign outcome values from each player's perspective
        experiences = _assign_values(positions, result)

        pgn = _build_pgn(env.move_history, result, self._game_counter)

        return GameRecord(
            game_id=self._game_counter,
            moves=san_moves,
            result=result,
            num_moves=ply,
            experiences=experiences,
            pgn=pgn,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _moves_to_san(move_history: List[chess.Move]) -> List[str]:
    board = chess.Board()
    san   = []
    for move in move_history:
        try:
            san.append(board.san(move))
            board.push(move)
        except Exception:
            san.append(str(move))
    return san


def _assign_values(
    positions: List[Tuple[np.ndarray, np.ndarray]],
    result:    str,
) -> List[Experience]:
    """
    Turn each (state, policy) into an Experience with the correct value target.

    White moves on even plies (0, 2, 4, ...).
    """
    experiences = []
    for i, (state, policy) in enumerate(positions):
        player_is_white = (i % 2 == 0)
        if result == "draw":
            value = -0.1
        elif result == "white":
            value = 1.0 if player_is_white else -1.0
        else:   # black wins
            value = -1.0 if player_is_white else 1.0
        experiences.append(Experience(state=state, policy=policy, value=value))
    return experiences


def _build_pgn(
    move_history: List[chess.Move],
    result:       str,
    game_id:      int,
) -> str:
    result_str = {"white": "1-0", "black": "0-1", "draw": "1/2-1/2"}.get(result, "*")
    board  = chess.Board()
    tokens = [
        f'[Event "Archon Self-Play"]',
        f'[Game "{game_id}"]',
        f'[Result "{result_str}"]',
        "",
    ]
    move_tokens = []
    for i, move in enumerate(move_history):
        if i % 2 == 0:
            move_tokens.append(f"{i // 2 + 1}.")
        try:
            move_tokens.append(board.san(move))
            board.push(move)
        except Exception:
            move_tokens.append(str(move))
    move_tokens.append(result_str)
    tokens.append(" ".join(move_tokens))
    return "\n".join(tokens)
