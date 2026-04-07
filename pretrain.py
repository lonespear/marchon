"""
pretrain.py — Supervised pre-training of MarchonNet on historical PGN games.

Usage
-----
  # Basic (plain PGN, 1 epoch)
  python pretrain.py --pgn data/games.pgn --output checkpoints/pretrained.pt

  # Larger run with AMP and progress bar
  python pretrain.py --pgn data/lichess.pgn --epochs 2 --batch-size 512 \\
      --output checkpoints/pretrained.pt

  # Limit positions (quick smoke-test)
  python pretrain.py --pgn data/games.pgn --max-positions 100000 \\
      --output checkpoints/pretrained.pt

Why pre-train?
--------------
AlphaZero from random weights takes millions of self-play games to learn
not to hang pieces.  Pre-training on human games gives the network a warm
prior so MCTS self-play starts from a strategically reasonable baseline.

The checkpoint saved here is compatible with main.py --checkpoint, so you
can immediately resume self-play fine-tuning after pre-training.

For .pgn.zst files (Lichess open database):
  zstd -d lichess_db_standard_rated_YYYY-MM.pgn.zst
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import config
from model.network import MarchonNet
from utils.pgn_dataset import PGNIterableDataset


def main():
    parser = argparse.ArgumentParser(
        description="Marchon supervised pre-training on PGN games"
    )
    parser.add_argument(
        "--pgn", required=True, nargs="+",
        help="Path(s) to .pgn file(s)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of passes over the dataset (default: 1)",
    )
    parser.add_argument(
        "--output", type=str, default="checkpoints/pretrained.pt",
        help="Output checkpoint path (default: checkpoints/pretrained.pt)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="Mini-batch size (default: 512)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--max-positions", type=int, default=None,
        help="Stop after this many positions per epoch (default: all)",
    )
    parser.add_argument(
        "--skip-plies", type=int, default=0,
        help="Skip the first N plies of each game (default: 0)",
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="DataLoader worker processes (default: 2)",
    )
    args = parser.parse_args()

    # ── Logging ───────────────────────────────────────────────────────────────
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        handlers= [logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("Marchon.Pretrain")

    cfg = config

    # ── Network ───────────────────────────────────────────────────────────────
    network = MarchonNet(
        num_planes     = cfg.num_planes,
        num_res_blocks = cfg.num_res_blocks,
        channels       = cfg.channels,
        num_moe_blocks = cfg.num_moe_blocks,
        num_experts    = cfg.num_experts,
        top_k          = cfg.top_k,
    ).to(cfg.device)

    log.info(f"MarchonNet: {network.count_parameters():,} parameters on {cfg.device}")

    if cfg.use_compile:
        try:
            network = torch.compile(network)
            log.info("torch.compile() applied.")
        except Exception as exc:
            log.warning(f"torch.compile() unavailable: {exc}")

    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=1e-4)
    use_amp   = cfg.use_amp and cfg.device == "cuda"
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Output dir ────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        log.info(f"Epoch {epoch}/{args.epochs}")

        dataset = PGNIterableDataset(
            pgn_paths     = args.pgn,
            max_positions = args.max_positions,
            skip_plies    = args.skip_plies,
        )
        loader = DataLoader(
            dataset,
            batch_size  = args.batch_size,
            num_workers = args.workers,
            pin_memory  = (cfg.device == "cuda"),
        )

        network.train()
        total_p = total_v = total_batches = 0

        for batch_idx, (states, policies, values) in enumerate(loader):
            s = states.to(cfg.device)
            p = policies.to(cfg.device)
            v = values.unsqueeze(1).to(cfg.device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    policy_logits, value_pred, aux_loss = network(s)
                    log_p  = F.log_softmax(policy_logits, dim=1)
                    p_loss = -(p * log_p).sum(dim=1).mean()
                    v_loss = F.mse_loss(value_pred, v)
                    loss   = p_loss + v_loss + cfg.load_balance_coeff * aux_loss
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                policy_logits, value_pred, aux_loss = network(s)
                log_p  = F.log_softmax(policy_logits, dim=1)
                p_loss = -(p * log_p).sum(dim=1).mean()
                v_loss = F.mse_loss(value_pred, v)
                loss   = p_loss + v_loss + cfg.load_balance_coeff * aux_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()

            total_p       += p_loss.item()
            total_v       += v_loss.item()
            total_batches += 1

            if (batch_idx + 1) % 500 == 0:
                log.info(
                    f"  batch {batch_idx+1:6d}  "
                    f"policy={total_p/total_batches:.4f}  "
                    f"value={total_v/total_batches:.4f}"
                )

        if total_batches:
            log.info(
                f"Epoch {epoch} done — "
                f"avg policy={total_p/total_batches:.4f}  "
                f"avg value={total_v/total_batches:.4f}  "
                f"positions={total_batches * args.batch_size:,}"
            )

    # ── Save checkpoint ───────────────────────────────────────────────────────
    # torch.compile() prefixes all keys with "_orig_mod." — strip it so the
    # checkpoint is loadable by an uncompiled network in the trainer.
    network.eval()
    raw_state = network.state_dict()
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
    torch.save(
        {
            "iteration":       0,
            "model_state":     clean_state,
            "optimizer_state": optimizer.state_dict(),
            "policy_losses":   [],
            "value_losses":    [],
            "elo":             1000.0,
            "total_games":     0,
        },
        out_path,
    )
    log.info(f"Checkpoint saved -> {out_path}")


if __name__ == "__main__":
    main()
