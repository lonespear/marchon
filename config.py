"""
config.py — Central configuration for Marchon.

All tunable hyperparameters live here so you can experiment
without hunting through source files.
"""
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).parent


@dataclass
class MarchonConfig:
    # ── Paths ──────────────────────────────────────────────────────────────────
    checkpoint_dir: Path = BASE_DIR / "checkpoints"
    games_dir:      Path = BASE_DIR / "games"
    log_file:       Path = BASE_DIR / "marchon.log"

    # ── Model architecture ─────────────────────────────────────────────────────
    # Backbone: 4 standard ResBlocks + num_moe_blocks MoE blocks.
    # Channels raised to 128 for better GPU utilisation vs Archon's 64.
    num_planes:     int = 19     # input feature planes (see chess_env.py)
    num_res_blocks: int = 4      # standard residual blocks before MoE tower
    channels:       int = 128    # conv channel width throughout

    # ── MoE settings ──────────────────────────────────────────────────────────
    # Each MoEResBlock has num_experts expert ResBlocks routed by a small
    # linear network.  top_k experts are blended per position.
    num_moe_blocks:     int   = 2     # MoE blocks appended after res_tower
    num_experts:        int   = 4     # experts per MoE block
    top_k:              int   = 2     # experts activated per forward pass
    load_balance_coeff: float = 0.01  # weight on expert load-balancing aux loss

    # ── MCTS ───────────────────────────────────────────────────────────────────
    num_simulations:   int   = 100
    c_puct:            float = 1.4
    dirichlet_alpha:   float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature:       float = 1.0
    temp_threshold:    int   = 120

    # ── Self-play ──────────────────────────────────────────────────────────────
    games_per_iteration: int = 16    # raised from 10 — more GPU parallelism
    max_game_length:     int = 200

    # ── Training ───────────────────────────────────────────────────────────────
    batch_size:              int   = 256       # raised from 128 — better GPU utilisation
    learning_rate:           float = 2e-3
    weight_decay:            float = 1e-4
    replay_buffer_capacity:  int   = 100_000  # raised from 50k
    min_buffer_size:         int   = 500
    train_steps_per_iter:    int   = 80        # raised from 60

    # ── Speed ──────────────────────────────────────────────────────────────────
    use_amp:     bool = True    # torch AMP mixed precision (CUDA only)
    use_compile: bool = False  # torch.compile() disabled — deadlocks with spawn workers

    # ── Evaluation & ELO ───────────────────────────────────────────────────────
    checkpoint_every_n_iters: int   = 50
    eval_every_n_iters:       int   = 5
    eval_games:               int   = 20
    elo_k_factor:             float = 32.0

    # ── Anchor evaluation ──────────────────────────────────────────────────────
    anchor_checkpoint:         str = ""
    anchor_eval_every_n_iters: int = 250

    # ── Device ─────────────────────────────────────────────────────────────────
    device: str = "cpu"


# Keep the old name as an alias so play.py imports don't break
ArchonConfig = MarchonConfig


def build_config() -> MarchonConfig:
    import torch
    cfg = MarchonConfig()
    if torch.cuda.is_available():
        cfg.device = "cuda"
    return cfg


config = build_config()
