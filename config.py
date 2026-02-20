"""
config.py — Central configuration for Archon.

All tunable hyperparameters live here so you can experiment
without hunting through source files.
"""
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).parent


@dataclass
class ArchonConfig:
    # ── Paths ──────────────────────────────────────────────────────────────────
    checkpoint_dir: Path = BASE_DIR / "checkpoints"
    games_dir:      Path = BASE_DIR / "games"
    log_file:       Path = BASE_DIR / "archon.log"

    # ── Model architecture ─────────────────────────────────────────────────────
    # Increasing these improves strength but slows training.
    # Good starter values for CPU / Raspberry Pi: 6 blocks, 64 channels.
    num_planes:     int = 18    # input feature planes (see chess_env.py)
    num_res_blocks: int = 6     # depth of residual tower
    channels:       int = 64    # conv channel width

    # ── MCTS ───────────────────────────────────────────────────────────────────
    # Reference (full strength): num_simulations=50
    num_simulations:   int   = 25    # simulations per move (higher = stronger)
    c_puct:            float = 1.4   # exploration vs exploitation balance
    dirichlet_alpha:   float = 0.3   # noise concentration at root
    dirichlet_epsilon: float = 0.25  # fraction of noise mixed into root priors
    temperature:       float = 1.0   # move diversity in early game
    temp_threshold:    int   = 20    # plies before switching to greedy selection

    # ── Self-play ──────────────────────────────────────────────────────────────
    # Reference (full strength): games_per_iteration=10, max_game_length=300
    games_per_iteration: int = 8     # self-play games per training cycle
    max_game_length:     int = 200   # cap before adjudicating draw

    # ── Training ───────────────────────────────────────────────────────────────
    # Reference (full strength): train_steps_per_iter=100
    batch_size:              int   = 128
    learning_rate:           float = 2e-3
    weight_decay:            float = 1e-4
    replay_buffer_capacity:  int   = 50_000
    min_buffer_size:         int   = 500    # start training after this many samples
    train_steps_per_iter:    int   = 60     # gradient steps per iteration

    # ── Evaluation & ELO ───────────────────────────────────────────────────────
    checkpoint_every_n_iters: int   = 1    # save model weights every N iterations
    eval_every_n_iters:       int   = 5    # run ELO evaluation games every N iterations
    eval_games:               int   = 10
    elo_k_factor:             float = 32.0

    # ── Device ─────────────────────────────────────────────────────────────────
    # Will auto-detect CUDA if available; falls back to CPU.
    device: str = "cpu"


def build_config() -> ArchonConfig:
    import torch
    cfg = ArchonConfig()
    if torch.cuda.is_available():
        cfg.device = "cuda"
    return cfg


config = build_config()
