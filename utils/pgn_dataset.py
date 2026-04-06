"""
pgn_dataset.py — PyTorch IterableDataset that streams chess positions from PGN files.

Used by pretrain.py to bootstrap MarchonNet on human games before self-play.

Each yielded sample: (state, policy, value)
  state  : (19, 8, 8) float32 board encoding
  policy : (4100,) float32 one-hot on the actual move played
  value  : float32 game outcome from the moving player's perspective

Draw value is -0.3 (same as self-play) to avoid the zero-attractor failure mode.
"""

import chess
import chess.pgn
import numpy as np
from torch.utils.data import IterableDataset
from typing import Iterator, List, Union

from env.chess_env import ChessEnv


class PGNIterableDataset(IterableDataset):
    """
    Streams (state, policy, value) tuples from one or more PGN files.

    Parameters
    ----------
    pgn_paths     : path or list of paths to .pgn files
    max_positions : stop after yielding this many positions (None = all)
    skip_plies    : skip the first N plies of each game (0 = include all)

    Notes
    -----
    - Supports plain .pgn files.  For .pgn.zst (Lichess), decompress first:
        zstd -d lichess_db_standard.pgn.zst
    - For DataLoader with num_workers > 1, each worker streams the full file
      independently.  Use worker_init_fn or set num_workers=1 if you want
      exact position counts.
    """

    def __init__(
        self,
        pgn_paths:     Union[str, List[str]],
        max_positions: int = None,
        skip_plies:    int = 0,
    ):
        if isinstance(pgn_paths, str):
            pgn_paths = [pgn_paths]
        self.pgn_paths     = pgn_paths
        self.max_positions = max_positions
        self.skip_plies    = skip_plies

    def __iter__(self) -> Iterator:
        count = 0
        for pgn_path in self.pgn_paths:
            with open(pgn_path, encoding="utf-8", errors="ignore") as fh:
                while True:
                    game = chess.pgn.read_game(fh)
                    if game is None:
                        break

                    result_tag = game.headers.get("Result", "*")
                    if result_tag not in ("1-0", "0-1", "1/2-1/2"):
                        continue

                    # White's value for each position
                    if result_tag == "1-0":
                        white_val = 1.0
                    elif result_tag == "0-1":
                        white_val = -1.0
                    else:
                        white_val = -0.3  # draw penalty, same as self-play

                    env = ChessEnv()
                    env.reset()
                    ply = 0

                    for move in game.mainline_moves():
                        if ply >= self.skip_plies:
                            state  = env.encode_state()
                            action = env.move_to_action(move)

                            if action < ChessEnv.ACTION_SIZE:
                                policy         = np.zeros(ChessEnv.ACTION_SIZE, dtype=np.float32)
                                policy[action] = 1.0

                                player_is_white = (ply % 2 == 0)
                                if result_tag == "1/2-1/2":
                                    value = -0.3
                                elif player_is_white:
                                    value = white_val
                                else:
                                    value = -white_val

                                yield state, policy, np.float32(value)
                                count += 1

                                if self.max_positions and count >= self.max_positions:
                                    return

                        # Advance environment (must happen even when skipping)
                        if move in env.board.legal_moves:
                            env.step(move)
                        ply += 1
