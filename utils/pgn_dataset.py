"""
pgn_dataset.py — PyTorch IterableDataset that streams chess positions from PGN files.

Used by pretrain.py to bootstrap MarchonNet on human games before self-play.

Supports both plain .pgn and Lichess-style .pgn.zst files.
.pgn.zst is streamed directly — the file is never fully decompressed to disk.

Each yielded sample: (state, policy, value)
  state  : (19, 8, 8) float32 board encoding
  policy : (4100,) float32 one-hot on the actual move played
  value  : float32 game outcome from the moving player's perspective

Draw value is -0.3 (same as self-play) to avoid the zero-attractor failure mode.
"""

import io
import chess
import chess.pgn
import numpy as np
from pathlib import Path
from torch.utils.data import IterableDataset
from typing import Iterator, List, Union

from env.chess_env import ChessEnv


def _open_pgn(path: str):
    """
    Return a text-mode file handle for a .pgn or .pgn.zst file.
    For .zst the stream is decompressed on-the-fly — no temp file written.
    Requires: pip install zstandard
    """
    p = Path(path)
    if p.suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError:
            raise ImportError(
                "zstandard package required for .pgn.zst files.\n"
                "  pip install zstandard"
            )
        cctx   = zstd.ZstdDecompressor()
        raw_fh = open(path, "rb")
        stream = cctx.stream_reader(raw_fh)
        return io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")
    else:
        return open(path, encoding="utf-8", errors="ignore")


class PGNIterableDataset(IterableDataset):
    """
    Streams (state, policy, value) tuples from one or more PGN / PGN.ZST files.

    Parameters
    ----------
    pgn_paths     : path or list of paths (.pgn or .pgn.zst)
    max_positions : stop after yielding this many positions (None = stream forever)
    skip_plies    : skip the first N plies of each game (0 = include all)

    Notes
    -----
    - With .pgn.zst the file is never written to disk uncompressed.
    - Use --max-positions in pretrain.py to cap training; 2-5M positions is
      enough to bootstrap past random-weight behaviour before self-play.
    - num_workers > 1 in DataLoader causes each worker to read the full file
      independently, so set num_workers=1 (or 0) for exact position counts.
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
            with _open_pgn(pgn_path) as fh:
                while True:
                    game = chess.pgn.read_game(fh)
                    if game is None:
                        break

                    result_tag = game.headers.get("Result", "*")
                    if result_tag not in ("1-0", "0-1", "1/2-1/2"):
                        continue

                    if result_tag == "1-0":
                        white_val = 1.0
                    elif result_tag == "0-1":
                        white_val = -1.0
                    else:
                        white_val = -0.3  # draw penalty matches self-play

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

                        if move in env.board.legal_moves:
                            env.step(move)
                        ply += 1
