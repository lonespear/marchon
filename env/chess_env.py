"""
chess_env.py — Chess environment for reinforcement learning.

Wraps python-chess into an RL-friendly interface:
  - encode_state()    → float32 tensor the neural network can read
  - step(move)        → (next_state, reward, done)
  - move_to_action()  → integer index into action space
  - render_unicode()  → pretty board string for debugging

Board encoding (18 planes, each 8×8):
  Planes 0-5:  White pieces  (P, N, B, R, Q, K)
  Planes 6-11: Black pieces  (P, N, B, R, Q, K)
  Plane 12:    Side to move  (1 = White)
  Planes 13-16: Castling rights (WK, WQ, BK, BQ)
  Plane 17:    Halfmove clock (normalized 0-1 for 50-move rule)
"""

import chess
import numpy as np
from typing import List, Optional, Tuple


class ChessEnv:
    """RL wrapper around a python-chess Board."""

    NUM_PLANES   = 18
    ACTION_SIZE  = 4100   # 64×64 from/to squares + 4 under-promotions

    _PIECE_TO_PLANE = {
        (chess.PAWN,   chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK,   chess.WHITE): 3,
        (chess.QUEEN,  chess.WHITE): 4,
        (chess.KING,   chess.WHITE): 5,
        (chess.PAWN,   chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK,   chess.BLACK): 9,
        (chess.QUEEN,  chess.BLACK): 10,
        (chess.KING,   chess.BLACK): 11,
    }

    def __init__(self):
        self.board        = chess.Board()
        self.move_history: List[chess.Move] = []
        self.done         = False
        self.result: Optional[str] = None   # 'white' | 'black' | 'draw'

    # ── Core RL interface ──────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self.board        = chess.Board()
        self.move_history = []
        self.done         = False
        self.result       = None
        return self.encode_state()

    def step(self, move: chess.Move) -> Tuple[np.ndarray, float, bool]:
        """
        Apply *move* and return (next_state, reward, done).

        reward is from the perspective of the player who just moved:
          +1 if that player won, -1 if they lost, 0 for draw / ongoing.
        """
        assert not self.done, "Game already over — call reset() first."
        assert move in self.board.legal_moves, f"Illegal move: {move}"

        self.board.push(move)
        self.move_history.append(move)

        reward = 0.0
        if self.board.is_game_over():
            self.done = True
            outcome = self.board.outcome()
            if outcome.winner is None:
                reward      = 0.0
                self.result = "draw"
            elif outcome.winner == chess.WHITE:
                self.result = "white"
                # self.board.turn is now the player whose turn it *would* be
                reward = 1.0 if not self.board.turn else -1.0
            else:
                self.result = "black"
                reward = 1.0 if self.board.turn else -1.0

        return self.encode_state(), reward, self.done

    # ── State encoding ─────────────────────────────────────────────────────────

    def encode_state(self) -> np.ndarray:
        """Return (NUM_PLANES, 8, 8) float32 array."""
        planes = np.zeros((self.NUM_PLANES, 8, 8), dtype=np.float32)

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                rank  = chess.square_rank(square)
                file  = chess.square_file(square)
                plane = self._PIECE_TO_PLANE[(piece.piece_type, piece.color)]
                planes[plane][rank][file] = 1.0

        planes[12] = float(self.board.turn == chess.WHITE)
        planes[13] = float(self.board.has_kingside_castling_rights(chess.WHITE))
        planes[14] = float(self.board.has_queenside_castling_rights(chess.WHITE))
        planes[15] = float(self.board.has_kingside_castling_rights(chess.BLACK))
        planes[16] = float(self.board.has_queenside_castling_rights(chess.BLACK))
        planes[17] = self.board.halfmove_clock / 100.0

        return planes

    # ── Action space ───────────────────────────────────────────────────────────

    def move_to_action(self, move: chess.Move) -> int:
        """
        Encode a Move as an integer in [0, ACTION_SIZE).

        Normal moves: from_square * 64 + to_square  → [0, 4095]
        Under-promotions (R/B/N): 4096 + offset      → [4096, 4098]
        Queen promotions are treated as normal moves (no offset needed).
        """
        if move.promotion and move.promotion != chess.QUEEN:
            offset = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}
            return 4096 + offset.get(move.promotion, 0)
        return move.from_square * 64 + move.to_square

    def legal_moves_with_actions(self) -> List[Tuple[chess.Move, int]]:
        """Return [(move, action_index), ...] for all legal moves."""
        return [(m, self.move_to_action(m)) for m in self.board.legal_moves]

    def get_legal_moves(self) -> List[chess.Move]:
        return list(self.board.legal_moves)

    # ── Rendering ──────────────────────────────────────────────────────────────

    def render_unicode(self) -> str:
        """Return a printable board string (useful in headless / debug mode)."""
        symbols = {
            "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
            "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
        }
        rows = ["  a b c d e f g h"]
        for rank in range(7, -1, -1):
            row = f"{rank + 1} "
            for file in range(8):
                sq    = chess.square(file, rank)
                piece = self.board.piece_at(sq)
                row  += (symbols.get(piece.symbol(), "?") if piece else "·") + " "
            rows.append(row + f"{rank + 1}")
        rows.append("  a b c d e f g h")
        return "\n".join(rows)
