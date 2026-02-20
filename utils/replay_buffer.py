"""
replay_buffer.py — Experience replay for Archon training.

Why replay buffers matter in RL:
  - Training on a single game's positions is inefficient (highly correlated).
  - By storing a large pool of past experiences and sampling randomly, we
    break temporal correlations and get more stable gradient estimates.
  - This is the same trick used in DQN (DeepMind, 2015).

An Experience stores:
  state  : the board encoding at the time
  policy : the MCTS visit distribution (our policy target)
  value  : the game outcome from *this player's* perspective (+1/-1/0)
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Experience:
    """One training sample produced by a single board position in self-play."""
    state:  np.ndarray   # (18, 8, 8)   board encoding
    policy: np.ndarray   # (4100,)      MCTS visit distribution
    value:  float        # game outcome from the moving player's perspective


class ReplayBuffer:
    """
    Fixed-capacity circular buffer.

    Once full, the oldest experiences are silently discarded to make
    room for new ones (deque with maxlen).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: deque = deque(maxlen=capacity)

    # ── Writes ─────────────────────────────────────────────────────────────────

    def push(self, experiences: List[Experience]):
        """Add a batch of experiences (typically one full game)."""
        self._buf.extend(experiences)

    # ── Reads ──────────────────────────────────────────────────────────────────

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample *batch_size* experiences uniformly at random.

        Returns
        -------
        states   : (B, 18, 8, 8)
        policies : (B, 4100)
        values   : (B,)
        """
        batch    = random.sample(self._buf, min(batch_size, len(self._buf)))
        states   = np.stack([e.state  for e in batch])
        policies = np.stack([e.policy for e in batch])
        values   = np.array([e.value  for e in batch], dtype=np.float32)
        return states, policies, values

    # ── Metadata ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def fill_fraction(self) -> float:
        return len(self._buf) / self.capacity
