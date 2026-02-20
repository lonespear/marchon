"""
elo.py — ELO rating tracker for Archon.

ELO basics:
  - Each player has a rating R.  The expected score against an opponent
    with rating R_opp is:  E = 1 / (1 + 10^((R_opp - R) / 400))
  - After a game: R_new = R_old + K * (score - E)
    where score = 1 (win), 0.5 (draw), 0 (loss)  and K controls sensitivity.

In self-play we treat each new network version as a separate "opponent."
We start at a nominal 1000 and adjust based on win rate vs. the previous
checkpoint in evaluation games.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ELOTracker:
    """
    Running ELO estimate for a single player (the current Archon model).

    Attributes
    ----------
    rating   : current ELO estimate
    k_factor : controls how much each batch of games moves the rating
    history  : list of all past ratings (for plotting the sparkline)
    """

    rating:   float = 1000.0
    k_factor: float = 32.0
    history:  List[float] = field(default_factory=list)

    def __post_init__(self):
        self.history.append(self.rating)

    # ── Single update ──────────────────────────────────────────────────────────

    def update(self, score: float, opponent_rating: float) -> float:
        """
        Update rating from a single game result.

        Parameters
        ----------
        score            : 1.0 = win, 0.5 = draw, 0.0 = loss
        opponent_rating  : ELO of the opponent
        """
        expected      = self._expected(opponent_rating)
        self.rating  += self.k_factor * (score - expected)
        self.history.append(self.rating)
        return self.rating

    # ── Batch update (more common in self-play eval) ───────────────────────────

    def update_from_match(
        self,
        wins:             int,
        draws:            int,
        losses:           int,
        opponent_rating:  float,
    ) -> float:
        """
        Update once from a multi-game evaluation match.
        Treats the whole match as a single ELO update using aggregate score.
        """
        total = wins + draws + losses
        if total == 0:
            return self.rating
        score = (wins + 0.5 * draws) / total
        return self.update(score, opponent_rating)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _expected(self, opponent_rating: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((opponent_rating - self.rating) / 400.0))

    @property
    def delta(self) -> float:
        """Change since the previous update (0 if no updates yet)."""
        if len(self.history) < 2:
            return 0.0
        return self.history[-1] - self.history[-2]
