"""
network.py — Dual-headed neural network for Archon.

Architecture (simplified AlphaZero):

  Input (batch, 19, 8, 8)
    │
    ▼
  Input convolution  [conv 3×3 → BN → ReLU]
    │
    ▼
  Residual tower     [N × ResidualBlock]
    │
    ├──────────────────────┐
    ▼                      ▼
  Policy head            Value head
  conv 1×1 → BN → ReLU  conv 1×1 → BN → ReLU
  flatten → Linear       flatten → Linear(256) → ReLU → Linear(1) → Tanh
    │                      │
    ▼                      ▼
  logits (4100,)         scalar in [-1, 1]

Teaching notes:
  - Policy head outputs *un-normalised* logits; apply masked softmax at
    inference time so illegal moves get zero probability.
  - Value head outputs a scalar in [-1, 1] estimating the position's
    outcome from the current player's perspective (+1 = win, -1 = loss).
  - Residual connections let gradients flow easily through the deep tower,
    which is why deep networks became practical after He et al. (2015).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """
    Standard pre-activation residual block:
        x → conv → BN → ReLU → conv → BN → (+x) → ReLU
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ArchonNet(nn.Module):
    """
    The brain of Archon.

    Parameters
    ----------
    num_planes    : number of input feature planes (19 in our encoding)
    num_res_blocks: depth of the residual tower
    channels      : width of every conv layer in the tower
    """

    ACTION_SIZE = 4100  # 64×64 + 4 under-promotions  (matches ChessEnv)

    def __init__(
        self,
        num_planes:     int = 19,
        num_res_blocks: int = 6,
        channels:       int = 64,
    ):
        super().__init__()
        self.num_planes = num_planes

        # ── Shared body ───────────────────────────────────────────────────────
        self.input_conv = nn.Sequential(
            nn.Conv2d(num_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        # ── Policy head ───────────────────────────────────────────────────────
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(32 * 8 * 8, self.ACTION_SIZE)

        # ── Value head ────────────────────────────────────────────────────────
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, num_planes, 8, 8)

        Returns
        -------
        policy_logits : (batch, ACTION_SIZE)  — raw, un-masked
        value         : (batch, 1)            — tanh output in [-1, 1]
        """
        x = self.input_conv(x)
        x = self.tower(x)

        # Policy head
        p = self.policy_conv(x).flatten(1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = self.value_conv(x).flatten(1)
        value = self.value_fc(v)

        return policy_logits, value

    # ── Inference helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        state:             np.ndarray,
        legal_action_mask: np.ndarray,
        device:            str = "cpu",
    ) -> Tuple[np.ndarray, float]:
        """
        Single-position inference (no gradient).

        Parameters
        ----------
        state             : (19, 8, 8) float32 array from ChessEnv.encode_state()
        legal_action_mask : bool array of shape (ACTION_SIZE,); True = legal
        device            : torch device string

        Returns
        -------
        policy : (ACTION_SIZE,) float32 — masked & softmax-normalised
        value  : float in [-1, 1]
        """
        self.eval()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        logits, val = self.forward(state_t)

        # Zero-out illegal moves before softmax
        mask_t = torch.BoolTensor(legal_action_mask).to(device)
        logits[0][~mask_t] = float("-inf")
        policy = F.softmax(logits[0], dim=0).cpu().numpy()

        return policy, float(val.item())

    def get_legal_mask(self, env) -> np.ndarray:
        """Build a boolean mask over ACTION_SIZE for all legal moves in *env*."""
        mask = np.zeros(self.ACTION_SIZE, dtype=bool)
        for move, action in env.legal_moves_with_actions():
            if action < self.ACTION_SIZE:
                mask[action] = True
        return mask

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
