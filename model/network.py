"""
network.py — Dual-headed neural network for Marchon.

Architecture:

  Input (batch, 19, 8, 8)
    |
    v
  Input convolution  [conv 3x3 -> BN -> ReLU]
    |
    v
  Residual tower     [num_res_blocks x ResidualBlock]
    |
    v
  MoE tower          [num_moe_blocks x MoEResBlock]
    |
    +---------------------------+
    v                           v
  Policy head               Value head
  conv 1x1 -> BN -> ReLU    conv 1x1 -> BN -> ReLU
  flatten -> Linear         flatten -> Linear(256) -> ReLU -> Linear(1) -> Tanh
    |                           |
    v                           v
  logits (4100,)            scalar in [-1, 1]

MoEResBlock routes each board position (via global avg pool) to the top-k
of num_experts ResidualBlock experts and blends their outputs.  An auxiliary
load-balancing loss encourages uniform expert utilisation.

forward() returns (policy_logits, value, aux_loss).
Callers that only need two values (e.g. predict()) may ignore aux_loss.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """Standard residual block: conv -> BN -> ReLU -> conv -> BN -> (+x) -> ReLU."""

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


class MoEResBlock(nn.Module):
    """
    Mixture-of-Experts residual block.

    A small linear router receives the global-average-pooled feature vector
    and produces per-position expert weights.  The top_k highest-weighted
    experts (each a full ResidualBlock) are computed and blended.

    All num_experts forward passes are always computed so every expert
    receives gradient signal, even those not in the top-k for a given
    position.  This is efficient for small num_experts (4-8) and avoids
    expert collapse.

    Returns
    -------
    out      : (B, C, H, W) -- weighted mixture of expert outputs
    aux_loss : scalar tensor -- load-balancing penalty; add to total loss
               weighted by config.load_balance_coeff
    """

    def __init__(self, channels: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = min(top_k, num_experts)
        self.router  = nn.Linear(channels, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B      = x.shape[0]
        device = x.device

        # Routing via global avg pool
        pooled        = x.mean(dim=(-2, -1))              # (B, C)
        router_logits = self.router(pooled)                # (B, E)
        router_probs  = F.softmax(router_logits, dim=-1)   # (B, E)

        # Load-balance loss: penalises collapsed routing.
        # Equals num_experts when uniform, higher when concentrated.
        avg_probs = router_probs.mean(0)                   # (E,)
        aux_loss  = self.num_experts * (avg_probs * avg_probs).sum()

        # Top-k selection
        topk_w, topk_idx = router_probs.topk(self.top_k, dim=-1)  # (B, k)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)         # renormalise

        # Compute all expert outputs (ensures gradient to all experts)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B,E,C,H,W)

        # Weighted blend of top-k experts
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = topk_idx[:, k]                                    # (B,)
            w   = topk_w[:, k].view(B, 1, 1, 1)                    # (B,1,1,1)
            sel = expert_outs[torch.arange(B, device=device), idx]  # (B,C,H,W)
            out = out + w * sel

        return out, aux_loss


class MarchonNet(nn.Module):
    """
    The brain of Marchon: residual backbone + MoE tower + dual heads.

    Parameters
    ----------
    num_planes    : number of input feature planes (19 in our encoding)
    num_res_blocks: depth of the standard residual tower before MoE
    channels      : conv channel width throughout
    num_moe_blocks: number of MoEResBlock layers after the residual tower
    num_experts   : experts per MoE block
    top_k         : experts blended per position per block
    """

    ACTION_SIZE = 4100  # 64x64 + 4 under-promotions  (matches ChessEnv)

    def __init__(
        self,
        num_planes:     int = 19,
        num_res_blocks: int = 4,
        channels:       int = 128,
        num_moe_blocks: int = 2,
        num_experts:    int = 4,
        top_k:          int = 2,
    ):
        super().__init__()
        self.num_planes = num_planes

        self.input_conv = nn.Sequential(
            nn.Conv2d(num_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.res_tower = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )
        # ModuleList (not Sequential) because each block returns a tuple.
        self.moe_tower = nn.ModuleList(
            [MoEResBlock(channels, num_experts, top_k)
             for _ in range(num_moe_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(32 * 8 * 8, self.ACTION_SIZE)

        # Value head
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

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x : (batch, num_planes, 8, 8)

        Returns
        -------
        policy_logits : (batch, ACTION_SIZE)  -- raw, un-masked
        value         : (batch, 1)            -- tanh output in [-1, 1]
        aux_loss      : scalar tensor         -- MoE load-balancing penalty
        """
        x = self.input_conv(x)
        x = self.res_tower(x)

        total_aux = torch.zeros(1, device=x.device, dtype=x.dtype)
        for block in self.moe_tower:
            x, aux = block(x)
            total_aux = total_aux + aux

        p = self.policy_conv(x).flatten(1)
        policy_logits = self.policy_fc(p)

        v = self.value_conv(x).flatten(1)
        value = self.value_fc(v)

        return policy_logits, value, total_aux

    @torch.no_grad()
    def predict(
        self,
        state:             np.ndarray,
        legal_action_mask: np.ndarray,
        device:            str = "cpu",
    ) -> Tuple[np.ndarray, float]:
        """
        Single-position inference (no gradient).  aux_loss is ignored.

        Parameters
        ----------
        state             : (19, 8, 8) float32 from ChessEnv.encode_state()
        legal_action_mask : bool array of shape (ACTION_SIZE,)
        device            : torch device string

        Returns
        -------
        policy : (ACTION_SIZE,) float32 -- masked & softmax-normalised
        value  : float in [-1, 1]
        """
        self.eval()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        logits, val, _ = self.forward(state_t)

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


# Backwards-compat alias so play.py (which imports ArchonNet) still works
ArchonNet = MarchonNet
