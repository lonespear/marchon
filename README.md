# Marchon — MoE Chess Reinforcement Learning

Marchon is a from-scratch implementation of [AlphaZero](https://www.nature.com/articles/nature24270)-style
self-play reinforcement learning applied to chess, extended with a **Mixture-of-Experts (MoE) residual
tower**. A dual-headed neural network learns through self-play — guided by Monte Carlo Tree Search (MCTS) —
and optionally warm-starts from supervised pre-training on human games.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Board Representation](#board-representation)
3. [Neural Network Architecture](#neural-network-architecture)
4. [Mixture of Experts](#mixture-of-experts)
5. [Monte Carlo Tree Search](#monte-carlo-tree-search)
6. [Self-Play & Training Data](#self-play--training-data)
7. [Learning — The Math](#learning--the-math)
8. [The Full Training Loop](#the-full-training-loop)
9. [Pre-Training on Human Games](#pre-training-on-human-games)
10. [Hyperparameters](#hyperparameters)
11. [Project Layout](#project-layout)

---

## Quick Start

```bash
pip install -r requirements.txt

# (Optional) Pre-train on Lichess PGN data before self-play
python pretrain.py --pgn data/lichess.pgn --output checkpoints/pretrained.pt

# Train headless, logs to marchon.log
python main.py --headless --iterations 500

# Resume from a checkpoint
python main.py --headless --checkpoint checkpoints/marchon_iter_0050.pt

# Train with live web dashboard at http://localhost:5000
python main.py

# Play against a saved checkpoint in your browser (port 5001)
python play.py --checkpoint checkpoints/marchon_iter_0050.pt --sims 20
```

---

## Board Representation

The network reads the board as a **3-D tensor of shape `(19, 8, 8)`** — 19 binary or
normalised feature planes, each covering all 64 squares.

| Planes | Content |
|--------|---------|
| 0 – 5  | White pieces: P, N, B, R, Q, K (1 where piece present) |
| 6 – 11 | Black pieces: P, N, B, R, Q, K |
| 12     | Side to move (1.0 = White, 0.0 = Black) |
| 13     | White kingside castling rights |
| 14     | White queenside castling rights |
| 15     | Black kingside castling rights |
| 16     | Black queenside castling rights |
| 17     | Half-move clock ÷ 50 (0 → 1 as 50-move draw approaches) |
| 18     | Repetition count — 0.5 if seen once before, 1.0 if seen twice (draw) |

**Action space:** 4 100 discrete actions — 64 × 64 = 4 096 (from-square, to-square) pairs
for normal moves, plus 4 under-promotion choices (R, B, N). Queen promotions are encoded
as normal moves.

---

## Neural Network Architecture

MarchonNet is a **dual-headed network** — a standard residual tower followed by a
Mixture-of-Experts tower, then split into a policy head and a value head.

```
Input  (batch, 19, 8, 8)
  │
  ▼
Input conv   Conv2d(19→128, 3×3)  →  BatchNorm  →  ReLU
  │
  ▼
Residual Tower  ×4 blocks
  │  Each block:
  │    Conv2d(128→128, 3×3) → BN → ReLU
  │    Conv2d(128→128, 3×3) → BN
  │    skip-add → ReLU
  │
  ▼
MoE Tower  ×2 blocks
  │  Each MoEResBlock:
  │    Global avg-pool  →  Router (Linear 128→4)  →  Softmax
  │    Top-2 experts selected per position
  │    All 4 experts computed (ensures gradients reach every expert)
  │    Weighted blend of top-2 outputs returned
  │    + Load-balance aux loss
  │
  ├─────────────────────────┐
  ▼                         ▼
Policy head               Value head
Conv2d(128→32, 1×1)       Conv2d(128→1, 1×1)
BN → ReLU                 BN → ReLU
Flatten (32×8×8 = 2048)   Flatten (1×8×8 = 64)
Linear(2048→4100)         Linear(64→256) → ReLU → Linear(256→1) → Tanh
  │                         │
  ▼                         ▼
logits (4100,)            scalar ∈ [-1, 1]
```

**11 990 343 parameters** (4 residual blocks + 2 MoE blocks, 128 channels).

- The **policy head** outputs raw logits over all 4 100 actions. At inference, illegal
  moves are masked to −∞ before softmax.
- The **value head** outputs a scalar $v \in [-1, 1]$ estimating the game outcome from
  the perspective of the player to move (+1 = win, −1 = loss).
- The **aux loss** from each MoE block is summed and added to the total loss weighted
  by `load_balance_coeff` (0.01).

---

## Mixture of Experts

The MoE tower replaces the final residual blocks with **expert-routed** residual blocks.
Each `MoEResBlock` contains `num_experts=4` full `ResidualBlock` experts and a small
linear router.

### How routing works (per forward pass)

```
1. Global average pool:  (B, 128, 8, 8)  →  (B, 128)
2. Router linear:        (B, 128)         →  (B, 4)   logits
3. Softmax:              (B, 4)           →  (B, 4)   probabilities
4. Top-k selection:      pick top 2 experts per position in the batch
5. Re-normalise weights: w_1 + w_2 = 1
6. All 4 experts run forward (guarantees gradient to every expert)
7. Weighted blend:       out = w_1 * expert_1(x) + w_2 * expert_2(x)
```

### Why run all experts if only top-k are used?

If only top-k experts ran, the bottom experts would receive zero gradient and stagnate —
a problem called **expert collapse**. By computing all expert outputs but only blending
the top-k, every expert participates in the backward pass and continues to improve.

### Load-balance loss

Without a penalty, the router collapses: it learns to always send all positions to the
same 1–2 experts, wasting capacity. The auxiliary load-balance loss penalises this:

$$\mathcal{L}_{\text{aux}} = E \cdot \sum_{e=1}^{E} \bar{p}_e^2$$

where $\bar{p}_e$ is the mean routing probability for expert $e$ over the batch, and
$E = 4$ is the number of experts. When routing is perfectly uniform, $\bar{p}_e = 1/E$
and $\mathcal{L}_{\text{aux}} = 1.0$. When collapsed to one expert, it reaches $E = 4.0$.
The loss is minimised when experts share load equally.

This is added to the main loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{value}} + 0.01 \cdot \mathcal{L}_{\text{aux}}$

### Why MoE for chess?

Chess positions fall into distinct regimes — openings, middlegames, endgames, tactical
vs. positional play. A single monolithic residual tower must represent all of these
with shared weights. MoE allows the network to route different position types to
specialised experts, potentially learning a richer representation with similar
parameter efficiency.

---

## Monte Carlo Tree Search

MCTS is the **planning algorithm** that turns a weak raw network into a strong move
selector. Each call to `mcts.search()` runs `num_simulations=100` simulations:

```
for each simulation (×100 per move):
  ┌─ SELECT   Walk down the tree, always choosing the child with the
  │           highest PUCT score, until reaching an unexpanded leaf.
  │
  ├─ EXPAND   Call the neural network on the leaf position.
  │           Create child nodes for every legal move, storing the
  │           network's policy output p(a|s) as the prior probability.
  │
  ├─ EVALUATE Use the network's value head output as the position
  │           estimate (no random rollout needed).
  │
  └─ BACKUP   Walk back to the root, adding the value to every
              ancestor's total and incrementing visit counts.
              The sign flips at each ply (zero-sum perspective).
```

### PUCT Score

$$\text{PUCT}(s, a) = Q(s,a) + c_{\text{puct}} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

| Symbol | Meaning |
|--------|---------|
| $Q(s,a) = W(s,a)/N(s,a)$ | Mean action-value over all simulations through this edge |
| $P(s,a)$ | Network prior probability for action $a$ |
| $N(s)$ | Total visit count of the parent node |
| $N(s,a)$ | Visit count of this child |
| $c_{\text{puct}}$ | Exploration constant (1.4) |

### Dirichlet Noise

Added to root priors during self-play to guarantee exploration:

$$P'(s,a) = (1 - \varepsilon)\,P(s,a) + \varepsilon\,\eta_a, \quad \eta \sim \text{Dir}(\alpha)$$

with $\alpha = 0.3$, $\varepsilon = 0.25$. Applied for the first `temp_threshold=120` plies.

### Policy Extraction

After all simulations, the improved policy is derived from visit counts:

$$\pi(a \mid s) = \frac{N(s,a)^{1/\tau}}{\sum_{a'} N(s,a')^{1/\tau}}$$

$\tau = 1$ for the first 120 plies (diverse exploration), $\tau \to 0$ (greedy) after.

---

## Self-Play & Training Data

One complete self-play game produces:

$$\mathcal{D}_{\text{game}} = \{(s_t,\, \pi_t,\, z_t)\}_{t=0}^{T}$$

- $s_t$ — board state tensor `(19, 8, 8)` at ply $t$
- $\pi_t$ — MCTS visit-count policy at ply $t$
- $z_t$ — game outcome from the perspective of the player to move at ply $t$:
  $+1$ (won), $-1$ (lost), $-0.3$ (draw)

The draw value of $-0.3$ (not $0$) is deliberate: scoring draws as $0$ lets the value
head find a trivial minimum by outputting a constant near zero, killing the learning
signal. At $-0.3$, both sides have a weak incentive to play for a win, and the value
head always has a non-zero target.

Experiences are pushed into a **replay buffer** (capacity 100 000). Mini-batches are
sampled uniformly, breaking temporal correlations and stabilising training.

---

## Learning — The Math

### Loss Function

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{value}} + 0.01 \cdot \mathcal{L}_{\text{aux}}$$

**Policy loss — cross-entropy:**

$$\mathcal{L}_{\text{policy}} = -\sum_{a} \pi(a \mid s)\,\log p_\theta(a \mid s)$$

Minimises KL divergence between the MCTS-improved policy $\pi$ and the network's raw
output. This causes the network to internalise MCTS look-ahead, so it needs less search
to reach the same quality over time.

**Value loss — MSE:**

$$\mathcal{L}_{\text{value}} = \bigl(z - v_\theta(s)\bigr)^2$$

**Gradient update** (Adam, lr=2×10⁻³, weight decay 10⁻⁴, gradient clip 1.0):

$$\theta \leftarrow \theta - \eta\,\nabla_\theta\,\mathcal{L}(\theta)$$

### Healthy Training Signals

| Signal | Healthy | Warning |
|--------|---------|---------|
| Value loss | 0.02–0.08, oscillating | Collapses to 0.0000 and stays |
| Policy loss | Decreasing (even slowly) | Flat for 100+ iters |
| Decisive game rate | 15–40% | < 5% sustained |
| ELO | Flat early, then rising | Flat 300+ iters after buffer full |

ELO is a **lagging indicator** — evaluation games run greedy/noise-free, so both
checkpoints find the same drawing lines until the network is genuinely stronger.
Decisive game rate is the leading signal.

---

## The Full Training Loop

```
for each iteration:
  ┌─────────────────────────────────────────────────────────────┐
  │  1. SELF-PLAY                                               │
  │     Play 4 games sequentially using MCTS + current network. │
  │     Each move: 100 MCTS simulations → visit-count policy.   │
  │     Push (state, policy, outcome) triples to replay buffer. │
  │                                                             │
  │  2. TRAIN  (once buffer ≥ 500 samples)                      │
  │     Sample 256-example mini-batches × 80 steps.             │
  │     Minimise L_policy + L_value + 0.01*L_aux via Adam.      │
  │                                                             │
  │  3. CHECKPOINT  (every 50 iterations)                       │
  │     Save model weights + optimizer state to .pt file.       │
  │                                                             │
  │  4. EVALUATE  (every 5 iterations)                          │
  │     Play 20 games: current network vs previous checkpoint.  │
  │     Update ELO estimate (K=32).                             │
  └─────────────────────────────────────────────────────────────┘
```

Self-play runs sequentially in a single process — no multiprocessing — to keep the
GPU fully utilised by one MCTS search at a time.

---

## Pre-Training on Human Games

Before self-play, the network can be warm-started on human PGN data (e.g. the
[Lichess open database](https://database.lichess.org/)) using supervised learning.
This gives the network a head start: it already knows not to hang pieces and has basic
positional intuitions before any self-play begins.

```bash
python pretrain.py --pgn data/lichess.pgn --epochs 2 --batch-size 512 \
    --output checkpoints/pretrained.pt

# Then hand off to self-play
python main.py --headless --checkpoint checkpoints/pretrained.pt
```

The `pretrained.pt` checkpoint is format-compatible with `main.py --checkpoint`;
self-play fine-tuning resumes from iteration 0 using the pretrained weights.

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_res_blocks` | 4 | Standard residual tower depth (before MoE) |
| `channels` | 128 | Conv channel width throughout |
| `num_moe_blocks` | 2 | MoE blocks appended after residual tower |
| `num_experts` | 4 | Expert ResBlocks per MoE block |
| `top_k` | 2 | Experts blended per position per block |
| `load_balance_coeff` | 0.01 | Weight on MoE aux loss |
| `num_simulations` | 100 | MCTS simulations per move |
| `c_puct` | 1.4 | Exploration constant |
| `dirichlet_alpha` | 0.3 | Noise concentration |
| `dirichlet_epsilon` | 0.25 | Noise fraction at root |
| `temperature` | 1.0 | Move diversity (early game) |
| `temp_threshold` | 120 | Plies before switching to greedy |
| `games_per_iteration` | 4 | Self-play games per cycle |
| `max_game_length` | 200 | Cap before draw adjudication |
| `batch_size` | 256 | Mini-batch size |
| `learning_rate` | 2×10⁻³ | Adam LR |
| `weight_decay` | 1×10⁻⁴ | L2 regularisation |
| `replay_buffer_capacity` | 100 000 | Uniform experience replay |
| `min_buffer_size` | 500 | Delay training until buffer populated |
| `train_steps_per_iter` | 80 | Gradient steps per iteration |
| `use_amp` | True | Mixed precision (CUDA only) |

---

## Project Layout

```
marchon/
├── main.py                  Entry point, argument parsing, logging
├── play.py                  Play vs AI in browser (port 5001, any checkpoint)
├── pretrain.py              Supervised pre-training on PGN games
├── config.py                All hyperparameters in one dataclass
├── env/
│   └── chess_env.py         RL wrapper: board encoding, action space, step()
├── model/
│   └── network.py           MarchonNet: ResNet + MoE tower + policy/value heads
├── mcts/
│   └── mcts.py              MCTS: select / expand / evaluate / backup + PUCT
├── training/
│   ├── self_play.py         Self-play game loop, experience generation, PGN export
│   └── trainer.py           Training loop: self-play → train → checkpoint → eval
├── utils/
│   ├── replay_buffer.py     Fixed-capacity uniform replay buffer
│   ├── pgn_dataset.py       Iterable PGN dataset for pre-training
│   └── elo.py               ELO tracker for self-evaluation
├── ui/
│   ├── dashboard.py         SharedState (thread-safe producer/consumer)
│   ├── web_app.py           Flask app + SSE endpoint
│   ├── templates/
│   │   ├── index.html       Live training dashboard
│   │   └── play.html        Play-vs-AI browser UI
│   └── static/              Locally-served JS/CSS/piece images
├── checkpoints/             Saved model weights (.pt)
└── games/                   PGN files for every self-play game
```

---

## References

- Silver et al., *Mastering the game of Go without human knowledge* (AlphaZero), Nature 2017
- Silver et al., *A general reinforcement learning algorithm that masters chess, shogi and Go*, Science 2018
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
- Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, ICLR 2017
- Rosin, *Multi-armed bandits with episode context* (PUCT derivation), 2011
