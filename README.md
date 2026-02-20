# Archon — Chess Reinforcement Learning

Archon is a from-scratch implementation of [AlphaZero](https://www.nature.com/articles/nature24270)-style
self-play reinforcement learning applied to chess. A dual-headed residual neural network learns
**entirely through self-play** — no human games, no handcrafted evaluation function — guided by
Monte Carlo Tree Search (MCTS).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Board Representation](#board-representation)
3. [Neural Network Architecture](#neural-network-architecture)
4. [Monte Carlo Tree Search](#monte-carlo-tree-search)
5. [Self-Play & Training Data](#self-play--training-data)
6. [Learning — The Math](#learning--the-math)
7. [The Full Training Loop](#the-full-training-loop)
8. [Hyperparameters](#hyperparameters)
9. [Project Layout](#project-layout)

---

## Quick Start

```bash
pip install -r requirements.txt

# Train (headless, logs to archon.log)
python main.py --headless --iterations 2000

# Train with live web dashboard at http://localhost:5000
python main.py

# Resume from a checkpoint
python main.py --headless --checkpoint checkpoints/archon_iter_0100.pt
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
for normal moves, plus 4 under-promotion choices (R, B, N for each colour). Queen
promotions are encoded as normal moves.

---

## Neural Network Architecture

ArchonNet is a **dual-headed residual network** — the same family used by AlphaZero and
Leela Chess Zero.

```
Input  (batch, 19, 8, 8)
  │
  ▼
Input conv  Conv2d(19→64, 3×3)  →  BatchNorm  →  ReLU
  │
  ▼
Residual Tower  ×6 blocks
  │  Each block:
  │    Conv2d(64→64, 3×3) → BN → ReLU
  │    Conv2d(64→64, 3×3) → BN
  │    skip-add → ReLU
  │
  ├─────────────────────┐
  ▼                     ▼
Policy head           Value head
Conv2d(64→32, 1×1)    Conv2d(64→1, 1×1)
BN → ReLU             BN → ReLU
Flatten               Flatten
Linear(2048→4100)     Linear(64→256) → ReLU
                      Linear(256→1) → Tanh
  │                     │
  ▼                     ▼
logits (4100,)        scalar ∈ [-1, 1]
```

**8 874 951 parameters** (6 residual blocks, 64 channels).

- The **policy head** outputs raw logits over all 4 100 actions. At inference, illegal
  moves are masked to −∞ before softmax, so the network only assigns probability to
  legal moves.
- The **value head** outputs a single scalar $v \in [-1, 1]$ estimating the game outcome
  from the perspective of the player to move (+1 = win, −1 = loss, 0 = draw).

Residual connections allow gradients to flow cleanly through the deep tower during
training (He et al., 2015).

---

## Monte Carlo Tree Search

MCTS is the **planning algorithm** that turns a weak raw network into a strong move
selector. It builds a search tree from the current position by repeatedly running
four phases:

```
for each simulation (×25 per move):
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
              The sign flips at each ply because each node is from
              the opposite player's perspective.
```

### PUCT Score

The **PUCT formula** (Predictor + Upper Confidence bound for Trees) governs which
node is selected at each step:

$$\text{PUCT}(s, a) = Q(s,a) + c_{\text{puct}} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

| Symbol | Meaning |
|--------|---------|
| $Q(s,a) = W(s,a)/N(s,a)$ | Mean action-value over all simulations through this edge |
| $P(s,a)$ | Network prior probability for action $a$ in state $s$ |
| $N(s)$ | Total visit count of the **parent** node |
| $N(s,a)$ | Visit count of **this** child |
| $c_{\text{puct}}$ | Exploration constant (1.4 in Archon) |

The second term is the **exploration bonus** $U(s,a)$. It is large when:
- the network thinks the move is promising ($P$ large), and
- the move hasn't been explored much yet ($N(s,a)$ small).

As search continues, heavily-visited nodes have $U \to 0$, forcing the search to
explore alternatives. In the limit of infinite simulations, PUCT converges to the
minimax-optimal action.

### Dirichlet Noise

To prevent the search from fixating on the highest-prior move, **Dirichlet noise** is
added to the root node's priors at the start of each move:

$$P'(s,a) = (1 - \varepsilon)\,P(s,a) + \varepsilon\,\eta_a, \quad \eta \sim \text{Dir}(\alpha)$$

with $\alpha = 0.3$ and $\varepsilon = 0.25$. This guarantees every legal move has
some chance of being explored early in training.

### Policy Extraction

After all simulations, the **improved policy** $\pi$ is derived from visit counts:

$$\pi(a \mid s) = \frac{N(s,a)^{1/\tau}}{\displaystyle\sum_{a'} N(s,a')^{1/\tau}}$$

Temperature $\tau$ controls exploration:
- $\tau = 1$: sample proportionally to visit counts (used for the first 20 plies)
- $\tau \to 0$: always pick the most-visited move (greedy; used after ply 20)

Because MCTS has searched 25 positions ahead before computing $\pi$, this distribution
is a strictly *better* policy than the network's raw output — and that gap is what
the network learns to close.

---

## Self-Play & Training Data

One complete self-play game produces a sequence of training examples:

$$\mathcal{D}_{\text{game}} = \{(s_t,\, \pi_t,\, z_t)\}_{t=0}^{T}$$

- $s_t$ — board state tensor at ply $t$
- $\pi_t$ — MCTS visit-count policy at ply $t$
- $z_t$ — game outcome **from the perspective of the player to move at ply $t$**:
  $+1$ (that side won), $-1$ (that side lost), $0$ (draw)

The sign of $z_t$ alternates with each ply because the players alternate. All examples
from a game are pushed into a **replay buffer** (capacity 50 000). Mini-batches are
sampled uniformly from the buffer during training, which breaks temporal correlations
and stabilises learning.

---

## Learning — The Math

### Connection to Reinforcement Learning

Chess is a finite, episodic, two-player zero-sum game. Cast in the RL framework:

- **State** $s$: board position
- **Action** $a$: legal move
- **Reward**: 0 at every step; $z \in \{+1, -1, 0\}$ at the terminal state
- **Discount** $\gamma = 1$ (no discounting in a finite game)

The **Bellman optimality equation** for the state-value function is:

$$V^{*}(s) = \max_{a} \left[ -V^{*}\bigl(T(s,a)\bigr) \right]$$

The negation reflects the zero-sum perspective flip: what is good for the player who
just moved is bad for the player who must move next. At terminal states,
$V^{*}(s_T) = z \in \{+1, -1, 0\}$.

ArchonNet's value head approximates $V^{*}$ directly: $v_\theta(s) \approx V^*(s)$.
MCTS implements **one step of policy improvement** by computing a look-ahead estimate
that is closer to the Bellman fixed point than $v_\theta$ alone.

### Loss Function

Both heads are trained simultaneously with a single combined loss:

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{value}}$$

**Policy loss — cross-entropy:**

$$\mathcal{L}_{\text{policy}} = -\sum_{a} \pi(a \mid s)\,\log p_\theta(a \mid s)$$

This minimises the KL divergence between the MCTS-improved distribution $\pi$ and the
network's policy $p_\theta$. Because MCTS is a strictly better policy oracle than the
raw network, minimising this loss causes the network to internalise the look-ahead
reasoning — so it needs *less* search to reach the same quality next time.

**Value loss — mean squared error:**

$$\mathcal{L}_{\text{value}} = \bigl(z - v_\theta(s)\bigr)^2$$

This trains the value head to predict the actual game outcome from each position,
which is the Monte Carlo estimate of $V^{*}(s)$ under the current policy.

**Gradient update** (Adam optimiser, weight decay $10^{-4}$, gradient clipping at 1.0):

$$\theta \leftarrow \theta - \eta\,\nabla_\theta\,\mathcal{L}(\theta)$$

Gradient clipping prevents the exploding-gradient problem that can occur when the
network has not yet learned to separate won from lost positions.

### Why This Converges

The key insight from AlphaZero is that this forms a **self-improving loop**:

1. MCTS uses the *current* $v_\theta$ and $p_\theta$ to produce a better policy $\pi$.
2. The network is trained to match $\pi$ and $z$.
3. A stronger network produces a better MCTS policy in the next iteration.

Under mild conditions this iteration converges to the Nash equilibrium of the game —
i.e., perfect play. In practice, with limited compute, it converges to a strong
approximation.

---

## The Full Training Loop

```
for each iteration:
  ┌─────────────────────────────────────────────────────────┐
  │  1. SELF-PLAY                                           │
  │     Play 8 games using MCTS + current network.          │
  │     Each move: 25 MCTS simulations → visit-count policy.│
  │     Push (state, policy, outcome) triples to buffer.    │
  │                                                         │
  │  2. TRAIN  (once buffer ≥ 500 samples)                  │
  │     Sample 128-example mini-batches × 60 steps.         │
  │     Minimise L_policy + L_value via Adam.               │
  │                                                         │
  │  3. CHECKPOINT  (every iteration)                       │
  │     Save model weights + optimizer state to .pt file.   │
  │                                                         │
  │  4. EVALUATE  (every 5 iterations)                      │
  │     Play 10 games: current network vs previous version. │
  │     Update ELO estimate (K=32).                         │
  └─────────────────────────────────────────────────────────┘
```

At ~45 seconds per iteration on a single GPU, 2 000 iterations ≈ 25 hours of training.

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_res_blocks` | 6 | Residual tower depth |
| `channels` | 64 | Conv channel width |
| `num_simulations` | 25 | MCTS simulations per move |
| `c_puct` | 1.4 | Exploration constant |
| `dirichlet_alpha` | 0.3 | Noise concentration (matches chess branching factor) |
| `dirichlet_epsilon` | 0.25 | Noise fraction at root |
| `temperature` | 1.0 | Move diversity (early game) |
| `temp_threshold` | 20 | Plies before switching to greedy |
| `games_per_iteration` | 8 | Self-play games per cycle |
| `max_game_length` | 200 | Cap before draw adjudication |
| `batch_size` | 128 | Mini-batch size |
| `learning_rate` | 2 × 10⁻³ | Adam LR |
| `weight_decay` | 1 × 10⁻⁴ | L2 regularisation |
| `replay_buffer_capacity` | 50 000 | Uniform experience replay |
| `min_buffer_size` | 500 | Delay training until buffer is populated |
| `train_steps_per_iter` | 60 | Gradient steps per iteration |

---

## Project Layout

```
archon/
├── main.py                  Entry point, argument parsing, logging
├── config.py                All hyperparameters in one place
├── env/
│   └── chess_env.py         RL wrapper: board encoding, action space, step()
├── model/
│   └── network.py           ArchonNet: ResNet body + policy/value heads
├── mcts/
│   └── mcts.py              MCTS: select / expand / evaluate / backup + PUCT
├── training/
│   ├── self_play.py         Self-play game loop, experience generation, PGN export
│   └── trainer.py           Training loop: self-play → train → checkpoint → eval
├── utils/
│   ├── replay_buffer.py     Fixed-capacity uniform replay buffer
│   └── elo.py               ELO tracker for self-evaluation
├── ui/
│   ├── dashboard.py         SharedState (thread-safe producer/consumer)
│   ├── web_app.py           Flask app + SSE endpoint
│   └── templates/index.html Live training dashboard
├── checkpoints/             Saved model weights (.pt)
└── games/                   PGN files for every self-play game
```

---

## References

- Silver et al., *Mastering the game of Go without human knowledge* (AlphaZero), Nature 2017
- Silver et al., *A general reinforcement learning algorithm that masters chess, shogi and Go*, Science 2018
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
- Rosin, *Multi-armed bandits with episode context* (PUCT derivation), 2011
