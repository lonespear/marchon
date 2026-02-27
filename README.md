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

# Play against a saved checkpoint in your browser (port 5001)
python play.py --checkpoint checkpoints/archon_iter_0100.pt --sims 20
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
for each simulation (×50 per move):
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
- $\tau = 1$: sample proportionally to visit counts (used for the first 120 plies)
- $\tau \to 0$: always pick the most-visited move (greedy; used after ply 120)

Because MCTS has searched 25 positions ahead before computing $\pi$, this distribution
is a strictly *better* policy than the network's raw output — and that gap is what
the network learns to close.

### How 50 Simulations Become a Move

A common point of confusion: the 50 simulations do **not** produce 50 candidate moves.
They produce **visit counts across all legal moves** from the root position.

Each simulation is one traversal of the tree — select a path down, expand a leaf,
evaluate it with the network, back up the value. The same move can be visited on
multiple simulations or zero times. After 50 simulations a typical distribution might
look like:

```
e2e4 → visited 18 times    ← most explored
d2d4 → visited 14 times
g1f3 → visited 10 times
c2c4 → visited 4 times
...  → 0–2 visits each
```

The move is then chosen from this distribution (sampled or greedy depending on
temperature). The network's raw policy output never directly picks a move — it only
sets the **initial priors** $P(s,a)$ that guide which branches get explored first.

The network is called exactly **once per simulation** on the leaf node (plus once at
root expansion before the loop begins — 51 network calls total). The policy head and
value head serve distinct roles:

- **Policy head → priors $P(s,a)$**: steers simulations toward promising branches early
- **Value head → $Q(s,a)$**: accumulates as visit counts grow, refining the estimate

Both signals feed into the PUCT score. The visit counts that result from resolving
those signals over 25 simulations are what ultimately determine move selection.

---

## Self-Play & Training Data

One complete self-play game produces a sequence of training examples:

$$\mathcal{D}_{\text{game}} = \{(s_t,\, \pi_t,\, z_t)\}_{t=0}^{T}$$

- $s_t$ — board state tensor at ply $t$
- $\pi_t$ — MCTS visit-count policy at ply $t$
- $z_t$ — game outcome **from the perspective of the player to move at ply $t$**:
  $+1$ (that side won), $-1$ (that side lost), $-0.3$ (draw, not 0)

The draw value of $-0.3$ rather than $0$ is a deliberate design choice. With draws
scored as $0$, the value head finds a trivial minimum by outputting a constant near
zero, which gives near-zero MSE loss on a draw-saturated buffer and stops learning.
Scoring draws as $-0.3$ means both sides have a weak incentive to play for a win,
and the value head always has a non-zero target to train against. The reward signal
remains purely outcome-based — no board evaluation or heuristics are involved.

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

$$V^{\ast}(s) = \max_{a} \left[ -V^{\ast}\bigl(T(s,a)\bigr) \right]$$

The negation reflects the zero-sum perspective flip: what is good for the player who
just moved is bad for the player who must move next. At terminal states,
$V^{\ast}(s_T) = z \in \{+1, -1, -0.3\}$ (win, loss, draw).

ArchonNet's value head approximates $V^{\ast}$ directly: $v_\theta(s) \approx V^{\ast}(s)$.
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
which is the Monte Carlo estimate of $V^{\ast}(s)$ under the current policy.

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

## How the Network Actually Learns

The math above describes what *should* happen. This section describes what you
*actually observe* in the logs, and what each signal means during a real training run.

### Reading the Logs

Each iteration prints three numbers worth watching:

```
Self-play: 2W / 6D / 0L   total games=256
Loss - policy: 2.7769     value: 0.0291
```

**Decisive game rate** (`W` + `L` out of 8) is the most direct health signal.
It measures whether MCTS is generating games with meaningful outcomes for the
network to learn from. A rate of 1–3 decisive games per iteration (~15–35%) is
healthy. Zero decisive games every iteration is the warning sign of a draw collapse
(see `TRAINING_NOTES.md`).

**Value loss** should be non-zero and stable. If it collapses to exactly `0.0000`
and stays there, the value head has found the trivial minimum of outputting a
constant, meaning MCTS is flying blind — it has no signal about which positions
are better or worse. When value loss is healthy (e.g. `0.03–0.06`), it oscillates
rather than monotonically decreasing, because every new batch of self-play games
shifts the target distribution.

**Policy loss** starts very high (4+ from a random network) and decreases as the
network learns to predict which moves MCTS will favour. It tends to plateau for
long stretches before finding a new downward slope — this is normal. The policy
target $\pi$ is itself noisy (it includes Dirichlet noise during self-play), so the
gradient signal is inherently noisier than supervised learning.

### The Phases of Learning

Training from scratch passes through roughly three phases. The boundaries are
approximate and depend on hardware and hyperparameters.

**Phase 1 — Random exploration (iter 1–~50)**

The network knows nothing. Policy priors are nearly uniform, so MCTS explores
almost randomly. Games are decided by whoever stumbles into a checkmate or
blunder first. Value loss is high and volatile. Policy loss falls steeply from
its initial value (~4+) as the network learns the most basic structure: that some
moves are legal and some squares matter.

What you see in self-play: erratic results, mix of wins/losses/draws. Games may
be short (tactical blunders) or hit the move cap (neither side can finish).

**Phase 2 — Pattern recognition (~iter 50–200)**

The network begins internalising basic chess knowledge purely from outcome signals —
that keeping more pieces tends to lead to winning, that king safety matters, that
connected pawns are useful. It did not receive any of this explicitly; it emerges
from the statistics of which positions correlate with +1 and −1 outcomes.

Policy loss plateaus and moves slowly. Value loss stabilises at a non-zero floor.
ELO will still appear flat at this stage because the **evaluation games run without
noise and temperature=0** (fully greedy, deterministic play). Both the new and old
network find the same solid drawing lines in greedy play even if one has genuinely
better positional understanding. The self-play decisive game rate is a more honest
signal of improvement than ELO at this stage.

**Phase 3 — Positional strength (~iter 200+)**

The policy has converged enough that MCTS consistently finds better plans than
random stumbling. The value head accurately scores positions. Greedy play starts
producing decisive games even without noise, which is when ELO finally begins to
move. Policy loss resumes a downward trend as the network approaches the quality
of its own MCTS oracle.

### Why ELO Lags Behind

ELO is measured by pitting the current network against the previous checkpoint in
**noise-free, greedy games** — both sides always play their single highest-visited
move. This is deliberately conservative: it measures whether the network is
genuinely stronger, not just more exploratory.

The consequence is that ELO is a **lagging indicator**. A network can be learning
meaningfully for 100+ iterations while ELO shows no change, because both versions
still find the same drawing sequences under deterministic play. When ELO does start
moving, the improvement is real and durable.

There is a secondary measurement gap: ELO compares the current checkpoint against
a snapshot taken only 5 iterations earlier. Two checkpoints 5 iterations apart are
nearly identical — they will almost always draw each other. Even if the network
improved significantly over 200 iterations, no single 5-iter comparison captures
it. ELO should be interpreted as a cumulative signal over many evaluations, not
a per-eval metric.

The self-play decisive game rate is the **leading indicator** — it tells you whether
the training data is rich enough to drive learning, regardless of whether that
learning has yet translated into measurable strength.

### What Healthy Training Looks Like

| Signal | Healthy | Warning |
|--------|---------|---------|
| Value loss | 0.02–0.08, oscillating | Collapses to 0.0000 and stays |
| Policy loss | Decreasing (even slowly) | Flat for 100+ iters with no change |
| Decisive game rate | 1–3 per iteration (~10–30%) at 50 sims | < 5% sustained at 50+ sims |
| Iteration speed | ~4 min at 50 sims | < 2 min (exploit) or > 8 min (stuck) |
| ELO | Flat early, then rising | Flat 300+ iters after buffer full |

---

## The Full Training Loop

```
for each iteration:
  ┌─────────────────────────────────────────────────────────┐
  │  1. SELF-PLAY                                           │
  │     Play 10 games using MCTS + current network.         │
  │     Each move: 50 MCTS simulations → visit-count policy.│
  │     Push (state, policy, outcome) triples to buffer.    │
  │                                                         │
  │  2. TRAIN  (once buffer ≥ 500 samples)                  │
  │     Sample 128-example mini-batches × 60 steps.         │
  │     Minimise L_policy + L_value via Adam.               │
  │                                                         │
  │  3. CHECKPOINT  (every 25 iterations)                   │
  │     Save model weights + optimizer state to .pt file.   │
  │                                                         │
  │  4. EVALUATE  (every 5 iterations)                      │
  │     Play 10 games: current network vs previous version. │
  │     Update ELO estimate (K=32).                         │
  └─────────────────────────────────────────────────────────┘
```

At ~4 minutes per iteration on a single GPU, 2 000 iterations ≈ 5–6 days of training.

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_res_blocks` | 6 | Residual tower depth |
| `channels` | 64 | Conv channel width |
| `num_simulations` | 50 | MCTS simulations per move |
| `c_puct` | 1.4 | Exploration constant |
| `dirichlet_alpha` | 0.3 | Noise concentration (matches chess branching factor) |
| `dirichlet_epsilon` | 0.25 | Noise fraction at root |
| `temperature` | 1.0 | Move diversity (early game) |
| `temp_threshold` | 120 | Plies before switching to greedy |
| `games_per_iteration` | 10 | Self-play games per cycle |
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
├── play.py                  Play vs AI in browser (port 5001, any checkpoint)
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
│   ├── templates/
│   │   ├── index.html       Live training dashboard
│   │   └── play.html        Play-vs-AI browser UI
│   └── static/              Locally-served JS/CSS/piece images (no CDN required)
│       ├── jquery-3.7.1.min.js
│       ├── chess.min.js
│       ├── chessboard-1.0.0.min.js
│       ├── chessboard-1.0.0.min.css
│       └── img/chesspieces/wikipedia/
├── checkpoints/             Saved model weights (.pt)
└── games/                   PGN files for every self-play game
```

---

## References

- Silver et al., *Mastering the game of Go without human knowledge* (AlphaZero), Nature 2017
- Silver et al., *A general reinforcement learning algorithm that masters chess, shogi and Go*, Science 2018
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
- Rosin, *Multi-armed bandits with episode context* (PUCT derivation), 2011
