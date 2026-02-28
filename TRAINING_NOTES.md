# Archon Training Notes

A running log of training problems encountered, diagnosed, and fixed.

---

## The Draw Vortex (Feb 2026, ~iter 187–237)

### Symptoms
By iteration 187 the training had fallen into a self-reinforcing failure mode:

- **All self-play games drew** — `0W / 8D / 0L` every single iteration
- **Value loss = exactly 0.0000** — the value head stopped learning
- **Policy loss plateaued** around 0.85–0.88, oscillating without improvement
- **ELO stuck at 1000** — eval games also `0W/10D/0L` every time

### Root Cause
A classic draw death spiral driven by three compounding factors:

1. **Zero attractor in the value head**
   `_assign_values` scored draws as `0.0`. With all games drawing, every value
   target in the replay buffer was `0.0`. The value head discovered that outputting
   `~0` gave perfect MSE loss, so it stopped learning. Once value = 0 everywhere,
   MCTS received no signal about which positions were better or worse.

2. **Dirichlet noise vanished after move 2**
   `add_noise = ply < 4` (hardcoded) meant Dirichlet noise was only applied to
   the first 4 plies (2 moves each side). After that, play was essentially
   deterministic. A converged-to-draw policy would then play the same drawing
   lines every game.

3. **Greedy play kicked in too early**
   `temp_threshold = 60` — temperature dropped to 0 (fully greedy) after 60
   plies, leaving 140 plies of deterministic play per game out of the 200-ply
   cap. Combined with a draw-biased policy this locked in drawing outcomes.

The replay buffer (12k samples at the time) was entirely contaminated with
draw experience, reinforcing the cycle on every training step.

**Note:** The environment is purely outcome-based — no heuristic rewards for
captures, checks, or material. The draw trap was a training dynamics problem,
not a reward design problem.

---

## Fix Attempt 1 — Exploration hyperparameters (Feb 2026)

**Hypothesis:** Extending exploration would generate decisive games, seeding
the buffer with non-zero value targets and breaking the cycle.

### Changes
| File | Setting | Before | After |
|---|---|---|---|
| `config.py` | `num_simulations` | 100 | 25 |
| `config.py` | `temp_threshold` | 60 | 120 |
| `training/self_play.py` | noise window | `ply < 4` | `ply < self.config.temp_threshold` |

Also cleared the contaminated replay buffer by restarting the process
(buffer is in-memory only).

Restarted from **checkpoint iter 200**.

### Result
Partial improvement but insufficient:
- Value loss broke out of 0.0000 → peaked at ~0.09 then gradually fell to ~0.017
- Occasional decisive game appeared (2 out of ~280 games over 37 iterations)
- Buffer refilled with draws, value loss kept declining back toward zero
- Self-play still predominantly `0W / 8D / 0L`

The draw behavior was too deeply encoded in the iter 200 weights for
noise alone to overcome.

---

## Fix Attempt 2 — Earlier checkpoint (Feb 2026)

**Hypothesis:** The iter 200 weights were too draw-biased. Rolling back to
iter 25 (before draw convergence was entrenched) would give the noise changes
room to work.

Restarted from **checkpoint iter 25**.

### Result
Failed. Value loss collapsed even faster — from 0.03 to 0.001 within 8
iterations. Even the iter 25 weights produced almost entirely drawing
self-play at 25 simulations. The draw vortex re-established itself
immediately.

**Conclusion:** The structural cause (draws = 0.0 value target) meant that
any checkpoint with mostly-drawing self-play would collapse the value head
regardless of the exploration settings. Hyperparameter changes alone
cannot fix a zero attractor.

---

## Fix Attempt 3 — Structural fix (Feb 2026)

**Hypothesis:** The zero attractor must be eliminated. Draws must carry a
non-zero value target so the value head always has a gradient to train on.

### Changes
| File | Setting | Before | After |
|---|---|---|---|
| `training/self_play.py` | draw value | `0.0` | `-0.1` |
| `config.py` | `dirichlet_epsilon` | 0.25 | 0.50 |

**Why `-0.1` for draws is not a heuristic:**
The reward signal is still purely outcome-based — the network only receives
information about game results, not about board positions, material, or
tactics. Changing the draw score from 0 to -0.1 is a game-theory design
choice (analogous to komi in Go): both sides are mildly incentivized to
play for a win rather than a safe repetition. The value head can no longer
achieve zero loss by outputting a constant, so it must learn.

**Why epsilon 0.50:**
With 50% noise weight at the root, policy priors from the network are
substantially overridden during self-play, preventing early convergence
to drawing lines before the value head has learned to distinguish positions.

Restarted from **iteration 1** (fresh weights) so no draw-biased priors
could interfere.

### Expected outcome
- Value loss should stabilize at a non-zero floor (~0.05+) and not collapse
- Decisive games should appear within the first 10–20 iterations
- ELO should begin moving once the value head provides real position estimates

---

## Fix Attempt 4 — Policy plateau from excess noise (Feb 2026)

**Problem:** After Fix 3, policy loss went completely flat at ~2.82 for 90+
iterations. With `dirichlet_epsilon = 0.50`, half of every policy target was
random Dirichlet noise. The network couldn't learn to predict visit counts
that were 50% random — the gradient signal was too noisy.

### Changes
| File | Setting | Before | After |
|---|---|---|---|
| `config.py` | `dirichlet_epsilon` | 0.50 | 0.25 |

The draw vortex was now structurally fixed by `draw = -0.1` rather than by
noise, so epsilon could safely return to the standard AlphaZero value.

Restarted from **checkpoint iter 100** of the fresh run.

### Result
Policy loss resumed declining: 2.82 → 1.38 over ~1800 iterations.
Value loss stable at 0.02–0.05. But a new problem emerged (see below).

---

## Fix Attempt 5 — Repetition exploit (Feb 2026)

**Problem:** By iter ~1800 the model had learned a degenerate strategy:
shuffle a piece A→B→A→B→A to trigger threefold repetition in ~8 moves.
With `claim_draw=True`, any repeated position instantly terminated as a draw.
Draw = -0.1 was better than risking a -1 loss, so the model preferred
near-instant draws over playing chess.

**Signatures:** iteration time dropped from 3–5 min to 30–60 sec; games were
5–10 moves; decisive game rate near 0%; policy loss improved but ELO never moved.

### Changes
| File | Setting | Before | After |
|---|---|---|---|
| `env/chess_env.py` | `claim_draw` in `is_game_over()` + `outcome()` | `True` | `False` |
| `mcts/mcts.py` | `claim_draw` in terminal check | `True` | `False` |
| `training/self_play.py` | draw value target | `-0.1` | `-0.3` |

With `claim_draw=False`, threefold repetition is still legal but must be
*claimed* by a player — the engine never auto-terminates on it. The stronger
draw penalty (-0.3) makes repetition-fishing a less attractive strategy.

Restarted from **checkpoint iter 1925** (which had policy loss 1.38 already
learned, but repetition-biased behaviour).

### Result
Iteration speed returned to 3–5 min immediately (games became real chess again).
Fresh run restarted from **iter 1** with all fixes in place.

---

## Summary of all config changes from defaults

| Setting | Original | Current | Reason |
|---|---|---|---|
| `num_simulations` | 100 | 50 | Raised 25→50 at iter ~1756 to generate organically decisive games |
| `temp_threshold` | 60 | 120 | Keeps stochastic sampling for more of the game |
| `dirichlet_epsilon` | 0.25 | 0.25 | Raised to 0.50 at iter ~1356 to break draw drift, reverted at iter ~1546 |
| draw value target | 0.0 | -0.3 | Breaks zero attractor; raised from -0.1 to close repetition exploit |
| noise window | `ply < 4` | `ply < temp_threshold` | Noise active throughout exploratory phase |
| `claim_draw` | `True` | `False` | Closes threefold-repetition exploit |

---

## Current training run (started Feb 22, 2026 — all fixes applied)

Fresh start from iter 1 with the full set of fixes. Key milestones:

| Iteration | Event |
|---|---|
| ~65 | ELO first moves: 1000 → 1008 |
| ~90 | ELO hits 1016, drops back at ~95 (oscillation) |
| ~390 | ELO holds at 1016 for first time |
| ~495 | ELO hits 1024 (high watermark) |
| ~500 | ELO drops back to 1016 |
| ~575 | ELO 1016, policy loss ~1.52 |
| ~650 | ELO regresses to 1008 and stalls |
| ~1356 | ELO still 1008, policy loss ~1.41 — soft draw drift (see below) |
| ~1426 | epsilon raised 0.25→0.50; decisive rate climbs from 4% to 15%+ |
| ~1490 | ELO drops to 992 — policy noise side effect |
| ~1525 | ELO drops to 984; decisive rate holding 15-16%; epsilon reverted 0.50→0.25 |
| ~1756 | Decisive rate reverted to ~4%, ELO 992 — noise failed (see below) |
| ~1756 | **Restart from iter 1425.pt**: sims 25→50, games/iter 8→10 |
| ~1490 | Buffer full at 50k; decisive rate settled to 6-7% |
| ~1605 | Decisive rate begins organic climb: 5% → 10% → 13.8% by iter 1785 |
| ~1785 | Decisive rate plateaus at ~13-14% for ~94 iters; policy loss 1.10→1.22 |
| **~1879** | **Organic breakthrough: decisive rate crosses 20% with no config change** |
| ~1946 | Decisive rate 33%; ELO moves 1008→1016 via 5W/5D/0L eval — first real eval win |
| ~2000 | Run cap hit; restarted from archon_iter_2000.pt |
| ~2113 | Decisive rate **43% and still climbing**; ELO 1016; policy loss ~1.60; value loss ~0.07-0.08 |

Policy loss trajectory: 4.78 (iter 1) → ~1.41 (iter 1356) → ~2.15 (iter 1546, noise phase) → ~1.34 (iter 1756) → ~1.22 (iter 1830) → ~1.60 (iter 2113, tactical learning phase).

---

## Soft Draw Drift (Feb 25, 2026, iter ~650–1356)

### Symptoms
- Self-play decisive rate fell to ~4% (15W / 384D / 1L over 50 iterations)
- ELO stuck at 1008 for 700+ iterations; eval almost always `0W/10D/0L`
- Policy loss continued declining (good) but ELO did not follow
- Value loss healthy (0.007–0.015), not approaching zero attractor

### Root Cause
The model converged to solid, drawish play. Policy head improved at predicting
MCTS visit counts, but all MCTS visits occur in draw territory — the policy is
getting better at drawing, not winning. The value head has signal (draw=-0.3)
but not enough decisive-game experience to push ELO higher.

This is a mild version of the Draw Vortex: structurally stable but stuck in a
local equilibrium. The ELO evaluation is also blind here since both sides play
identically at temperature=0 with no noise.

### Fix applied (iter ~1356)
| File | Setting | Before | After |
|---|---|---|---|
| `config.py` | `dirichlet_epsilon` | 0.25 | 0.50 |
| `training/trainer.py` | decisive-rate logging | — | added (rolling 50-iter window) |

**Reverted epsilon to 0.25 at iter ~1546** (120 iters post-raise). Decisive rate
reached 15–16% as planned but ELO dropped 984 (from 1008) — policy targets were
50% noise so the network adapted to noise-corrupted distributions. Policy loss
rose 1.38 → 2.15 confirming degradation.

**Noise fix failed.** After revert, decisive rate returned cleanly to ~4% by iter
~1700 and ELO settled at 992. The buffer seeding did not stick — the decisive
games during the noise phase were noise-driven blunders, not learned patterns.
The network weights never internalized decisive play.

---

## Epsilon Noise as Draw-Drift Fix: Post-Mortem

### Verdict: Ineffective at 25 simulations

With only 25 sims, MCTS produces a shallow tree. Decisive games generated under
high epsilon were largely random tactical blunders from noise overriding the
policy, not genuine tactical sequences the network could learn from. The model
could not distinguish "I won because noise forced a blunder" from "I won because
I found a forced win." Once noise was removed, the confident draw-biased policy
reasserted immediately.

Lesson: epsilon bumps can only work if the base MCTS is already generating
meaningful decisive games that the network can learn from. At 25 sims they
cannot — the tree isn't deep enough to find real forcing lines.

### Structural fix: increase num_simulations 25 → 50

Deeper search generates decisive games organically — no policy corruption,
no noise-driven blunders. The model can actually learn the decisive patterns
because they arise from genuine tactical depth rather than forced randomness.

games_per_iteration also raised 8 → 10 to partially offset the slower iteration
speed (~4 min vs ~2 min per iter).

Restarting from **archon_iter_1425.pt** (pre-noise, ELO 1008, clean weights).

---

## Organic Decisive Rate Breakthrough (Feb 27–28, 2026)

### What happened
After ~94 iterations of plateau at 13-14% decisive rate (iter ~1785–1879), the rate
broke through 20% with **no config changes whatsoever**. It then climbed continuously:

| Total games | Iter (approx) | Decisive rate |
|---|---|---|
| ~15,150 | ~1783 | 13.2% (plateau) |
| ~15,940 | **~1879** | **20.4%** ← breakthrough |
| ~16,610 | ~1946 | 33.2% |
| ~17,150 | ~2000 | ~35% (cap hit) |
| ~18,270 | ~2113 | **43.2% and rising** |

The 94-iter plateau was not stagnation — it was the buffer accumulating enough
tactical experience to cross a learning threshold. Once enough decisive-game
positions were in the training distribution, the policy head started finding
forcing moves consistently and the decisive rate snowballed.

### ELO response
At ~33% decisive rate (iter ~1946), even the 5-iter eval comparison began producing
decisive results: `5W/5D/0L`. ELO moved 1008→1016 organically. This confirmed the
improvement was real — the greedy policy became strong enough to beat a 5-iter-older
version of itself.

### Current signals (iter ~2113)
- Decisive rate: **43%+** and still climbing
- Policy loss: ~1.60 (elevated — model in rapid tactical learning phase; expected)
- Value loss: ~0.07-0.08 (healthy for this decisive rate; warning threshold scales up)
- ELO: 1016 (eval games still `0W/10D/0L` — greedy policy lags noisy self-play)

### Self-play W/D/L clarification
The W/D/L in self-play logs (`0W / 3D / 7L`) is **not** the model losing to a
previous version. It is the current model playing **against itself** — W = White
won, L = Black won, same network on both sides. The current loss skew (Black winning
more than White) reflects a self-play color asymmetry: the model has developed
strong second-player responses that punish White's first moves. This is normal
and not a sign of regression.

Eval (`Eval vs prev: 0W/10D/0L`) is where the current network plays a previous
checkpoint. Those are still drawing because greedy (no-noise) deterministic play
is more conservative than noisy self-play. ELO will move again once the tactical
patterns consolidate into the greedy policy.

---

## ELO Measurement Gap Fix (Feb 27, 2026)

### Problem
ELO was frozen at 1008 with `0W/10D/0L` on every eval for 300+ iterations, even
as the self-play decisive rate climbed from 4% to 13-14%. Root cause: `prev_network`
was snapshotted every `eval_every_n_iters=5` iterations. Two checkpoints 5 iterations
apart are nearly identical — they always draw each other at temp=0, no noise. ELO
never accumulated any signal.

### Fix
Decouple the `prev_network` refresh from the eval interval. Snapshot is now taken
only at checkpoint intervals (`checkpoint_every_n_iters=25`). The 5 evals between
each pair of checkpoints all compare against the same 25-iter-old baseline, giving
real divergence time. Two networks 25 iterations apart at 13%+ decisive rate have
a meaningful chance of producing non-draw eval games.

**File changed:** `training/trainer.py` — separated `prev_network = deepcopy(...)`
from the eval block into the checkpoint block.

---

## Other changes made during this period

### play.py + ui/templates/play.html (Feb 2026)
Added a standalone Flask server (port 5001) for playing against any saved
checkpoint in a browser. Human plays White, AI plays Black. Includes an
eval bar, move history, and new-game button. Runs independently of the
training dashboard on port 5000.

Usage:
```bash
python play.py --checkpoint checkpoints/archon_iter_0575.pt --sims 20
# Open http://localhost:5001
```

### play.py — static file serving (Feb 23, 2026)
Original play.html loaded jQuery, chess.js, and chessboard.js from CDN.
On networks that block unpkg/jsdelivr the board wouldn't render at all.
Fixed by downloading all JS/CSS/piece-image assets to `ui/static/` and
serving them locally via Flask's static folder. No internet connection
required to run the UI.

### play.py — eval bar non-blocking (Feb 23, 2026)
`/api/state` originally called `_get_value_estimate()` (a NN forward pass)
on every page load, blocking the board from rendering until inference
completed. Fixed by caching the last computed value and only updating it
after moves — `/api/state` now returns instantly.

### play.py — piece stays during AI thinking (Feb 23, 2026)
After dropping a piece the board would snap it back, then animate both the
human and AI moves together once the AI responded. Fixed by keeping the
chess.js move applied (not undoing it) in `onDrop`, and making `onSnapEnd`
a no-op. The board now updates once to the final FEN after the AI responds.

### play.py — chess.js version fix (Feb 23, 2026)
`chess.js@0.13.4` from jsdelivr is an ES module and throws
`Uncaught SyntaxError: Unexpected token 'export'` in a plain browser script
context. Replaced with `chess.js@0.10.2` which exports a global `Chess`
constructor compatible with chessboard.js 1.0.0.

### play.py bug fix
The trainer saves weights under key `"model_state"` but the initial
`play.py` looked for `"model_state_dict"`. Fixed to check both keys.
