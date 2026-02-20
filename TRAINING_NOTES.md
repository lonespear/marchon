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

## Summary of all config changes from defaults

| Setting | Original | Current | Reason |
|---|---|---|---|
| `num_simulations` | 100 | 25 | Less deterministic search; faster iteration |
| `temp_threshold` | 60 | 120 | Keeps stochastic sampling for more of the game |
| `dirichlet_epsilon` | 0.25 | 0.50 | Noise dominates priors during early training |
| draw value target | 0.0 | -0.1 | Breaks zero attractor in value head |
| noise window | `ply < 4` | `ply < temp_threshold` | Noise active throughout exploratory phase |

---

## Other changes made during this period

### play.py + ui/templates/play.html (Feb 2026)
Added a standalone Flask server (port 5001) for playing against any saved
checkpoint in a browser. Human plays White, AI plays Black. Includes an
eval bar, move history, and new-game button. Runs independently of the
training dashboard on port 5000.

Usage:
```bash
python play.py --checkpoint checkpoints/archon_iter_0100.pt --sims 20
```

### play.py bug fix
The trainer saves weights under key `"model_state"` but the initial
`play.py` looked for `"model_state_dict"`. Fixed to check both keys.
