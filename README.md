# ABI-PPO — ARCbound Intelligence

**A production reinforcement-learning system for real-time multiplayer game AI.**

ABI-PPO is the complete AI stack behind the bots in *Arcbound* (the browser-based
top-down multiplayer arena shooter by [ArcBound Interactive](https://arcboundinteractive.com)): a custom PPO trainer, a
rule-based decision layer (the Orch-OR substrate) with the neural policy holding
bounded authority over a few action heads, a zero-dependency inference engine
that runs inside the game server, and a closed-loop collect → train → judge →
deploy pipeline that retrains the policy on its own gameplay.

This repository overview is a public showcase of a private codebase. The
algorithms are described at concept level; implementations remain private.

---

## Novel Systems & Methods

The project's core contribution is making PPO *work* in a setting where
textbook PPO fails: long-horizon (~10,000-tick) episodes, sparse terminal
rewards, and a live game server as the only environment. Nine original systems
carry that:

### 1. Decoupled Dual-Optimizer PPO
Actor and critic get fully separate network backbones, separate Adam
optimizers, and separate learning rates. In shared-optimizer PPO the value loss
(magnitude ~1000) corrupts the shared momentum state and drowns the policy
gradient (magnitude ~0.03). Decoupling removed that interference outright.
**Result:** 15 ABI-PPO epochs achieved what 800 epochs of the legacy trainer
could not — value-function explained variance went from −3.3 (worse than a
constant) to +0.49, and value loss converged from ~420 to ~0.9 where the legacy
trainer plateaued near 945.

### 2. Reward-Channel Decomposition
Scalar rewards are decomposed into three orthogonal channels — *survival*,
*combat*, and *strategy* — and each channel's advantages are normalized
independently, so a rare +1000 round-win can no longer erase the dense
per-tick dodge and aim signal.

Attribution used to be *retroactive*: the server emitted one scalar per tick and
the trainer classified each reward back to its channel from the known reward
taxonomy by magnitude. That approach is inexact in principle — two different
events can share a value, and a tick holding several events sums to a number
that matches a third — and the failure is silent, because a misrouted reward
still trains, just the wrong head. The experience format now carries the split
**written at source**: the collector knows which channel each component came
from, so it records all three alongside the scalar they sum to. The
classifier survives only as the fallback for historical data, and a build-time
guard asserts every reward weight against its declared channel so a new weight
cannot be added without one.

### 3. Staged Skill Curriculum with Head Masking
Training runs in three phases — **Movement → Combat → Strategy** — and each
phase masks the action heads and reward channels irrelevant to the skill being
learned (movement trains only the move head on survival rewards; combat trains
fire + aim on combat rewards; strategy unlocks everything). This collapses the
credit-assignment horizon from ~10,000 ticks to ~200 per phase. Phase
transitions are *convergence-gated*, not epoch-scheduled: a phase advances when
policy loss flattens, and the critic-priority phase additionally requires the
value function's explained variance to plateau, so the long-horizon value head
is never cut off early.

### 4. Bounded-Authority Hybrid Control
The shipped bots are not a pure neural agent. The rules decide — which
objective, where to go, when to commit, when to fire — and the PPO policy holds
*bounded authority* over a few action heads. Since September 2026 the rule
layer in every room with player-vs-player bots is a utility-scoring decision
substrate (*Orch-OR*: candidate objectives are scored, held and committed by
rule), released in numbered versions; the older per-role state machine drives
only legacy bots. Which heads the policy owns is part of each release, and the
aim head predicts a *target-relative offset* (a correction around the rule
layer's own firing solution) rather than an absolute angle.

Authority is granted only by measurement. In the current release the policy
picks the dodge on a share of evasion windows and holds a fire bit that can
only *add* shots to the rule layer's own; its aim offset is discarded, because
a study across 194 maps found the re-applied offset lost rounds (46.4 %) and
the exchange (0.89×). The net cannot throw a match by ignoring an objective,
and it gains a head only by beating the rules at it in a controlled trial.

The rule layer's own firing solution was re-measured in October 2026. Against
pilots who dodge, the shot aimed at the full intercept is the one a target
swerves out of: in a controlled trial (same skill on both sides, 60 matches)
leading at 0.6 of the intercept landed 2.5 more lasers per hundred and traded
1.7× better than leading at 0.95, still won when the target's dodge was
delayed by 400 ms (as a networked player's is), and won or tied on five more
maps. Every skill tier now leads at 0.6. The same trials showed why a
hindsight label — the correction that would have hit, read off the target's
real path — cannot train the aim head here: the target reacted to the shot
that was fired, so every other angle looks better in hindsight. Per-shot
outcomes of randomised leads give the unbiased signal instead.

### 5. Map-Context Conditioning ("Map Digest")
A compact static descriptor of the current map and mode — mode one-hot,
topology flags, arena scalars, a coarse wall-density grid, objective anchor
positions — is appended to the observation. It is computed once per map (zero
per-tick cost) and cures cross-map strategy bleed: one shared policy plays
every map and mode instead of averaging them into mush. The append is
compatibility-safe (the legacy observation prefix is byte-identical) and a
dimension guard at load time refuses any policy whose input width doesn't
match, falling back to pure rules rather than shipping garbage.

### 6. Shard-Rotating Full-Corpus Training
A trainer that pre-allocates its dataset can only ever see a RAM-sized slice —
against a pool several times larger than memory, most of what was collected is
never trained on and then deleted. The obvious workaround, training each slice
to convergence and chaining the checkpoints, was measured and **lost**: every
curriculum phase regressed. The reason is a property of this data — the bots
never learn during play, so an entire pool is written by one frozen behaviour
policy, and chaining converged fits against a policy that never moves simply
walks the learner off the data supporting it.

The working form is one bounded run whose *resident shard rotates* every few
epochs, so the policy sees the whole corpus at approximately one policy state
instead of N policies fitted in sequence. Rotation is not free: swapping shards
invalidates everything derived from the rows, so the advantage estimates, the
normalization, the action masks and the old-policy reference are all rebuilt
against the incoming shard. Shards are dealt bucket-balanced — each carries a
proportional slice of every mode and map, because reward scale is not comparable
across modes — and the load unit is a whole experience file, never a random row,
since advantage estimation runs over contiguous per-agent trajectories. A phase
is forbidden from declaring convergence before it has seen every shard, or it
would converge on a fraction of the corpus and call it done.

### 7. Human-Anchored Training from Replays
A replay-ingestion path reconstructs the full training observation,
slice-for-slice, from years of recorded *human* games — including encoding
quirks like mixed text encodings in legacy replays and fire events that fall
between movement samples. The resulting behavior-cloning corpus is used two
ways: to initialize a fresh policy from human play, and as a *frozen KL
anchor* — a divergence penalty toward the human-cloned reference — so PPO can
optimize reward without drifting into inhuman play ("human ↔ optimal" as a
single tunable knob).

### 8. Round-Quality Reward Shaping
Reward design here was outcome-based — win, kill, capture — which measured
*whether* a round was won and never *how* it was played. Two thirds of all
reward mass sat in terminal events while the dense per-tick channel summed
negative, so nothing in the objective distinguished a won exchange from a lost
one, and the bots settled at roughly half the damage ratio a strong human
sustains.

The addition is a terminal quality term, and its form matters more than its
weight. Per-tick accumulators reward *volume* — the same damage pays the same
whether it was traded well or badly — so quality is scored once, at the round
boundary, from ratios that volume cannot inflate: a normalized damage
differential and a landed-shot rate. The differential is deliberately
scale-free, so identical play scores identically regardless of a map's weapon
damage, and both terms are floored on engagement — without that, the maximum is
obtained by disengaging, which is precisely the behaviour the change is meant to
discourage. The term is emitted as its own experience row rather than folded
into the terminal, because the trainer compresses large rewards and folding it in
would have reduced it to noise beside the round-win value.

### 9. Pre-Registered Head-to-Head Judging
A trained model never ships on its loss curve. A headless harness runs the real
server code at many times real speed, puts two models (or a model and a
rules-only arm) on opposite squads of the same match, and swaps sides every
match, so map, opponent and tempo are held constant inside the comparison. The
verdict bars are written down before any result is seen: a model earns more
authority only with at least 57 % of 200+ decisive rounds and a 95 % Wilson
interval clear of 50 %, a damage trade at least 10 % better with no more
deaths, and a winning share on *each* side of the map. Model-against-model
trials call a loss only when the whole interval sits below 50 %, since two
near-equal nets land under 50 % about half the time by chance.

The same harness checks the trainer itself. In October 2026 an audit found
that the curriculum's per-phase learning rates had applied only in each phase's
first epoch, so every policy had trained at one rate pair; the schedule now
holds, a tripwire stops any run whose optimizers drift off it, and an A/B of
the designed per-phase rates against the de-facto ones on the same data tied
in three 190-match trials, so the de-facto rates became the declared ones.

**Also notable:** a self-healing deployment loop (pre-deploy sanity gate that
mirrors the runtime's dimension guard, atomic policy + normalizer deploy so the
two can never mismatch, checksum-verified go-live, automatic backup and
one-command rollback), and a pure-TypeScript inference engine — the trained
network runs inside the Node game server with no ML runtime dependency at all,
from weights exported as JSON. Invalid-action masking follows the published
MaskablePPO approach (Huang & Ontañón, 2020) in an independent implementation.

---

## Status

**Live in production.** The public bots run Orch-OR v0.0.8 (October 2026) with
the policy holding the bounded authority above, on a 462-float observation
(layout v12). Lifetime training ledger as of the last production run
(29 Sep 2026): **53 training runs**, **4,564 epochs** and **248.5 billion
cumulative training samples**, all three curriculum phases.

Versions — observation, reward set, experience format, corpus epoch and
checkpoint — are kept in one table, after a period in which they had drifted to
four different answers and a reward re-tune reached the game server without
reaching the trainer that reads its data. Agreement between the two is now
asserted rather than assumed: the trainer rejects experience files in a format
it does not know, and a self-test that every training cycle runs fails when the
trainer's reward constants differ from the server's.

Open work: giving the net a decision it can measurably improve on before
reinforcement learning moves inside the decision substrate, side balance and
positioning in the rule layer, and an aim head trained on per-shot outcomes
(randomised leads) rather than on hindsight geometry.

**Stats:** Python + TypeScript · ~15 K lines of Python (trainer + pipeline)
and ~36 K lines of TypeScript (runtime AI), plus ~14 K lines of tests, the
deployed model artifacts and format documentation.

---

## What's in the (private) repo

| Area | Contents |
|---|---|
| Trainer | The ABI-PPO trainer: dual-optimizer PPO, reward decomposition, staged curriculum, action masking, BC anchoring |
| Runtime | TypeScript game-server AI: the Orch-OR decision substrate and its versioned arms, the legacy rule brain, perception, pathfinding, per-opponent dodge modeling, weapon ballistics solvers, RL glue (observation encoding, inference, experience collection, reward shaping) |
| Pipeline | Closed-loop automation: pull experience → validate → train → judge → sanity-gate → atomic deploy → verify → rollback |
| Models | The actual deployed policy, its paired observation normalizer, training checkpoints and history |
| Docs | Architecture deep-dive, binary experience/replay format specs, training-compatibility audits |

### Trainer at a glance

```bash
python abi_ppo.py --epochs 300 --phase all         # full 3-phase curriculum
python abi_ppo.py --resume <checkpoint> --phase combat --epochs 50
python abi_ppo.py --info                           # inspect corpus + checkpoint
```

Each curriculum phase is declared as data — reward filter, head mask, and
per-phase optimization settings — and the loop just walks the list:

```python
class TrainingPhase:
    """One curriculum stage: what it trains, on which rewards, until when."""
    name: str
    reward_filter: Callable      # scalar rewards -> this phase's channel
    active_heads: dict           # {move: bool, fire: bool, aim: bool}
    actor_lr: float; critic_lr: float
    entropy_coeffs: dict         # per-head exploration pressure
    epochs_min: int; epochs_max: int
    convergence_threshold: float # plateau detector for phase advance
    value_coeff: float
```

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the full system design and
**[SETUP.md](SETUP.md)** for the toolchain a comparable stack requires.

## License

The private codebase is MIT-licensed. This showcase describes it; it does not
reproduce it.
