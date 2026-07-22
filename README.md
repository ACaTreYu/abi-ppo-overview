# ABI-PPO — ARCbound Intelligence

**A production reinforcement-learning system for real-time multiplayer game AI.**

ABI-PPO is the complete AI stack behind the bots in *Arcbound* (the browser-based
top-down multiplayer arena shooter by [ArcBound Interactive](https://arcboundinteractive.com)): a custom PPO trainer, a
rule-based tactical brain with the neural policy blended in as an advisor, a
zero-dependency inference engine that runs inside the game server, and a fully
closed-loop collect → train → deploy pipeline that retrains the live policy on
its own gameplay.

This repository overview is a public showcase of a private codebase. The
algorithms are described at concept level; implementations remain private.

---

## Novel Systems & Methods

The project's core contribution is making PPO *work* in a setting where
textbook PPO fails: long-horizon (~10,000-tick) episodes, sparse terminal
rewards, and a live game server as the only environment. Seven original systems
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
per-tick dodge and aim signal. A companion technique performs *retroactive*
attribution: because the server emits one scalar per tick, the trainer
classifies each reward back to its source channel from the known reward
taxonomy, with deliberate "permissive overlap" at ambiguous values (a signal
that plausibly belongs to two channels feeds both — extra gradient, not noise).

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

### 4. Advisor-Blend Hybrid Control
The shipped bots are not a pure neural agent. A rule-based state machine owns
*strategy* (objectives, roles, routes) and the PPO policy rides on top as an
*advisor* shaping moment-to-moment combat feel — strafing, dodging, aim
micro-correction. The blend is continuous: a strategy-urgency weight decides
how much the net's movement counts each tick, and the aim head predicts a
*target-relative offset* (a ±90° correction around the rule layer's chosen
target) rather than an absolute angle. Difficulty tiers scale the same
influence dial. This yields human-feeling combat without ever letting the net
throw a match by ignoring an objective.

### 5. Map-Context Conditioning ("Map Digest")
A compact static descriptor of the current map and mode — mode one-hot,
topology flags, arena scalars, a coarse wall-density grid, objective anchor
positions — is appended to the observation. It is computed once per map (zero
per-tick cost) and cures cross-map strategy bleed: one shared policy plays
every map and mode instead of averaging them into mush. The append is
compatibility-safe (the legacy observation prefix is byte-identical) and a
dimension guard at load time refuses any policy whose input width doesn't
match, falling back to pure rules rather than shipping garbage.

### 6. Reward-Ranked, Per-Bucket Data Curation
Experience files are bucketed by mode and map. When the corpus exceeds RAM,
the trainer keeps the highest-mean-reward files *ranked within each bucket,
never globally* — reward scales aren't comparable across game modes, and a
global ranking would let one mode crowd out the rest. Selection reads only
file headers plus the reward slice, so ranking millions of transitions costs
almost nothing, and it mirrors the same score the server's own eviction policy
curates on. Every map stays represented; only the best play per bucket trains.

### 7. Human-Anchored Training from Replays
A replay-ingestion path reconstructs the full training observation,
slice-for-slice, from years of recorded *human* games — including encoding
quirks like mixed text encodings in legacy replays and fire events that fall
between movement samples. The resulting behavior-cloning corpus is used two
ways: to initialize a fresh policy from human play, and as a *frozen KL
anchor* — a divergence penalty toward the human-cloned reference — so PPO can
optimize reward without drifting into inhuman play ("human ↔ optimal" as a
single tunable knob).

**Also notable:** a self-healing deployment loop (pre-deploy sanity gate that
mirrors the runtime's dimension guard, atomic policy + normalizer deploy so the
two can never mismatch, checksum-verified go-live, automatic backup and
one-command rollback), and a pure-TypeScript inference engine — the trained
network runs inside the Node game server with no ML runtime dependency at all,
from weights exported as JSON. Invalid-action masking follows the published
MaskablePPO approach (Huang & Ontañón, 2020) in an independent implementation.

---

## Status

**Live in production.** The current-generation policy (356-dimensional
observation, v7) ships in the game's public showcase servers and is retrained
on multi-day, reward-curated data-collection cycles. Lifetime training ledger:
**~110 M unique transitions ingested** and **~15.7 billion cumulative training
samples** over 966 global epochs across all three curriculum phases.
Movement and combat heads are strong; ongoing work targets the long-horizon
strategy value head and a utility-scoring decision substrate that runs in
shadow alongside the rule brain.

**Stats:** 79 commits · Python + TypeScript · ~7 K lines of Python
(trainer + pipeline) and ~21 K lines of TypeScript (runtime AI), plus the
deployed model artifacts and format documentation.

---

## What's in the (private) repo

| Area | Contents |
|---|---|
| Trainer | The ABI-PPO trainer: dual-optimizer PPO, reward decomposition, staged curriculum, action masking, BC anchoring |
| Runtime | TypeScript game-server AI: rule brain, perception, pathfinding, per-opponent dodge modeling, weapon ballistics solvers, RL glue (observation encoding, inference, experience collection, reward shaping) |
| Pipeline | Closed-loop automation: pull experience → validate → train → sanity-gate → atomic deploy → verify → rollback |
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
