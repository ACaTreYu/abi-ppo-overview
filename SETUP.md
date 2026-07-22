# SETUP — Toolchain & Environment

What you would need to build and operate a comparable stack. This is a generic
description of the environment, not a copy of the project's private
configuration.

---

## Training machine (Python)

| Requirement | Notes |
|---|---|
| Python | 3.9+ (the project runs current 3.x) |
| NumPy | ≥ 1.24 — corpus loading, reward filtering, binary file parsing |
| PyTorch | ≥ 2.0, **CUDA build** — training runs on GPU; pick the wheel matching your CUDA toolchain from pytorch.org |
| GPU | Any recent CUDA-capable card; the network is small (two 356→256→256 MLPs), so VRAM is not the constraint — corpus size in RAM is |
| RAM | The trainer auto-sizes its file budget to ~75% of free RAM; more RAM = more of the experience corpus per run |
| OS | Windows or Linux both work; on Windows, force UTF-8 stdout for long unattended runs so a single log line can't kill a session |

The dependency surface is deliberately tiny — NumPy + PyTorch is the entire
training stack. No RL framework, no gym wrapper, no experiment tracker
required: the environment is a live game server, and experience arrives as
self-describing binary files.

```text
# requirements.txt (shape of it)
numpy>=1.24
torch>=2.0   # install the CUDA wheel matching your toolchain
```

## Game server (TypeScript runtime)

| Requirement | Notes |
|---|---|
| Node.js | Current LTS |
| TypeScript | Standard toolchain; the AI runtime compiles with the game server |
| ML runtime | **None.** Inference is hand-rolled TypeScript (matrix-vector forward pass, LayerNorm, ReLU, softmax) loading weights from JSON |
| Test runner | The runtime AI carries unit tests (pathfinding, perception, ballistics solvers, route validation) that run real map data in CI |

The only contract between trainer and server is a pair of files — the policy
weights (JSON) and their paired observation-normalizer stats — plus a shared,
versioned observation layout. Keep the width-check on both ends.

## Deployment target (production host)

| Requirement | Notes |
|---|---|
| Host | A small Linux VPS is sufficient for the game server + policy inference (inference is CPU, milliseconds per bot per tick) |
| Process manager | pm2 (or equivalent) — the pipeline needs a way to restart or hot-reload the server process |
| Transfer | SSH/SCP (the pipeline uses `paramiko` for scripted pulls and deploys) |
| Admin endpoint | An authenticated admin HTTP endpoint on the game server for hot-reloading the policy without a restart; key supplied via environment variable, never committed |

## Operational conventions worth copying

- **Secrets live outside the repo.** Host/credentials go in a gitignored local
  config created from a committed `*.example` template; API keys come from the
  environment. Nothing sensitive is ever a tracked file.
- **Validate before you destroy.** Every pulled experience file's header is
  checked (version, observation size) *before* the server-side pool is wiped,
  and the pull aborts on unexpected corpus shrinkage.
- **Deploy atomically, verify closed-loop.** Upload policy + normalizer to
  temp names and rename together; then compare on-disk checksums against the
  training output and confirm the process actually reloaded. Snapshot the live
  policy before every deploy so rollback is one command.
- **Version everything that crosses the wire.** Binary experience files carry
  a header (count, obs size, action size, version); loaders skip mismatches
  instead of crashing, and the runtime refuses mismatched policies instead of
  running them.

## Typical workflow

```bash
# one full closed-loop cycle: pull → validate → train → gate → deploy → verify
python ai_training_pipeline.py --once

# targeted trainer runs
python abi_ppo.py --epochs 300 --phase all
python abi_ppo.py --resume <checkpoint> --phase combat --epochs 50

# restore the previous policy and make it live
python ai_training_pipeline.py --rollback
```

Expect multi-day cadence per cycle: days of live data collection, a GPU
training session, then an automated gated deploy.
