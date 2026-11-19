# PROJECT_STATUS.md — AuditLab Engineering Status

**Product**: AuditLab — *An autonomous financial audit simulation and AI training environment.*  
**Schema Version**: `1.0.0`  
**Runtime**: Python 3.11+ pure standard library (zero external runtime dependencies, 100% offline, CPU-only)

---

## 1. Git & Pull Request Status

- **Current Git Branch**: `main` (tracking `origin/main`)
- **Open Pull Requests**: None (active milestone changes being integrated and verified locally before PR/push)

---

## 2. Completed Work & Current Implementation State

| Subsystem | Files | Status |
|---|---|---|
| Schemas & Causal Synthetic Generator | `audit_rl/schemas.py`, `audit_rl/generator.py`, `audit_rl/__init__.py` | Implemented & verified |
| Safe AST Decimal Calculator | `audit_rl/calculator.py` | Implemented & verified |
| Deterministic 1-to-1 Kuhn Matcher & Reward | `audit_rl/matcher.py` | Implemented & verified |
| Gymnasium Env & Out-of-Process IPC Worker | `audit_rl/env.py`, `audit_rl/eval_worker.py` | Implemented & verified |
| Baselines, CLI & Verification Suite | `audit_rl/policies.py`, `audit_rl/cli.py`, `audit_rl/__main__.py`, `audit_rl/rubric.py` | Implemented & verified |
| Versioned Dataset Splits & Trajectories | `configs/{train,dev,val,eval}_configs.json`, `data/train/`, `data/dev/`, `trajectories/trajectories.jsonl` | Implemented & verified |
| SQLite Experiment Store & RL Trainer | `audit_rl/storage.py`, `audit_rl/trainer.py` | In final integration & verification |
| 8-Page Offline Web Application | `audit_rl/gui.py` | In final integration & verification |
| Governance Docs & 8 Custom Agent Skills | `README.md`, `ARCHITECTURE.md`, `DESIGN.md`, `DEMO.md`, `DECISIONS.md`, `IMPLEMENTATION_PLAN.md`, `CONTRIBUTING.md`, `PROJECT_STATUS.md`, `.agents/skills/*/SKILL.md` | Complete |
| Offline Container Packaging | `Dockerfile`, `pyproject.toml`, `requirements.txt`, `LICENSE` | Complete |

---

## 3. Verified Test & Benchmark Results

Last full verification baseline (`python3 -m audit_rl verify`):
- **Overall Result**: **PASS (2.74 s)**
- **Unit & Integration Tests** (`python3 -m unittest discover -s tests -v`): **61 / 61 OK (1.22 s)** across 5 core suites (plus 2 new suites `tests/test_training_and_storage.py` and `tests/test_branding_and_security.py` being added in the current pass).
- **Eval Integrity Checks**: **20 / 20 PASS** (`verify_episode_integrity` on `configs/eval_configs.json`).
- **Permutation & ID-Remapping Invariance**: **20 / 20 PASS** (`permute_and_relabel_episode`, `Oracle F1 = 1.0`, `Return = 1.0`).
- **Trajectory Leak Scan**: **PASS** (`trajectories/trajectories.jsonl`, 30 runs, 341 steps, 0 leaks of `truth`, `provenance`, `seed`, or local paths).

Measured held-out eval baselines (20 episodes, `SubprocessAuditEnv`, `mode="eval"`):

| Policy | Precision | Recall | F1 | Return | Steps | Avg Runtime |
|---|---:|---:|---:|---:|---:|---:|
| `oracle` | `1.0000` | `1.0000` | `1.0000` | `+1.0000` | `1.00` | `0.14 ms` |
| `simple_reconciliation` | `1.0000` | `1.0000` | `1.0000` | `+0.9676` | `17.20` | `4.98 ms` |
| `random_explorer` | `0.1000` | `0.1000` | `0.1000` | `-0.8019` | `12.60` | `1.78 ms` |
| `flag_everything_attack` | `0.0017` | `0.0250` | `0.0031` | `-0.9940` | `4.00` | `1.71 ms` |

---

## 4. Known Limitations

1. **Synthetic Accounting Scope**: Models invoices, payments, FX rates, and double-entry ledger postings across 4 canonical error types and 2 interacting scenarios; does not model full IFRS/GAAP (tax, depreciation, consolidations, derivatives).
2. **Not Professional Audit Advice**: Built exclusively for AI/RL research, simulation, and agent evaluation—not real-world financial auditing.
3. **Process Isolation vs. Kernel Sandboxing**: `SubprocessAuditEnv` provides Python process/API memory isolation, not an OS-level kernel sandbox (seccomp/VM) for untrusted hostile native binaries.
4. **Lightweight Reference RL Trainer**: `audit_rl/trainer.py` optimizes structured policy parameters on CPU with stdlib only; deep/LLM RL algorithms connect externally via `SubprocessAuditEnv`.

---

## 5. Remaining Work

- Complete final verification pass of `tests/test_training_and_storage.py` and `tests/test_branding_and_security.py` alongside `audit_rl/storage.py`, `audit_rl/trainer.py`, and the 8-page `audit_rl/gui.py`.
- Record measured `untrained_eval` vs `trained_eval` delta from `python3 -m audit_rl train` in `README.md` once the final verification run completes.

---

## 6. Exact Resume Commands

```bash
# Run full unit & integration test suite
python3 -m unittest discover -s tests -v

# Run single-command platform verification
python3 -m audit_rl verify

# Run deterministic RL training experiment (persists to data/platform.db)
python3 -m audit_rl train --epochs 2 --seed 42

# Launch the 8-page local offline Web UI
python3 -m audit_rl gui --host 127.0.0.1 --port 8765
```
