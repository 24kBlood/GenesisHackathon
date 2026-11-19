# AuditLab

**An autonomous financial audit simulation and AI training environment.**

**AuditLab** (`audit-rl`) is a deterministic, verifiable reinforcement learning (RL) environment and local research platform built on **pure Python 3.11+ standard library** (zero external runtime dependencies, 100% offline, CPU-only). It generates causally consistent double-entry synthetic company books with pre-verified accounting invariants, injects verifiable financial discrepancies, trains and benchmarks autonomous audit agents, and evaluates submissions in an isolated process using an open 1-to-1 bipartite matcher.

---

## Key Capabilities

- **Zero External Runtime Dependencies & 100% Offline**: Pure Python 3.11+ stdlib (`decimal`, `ast`, `sqlite3`, `http.server`, `subprocess`, `unittest`); runs comfortably within `2 CPU / 4 GB RAM` and no GPU.
- **Out-of-Process Evaluator Isolation**: `AuditEnv(mode="eval")` hides rewards and ground truth from `observation`/`info`, while `SubprocessAuditEnv` + `audit_rl/eval_worker.py` isolates `truth.json`, `provenance.json`, and `seed` in a separate process over JSON-lines IPC.
- **Causal Synthetic Accounting & Pre-Injection Invariants**: Verifies double-entry balance (`sum(debit_minor) == sum(credit_minor)`) before injecting errors; includes 5 clean distractor classes (split payments, refunds/credit notes, legitimate recurring identical amounts, valid multi-currency conversions, valid boundary timestamps).
- **4 Canonical Error Types + 2 Interacting Multi-Document Scenarios**:
  - `duplicate_invoice`
  - `payment_allocation_mismatch`
  - `fx_conversion_error`
  - `period_cutoff_error`
  - Interacting scenarios: `fx_period_partial_payment` and `duplicate_invoice_fx_mismatch`.
- **Narrow 5-Tool Agent Interface & Safe AST `Decimal` Calculator**: `list_records` (`<= 50` rows/page), `get_record`, `get_document`, `calculator` (AST-only `Decimal`, no `eval()`), and `submit` within a 100-step budget.
- **Evidence-First Auditing & Explainability**: Separates observed open records (`Observed:`), exact arithmetic (`Calculated:`), and policy rules (`Rule:`) with strict `acceptable_evidence_sets` verification.
- **Pure-Stdlib RL Training & SQLite Persistence**: `audit_rl/trainer.py` trains `LearnableAuditPolicy` strictly on the `train` split using terminal rewards and persists runs/policy versions to `data/platform.db` (`audit_rl/storage.py`).
- **8-Page Offline Web Application**: Local `ThreadingHTTPServer` dashboard (`python3 -m audit_rl gui`) with Dark/Light theme toggle and zero external CDNs.

---

## Architecture Overview

```text
Web UI (8-page offline SPA) / CLI (audit_rl/cli.py)
        │
        ├── Policies & RL Trainer (policies.py, trainer.py)
        ├── SQLite Persistence (storage.py -> data/platform.db)
        └── JSON-lines IPC ──▶ SubprocessAuditEnv / eval_worker.py
                                 ├── AuditEnv (train / eval)
                                 ├── generator.py (open + private truth/provenance)
                                 ├── calculator.py (safe AST Decimal)
                                 └── matcher.py (Kuhn 1-to-1 + terminal reward)
```

For full architectural details, threat model disclosures, and ADRs, see [ARCHITECTURE.md](ARCHITECTURE.md), [DESIGN.md](DESIGN.md), and [DECISIONS.md](DECISIONS.md).

---

## Repository Structure

```text
.
├── audit_rl/
│   ├── __init__.py          # Public package exports
│   ├── __main__.py          # CLI entrypoint (`python3 -m audit_rl`)
│   ├── schemas.py           # Constants, EpisodeConfig, payload validation, serialization
│   ├── generator.py         # Causal generator, verify_episode_integrity, permute_and_relabel_episode
│   ├── calculator.py        # Safe AST Decimal calculator (no eval)
│   ├── matcher.py           # Normalization, 1-to-1 Kuhn bipartite matcher, terminal reward
│   ├── env.py               # AuditEnv (train/eval) & SubprocessAuditEnv (IPC)
│   ├── eval_worker.py       # Out-of-process evaluator worker
│   ├── policies.py          # Baseline policies + leak-free trajectory recording
│   ├── trainer.py           # LearnableAuditPolicy & deterministic stdlib RL trainer
│   ├── storage.py           # SQLite PlatformStore (data/platform.db)
│   ├── rubric.py            # 100-point Platform Verification Suite
│   ├── gui.py               # 8-page offline local Web Application
│   └── cli.py               # CLI commands: verify, train, run-baselines, gui, demo, generate-datasets
├── configs/
│   ├── train_configs.json   # 10 training episodes (seeds 1001..1010)
│   ├── dev_configs.json     # 10 development episodes (seeds 2001..2010)
│   ├── val_configs.json     # 10 validation episodes (distinct seeds)
│   └── eval_configs.json    # 20 held-out eval episodes (seeds 9001..9020)
├── data/                    # Generated open datasets & local SQLite store
├── trajectories/            # Recorded leak-scanned trajectories (trajectories.jsonl)
├── tests/                   # Pure-stdlib unittest suites
├── .agents/skills/          # 8 custom engineering Agent Skills
├── ARCHITECTURE.md          # System architecture & explainability layers
├── DESIGN.md                # Technical design specification
├── DEMO.md                  # Interactive walkthrough & episode trace
├── DECISIONS.md             # Architecture Decision Records (ADRs)
├── IMPLEMENTATION_PLAN.md   # Phased implementation roadmap
├── PROJECT_STATUS.md        # Current engineering status & test metrics
├── CONTRIBUTING.md          # Branching, PR checklist, and security rules
├── Dockerfile               # Offline CPU-only container (python:3.11-slim)
├── pyproject.toml           # PEP 621 metadata & `audit-rl` console script
├── requirements.txt         # Zero external runtime dependencies note
└── LICENSE                  # MIT License (2026)
```

---

## Quickstart & CLI Commands

### 1. Single-Command Platform Verification
```bash
python3 -m audit_rl verify
```
Runs the complete verification pipeline: all pure-stdlib `unittest` suites, 20 held-out eval integrity checks, 20 paired permutation & ID-remapping checks, trajectory leak scan, and all baseline evaluations in `SubprocessAuditEnv(mode="eval")`.

### 2. All CLI Commands
```bash
# Train LearnableAuditPolicy on train split (persists to data/platform.db)
python3 -m audit_rl train --epochs 2 --seed 42

# Launch the 8-page offline Web Application (http://127.0.0.1:8765)
python3 -m audit_rl gui --host 127.0.0.1 --port 8765

# Run baseline benchmark suite on 20 held-out eval episodes
python3 -m audit_rl run-baselines

# Step-by-step CLI walkthrough of a single audit episode
python3 -m audit_rl demo

# Regenerate open train/dev datasets and trajectories/trajectories.jsonl
python3 -m audit_rl generate-datasets

# Run pure-stdlib unit & integration tests directly
python3 -m unittest discover -s tests -v
```

### 3. Offline Docker Execution (`2 CPU / 4 GB RAM`, No Network, No GPU)
```bash
docker build -t auditlab:latest .
docker run --rm --network=none --cpus=2 --memory=4g auditlab:latest
docker run --rm --network=none --cpus=2 --memory=4g auditlab:latest python -m audit_rl demo
```

---

## 8-Page Offline Web Application (`python3 -m audit_rl gui`)

1. **Overview**: Dataset totals across `train`/`dev`/`val`/`eval`, persisted benchmark & training metrics from `PlatformStore`, recent experiments, and live system status.
2. **Company Explorer**: Browse any episode's company profile, reporting period, accounting policy, and searchable paginated tables (`invoices`, `payments`, `ledger`, `fx`).
3. **Agent Workspace**: Interactive manual investigation tools (`list_records`, `get_record`, `get_document`, safe AST `calculator`, `submit`) or one-click autonomous policy run with step-budget tracking.
4. **Evaluation Results**: Submitted findings, TP/FP/FN breakdown, clickable Evidence ID lookup drawer (`Observed:` / `Calculated:` / `Rule:`), and the 100-point Platform Verification Suite.
5. **Training**: Launch deterministic RL training runs, inspect persisted policy versions, view the real SVG reward curve, and compare `Untrained` vs `Trained` vs `Rule-Based` held-out F1/Return.
6. **Benchmarks**: Multi-policy comparison across held-out `eval` episodes with per-category breakdown (`clean`, `single`, `multi`, `interacting`), SVG grouped bar chart, and 20-episode heatmap.
7. **Trajectory Explorer**: Inspect real recorded episodes from `trajectories/trajectories.jsonl` (and live runs) with a step-by-step replay scrubber and JSONL export.
8. **Settings**: Inspect active environment constants (`DEFAULT_MAX_STEPS=100`, `MAX_PAGE_LIMIT=50`, `MAX_FINDINGS=30`, `SCHEMA_VERSION`), dataset split counts, and isolation configuration.

---

## Dataset Splits & Trajectory Format

| Split | Config | Episodes | Purpose |
|---|---|---:|---|
| `train` | [configs/train_configs.json](configs/train_configs.json) | 10 | Policy training & RL reward optimization |
| `dev` | [configs/dev_configs.json](configs/dev_configs.json) | 10 | Local development & debugging |
| `val` | [configs/val_configs.json](configs/val_configs.json) | 10 | Validation & hyperparameter checks |
| `eval` | [configs/eval_configs.json](configs/eval_configs.json) | 20 | Strictly held-out evaluation (5 clean, 5 interacting, 10 single/multi-error) |

Recorded trajectories in `trajectories/trajectories.jsonl` store one JSON object per step (`episode_id`, `policy`, `step`, `action`, `observation`, `reward`, `terminated`, `truncated`) and are scanned to guarantee **0 leaks** of `truth`, `provenance`, `seed`, or local filesystem paths.

---

## Measured Verification & Benchmark Results

Verified via `python3 -m audit_rl verify` (**Overall PASS in 2.74 s**):
- **Core Unit & Integration Tests**: **61 / 61 OK (1.22 s)** (extended with `test_training_and_storage.py` and `test_branding_and_security.py`).
- **Eval Integrity Checks**: **20 / 20 PASS** (`verify_episode_integrity`).
- **Permutation & ID-Remapping Invariance**: **20 / 20 PASS** (`permute_and_relabel_episode`, `Oracle F1 = 1.0`, `Return = 1.0`).
- **Trajectory Leak Scan**: **PASS** (`trajectories/trajectories.jsonl`: 30 runs, 341 steps, **0 leaks**).

Measured baselines on 20 held-out `eval` episodes (`SubprocessAuditEnv`, `mode="eval"`):

| Policy | Mean Precision | Mean Recall | Mean F1 | Mean Return | Mean Steps | Mean Runtime (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `oracle` (evaluator-only ceiling) | `1.0000` | `1.0000` | `1.0000` | `+1.0000` | `1.00` | `0.14` |
| `simple_reconciliation` (rule-based) | `1.0000` | `1.0000` | `1.0000` | `+0.9676` | `17.20` | `4.98` |
| `random_explorer` | `0.1000` | `0.1000` | `0.1000` | `-0.8019` | `12.60` | `1.78` |
| `flag_everything_attack` (adversarial) | `0.0017` | `0.0250` | `0.0031` | `-0.9940` | `4.00` | `1.71` |

---

## Failure Analysis & Honest Limitations

| Failure Mode / Limitation | Measured Behavior / Root Cause | Mitigation & Scope Note |
|---|---|---|
| **Specification gaming (`flag_everything_attack`)** | `P = 0.0017`, `F1 = 0.0031`, `Return = -0.9940` | 1-to-1 Kuhn matcher + strict `acceptable_evidence_sets` + clean-book penalty (`-1.0`) neutralize spam attacks. |
| **Untrained / random exploration (`random_explorer`)** | `F1 = 0.1000`, `Return = -0.8019` | Only scores when randomly submitting empty findings on clean books; cannot reconcile cross-table records. |
| **Step-budget trade-off (`simple_reconciliation`)** | `F1 = 1.0000`, `Return = +0.9676` (`17.20 steps`) | `-0.0324` gap vs `oracle` reflects the step-economy penalty `0.002 * inspection_steps`. |
| **Synthetic accounting scope** | Covers 4 error classes & 2 interacting scenarios | Research simulation only; not full IFRS/GAAP and not professional audit advice. |
| **Process isolation boundary** | `SubprocessAuditEnv` isolates Python memory/API state | Not an OS-level kernel/seccomp sandbox for untrusted native binaries; use containers for hostile code. |

---

## Documentation & License

- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture, data model, and explainability layers
- [DESIGN.md](DESIGN.md) — Detailed technical specification
- [DEMO.md](DEMO.md) — Step-by-step walkthrough & sample episode trace
- [DECISIONS.md](DECISIONS.md) — Architecture Decision Records (ADRs)
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — Phased roadmap
- [PROJECT_STATUS.md](PROJECT_STATUS.md) — Engineering status & resume commands
- [CONTRIBUTING.md](CONTRIBUTING.md) — Contribution & security guidelines
- [LICENSE](LICENSE) — MIT License (2026)
