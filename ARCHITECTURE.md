# ARCHITECTURE.md — AuditLab System Architecture

**AuditLab** — *An autonomous financial audit simulation and AI training environment.*

---

## 1. Architectural Philosophy: Explainable Four-Layer Separation

AuditLab separates financial simulation, exact arithmetic, agent policy reasoning, and deterministic evaluation into four distinct layers so that every audit finding is traceable and independently verifiable:

| Layer | Responsibility | Implementation |
|---|---|---|
| **1. Deterministic Accounting Rules & Data** | Generates causally consistent double-entry books, accounting policies, clean distractors, and verifiable anomalies. | `audit_rl/schemas.py`, `audit_rl/generator.py` |
| **2. Exact Arithmetic Sandbox** | Executes decimal financial calculations safely via AST parsing and `decimal.Decimal` (no IEEE-754 float drift, no dynamic code execution). | `audit_rl/calculator.py` |
| **3. Policy Reasoning & Training** | Interacts exclusively with open records via the narrow 5-tool interface; learns policy parameters on `train` split using terminal rewards without touching `eval`. | `audit_rl/policies.py`, `audit_rl/trainer.py` |
| **4. Independent Deterministic Evaluation** | Validates submission schemas, performs deterministic 1-to-1 Kuhn bipartite matching against private `truth.json`, and computes terminal rewards out-of-process. | `audit_rl/matcher.py`, `audit_rl/env.py`, `audit_rl/eval_worker.py` |

### Separation of Facts, Inferences, and Conclusions (Evidence-First Auditing)
Every audit finding produced by honest policies and inspected in the UI explicitly distinguishes:
- **Observed facts (`Observed:`)** — raw values read directly from open primary records (`invoices.jsonl`, `payments.jsonl`, `ledger.csv`, `fx.csv`) referenced by `evidence_ids`.
- **Calculated facts (`Calculated:`)** — exact `Decimal` / integer minor-unit arithmetic derived from observed records (e.g., expected FX conversion vs posted ledger amount, unallocated payment delta).
- **Rule & Conclusion (`Rule:`)** — the specific accounting rule from `policy.json` / `manifest.json` that was violated, mapped to one of the canonical `ERROR_TYPES`.

---

## 2. High-Level System Topology

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                        AuditLab Local Web UI (8 Pages)                   │
│       audit_rl/gui.py (ThreadingHTTPServer + Offline Embedded SPA)       │
└──────────────┬────────────────────────┬────────────────────────┬─────────┘
               │                        │                        │
               ▼                        ▼                        ▼
┌─────────────────────────┐  ┌─────────────────────┐  ┌────────────────────┐
│  Agent / Trainer Layer  │  │  SQLite Persistence │  │  CLI & Verifier    │
│  - policies.py          │  │  audit_rl/storage.py│  │  audit_rl/cli.py   │
│  - trainer.py (RL)      │  │  data/platform.db   │  │  audit_rl/rubric.py│
└──────────────┬──────────┘  └─────────────────────┘  └────────────────────┘
               │  JSON-lines IPC (stdin/stdout)
               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│            Out-of-Process Evaluator (SubprocessAuditEnv)                 │
│  audit_rl/eval_worker.py -> AuditEnv(mode="eval")                        │
│  - generator.py (open records + private truth.json / provenance.json)    │
│  - calculator.py (AST Decimal)                                           │
│  - matcher.py (Kuhn 1-to-1 bipartite matcher + terminal reward)          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Model: Open vs. Private Episode Artifacts

Schema version: `SCHEMA_VERSION = "1.0.0"`.

### Open Files (Accessible via Narrow Tool Interface)
| Artifact | Format | Key Fields & Value Guarantees |
|---|---|---|
| `manifest.json` | JSON | `episode_id`, `schema_version`, `period_start`, `period_end`, `timezone_offset`, `base_currency`, `currencies`, table counts |
| `policy.json` | JSON | `fx_date_policy` (`invoice_issue_date` / `payment_date` / `posting_date`), `rounding_rule` (`ROUND_HALF_EVEN` / `ROUND_HALF_UP`), cutoff rules |
| `ledger.csv` | CSV | `row_id,timestamp,account,debit_minor,credit_minor,currency,entity_id,invoice_id,payment_id` (empty string -> `None`) |
| `invoices.jsonl` | JSONL | `doc_id,invoice_id,entity_id,issued_at,due_at,currency,total_minor,line_items` (`description,quantity,unit_price_minor,amount_minor,service_date`) |
| `payments.jsonl` | JSONL | `doc_id,payment_id,timestamp,currency,amount_minor,invoice_refs` (list of `{invoice_id, allocated_minor}` or `null`) |
| `fx.csv` | CSV | `date,source_currency,target_currency,rate` (exact decimal string, key `"{date}:{source_currency}:{target_currency}"`) |

### Private Files (Evaluator-Only)
| Artifact | Format | Contents |
|---|---|---|
| `truth.json` | JSON | Canonical `issues` list (`issue_id`, `type`, `affected_ids`, `expected_amount_minor`, `currency`, `tolerance_minor`, `acceptable_evidence_sets`, `description`) |
| `provenance.json` | JSON | Generation config, `pre_injection_checks` (double-entry balance proof before error injection), and causal injection trace |

### Dataset Splits (`configs/`)
- `configs/train_configs.json` — 10 training episodes (seeds `1001..1010`).
- `configs/dev_configs.json` — 10 development episodes (seeds `2001..2010`).
- `configs/val_configs.json` — 10 validation episodes (distinct seeds, used for model selection).
- `configs/eval_configs.json` — 20 held-out evaluation episodes (seeds `9001..9020`: 5 clean, 5 interacting, 10 single/multi-error).

---

## 4. Out-of-Process Evaluator Isolation & Threat Model

- **`AuditEnv(mode="eval")`**: Always returns `reward = 0.0` from `step()` and strips `truth`, `provenance`, `seed`, and filesystem paths from `observation` and `info`. Metrics are retrievable only after termination via `get_evaluator_report()`.
- **`SubprocessAuditEnv` + `audit_rl/eval_worker.py`**: Spawns a dedicated child Python process communicating strictly over JSON-lines (`stdin`/`stdout`). The agent process never holds `truth.json`, `provenance.json`, or `seed` in its Python heap.
- **Honest Security Disclosure**: `SubprocessAuditEnv` enforces **process- and API-boundary isolation** suitable for preventing accidental or programmatic in-process state leakage by Python policies. It is **not** an OS-level kernel sandbox (e.g., seccomp/AppArmor/container namespace isolation) for executing untrusted hostile native binaries; running untrusted third-party agent code should additionally use container or VM isolation.

---

## 5. SQLite Persistence (`audit_rl/storage.py`)

`PlatformStore` manages a local SQLite database (`data/platform.db`, using Python stdlib `sqlite3` with parameterized queries) to persist:
- **Evaluation & Benchmark Runs**: split, policy name, aggregate `Precision`, `Recall`, `F1`, `Return`, `Steps`, `runtime_ms`, and timestamp.
- **RL Training Experiments & Policy Versions**: policy version ID, training seed, epochs, reward history curve, serialized parameters (`params_before` / `params_after`), and held-out `untrained_eval` vs `trained_eval` metrics.
- **Overview Aggregates**: live counts of persisted runs, evaluated policies, and latest benchmark summaries served to the Web UI without mock data.

---

## 6. 8-Page Offline Web Application (`audit_rl/gui.py`)

Served locally via stdlib `http.server.ThreadingHTTPServer` (`python3 -m audit_rl gui --host 127.0.0.1 --port 8765`) with an embedded HTML/CSS/JS SPA (zero external CDNs/fonts, Dark/Light theme toggle):

1. **Overview** (`GET /api/overview`) — Dataset split totals (`train`/`dev`/`val`/`eval`), persisted benchmark/training metrics from `PlatformStore`, recent experiments, and live system status.
2. **Company Explorer** (`POST /api/episode/inspect`) — Company profile, reporting period, accounting policy, and searchable paginated tables (`invoices`, `payments`, `ledger`, `fx`).
3. **Agent Workspace** (`POST /api/workspace/reset`, `POST /api/workspace/step`, `POST /api/episode/run`, `POST /api/calculator`) — Interactive manual investigation tools or one-click autonomous policy execution with live step-budget tracking.
4. **Evaluation Results** (`POST /api/record/lookup`, `POST /api/verify`) — Submitted findings, TP/FP/FN breakdown, Evidence ID lookup drawer (Observed / Calculated / Rule), and the full 100-point Platform Verification Suite.
5. **Training** (`POST /api/train`, `GET /api/training/history`) — Interactive RL training runner, persisted policy versions, inline SVG reward curve, and held-out comparison (`Untrained` vs `Trained` vs `Rule-Based`).
6. **Benchmarks** (`POST /api/baselines`) — Multi-policy comparison on held-out `eval` episodes with per-category breakdown (`clean`, `single`, `multi`, `interacting`), inline SVG grouped bar chart, and 20-episode heatmap.
7. **Trajectory Explorer** (`GET /api/trajectories`, `GET /api/trajectories/export`) — Real episode timelines from `trajectories/trajectories.jsonl` (and live runs) with step-by-step replay scrubber and JSONL export.
8. **Settings** (`GET /api/settings`) — Active environment constants (`DEFAULT_MAX_STEPS=100`, `MAX_PAGE_LIMIT=50`, `MAX_FINDINGS=30`, `SCHEMA_VERSION`), split counts, isolation mode, and training defaults.
