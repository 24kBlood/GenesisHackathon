# DECISIONS.md — Architecture Decision Records (ADRs)

This document records the key engineering decisions behind **AuditLab** and the trade-offs considered.

---

## ADR-001: Pure Python 3.11+ Standard Library (Zero Runtime Dependencies)

- **Context**: Financial audit RL environments must be trivially reproducible across laptops, CI runners, and air-gapped containers (`2 CPU / 4 GB RAM`, no GPU, `--network=none`).
- **Decision**: Implement the entire platform—generator, AST calculator, 1-to-1 matcher, Gymnasium-compatible environment, IPC worker, RL trainer, SQLite store, CLI, 8-page Web UI, and `unittest` suites—using only Python 3.11+ standard library modules (`decimal`, `ast`, `json`, `csv`, `sqlite3`, `subprocess`, `http.server`, `unittest`, `random`).
- **Consequences**:
  - *Pros*: Zero dependency drift, instant offline builds, minimal attack surface, <3s full verification.
  - *Cons*: Custom implementations are required for bipartite matching, SVG charting, and policy-search RL instead of pulling `scipy`, `matplotlib`, or `torch`.

---

## ADR-002: AST-Based `Decimal` Calculator vs. `eval()` / Floating-Point Math

- **Context**: Agents need exact financial arithmetic (FX conversion, rounding, invoice/payment reconciliation) without floating-point rounding drift or remote code execution risks.
- **Decision**: Implement `safe_calculate(expression)` in `audit_rl/calculator.py` using `ast.parse(expression, mode="eval")` with an explicit node whitelist (`+`, `-`, `*`, `/`, `//`, `%`, unary `+`/`-`, and `abs`, `min`, `max`, `round`), converting numeric literals to `decimal.Decimal` and capping expression length at 256 characters. Disallow `**` to prevent CPU/memory exhaustion.
- **Consequences**:
  - *Pros*: Eliminates `eval()` injection vectors and IEEE-754 float errors (`0.1 + 0.2 != 0.3`).
  - *Cons*: Complex symbolic math or nested comprehensions are intentionally unsupported.

---

## ADR-003: Deterministic 1-to-1 Kuhn Bipartite Matcher vs. Greedy Set Matching

- **Context**: An episode can contain interacting errors sharing documents (e.g., `duplicate_invoice_fx_mismatch`). Naive greedy matching can depend on submission order, and naive set-overlap scoring can be gamed by submitting duplicates.
- **Decision**: Implement maximum-cardinality 1-to-1 bipartite matching (Kuhn's augmenting-path algorithm) in deterministic sorted order of `(issue_id, finding_index)` in `audit_rl/matcher.py`, requiring all 5 criteria (`type`, `set(affected_ids)`, `currency`, `tolerance_minor`, and `acceptable_evidence_sets` subset).
- **Consequences**:
  - *Pros*: Order-independent, optimal 1-to-1 assignment, duplicate findings are automatically penalized as `FP`.
  - *Cons*: Slightly higher implementation complexity than greedy matching (still sub-millisecond for `<= 30` findings).

---

## ADR-004: Out-of-Process Evaluator Isolation (`SubprocessAuditEnv` + JSON-Lines IPC)

- **Context**: In-process RL environments allow buggy or adversarial Python agents to inspect `env._truth` or internal RNG state via reflection.
- **Decision**: Provide `SubprocessAuditEnv`, which spawns `python3 -m audit_rl.eval_worker` as a separate process communicating strictly via JSON-lines over `stdin`/`stdout`.
- **Consequences**:
  - *Pros*: `truth.json`, `provenance.json`, and `seed` never enter the agent's Python memory space; zero extra dependencies.
  - *Cons*: Small IPC serialization overhead (~1–5 ms/episode). Process/API isolation is not an OS kernel sandbox for hostile native binaries; containerization is recommended for untrusted external code.

---

## ADR-005: Local SQLite Store (`PlatformStore`) for Experiment & Policy Persistence

- **Context**: Users running benchmarks and RL training from the CLI or Web UI need persistent, queryable experiment history and policy versions across restarts without setting up an external database server.
- **Decision**: Use stdlib `sqlite3` in `audit_rl/storage.py` writing to `data/platform.db` with parameterized SQL queries.
- **Consequences**:
  - *Pros*: ACID persistence, zero external services, trivial temporary-directory isolation in unit tests.
  - *Cons*: Single-node local persistence; not intended for high-concurrency distributed training clusters.

---

## ADR-006: Pure-Stdlib Policy-Search / Evolutionary RL Trainer vs. GPU Deep-RL Frameworks

- **Context**: AuditLab must demonstrate real, reproducible RL training (improving a learnable policy from terminal rewards on `train` split and generalizing to unseen `eval` episodes) in a CPU-only, offline, zero-dependency environment.
- **Decision**: Implement `LearnableAuditPolicy` and `run_training_experiment` in `audit_rl/trainer.py` using deterministic policy-search / evolutionary optimization over structured audit policy parameters trained strictly on `train` split terminal rewards.
- **Consequences**:
  - *Pros*: Runs in seconds on CPU, 100% deterministic, verifiable held-out improvement (`trained_eval.f1 > untrained_eval.f1`), and clean separation of `train` vs `eval`.
  - *Cons*: Intended as a lightweight reference trainer; external LLM/PPO/GRPO frameworks can plug into `SubprocessAuditEnv` via the standard JSON Gymnasium interface.
