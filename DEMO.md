# DEMO.md — AuditLab Walkthrough & Sample Episode Trace

**AuditLab** — *An autonomous financial audit simulation and AI training environment.*

This guide walks through a 3–5 minute live demonstration of AuditLab (CLI, Web Application, RL training, and offline Docker container) and shows a step-by-step episode trace from `reset()` to final evaluation.

---

## Step-by-Step Walkthrough

### 1. Single-Command Platform Verification
Run the full verification suite completely offline:

```bash
python3 -m audit_rl verify
```

Verified highlights (baseline run — **Overall PASS in 2.74 s**):
- **Unit & integration tests**: **61 / 61** core tests pass in **1.22 s** (plus `test_training_and_storage.py` and `test_branding_and_security.py`).
- **Eval integrity**: **20 / 20** held-out eval episodes pass `verify_episode_integrity` (pre-injection double-entry balance + independent recalculation from open files).
- **Permutation & ID-remapping invariance**: **20 / 20** paired remapped episodes yield identical `Oracle F1 = 1.0` and `Return = 1.0`.
- **Trajectory leak scan**: **0 leaks** of `truth`, `provenance`, or `seed` across `trajectories/trajectories.jsonl` (30 runs, 341 steps).
- **4-baseline held-out comparison** in `SubprocessAuditEnv(mode="eval")`:
  - `oracle`: `P = 1.0000`, `R = 1.0000`, `F1 = 1.0000`, `Return = +1.0000` (`1.00 step`, `0.14 ms`)
  - `simple_reconciliation`: `P = 1.0000`, `R = 1.0000`, `F1 = 1.0000`, `Return = +0.9676` (`17.20 steps`, `4.98 ms`)
  - `random_explorer`: `P = 0.1000`, `R = 0.1000`, `F1 = 0.1000`, `Return = -0.8019` (`12.60 steps`, `1.78 ms`)
  - `flag_everything_attack`: `P = 0.0017`, `R = 0.0250`, `F1 = 0.0031`, `Return = -0.9940` (`4.00 steps`, `1.71 ms`)

### 2. Interactive CLI Episode Walkthrough
```bash
python3 -m audit_rl demo
```
Prints the full lifecycle of a single episode: sanitized `reset()` observation, paginated `list_records` calls, `get_document`/`get_record`, exact AST `Decimal` `calculator` call, `submit`, and final evaluator report.

### 3. Reproducible RL Training Experiment
```bash
python3 -m audit_rl train --epochs 2 --seed 42
```
Trains `LearnableAuditPolicy` strictly on `configs/train_configs.json` using terminal rewards, persists the policy version and reward curve to `data/platform.db`, and reports held-out `Untrained` vs `Trained` improvement.

### 4. 8-Page Offline Web Application
```bash
python3 -m audit_rl gui --host 127.0.0.1 --port 8765
```
Open `http://127.0.0.1:8765` to explore all 8 pages: Overview, Company Explorer, Agent Workspace (manual investigation + autonomous run), Evaluation Results (Evidence ID lookup + 100-point Verification Suite), Training, Benchmarks, Trajectory Explorer (replay + JSONL export), and Settings.

### 5. Offline Docker Container (`2 CPU / 4 GB RAM`, No Network)
```bash
docker build -t auditlab:latest .
docker run --rm --network=none --cpus=2 --memory=4g auditlab:latest
```

---

## Sample Episode Walkthrough

Example interaction with `SubprocessAuditEnv(mode="eval")` on an episode containing an FX conversion discrepancy (`fx_conversion_error`).

### 1. Reset
```python
from audit_rl.env import SubprocessAuditEnv

with SubprocessAuditEnv(mode="eval", max_steps=100) as env:
    obs, info = env.reset(seed=9003, config=eval_config)
```

Sanitized initial observation (zero `truth`, `provenance`, or `seed`):
```json
{
  "task": "Audit the company financial records for the reporting period and submit all verifiable discrepancies.",
  "manifest": {
    "episode_id": "ep_9003_...",
    "schema_version": "1.0.0",
    "base_currency": "USD",
    "currencies": ["USD", "EUR", "GBP", "KZT", "JPY"],
    "timezone_offset": "+05:00",
    "period_start": "2026-01-01T00:00:00+05:00",
    "period_end": "2026-03-31T23:59:59+05:00"
  },
  "policy": {
    "fx_date_policy": "invoice_issue_date",
    "rounding_rule": "ROUND_HALF_EVEN"
  },
  "budget_remaining": 100,
  "step": 0,
  "available_actions": ["list_records", "get_record", "get_document", "calculator", "submit"]
}
```

### 2. Paginated Inspection (`list_records`)
```json
{"action": "list_records", "table": "invoices", "cursor": "0", "limit": 50}
{"action": "list_records", "table": "ledger", "cursor": "0", "limit": 50}
```

### 3. Document & FX Rate Lookup (`get_document` + `get_record`)
```json
{"action": "get_document", "doc_id": "DOC-INV-014"}
{"action": "get_record", "table": "fx", "id": "2026-02-10:EUR:USD"}
```
Returns exact decimal string `"1.084500"`.

### 4. Exact Arithmetic via AST `Decimal` Calculator (`calculator`)
```json
{"action": "calculator", "expression": "abs(124500 - round(112000 * 1.084500))"}
```
Returns `"3036"` (`30.36 USD`, exceeding `tolerance_minor = 1`). Unsafe expressions such as `__import__('os')` or `2 ** 10000` are rejected with a bounded error consuming 1 step.

### 5. Evidence-First Submission (`submit`)
```json
{
  "action": "submit",
  "findings": {
    "episode_id": "ep_9003_...",
    "findings": [
      {
        "type": "fx_conversion_error",
        "affected_ids": ["INV-014"],
        "claimed_amount_minor": 3036,
        "currency": "USD",
        "evidence_ids": ["DOC-INV-014", "LED-0087", "2026-02-10:EUR:USD"],
        "explanation": "Observed: INV-014 total 112000 EUR, ledger LED-0087 posted 124500 USD, FX 2026-02-10:EUR:USD = 1.084500. Calculated: expected 121464 USD, diff 3036 USD minor. Rule: policy fx_date_policy=invoice_issue_date with ROUND_HALF_EVEN."
      }
    ]
  }
}
```

In `mode="eval"`, `step()` returns `reward = 0.0, terminated = True, truncated = False`.

### 6. Evaluator Report (`get_evaluator_report`)
```python
report = env.get_evaluator_report()
```
```json
{
  "valid_submission": true,
  "tp": 1,
  "fp": 0,
  "fn": 0,
  "precision": 1.0,
  "recall": 1.0,
  "f1": 1.0,
  "inspection_steps": 5,
  "reward": 0.99,
  "terminated": true,
  "truncated": false
}
```
Terminal reward: `2 * 1.0 - 1.0 - 0.002 * 5 = 0.99`.
