---
name: fintech-sandbox-engineering
description: Engineering guide for AuditLab's offline FinTech sandbox, out-of-process evaluator isolation (SubprocessAuditEnv), safe AST Decimal calculator, and narrow tool interface. Use when modifying audit_rl/env.py, audit_rl/eval_worker.py, audit_rl/calculator.py, or audit_rl/gui.py.
---

# FinTech Sandbox Engineering

## When to Use
- Modifying or extending the narrow agent tool interface (`list_records`, `get_record`, `get_document`, `calculator`, `submit`).
- Hardening or debugging out-of-process evaluation via `SubprocessAuditEnv` and `audit_rl/eval_worker.py`.
- Updating `audit_rl/calculator.py` or enforcing step/timeout/pagination budgets.
- Exposing sandbox endpoints in `audit_rl/gui.py`.

## Core Invariants
1. **Pure Python 3.11+ Stdlib, Offline-First**: Never introduce external runtime dependencies, network calls, or CDN URLs.
2. **Process & Memory Boundary**: In `eval` mode, `truth.json`, `provenance.json`, and `seed` must live exclusively inside `eval_worker.py` (communicating via stdin/stdout JSON-lines). Note honestly in docs that process/API isolation is not an OS-level kernel sandbox.
3. **Narrow Tool Contract**:
   - `list_records`: `1 <= limit <= 50` (`MAX_PAGE_LIMIT`), stable deterministic cursors (`"0"`, `"50"`, ...).
   - `get_record`: direct lookup by `row_id`, `invoice_id`, `payment_id`, or `"{date}:{source}:{target}"` FX key.
   - `get_document`: lookup by `doc_id`.
   - `calculator`: AST-only `Decimal` arithmetic (`ast.parse(expr, mode="eval")`), max 256 chars, no `eval()`, no `**`, no attributes/subscripts/imports.
   - `submit`: terminates episode (`terminated=True`, `truncated=False`).
4. **Bounded Errors**: Invalid actions, unknown IDs, out-of-range limits, or unsafe calculator expressions must return a structured error dict in `observation`, consume 1 step, and return `reward = 0.0`.
5. **Post-Termination Immutability**: Calling `step()` after `terminated` or `truncated` must return a bounded error without mutating final metrics or reward.

## Workflow
1. Inspect `audit_rl/env.py`, `audit_rl/eval_worker.py`, and `audit_rl/calculator.py` before making changes.
2. Ensure every new observation field is JSON-serializable and free of `truth`, `provenance`, `seed`, and local filesystem paths.
3. Verify `AuditEnv(mode="eval")` always returns `reward = 0.0` from `step()` and exposes diagnostics only via `get_evaluator_report()`.
4. Run targeted isolation and contract tests:
   ```bash
   python3 -m unittest tests/test_env_contract_and_isolation.py -v
   ```
5. Run full platform verification:
   ```bash
   python3 -m audit_rl verify
   ```

## Verification Checklist
- [ ] Zero external runtime imports added.
- [ ] `SubprocessAuditEnv` context manager cleanly spawns and terminates `eval_worker.py`.
- [ ] `calculator` rejects `__import__`, `open`, `eval`, `**`, attributes, and >256-char inputs.
- [ ] `tests/test_env_contract_and_isolation.py` passes 100%.
- [ ] `python3 -m audit_rl verify` exits 0 with 100.0/100.0 score.
