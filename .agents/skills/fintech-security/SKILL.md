---
name: fintech-security
description: Security checklist and hardening workflow for AuditLab's AST Decimal calculator, out-of-process evaluator IPC, local Web UI HTTP API, SQLite persistence, and zero-leakage guarantees. Use when editing calculator.py, env.py, eval_worker.py, gui.py, or storage.py.
---

# FinTech Security & Isolation Hardening

## When to Use
- Modifying expression parsing in `audit_rl/calculator.py`.
- Adding or changing HTTP endpoints in `audit_rl/gui.py`.
- Editing IPC serialization in `audit_rl/env.py` / `audit_rl/eval_worker.py` or SQL queries in `audit_rl/storage.py`.
- Reviewing code for secret/truth leakage, path traversal, or unsafe deserialization.

## Security Controls
1. **No Dynamic Code Execution**: Never use `eval()`, `exec()`, `compile()`, or `pickle` on untrusted input. `audit_rl/calculator.py` must parse via `ast.parse(..., mode="eval")` with a strict AST node & function whitelist (`abs`, `min`, `max`, `round`), rejecting `**` (CPU/memory DoS) and inputs `> 256` chars.
2. **Parameterized SQLite Queries**: Every query in `audit_rl/storage.py` must use `?` placeholders—never string interpolation.
3. **HTTP Server Hardening (`audit_rl/gui.py`)**:
   - Bind to `127.0.0.1` by default.
   - Enforce request body size limits and catch malformed JSON cleanly (return HTTP 400 JSON error, never stack traces).
   - Reject path traversal (`..`, absolute paths, null bytes) on any route or file lookup.
   - Zero external CDN scripts, fonts, or stylesheets in embedded HTML.
4. **Evaluator Privacy & Honest Threat Model**:
   - Never expose `truth`, `provenance`, `seed`, or local filesystem paths in `observation`, `info`, or `trajectories/trajectories.jsonl`.
   - Document honestly that `SubprocessAuditEnv` provides process/API boundary isolation—not an OS-level kernel/seccomp sandbox for hostile native binaries.

## Workflow
1. Run security, isolation, and branding checks:
   ```bash
   python3 -m unittest tests/test_env_contract_and_isolation.py tests/test_branding_and_security.py -v
   ```
2. Scan for forbidden dynamic execution or external CDN references:
   ```bash
   grep -rnE "(pickle\.load|eval\(|exec\(|https?://cdn|fonts\.googleapis)" audit_rl/
   ```
3. Run full verification:
   ```bash
   python3 -m audit_rl verify
   ```

## Verification Checklist
- [ ] Zero `eval()`, `exec()`, or `pickle` calls in `audit_rl/`.
- [ ] All SQLite queries use parameterized placeholders.
- [ ] Malformed JSON, oversized bodies, and path-traversal requests to `gui.py` return clean 4xx responses.
- [ ] Zero external CDN URLs in `audit_rl/gui.py`.
- [ ] `tests/test_branding_and_security.py` passes 100%.
