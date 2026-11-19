---
name: python-testing
description: Pure-stdlib unittest testing standards and commands for AuditLab covering generator invariants, env isolation, matcher/reward anti-gaming, permutation invariance, RL training, SQLite storage, and Web UI APIs. Use when writing or debugging tests in tests/.
---

# Python Testing Standards (Pure Stdlib `unittest`)

## When to Use
- Adding new unit or integration tests under `tests/`.
- Debugging test failures reported by `python3 -m unittest discover -s tests -v` or `python3 -m audit_rl verify`.
- Verifying new modules (`storage.py`, `trainer.py`, `gui.py`) before opening a PR.

## Testing Rules
1. **Zero External Test Dependencies**: Use only Python 3.11+ stdlib (`unittest`, `tempfile`, `urllib.request`, `http.client`, `json`, `pathlib`, `sqlite3`). Tests must pass offline without `pytest` installed.
2. **Isolation & Cleanup**:
   - Always use `tempfile.TemporaryDirectory()` for SQLite DBs and saved episodes—never mutate tracked `data/platform.db`, `data/train/`, or `trajectories/trajectories.jsonl` inside unit tests.
   - Always close `SubprocessAuditEnv` (via `with` context manager) and shut down ephemeral `ThreadingHTTPServer` instances in `addCleanup` or `try/finally`.
3. **Deterministic Seeds**: Pin all random seeds in tests so failures are 100% reproducible.
4. **Full Suite Coverage**:
   - `tests/test_generator_and_invariants.py`: clean balance, distractors, 4 error types, 2 interacting scenarios, schemas.
   - `tests/test_env_contract_and_isolation.py`: Gymnasium contract, pagination, AST calculator, `train`/`eval` isolation, `SubprocessAuditEnv`.
   - `tests/test_matcher_and_reward.py`: Kuhn 1-to-1 matching, tolerances, evidence subsets, anti-spam, malformed payloads.
   - `tests/test_reproducibility_and_permutation.py`: trace determinism + 20 paired permutation & ID-remapping checks.
   - `tests/test_policies_and_trajectories.py`: 4 baselines + trajectory schema and zero-leak scan.
   - `tests/test_training_and_storage.py`: `PlatformStore`, `val_configs.json`, `LearnableAuditPolicy` training improvement, Evidence-First explanations.
   - `tests/test_branding_and_security.py`: zero forbidden branding, GUI HTTP endpoints, input safety, 8 custom skills YAML frontmatter.

## Workflow
1. Run a single test file while iterating:
   ```bash
   python3 -m unittest tests/test_training_and_storage.py -v
   ```
2. Run the full test suite:
   ```bash
   python3 -m unittest discover -s tests -v
   ```
3. Run end-to-end platform verification:
   ```bash
   python3 -m audit_rl verify
   ```

## Verification Checklist
- [ ] All tests run cleanly with `python3 -m unittest discover -s tests -v` (zero external packages).
- [ ] No stray temporary files or background worker processes left behind.
- [ ] `git status` shows no unintended mutations to tracked datasets or DB files after test runs.
- [ ] `python3 -m audit_rl verify` passes completely.
