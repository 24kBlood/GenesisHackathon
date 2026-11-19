# CONTRIBUTING.md — Contributing to AuditLab

Thank you for contributing to **AuditLab** (*An autonomous financial audit simulation and AI training environment*).

---

## 1. Development Environment

AuditLab requires **Python 3.11+** and has **zero external runtime or test dependencies**.

```bash
# Clone and verify locally (works 100% offline)
python3 --version
python3 -m unittest discover -s tests -v
python3 -m audit_rl verify
```

---

## 2. Branch & Commit Conventions

Use descriptive, scoped branch names:
- `feat/<topic>` — new environment features, policies, training algorithms, or Web UI pages.
- `fix/<topic>` — bug fixes or security hardening.
- `test/<topic>` — additional test coverage or regression tests.
- `docs/<topic>` — architecture docs, ADRs, skills, or guides.

Keep commits focused and use concise imperative messages (e.g., `feat: add SQLite experiment persistence`, `fix: enforce request size limit in gui server`).

---

## 3. Testing & Verification Commands

Before opening a Pull Request, run all checks below:

```bash
# 1. Full pure-stdlib unit & integration suite
python3 -m unittest discover -s tests -v

# 2. Single-command platform verification (tests + 20 eval integrity + 20 permutation + leak scan + baselines)
python3 -m audit_rl verify

# 3. Verify RL training reproducibility
python3 -m audit_rl train --epochs 2 --seed 42

# 4. Optional offline Docker build & verification
docker build -t auditlab:latest .
docker run --rm --network=none --cpus=2 --memory=4g auditlab:latest
```

---

## 4. Security & Isolation Rules

1. **Zero Dynamic Code Execution**: Never introduce `eval()`, `exec()`, `compile()`, or `pickle` on untrusted data. Keep `audit_rl/calculator.py` strictly AST-based (`Decimal`).
2. **Zero External Runtime Dependencies / CDNs**: Keep `pyproject.toml` `dependencies = []` and ensure `audit_rl/gui.py` loads zero external scripts, fonts, or stylesheets.
3. **Evaluator Privacy**: Never expose `truth`, `provenance`, `seed`, or local filesystem paths in `observation`, `info`, or `trajectories/trajectories.jsonl`.
4. **SQL & HTTP Safety**: Use parameterized `?` queries in `audit_rl/storage.py`; validate JSON bodies, cap payload sizes, and block path traversal in `audit_rl/gui.py`.
5. **Split Hygiene**: Never train or tune policy parameters on `configs/eval_configs.json`.

---

## 5. Pull Request Checklist

- [ ] All `unittest` suites pass (`python3 -m unittest discover -s tests -v`).
- [ ] `python3 -m audit_rl verify` exits 0 with 100.0/100.0 score.
- [ ] No external runtime dependencies added.
- [ ] New or changed behavior is covered by deterministic pure-stdlib tests.
- [ ] Documentation (`README.md`, `ARCHITECTURE.md`, `PROJECT_STATUS.md`, `DECISIONS.md`) is updated to match implementation.
- [ ] No temporary files, secrets, or unintended mutations to tracked datasets are included.
