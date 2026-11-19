---
name: reproducible-benchmarking
description: Workflow for running deterministic benchmarks, verifying 100-point platform checks, persisting experiments to SQLite PlatformStore, and validating offline Docker execution in AuditLab. Use when running benchmarks, updating eval configs, or verifying release readiness.
---

# Reproducible Benchmarking

## When to Use
- Running held-out baseline comparisons across `configs/eval_configs.json`.
- Recording and comparing training/evaluation experiments in `data/platform.db` (`audit_rl/storage.py`).
- Auditing trajectory logs (`trajectories/trajectories.jsonl`) and verifying offline Docker reproducibility.

## Reproducibility Rules
1. **Versioned Splits**: Never modify `configs/eval_configs.json` mid-benchmark without bumping config version and re-running all baselines cleanly.
2. **Isolated Evaluation**: Always evaluate baselines through `SubprocessAuditEnv(mode="eval")` so agent code cannot inspect private state.
3. **Zero Fabricated Metrics**: All tables in `README.md`, `DESIGN.md`, `PROJECT_STATUS.md`, and the Web UI must reflect real measured runs—never placeholder or synthetic charts.
4. **Leak-Free Artifacts**: Run the trajectory leak scanner after regenerating `trajectories/trajectories.jsonl` to confirm zero `truth`, `provenance`, `seed`, or local path leaks.

## Workflow
1. Run single-command full verification (unit tests, 20 eval integrity checks, 20 permutation checks, trajectory leak scan, 4 baselines):
   ```bash
   python3 -m audit_rl verify
   ```
2. Run standalone held-out baselines:
   ```bash
   python3 -m audit_rl run-baselines
   ```
3. Run deterministic RL training and verify held-out delta:
   ```bash
   python3 -m audit_rl train --epochs 2 --seed 42 --json
   ```
4. Verify offline container execution (2 CPU / 4 GB RAM, no network):
   ```bash
   docker build -t auditlab:latest .
   docker run --rm --network=none --cpus=2 --memory=4g auditlab:latest
   ```

## Verification Checklist
- [ ] `python3 -m audit_rl verify` exits 0 with 100.0/100.0 score.
- [ ] Relative ordering holds: `oracle >= simple_reconciliation > random_explorer > flag_everything_attack`.
- [ ] `trained_eval.f1 > untrained_eval.f1` on unseen held-out episodes.
- [ ] Experiments persist cleanly in `data/platform.db` across restarts.
- [ ] Offline Docker run passes with `--network=none`.
