---
name: rl-environment-design
description: Guidelines for Gymnasium-compatible RL environment contracts, episode lifecycle, state isolation, Evidence-First policies, and stdlib RL training in AuditLab. Use when working on audit_rl/env.py, audit_rl/policies.py, or audit_rl/trainer.py.
---

# RL Environment & Agent Policy Design

## When to Use
- Modifying `AuditEnv` / `SubprocessAuditEnv` reset/step semantics or observation/info schemas.
- Implementing or tuning policies in `audit_rl/policies.py` (`RandomExplorerPolicy`, `SimpleReconciliationPolicy`, `FlagEverythingAttackPolicy`, `OraclePolicy`).
- Developing or extending `LearnableAuditPolicy` and `run_training_experiment` in `audit_rl/trainer.py`.

## Design Standards
1. **Gymnasium JSON Contract**:
   - `reset(seed, config) -> (obs, info)` fully clears prior episode state.
   - `step(action) -> (obs, reward, terminated, truncated, info)` uses strictly JSON-compatible payloads.
2. **Train vs Eval Separation**:
   - Training uses `train` (and optionally `dev`/`val`) episodes with terminal rewards (`mode="train"`).
   - Held-out `eval` episodes must NEVER be touched during training or parameter updates, and must always be evaluated via `SubprocessAuditEnv(mode="eval")`.
3. **Evidence-First Explainability**:
   - Policy findings must ground explanations in observed open records, calculated figures, and the violated policy rule (`Observed: ... Calculated: ... Rule: ...`, `<= 1000` chars).
   - Never fabricate IDs, leak ground truth, or read private env attributes inside non-oracle policies.
4. **Reproducible Stdlib Training**:
   - `LearnableAuditPolicy` and `run_training_experiment` must be deterministic for a fixed seed, run on CPU with zero external ML libraries, persist runs/versions to `PlatformStore` (`data/platform.db`), and demonstrate measurable held-out improvement (`trained_eval.f1 > untrained_eval.f1`).

## Workflow
1. Run a deterministic training experiment and inspect before/after held-out metrics:
   ```bash
   python3 -m audit_rl train --epochs 2 --seed 42
   ```
2. Run policy, training, and environment tests:
   ```bash
   python3 -m unittest tests/test_env_contract_and_isolation.py tests/test_policies_and_trajectories.py tests/test_training_and_storage.py -v
   ```
3. Confirm trajectory recording stays leak-free:
   ```bash
   python3 -m audit_rl verify
   ```

## Verification Checklist
- [ ] Identical `(seed, config, actions)` produce bit-for-bit identical observations and rewards.
- [ ] Non-oracle policies use only `obs` and narrow tool calls (no private attribute access).
- [ ] `SimpleReconciliationPolicy` emits `Observed:`, `Calculated:`, `Rule:` within 1000 chars.
- [ ] `train` split optimization improves held-out `eval` F1 without reading `eval` during training.
- [ ] `OraclePolicy` remains strictly evaluator-only.
