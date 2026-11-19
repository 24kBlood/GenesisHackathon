# IMPLEMENTATION_PLAN.md — AuditLab Phased Roadmap

This roadmap tracks implementation phases for **AuditLab** (*An autonomous financial audit simulation and AI training environment*).

## Phase 1: Core Schemas & Causal Synthetic Accounting Generator
- [x] Define `SCHEMA_VERSION = "1.0.0"`, `ERROR_TYPES`, `INTERACTING_SCENARIOS`, `CURRENCY_EXPONENTS`, and `EpisodeConfig` (`audit_rl/schemas.py`).
- [x] Implement deterministic causal generator (`random.Random(seed)`) with pre-injection double-entry balance invariants (`audit_rl/generator.py`).
- [x] Implement 5 clean distractor classes, 4 standalone error types, and 2 interacting multi-document scenarios.
- [x] Implement `verify_episode_integrity` and `permute_and_relabel_episode`.

## Phase 2: Safe Arithmetic, Open Matcher & Terminal Reward
- [x] Implement AST-only `Decimal` calculator (`audit_rl/calculator.py`) with strict node/function whitelist and length limit.
- [x] Implement deterministic 1-to-1 Kuhn bipartite matcher (`audit_rl/matcher.py`), normalization, tolerances, and `acceptable_evidence_sets` subset validation.
- [x] Implement bounded anti-gaming terminal reward `clip(2 * F1 - 1.0 - 0.002 * inspection_steps, -1.0, 1.0)`.

## Phase 3: Gymnasium RL Environment & Out-of-Process Isolation
- [x] Implement `AuditEnv` with `train` / `eval` modes, narrow 5-tool interface, pagination (`<= 50`), step budget (`100`), and post-termination immutability (`audit_rl/env.py`).
- [x] Implement `SubprocessAuditEnv` and `audit_rl/eval_worker.py` for JSON-lines IPC isolation.

## Phase 4: Baselines, Versioned Splits & Trajectory Recording
- [x] Implement 4 baseline policies (`RandomExplorerPolicy`, `SimpleReconciliationPolicy`, `FlagEverythingAttackPolicy`, `OraclePolicy`) (`audit_rl/policies.py`).
- [x] Create versioned `train` (10), `dev` (10), `val` (10), and held-out `eval` (20) split configs (`configs/`).
- [x] Generate open `data/train/`, `data/dev/`, and leak-scanned `trajectories/trajectories.jsonl`.

## Phase 5: Persistence & Pure-Stdlib RL Training
- [x] Implement SQLite `PlatformStore` (`audit_rl/storage.py`) for experiments, benchmarks, and policy versions (`data/platform.db`).
- [x] Implement `LearnableAuditPolicy` and deterministic `run_training_experiment` (`audit_rl/trainer.py`) optimizing strictly on `train` split rewards and evaluating on unseen held-out splits.
- [x] Upgrade `SimpleReconciliationPolicy` explanations to Evidence-First format (`Observed:`, `Calculated:`, `Rule:`).
- [x] Wire `python3 -m audit_rl train` into `audit_rl/cli.py`.

## Phase 6: 8-Page Offline Web Application
- [x] Build local `ThreadingHTTPServer` + embedded zero-CDN SPA (`audit_rl/gui.py`) with Dark/Light theme toggle.
- [x] Implement all 8 pages: Overview, Company Explorer, Agent Workspace (manual + autonomous), Evaluation Results (with Evidence ID lookup & 100-point Verification Suite), Training, Benchmarks, Trajectory Explorer (with replay & JSONL export), and Settings.

## Phase 7: Automated Verification & Security Hardening
- [x] Core pure-stdlib `unittest` suites: generator invariants, env isolation, matcher/reward, reproducibility/permutation, and policies/trajectories (61 core tests).
- [x] Extended suites: `tests/test_training_and_storage.py` and `tests/test_branding_and_security.py`.
- [x] Single-command `python3 -m audit_rl verify` achieving 100.0/100.0 score.

## Phase 8: Documentation, Governance & Project Agent Skills
- [x] Author 8 technical project skills under `.agents/skills/*/SKILL.md`.
- [x] Maintain `README.md`, `ARCHITECTURE.md`, `DESIGN.md`, `DEMO.md`, `DECISIONS.md`, `IMPLEMENTATION_PLAN.md`, `CONTRIBUTING.md`, and `PROJECT_STATUS.md`.
- [x] Verify offline CPU-only `Dockerfile` (`python:3.11-slim`, `--network=none`).

## Phase 9: Future Extensions (Post-v1.0)
- [ ] Add external LLM / tool-calling agent adapters over `SubprocessAuditEnv` JSON IPC.
- [ ] Expand accounting domain coverage (accruals, deferred revenue, multi-entity intercompany eliminations).
- [ ] Optional containerized seccomp execution wrapper for untrusted external agent binaries.
