# Genesis FinTech — 10-Agent Swarm Code Review

> **Repository:** [`24kBlood/GenesisHackathon`](https://github.com/24kBlood/GenesisHackathon) (`main` branch only)  
> **Reviewed Commit:** `72a9473c934031fe176ad3209c299b8e477c7946`  
> **Review Date:** 2026-11-21  
> **Methodology:** 10 independent specialized review agents (Phase 1 isolated read-only audits to prevent groupthink) → paired cross-agent verification & adversarial challenges (Phase 2) → direct coordinator source-code verification (Phase 3).  
> **Specification Note:** The file `Genesis_FinTech_Case_RU_light.pdf` was not present in the repository or local workspace during this audit; all hackathon criteria were evaluated strictly against the authoritative specification provided in the review rubric and verified directly against the repository's implementation, schemas, tests, datasets, and documentation. No PDF page numbers were fabricated.

---

## 1. Executive Summary

### Overall Repository Quality
**Strong engineering core wrapped around a scoring-contract contradiction and a toy-scale generator.** The repository (`~4,069` LOC of pure Python 3.11 standard-library runtime code, 14 JSON Schema Draft 2020-12 files, 147 `unittest` cases, 10 `train` + 10 `dev` episodes, and a digest-pinned ~50.7 MB Docker verification image) demonstrates genuine discipline in deterministic matching, bounded decimal arithmetic, JSONL subprocess framing, container hardening flags, and zero-network/zero-GPU offline execution.

### Main Architectural Assessment
The layering across `domain`, `generation`, `environment`, `evaluation`, `policies`, `trajectories`, and `cli` is compact and readable, with a clean Gymnasium-style JSON `reset(seed, config)` / `step(action)` interface and deferred closed-mode container teardown before scoring. However, three structural flaws undermine the system's core purpose:
1. Standalone error injection (`allocation.py`) and interacting error injection (`interactions.py`) emit **contradictory `affected_ids` sets** for the exact same `FindingType`, directly breaking exact-set matching (`matcher.py`).
2. The generator (`company.py`) is hardcoded to a **fixed 4-obligation skeleton** (max 44 ledger rows `< 50` page size), leaving pagination and `get_document`/`get_record`/`calculator` functionally dead on generated episodes and enabling a 5-step rule-based script to score `F1 = 1.000 / Return = +0.992`.
3. Containerized policy orchestration (`DockerPolicy`), episode export (`export_episode`), and baseline worker containerization are **disconnected from production CLI entrypoints**, while `genesis.cli verify --all` has 0% unit-test coverage and exits `0` when Docker isolation check `C18` is skipped.

### Most Serious Risks
1. **Honest Auditors Penalized with `F1 = 0, Reward = -1.0` (`F-01`, Critical):** `allocation.py` splats `doc_id`s and `bank.row_id` into `affected_ids`, contradicting `interactions.py` and `docs/DATA_CONTRACT.md:60`. Spec-compliant auditors score `TP=0, FP=1, FN=1` on every standalone `duplicate_invoice` and `payment_allocation` episode.
2. **Broken/Crashing Workers Earn Maximum `+1.0` Reward on Clean Episodes (`F-02`, High):** Transport crashes (`EOF`), malformed JSON (`BAD_FRAME`), timeouts (`TIMEOUT`), stderr floods (`STDERR_LIMIT`), and even missing Docker images (`PROCESS_ERROR`) trigger `env.truncate()`, earning `F1 = 1.0, reward = +1.0` on clean episodes (beating honest inspecting policies at `+0.992`).
3. **Untrusted-to-Trusted Serialization Crash via Lone UTF-16 Surrogates (`F-03`, High):** A schema-valid JSONL `submit` containing `"\ud800"` in `affected_ids`/`evidence_ids` passes `jsonl.py` and `AuditorEnv` and crashes `TrajectoryRecorder.to_jsonl()` and canonical submission serialization with unhandled `UnicodeEncodeError`.
4. **Incomplete Contest Submission Package (`F-08`, High):** `materials/submission/README.md` explicitly records that Task #26 was skipped and the demo video (`<= 3:00`), presentation, Google Drive master document, and portal submission are absent, with audit links pointing to pre-merge commit `47c8fb37`.
5. **Non-Identifiable Interacting Truth, Negative Bank Overdraft Leaks, and Oracle-Only Permutation Tautology (`F-05`, `F-06`, `F-07`, High):** Together, these weaken data validity, fairness, and generalization guarantees.

### Section 4.7 Security & Submission Blockers Check

| Blocker Criterion | Status | Evidence |
| --- | --- | --- |
| Hidden ground-truth leakage (`truth.json`, `provenance.json`, `seed`) | **PASS (No blocker)** | Public projections (`auditor_env.py:180-235`), post-teardown closed scoring (`judge.py:38-72`), and `records.py:96-104` private-field rejection prevent direct leakage. *(Note: indirect statistical/structural shortcuts exist in `F-04` and `F-06`.)* |
| Writing to evaluator-owned data | **PASS (No blocker)** | `DockerPolicy` mounts zero host paths (`--read-only`, `--network none`); `PublicTools` deep-copies public records. |
| Real financial records / PII / secrets | **PASS (No blocker)** | All datasets, fixtures, and trajectories are 100% synthetic (`Synthetic service`, synthetic IDs/vendors); zero secrets/tokens found. |
| Nonfunctional core environment | **PASS (No blocker)** | Core `AuditorEnv`, `PublicTools`, `matcher.py`, `reward.py`, and `unittest` suite (`145/147` passed, `2` Docker skipped locally) work deterministically. |
| Network dependence during evaluation | **PASS (No blocker)** | Zero runtime network imports (`urllib`/`http`/`socket`/`requests` absent from `src/genesis/`); container runs with `--network none`. |
| Paid API or GPU dependence | **PASS (No blocker)** | Pure Python 3.11 stdlib runtime; CPU-only; 80 local benchmark episodes complete in `78.3 ms` (`311 KiB` traced heap). |
| **Incomplete contest deliverables (video, slides, Drive link)** | **BLOCKER (`F-08`)** | Explicitly marked open (`комплект сдачи неполный`) in `materials/submission/README.md:3,13-16,30`. |
| **Scoring contract contradiction (`allocation.py` vs `interactions.py`/`DATA_CONTRACT.md`)** | **BLOCKER (`F-01`)** | Spec-compliant external policies score `F1 = 0, reward = -1.0` on standalone `duplicate_invoice` & `payment_allocation` episodes. |

### Hackathon & Production Readiness
- **Estimated Official Hackathon Score:** **67 / 100** (see Section 4 Scorecard).
- **Hackathon Readiness:** **Conditional / Needs P0 Fixes Before Submission.** Zero security/leakage/offline disqualifiers, but `F-01` (contradictory `affected_ids`), `F-02` (crash = `+1.0` on clean episodes), `F-03` (`\ud800` crash), and `F-08` (missing contest deliverables) must be resolved first.
- **Production / Real RL Training Readiness:** **Not Ready.** Requires parameterized episode scaling (`200–2,000` rows), non-redundant `get_document` payloads, `s_0` reset logging in trajectories, configurable Docker ML resource caps, and Python packaging (`pyproject.toml`).

### Top 5 Recommended Actions
1. **Fix `F-01` immediately:** Remove `doc_id`s and `bank["row_id"]` from `affected_ids` in `src/genesis/generation/allocation.py:143-181` to match `interactions.py` and `docs/DATA_CONTRACT.md:60`, add multiple minimal `acceptable_evidence_sets` in `interactions.py` (`F-05`), and regenerate `data/` + `checksums.json`.
2. **Fix `F-02` & `F-03`:** Penalize transport/protocol faults (`EOF`, `BAD_FRAME`, `FRAME_LIMIT`, `STDERR_LIMIT`, `TIMEOUT`) with `-1.0` (and abort/flag infrastructure `PROCESS_ERROR`) instead of clean-episode `+1.0`; reject lone UTF-16 surrogates (`\ud800`–`\udfff`) in `jsonl.py` and `auditor_env.py`.
3. **Fix `F-06` & `F-07`:** Add an opening cash reserve in `company.py:228-232` so FX mutations never push `bank` negative, and extend `reproducibility.py:202-230` so the 20 paired permutation/ID-renaming checks verify `simple_reconciliation` (and tool-using replays) rather than only 0-step `OraclePolicy`.
4. **Fix `F-09` & `F-10`:** Add unit tests for `genesis.cli verify --all`, surface `ImportError`/`ModuleNotFoundError` cleanly instead of opaque `ValueError`, wire `verify --all --require-docker` in CI, add a worker Dockerfile for baseline policies, and make `DockerPolicy` resource limits configurable (with container startup excluded from the 60 s episode clock).
5. **Complete `F-08`:** Record the `<= 3:00` demo video, finalize presentation & Google Drive package, re-run verification on final `main` HEAD, and update stale commit SHAs across `materials/submission/README.md`, `README.md`, and `docs/DESIGN_NOTE.md`.

---

## 2. Repository Architecture

### Main Components

| Layer / Directory | Files | Purpose & Key Modules |
| --- | ---: | --- |
| `src/genesis/domain/` | 4 | Enums (`FindingType`, `LedgerAccount`), `TypedDict` public/private models (`models.py`), `Decimal` FX conversion (`money.py`). |
| `src/genesis/generation/` | 8 | Deterministic company generator (`company.py`), error injectors (`allocation.py`, `fx_cutoff.py`, `interactions.py`), distractors (`distractors.py`), public dataset builder & checksums (`datasets.py`), atomic directory exporter (`export.py`). |
| `src/genesis/environment/` | 7 | Gymnasium-style `AuditorEnv` (`auditor_env.py`), `PublicTools` & scoped pagination (`tools.py`), AST-bounded decimal `calculator.py`, bounded JSONL stdio IPC (`jsonl.py`), hardened container runner (`docker_policy.py`), `Episode` builder (`episode.py`). |
| `src/genesis/evaluation/` | 10 | Bipartite 1-to-1 `matcher.py`, `metrics.py`, `reward.py`, `terminal.py`, `public_proofs.py`, `oracle.py`, episode harness `judge.py`, `permutation.py`, `benchmark.py`, `reproducibility.py`. *(Missing `__init__.py` — implicit namespace package.)* |
| `src/genesis/policies/` | 5 | Policy protocol (`protocol.py`), `RandomPolicy` & `MarkAllPolicy` (`baseline.py`), `FxReconciliationPolicy` (`reconcile.py`), stdio JSONL worker entrypoint (`worker.py`). |
| `src/genesis/trajectories/` | 2 | Public-only `TrajectoryRecorder` & validator (`records.py`). |
| `src/genesis/cli/` | 2 | Single command `python -m genesis.cli verify --all [--require-docker]` (`__main__.py`). |
| `schemas/` | 14 | JSON Schema Draft 2020-12 contracts for records, observations, actions, findings, truth, provenance, config, trajectories. |
| `data/` & `benchmarks/` & `trajectories/` | 127 | 10 `train` + 10 `dev` public episodes (`checksums.json`), 20 local evaluation profiles (`local_eval_profiles.json`), 1 five-step trajectory file (`trajectories.jsonl`). |
| `tests/` | 31 | 17 `unittest` modules (`147` tests total: `C01–C20`, `E01–E20`, generator/dataset/trajectory/policy/reproducibility suites), `requirements-schema.in`/`.lock`, `fixtures/`. |
| `.github/workflows/` & `Dockerfile` | 4 | `data-contract.yml`, `offline-verify.yml`, `policy-isolation.yml`, digest-pinned `python:3.11-slim` multi-stage `Dockerfile` (~50.7 MB CI-reported). |

### Technologies & Dependencies
- **Runtime (`src/genesis/`):** Pure Python 3.11 standard library (`asyncio`, `ast`, `decimal`, `hashlib`, `json`, `random`, `unittest`, `zoneinfo`). Zero network, ML, or GPU dependencies.
- **Test/Schema Lock (`tests/requirements-schema.lock`):** 9 pinned wheels (`jsonschema==4.25.1`, `tzdata==2025.2`, `attrs`, `referencing`, `rpds-py`, `rfc3339-validator`, `jsonschema-specifications`, `six`, `typing-extensions`). No SHA-256 hashes (`--require-hashes`).
- **Packaging:** No `pyproject.toml` or `setup.cfg`; relies on `PYTHONPATH=src`.

### Important Execution Flows
1. **Episode Generation:** `generate_episode(seed, config)` (`company.py`) → creates 4 hardcoded obligations + opening cash + optional `distractors.py` → sequentially mutates via `apply_allocation_errors`, `apply_fx_cutoff_errors`, `apply_interactions` → outputs `(public, truth, provenance)`.
2. **RL Interaction Loop:** `judge.run_episode(env, runner, seed, config)` → `env.reset(seed, config)` initializes `PublicTools` + starts monotonic deadline → `runner.open(deadline)` starts local/subprocess/`DockerPolicy` channel → loop `receive(action) -> env.step(action) -> send(transition)` → on terminal/truncation/transport error, tears down runner first, then evaluates `env.terminal_result` via `matcher.match()` + `reward.score_terminal()`.
3. **Verification Gate:** `python -m genesis.cli verify --all` discovers `tests/`, buckets `C01–C20` & `E01–E20` by method-name slicing, runs groups (optionally skipping `docker-policy-smoke` unless `--require-docker`), then runs `verify_reproducibility` (20 replays + 20 Oracle-only pairs) and `local_benchmark` (80 runs across 4 policies).

### Architectural Strengths & Weaknesses
- **Strengths:** Strict JSON-only wire contract; clean AST-sandboxed decimal calculator; polynomial maximum bipartite matcher (`0.41 ms` at 30×30); deferred closed-mode scoring after container teardown; strong `DockerPolicy` isolation flags (`--network none`, `--read-only`, `--cap-drop ALL`, `--security-opt no-new-privileges`, UID `65532`); tiny repo (`2.4 MiB`) and ultra-fast local runtime (`78.3 ms` for 80 benchmark runs).
- **Weaknesses:** Contradictory truth `affected_ids` contract between injectors (`F-01`); hardcoded 4-obligation generator ceiling (`F-04`); cyclic dependency between `environment` and `evaluation` (`F-19`); unwired `DockerPolicy`/`export_episode` and missing episode loader (`F-10`); duplicated domain helpers and schema-vs-code validation drift (`F-17`, `F-24`).

---

## 3. Findings by Severity

### Critical Findings

#### F-01: Contradictory `affected_ids` Contract Between `allocation.py`, `interactions.py`, and `docs/DATA_CONTRACT.md` Penalizes Spec-Compliant Auditors with `F1 = 0`
**Title:** Contradictory `affected_ids` contract between standalone `allocation.py`, interacting `interactions.py`, and `docs/DATA_CONTRACT.md:60`  
**Severity:** Critical  
**Confidence:** Confirmed (independently verified by 7 agents + coordinator)  
**Affected files:** `src/genesis/generation/allocation.py:143-181`, `src/genesis/generation/interactions.py:108-133`, `src/genesis/evaluation/matcher.py:86-118`, `docs/DATA_CONTRACT.md:60`, `tests/test_public_datasets.py:140-146`, `tests/test_interaction_scenarios.py:77-105`  
**Evidence:**
- `docs/DATA_CONTRACT.md:60` defines canonical `affected_ids`: *"Для распределения канонические затронутые ID — ID счёта, платежа и проводок AP и suspense. Для дубля — ID счёта и четырёх строк начисления."*
- `interactions.py:112` (`payment_allocation`): `affected = [invoice_id, payment_id, allocated["row_id"], suspense["row_id"]]` (**4 IDs** — matches docs).
- `interactions.py:129` (`duplicate_invoice`): `[invoice_id, *row_ids]` (**5 IDs** — matches docs).
- `allocation.py:143-146` (`duplicate_invoice`): `evidence = [candidate.invoice["doc_id"], *(row["row_id"] for row in original + copied)]` followed by `[candidate.invoice["invoice_id"], *evidence]` (**6 IDs — injects `doc_id`**).
- `allocation.py:170-177` (`payment_allocation`): splats `*evidence` (`invoice["doc_id"]`, `payment["doc_id"]`, `bank["row_id"]`) into `affected_ids` (**7 IDs for partial / 6 IDs for missing**).
- Empirical reproduction on `train:20260102` (`duplicate_invoice`) and `standard-05:20260315` (`payment_allocation`): submitting the exact `DATA_CONTRACT.md:60` `affected_ids` with exact type, amount, currency, and evidence returns `match(...) -> (tp=0, fp=1, fn=1)`, `reward = -1.0`. Across all 20 local profiles, a spec-compliant auditor drops to `mean F1 = 0.7375` (`0.000` on `standard-01`, `standard-02`, `standard-05`) from this mismatch alone (and `0.625` when combined with `F-05`).  
**Problem:** Standalone `allocation.py` splats `*evidence` into `affected_ids`, mixing document IDs and unaffected bank rows into canonical `affected_ids`. The exact same `FindingType` therefore requires mutually incompatible `affected_ids` sets depending on which generator module created the issue — even inside the same mixed episode (`tests/test_interaction_scenarios.py:137`).  
**Impact:** Because `matcher.py:103,117-118` requires exact set equality on `affected_ids`, any honest external agent or human auditor built from `docs/DATA_CONTRACT.md` receives `0 TP, 1 FP, 1 FN` (`reward = -1.0`) on every standalone `duplicate_invoice` and `payment_allocation` episode. Conversely, an RL agent trained on standalone episodes learns to include `doc_id`/`bank_row_id` and fails on interacting episodes. Directly compromises 50/100 rubric points (Sections 4.2 & 4.3).  
**Realistic scenario:** A hackathon participant reads `docs/DATA_CONTRACT.md:60`, builds a mathematically exact financial auditor that passes both interacting scenarios, and gets `-1.0` on `train:20260102`, `train:20260103`, `dev:20260202`, `dev:20260203`, `dev:20260208`, and `dev:20260209` purely because `allocation.py` copied `*evidence` into `affected_ids`.  
**Recommended fix:** Align `allocation.py:143-181` with `interactions.py` and `docs/DATA_CONTRACT.md:60` (exclude `doc_id`s and `bank["row_id"]` from `affected_ids`: `[invoice_id, *4_rows]` for `duplicate_invoice` and `[invoice_id, payment_id, *after_rows]` for `payment_allocation`), explicitly document canonical `affected_ids` for all 4 error types (including `missing` vs `partial` and `period_cutoff`), update `tests/test_public_datasets.py:142`, bump `generator_version`, and regenerate `data/` + `checksums.json`.

---

### High Findings

#### F-02: Clean-Episode Transport Crashes, Timeouts, Protocol Violations, and Missing Docker Images Earn Maximum Reward (`+1.0000`, `F1 = 1.0`)
**Title:** Transport faults, timeouts, step-limit exhaustion, and missing Docker images score up to `+1.0000` (`F1 = 1.0`) on clean episodes; invalid submissions still record `F1 = 1.0`  
**Severity:** High *(reconciled in Phase 2 from Medium/High/Critical across `security`, `devils-advocate`, `failure-analyst`)*  
**Confidence:** Confirmed  
**Affected files:** `src/genesis/evaluation/judge.py:66-71`, `src/genesis/environment/auditor_env.py:325-331,398-419`, `src/genesis/evaluation/reward.py:35-43`, `src/genesis/evaluation/metrics.py:24-28`, `tests/test_evaluator_isolation.py:147-170,205-215`  
**Evidence:**
- `judge.py:66-68` catches every `TransportError` (`EOF`, `BAD_FRAME`, `FRAME_LIMIT`, `STDERR_LIMIT`, `TIMEOUT`, `PROCESS_ERROR`) and calls `env.truncate(...)`.
- `auditor_env.py:408-414` sets `submission = empty`, `invalid_submission = terminated and submission is None` (`False` on truncation).
- `metrics.py:27` returns `Metrics(precision=0.0, recall=0.0, f1=1.0)` when `tp=fp=fn=0`, so `reward.py:41` returns `+1.0000` (at 0 steps) or `+0.8000` (at 100 garbage steps).
- `tests/test_evaluator_isolation.py:149-168,205-215` literally asserts `reward == 1` on clean episodes for crashing, sleeping past timeout, outputting `{bad-json}`, overflowing `512 KiB`, flooding `stderr`, and referencing a nonexistent Docker image (`PROCESS_ERROR`).
- Additionally, `reward.py:35-39` forces `reward = -1.0` when `invalid_submission=True` on a clean episode, **but leaves `metrics.f1 = 1.0`**, inflating macro `F1` to `0.25` for a 100%-invalid policy; and `metrics.py:25-27` (`P=0, R=0, F1=1` on clean books) caps perfect Oracle macro Precision/Recall at `0.7500` when 25% of episodes are clean.  
**Problem:** Truncation due to policy crash, protocol violation, timeout, or container startup failure is scored identically to an optimal zero-step clean-book submission, strictly outperforming honest policies that inspect the books (`+0.992`).  
**Impact:** In RL training (`mode="train"`), crashing or hanging produces higher reward (`+1.000`) than auditing (`+0.992`). In closed evaluation, a broken container that fails on import or a misconfigured missing Docker image silently earns `+1.0` on all 5 clean episodes (25% of the benchmark), beating `random` (`-0.504`) and `mark_all` (`-1.000`).  
**Realistic scenario:** An entrant's container crashes on startup (`ModuleNotFoundError` → `EOF` at step 0). On the 5 hidden clean episodes, the harness awards `F1 = 1.0, reward = +1.0` (tying Oracle), masking a startup failure as a partially successful benchmark run.  
**Recommended fix:** Treat transport/protocol faults (`EOF`, `BAD_FRAME`, `FRAME_LIMIT`, `STDERR_LIMIT`, `TIMEOUT`) and step-limit truncation without a valid `submit` as penalized failures (`reward = -1.0`, `f1 = 0.0`, matching `invalid_submission`), raise/abort infrastructure errors like `PROCESS_ERROR` (missing image) rather than scoring them, and set `metrics.f1 = 0.0` whenever `invalid_submission=True`.

#### F-03: Lone UTF-16 Surrogate (`\ud800`) in Valid JSONL `submit` Crashes Trusted Trajectory & Submission Serialization
**Title:** Untrusted policy `submit` containing a lone UTF-16 surrogate (`\ud800`) passes JSONL & `AuditorEnv` validation and crashes trusted UTF-8 serialization with `UnicodeEncodeError`  
**Severity:** High  
**Confidence:** Confirmed (discovered by `failure-analyst`, independently reproduced over real `JsonlProcess` by `security`, verified by coordinator)  
**Affected files:** `src/genesis/environment/jsonl.py:95-100`, `src/genesis/environment/auditor_env.py:124-160`, `src/genesis/trajectories/records.py:107-113`, `src/genesis/evaluation/reproducibility.py:102`  
**Evidence:**
- In Python, `json.loads('{"type":"submit","findings":{"episode_id":"...","findings":[{"issue_type":"fx_conversion","affected_ids":["\\ud800"],"claimed_amount_minor":1,"currency":"KZT","evidence_ids":["\\ud800"],"explanation":""}]}}')` decodes `\ud800` into a legal Python `str` containing a lone UTF-16 surrogate.
- This ASCII-safe JSONL frame passes `jsonl.py:95-98` (`frame.decode("utf-8")` + `json.loads`) and `auditor_env.py:124` (`_is_id` checks only `type(v) is str and bool(v.strip())`), completing `run_episode` cleanly (`transport_error=None, terminated=True, invalid_submission=False`).
- `TrajectoryRecorder.to_jsonl()` (`records.py:108-110`) then executes `json.dumps(row, ensure_ascii=False, ...).encode("utf-8")` and crashes with unhandled `UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800': surrogates not allowed`. Even placing `\ud800` only in `explanation` survives in `TerminalResult.submission`, crashing `reproducibility._canonical()` (`reproducibility.py:102`) or any trusted UTF-8 export.  
**Problem:** JSON escape decoding allows unpaired UTF-16 surrogates to cross the untrusted-policy → trusted-evaluator boundary, where `ensure_ascii=False` + `.encode("utf-8")` fails catastrophically.  
**Impact:** Any untrusted policy (malicious or buggy) can trigger a remote denial-of-service crash in the trusted evaluation harness during trajectory export or canonical result serialization.  
**Realistic scenario:** During closed evaluation of an external policy container, the policy outputs a JSON string containing an unpaired surrogate escape (`\ud800`–`\udfff`). The episode finishes cleanly, then crashes the judge's batch trajectory writer before scores are saved.  
**Recommended fix:** Reject strings containing surrogate codepoints (`0xD800 <= ord(c) <= 0xDFFF`, or `value.encode("utf-8")` check) in `jsonl.py:receive()` (converting to `TransportCode.BAD_FRAME`) AND defensively inside `auditor_env.py` string/ID validators and `records.py`.

#### F-04: Fixed 4-Obligation Micro-Template Generator (Max 44 Rows < 50 Page Limit), Dead Tools/Pagination, Step-0 `manifest` Leakage, and 5-Step `F1 = 1.0` Shortcut
**Title:** Hardcoded 4-obligation generator cannot scale, renders pagination and 3/4 inspection tools dead, leaks error counts on step 0, and is 100% solvable in 5 steps without ML  
**Severity:** High *(reconciled in Phase 2 across `architect`, `devils-advocate`, `failure-analyst`, `backend`)*  
**Confidence:** Confirmed  
**Affected files:** `src/genesis/generation/company.py:99-102,146-167,242-284,311-317`, `src/genesis/environment/tools.py:14,36-50`, `src/genesis/generation/fx_cutoff.py:96-99,130-141`, `src/genesis/generation/allocation.py:106-110,155-160`, `src/genesis/generation/interactions.py:44-58`  
**Evidence:**
- `company.py:146-167` hardcodes the exact same 4 base obligations on every seed: `("JPY", 3, 7), ("KZT", 2, 5), ("KZT", 4, 9), ("KZT", 0, None)`, fixed year `2026`, `"Synthetic service"`, `KZT`/`JPY`, `Asia/Almaty`, `payment_date`, `ROUND_HALF_EVEN`.
- Schema-valid configs with `fx_conversion: 2`, `period_cutoff: 2`, or `duplicate_invoice + payment_allocation > 2` crash with `ValueError`.
- Across all 40 committed/benchmark episodes, `max_ledger = 41` (theoretical max with all errors + distractors = `44`, strictly `< DEFAULT_PAGE_SIZE = 50` at `tools.py:14`), 5×–45× below the 200–2,000-row target (`docs/LIMITATIONS.md:6`).
- `tools.py:36-50` stores in `_documents[doc_id]` the **exact same dict objects** returned by `list_records("invoices")` and `list_records("payments")` (plus `reset()["policy"]`). Consequently, `get_document`, `get_record`, `calculator`, and pagination cursors provide zero new information while costing `-0.002` reward per call.
- Across 40/40 episodes (`10 train + 10 dev + 20 local`), step-0 `manifest.record_counts` deterministically leaks scenario structure: `fx` count is a 1-to-1 bijection to `interaction_scenarios` (`2 -> []`, `4 -> [dup_fx]`, `5 -> [partial_fx]`, `7 -> [both]`), `payments - invoices` (`-1` vs `+1`) reveals distractors, and `ledger - base == 2*dup + 1*partial`.
- Distribution gaps: `0/10 train` and `0/10 dev` contain a standalone `partial` allocation; `ROUND_HALF_EVEN` triggers `0/40` times (amounts are multiples of 100 with 2-dp rates); `fx_cutoff.py:140` (`else end` post-period cutoff) triggers `0/500` seeds (dead code).
- Two agents (`devils-advocate` and `failure-analyst`) independently demonstrated a ~90-line deterministic rule script achieving **`F1 = 1.0000, Return = +0.9920`** across 40/40 episodes in **5 steps** (`4 list_records + 1 submit`).  
**Problem:** The generator lacks parameterized scale, currency/policy diversity, and partial observability, making the 100-step budget and 5-tool action space largely decorative on generated data.  
**Impact:** Weakens both Sections 4.2 (data richness/realism) and 4.4 (resistance to pattern exploitation/generalization) and prevents using the environment as-is for meaningful sequential RL training. *(Note: downgraded from Critical to High because `docs/LIMITATIONS.md:6` candidly discloses the small v1.0.0 scale and `PublicTools` pagination is verified on synthetic >50-row fixtures.)*  
**Realistic scenario:** An RL policy trained on `data/train` learns a degenerate 5-step policy (dump 4 tables, apply hardcoded syntactic rules like `account == "suspense"`, ignore `get_document`/`calculator`/pagination) and fails to generalize to realistic multi-page audit books.  
**Recommended fix:** Parameterize `n_obligations` (supporting 200–2,000 rows / 20–200 docs), multi-currency pairs, and `policy` variants in `config.schema.json` + `company.py`; have `list_records` return summary fields while `get_document` returns line-item/contract details; add clean suspense-clearing entries and decoy FX quotes so `manifest.record_counts` is not a bijection to errors; fix `fx_cutoff.py:140` so post-`period_end` cutoffs actually occur.

#### F-05: Non-Identifiable `fx_conversion` `affected_ids` and Over-Constrained Single `acceptable_evidence_sets` in `partial_fx_payment_at_cutoff`
**Title:** Non-identifiable `fx_conversion` `affected_ids` and single over-constrained `acceptable_evidence_sets` in `partial_fx_payment_at_cutoff` reject valid minimal proofs  
**Severity:** High  
**Confidence:** Confirmed (verified by `backend`, `devils-advocate`, `qa-test`, `architect`)  
**Affected files:** `src/genesis/generation/interactions.py:99-164`, `src/genesis/evaluation/public_proofs.py:65-81`, `docs/DATA_CONTRACT.md:57-60`  
**Evidence:**
- In `partial_fx_payment_at_cutoff` (`train:20260106`), one foreign payment creates a balanced 3-leg entry (`Dr accounts_payable`, `Dr suspense`, `Cr bank`). `interactions.py:118,150,153` arbitrarily adds the FX `delta` to `suspense` and sets `fx_conversion.affected_ids = [bank["row_id"], suspense["row_id"]]`. From public records alone, an auditor cannot distinguish whether the FX error landed in `suspense`, `accounts_payable`, or the whole 3-leg payment posting.
- Across 40/40 episodes, **100% of truth issues have `len(acceptable_evidence_sets) == 1`** despite `truth.schema.json` and `matcher.py` supporting multiple alternative sets.
- `interactions.py:156-158` mandates the *erroneous* next-day quote `fx:<wrong_date>:JPY:KZT` inside the sole `fx_conversion` evidence set, and `interactions.py:108-111` mandates 8 IDs including `accrual["row_id"]` for `payment_allocation`. Empirically verified: removing both IDs still passes the repo's own `verify_public_proofs()` (`public_proofs.py:65-81`), yet `matcher.match()` rejects those proofs with `(tp=0, fp=1, fn=1)`.  
**Problem:** Truth requires an unobservable internal leg attribution and extraneous evidence IDs that are not mathematically required to prove either discrepancy.  
**Impact:** Honest domain auditors using minimal sufficient proofs (and citing all 3 payment legs or standard standalone FX evidence) score `F1 = 0` on `partial_fx_payment_at_cutoff` episodes (`interacting-01`, `interacting-03`, `train:20260106`, `dev:20260206`).  
**Realistic scenario:** An auditor computes `expected = 986 * 3.40 * 100 = 335,240` vs `bank = 354,960` (`delta = 19,720`) and submits `[payment_doc, policy, fx:2026-01-31:JPY:KZT, bank_row, suspense_row]` — identical to standalone `fx_conversion` — and receives `0 TP, 1 FP, 1 FN` solely for omitting the erroneous next-day quote `fx:2026-02-01:JPY:KZT`.  
**Recommended fix:** Populate multiple valid `acceptable_evidence_sets` in `interactions.py` (minimal `public_proofs`-sufficient sets alongside diagnostic sets containing `wrong_date` / `accrual`), and either include all 3 payment legs (`[bank, allocated, suspense]`) for 3-leg FX `affected_ids` or explicitly specify the leg-attribution convention in `policy`/`DATA_CONTRACT.md`.

#### F-06: FX Error Injection Pushes `bank` Into Negative Overdraft Across Published Datasets, Violating Clean Cash Invariant and Leaking FX Errors
**Title:** Zero opening cash buffer causes FX error injection to drive `bank` strictly negative in 3/10 `train`, 1/10 `dev`, and 7/20 local profiles  
**Severity:** High  
**Confidence:** Confirmed (independently reproduced by `backend` and `qa-test`)  
**Affected files:** `src/genesis/generation/company.py:228-232`, `src/genesis/generation/fx_cutoff.py:96-105`, `src/genesis/generation/interactions.py:144-152`, `tests/test_generator_and_invariants.py:298-305`  
**Evidence:**
- `company.py:228-232` sets `opening_balance` to the exact sum of clean obligations plus distractor cash — zero reserve buffer.
- All three FX injectors inflate `bank` credit (`fx_cutoff.py:97` `* 1.01`; `interactions.py:50` `3.40 -> 3.60`; `interactions.py:54` `3.20 -> 3.25`).
- Verified negative `bank` balances on **3/10 `train`** (`20260106`: `-14,320`, `20260109`: `-6,480`, `20260110`: `-29,340`), **1/10 `dev`** (`20260206`), and **7/20 `benchmarks/local_eval_profiles.json`** episodes.
- `tests/test_generator_and_invariants.py:298-305` asserts `cash >= 0` at every timestamp, but runs that check **only** on clean episodes (`error_counts={}`).  
**Problem:** Mutated books violate the environment's own non-negative cash invariant (`cash >= 0`) without modeling an overdraft facility, introducing impossible cash balances and a trivial negative-cash indicator that leaks FX error presence.  
**Impact:** Weakens accounting realism (Section 4.2: no impossible/contradictory data) and creates an unintended statistical shortcut for agents (`min(cash) < 0 => fx_conversion present`).  
**Realistic scenario:** A policy checks running `bank` balance without inspecting FX quotes; whenever cash goes negative, it immediately knows an FX over-credit occurred.  
**Recommended fix:** Add a deterministic positive opening cash buffer (e.g., 25%–50% of obligations) in `company.py:228-232` and extend the `cash >= 0` invariant test in `tests/test_generator_and_invariants.py` to cover mutated episodes as well.

#### F-07: 20 Paired Permutation/ID-Renaming Checks Only Execute Zero-Step `OraclePolicy`, and Benchmark Tests Ignore Non-Oracle Baselines
**Title:** Paired permutation & ID-renaming invariance suite (`20/20`) tests only zero-step `OraclePolicy` (0 tool calls), and benchmark test asserts zero invariants on non-Oracle policies  
**Severity:** High  
**Confidence:** Confirmed (verified by `qa-test`, `devils-advocate`, `backend`)  
**Affected files:** `src/genesis/evaluation/reproducibility.py:182-231`, `src/genesis/evaluation/oracle.py:36-39`, `tests/test_reproducibility_and_permutation.py:27-46`  
**Evidence:**
- `reproducibility.py:220-228` compares `first` vs `moved = permute_episode(first, ...)` using **only** `_oracle(first)` and `_oracle(moved)`. `OraclePolicy.act()` (`oracle.py:36-39`) reads `truth` directly and submits on step 1 with **0 calls** to `list_records`, `get_record`, `get_document`, or `calculator`.
- Meanwhile, the 20 determinism replays (`reproducibility.py:186-201`) run only `RandomPolicy`, which submits `findings: []` on every episode.
- `tests/test_reproducibility_and_permutation.py:32-46` asserts only `oracle`'s `f1 == 1.0` and `precision == 0.75`, making **zero assertions** on `simple_reconciliation`, `random`, or `mark_all`. *(Nuance noted by `backend`: `reproducibility.py:207` does run `verify_public_proofs(moved)` on permuted tables directly, so it checks slightly more than pure identity mapping, but the RL environment/tool/policy loop is completely untested.)*  
**Problem:** Permutation/ID-renaming invariance is never verified through `PublicTools`, `AuditorEnv` pagination/cursors, or any public non-Oracle policy.  
**Impact:** Under Section 4.4 (15 pts — 20 paired checks), regressions making `PublicTools`, `AuditorEnv`, or `FxReconciliationPolicy` sensitive to row ordering or ID prefixes would still pass `pairs 20/20` with green CI.  
**Realistic scenario:** A refactor makes `FxReconciliationPolicy` assume `row_id.startswith("row-")` or expect accruals before payments in `list_records`. On `qa-<hash>` renamed/shuffled evaluation episodes, recall drops to `0`, yet `verify --all` reports `PASS replay 20/20, pairs 20/20`.  
**Recommended fix:** Run `simple_reconciliation` (and a deterministic multi-tool research+submit replay mapped through `mapping`) on all 20 original-vs-permuted pairs in `reproducibility.py` and assert identical `(tp, fp, fn, reward, total_steps)` and inverse-mapped findings. Add baseline ordering assertions (`oracle > simple_reconciliation > random > mark_all`, `mark_all.return == -1.0`) in `tests/test_reproducibility_and_permutation.py`.

#### F-08: Incomplete Hackathon Contest Deliverables and Stale Commit SHAs Across Submission Documentation
**Title:** Final contest deliverables (demo video `<= 3:00`, presentation, Google Drive master document, portal submission) are missing, and submission/README audit links cite stale pre-merge SHAs  
**Severity:** High  
**Confidence:** Confirmed (verified by `devops-infra`, `frontend-ux`, `devils-advocate`, `performance`, coordinator)  
**Affected files:** `materials/submission/README.md:3-30,38,58,65-69`, `README.md:45`, `docs/DESIGN_NOTE.md:59`  
**Evidence:**
- `materials/submission/README.md:3` states: *"Статус на 21 ноября 2026 года: технический кандидат проверен, комплект сдачи неполный."*
- Lines `13-16,30`: Demo video (`<= 3:00`), presentation, Google Drive master doc, and portal submission are explicitly marked absent/open (`Задача #26 пропущена`, Deliverable #9 `"Открыто"`).
- Three different stale commit SHAs are cited across docs (`47c8fb37` in `materials/submission/README.md:3`, `63a01ad` in `README.md:45`, `bb3f928` in `docs/DESIGN_NOTE.md:59`) — none matching `main` HEAD (`72a9473`).  
**Problem:** Even if technical code were perfect, the hackathon submission package itself is incomplete and its audit trail is not pinned to `main` HEAD.  
**Impact:** Direct submission/deliverable deduction under Section 4.5 (working demo + submission completeness) and risk of disqualifying administrative failure if submitted before Task #26 artifacts exist.  
**Realistic scenario:** A judge opens the submission README and finds missing video/slides/Drive links plus hyperlinks pointing to an older Draft PR tree (`47c8fb37`) that lacked the MIT `LICENSE`.  
**Recommended fix:** Produce the `<= 3:00` working demo video, slide deck, and Google Drive master document; re-run `verify --all --require-docker` on final `main` HEAD; update all commit SHA references across `materials/submission/README.md`, `README.md`, and `docs/DESIGN_NOTE.md`.

#### F-09: Top-Level Verification Gate (`genesis.cli verify --all`) Has 0% Test Coverage, Exits `0` While Skipping `C18`, Masks `ModuleNotFoundError`, Runs Twice, and Uses Brittle Name Slicing
**Title:** `genesis.cli verify --all` is completely untested (0/105 lines), silently skips `C18` with exit code `0`, masks missing dependencies as `ValueError`, executes benchmarks twice, and slices test names by `test_C`/`test_E`  
**Severity:** High *(consolidated from `QA-01`, `UX-02`, `OPS-06`, `MAINT-04` in Phase 2)*  
**Confidence:** Confirmed (verified via `stdlib trace`, live execution, and coordinator inspection)  
**Affected files:** `src/genesis/cli/__main__.py:50-163`, `.github/workflows/data-contract.yml:28`, `.github/workflows/offline-verify.yml:29-37`, `.github/workflows/policy-isolation.yml:29`, `benchmarks/README.md:22`  
**Evidence:**
- `stdlib trace` over the full `unittest` suite confirms **0 / 105** statement lines of `src/genesis/cli/__main__.py` are covered by tests.
- Running `python -m genesis.cli verify --all` without `jsonschema` installed prints `FAIL verification setup: ValueError (test discovery or required groups)` because `TestLoader().discover()` wraps `ModuleNotFoundError` into `_FailedTest` objects, tripping `set(matrix_ids) != expected_ids` (`__main__.py:77-78`) and hiding the actual import error.
- `__main__.py:129-131` catches exceptions in `_verify_profiles()` and prints only `type(error).__name__`, stripping `reproducibility.py:124`'s safe public diff path (`f"{case}: {component} differs at {path}"`) promised by `benchmarks/README.md:22`.
- `__main__.py:63-66,95-97` moves `C18` (`DockerPolicyTests.test_C18_...`) into `docker-policy-smoke`, reporting `PASS contract-matrix: 19/19`, `SKIP docker-policy-smoke: 0/2`, and exiting `0` unless `--require-docker` is passed; meanwhile no CI workflow ever invokes `verify --all --require-docker`.
- `__main__.py:61-62,76-78` slices `method[5:8]` on any method starting with `test_C` or `test_E`, breaking `verify --all` if any future test is named `test_CSV_*` or `test_EOF_*`.
- `__main__.py:150-151` runs `test_reproducibility_and_permutation.py` (which executes `verify_reproducibility` + `local_benchmark`) and then `_verify_profiles()` runs both a **second time**.  
**Problem:** The single-command verification gate advertised as the primary judge entrypoint is untested, fragile, obscures setup errors, and silently omits 1 of the 20 contract checks (`C18`) by default.  
**Impact:** Misconfigured evaluation environments get an unactionable `ValueError`; CLI regressions ship unnoticed; `verify --all --require-docker` is never validated in CI.  
**Realistic scenario:** A judge runs `python -m genesis.cli verify --all` in a Python 3.11 environment without `tests/requirements-schema.lock` installed and sees only `FAIL verification setup: ValueError (test discovery or required groups)` with no hint that `jsonschema` is missing.  
**Recommended fix:** Add `tests/test_cli_verify.py` covering discovery, `_FailedTest` surfacing (reporting `ModuleNotFoundError` cleanly), `--require-docker`, and diff reporting; replace `test_C`/`test_E` string slicing with explicit decorator metadata/registries; avoid running `verify_reproducibility`/`local_benchmark` twice; run `verify --all --require-docker` in `policy-isolation.yml`.

#### F-10: Restrictive Hardcoded `DockerPolicy` Caps (`512m` RAM, `1` CPU, `32` PIDs), Cold Container Startup Billed to 60 s Clock, Unwired `DockerPolicy`/`export_episode`, and Missing Worker Dockerfile
**Title:** Hardcoded `512m`/1-CPU/32-PID Docker caps, per-episode cold container lifecycle charged to 60 s deadline, unwired `DockerPolicy`/`export_episode`, and missing baseline worker Dockerfile  
**Severity:** High *(High for containerized ML/RL policy evaluation & operational readiness; Medium for lightweight heuristic baselines — reconciled in Phase 2)*  
**Confidence:** Confirmed  
**Affected files:** `src/genesis/environment/docker_policy.py:30-37,47-51,90-140`, `src/genesis/evaluation/judge.py:38-46`, `src/genesis/environment/auditor_env.py:276-283`, `src/genesis/generation/export.py:45-70`, `src/genesis/cli/__main__.py:155-163`, `tests/fixtures/policy_smoke/Dockerfile:1-4`, `docs/BASELINE_POLICIES.md:18-22`, `docs/ISOLATION_AND_LEAKAGE.md:17`  
**Evidence:**
- `docker_policy.py:30-37` hardcodes `--read-only` (with no `--tmpfs /tmp`), `--pids-limit 32 --memory 512m --memory-swap 512m --cpus 1` with no constructor parameters to override — using only 12.5% of the official 4 GB RAM cap and 50% of the 2 CPU cap. *(Note: `docs/ISOLATION_AND_LEAKAGE.md:17` intentionally documents these tighter caps, narrowing `ADR 0002:9`'s 2 CPU / 4 GB envelope.)*
- `judge.py:38-45` calls `env.reset()` (starting the 60 s monotonic deadline at `auditor_env.py:276-283`) **before** `runner.open(deadline)` invokes uncached `docker image inspect` + `docker create` + `docker start` per episode (`docker_policy.py:119-128`).
- `DockerPolicy` and `export_episode` have **zero callers anywhere in `src/genesis/`** (only invoked in tests); there is no CSV/episode directory loader in `src/`, no CLI subcommand to run an external policy image or demo episode, and **no Dockerfile exists to containerize `genesis.policies.worker`** despite `docs/BASELINE_POLICIES.md:22`.  
**Problem:** Containerized policy evaluation is implemented as an isolated unit-tested library class rather than a wired, configurable evaluation runner capable of hosting real CPU ML/RL models within the official 2 CPU / 4 GB / 60 s budget.  
**Impact:** Real PyTorch/ONNX/NumPy RL policy containers can OOM at 512 MiB, hit `pids-limit 32` via OpenMP/BLAS thread pools, fail on read-only `/tmp`, or lose seconds per episode to container startup; evaluators cannot run containerized policies or built-in baselines via the CLI without writing custom glue scripts.  
**Realistic scenario:** A team submits a lightweight quantized LLM or PyTorch/ONNX policy container requiring 1.5 GB RAM and `/tmp` scratch space; `DockerPolicy` immediately OOM-kills it at `512m` (earning `+1.0` on clean episodes via `F-02` and `-1.0` on anomalous episodes).  
**Recommended fix:** Make `memory`, `cpus`, `pids_limit`, and bounded `--tmpfs /tmp` configurable on `DockerPolicy` (defaulting to or supporting the official `2 CPU / 4 GB` envelope); cache `image_id` per runner and start the episode timer after container startup (or enforce a separate startup deadline); add a pinned worker `Dockerfile` for `genesis.policies.worker` and CLI subcommands (`run-policy`, `demo`, `export-episode`).

#### F-11: Transport & Isolation Test Gaps (`C17`/`C19`/Docker): Faults Only Tested on Clean 0-Step Books, 6 Documented `jsonl.py` Guards Untested, and Flaky Wall-Clock Sleeps
**Title:** `C17`/`C19`/Docker tests never test transport faults on anomalous or multi-step episodes, leave 6 documented `jsonl.py` safety guards and 28/63 `docker_policy.py` lines uncovered, and use real wall-clock sleeps  
**Severity:** High  
**Confidence:** Confirmed (`stdlib trace` verified by `qa-test`)  
**Affected files:** `tests/test_evaluator_isolation.py:67-179,205-215,300-310`, `src/genesis/environment/jsonl.py:43-130`, `src/genesis/environment/docker_policy.py:53-140`  
**Evidence:**
- `test_C17_transport_accepts_valid_frame_and_bounds_failures` (`test_evaluator_isolation.py:162-168`) tests all 7 transport failure modes **only** on `config={"clean": True}` at `inspection_steps == 0` (`assertEqual(reward, 1)`), never on anomalous books (`clean=False`) or mid-episode faults after $k > 0$ steps.
- `stdlib trace` proves 6 documented `jsonl.py` guards (`docs/ISOLATION_AND_LEAKAGE.md:9`) are **never executed** by the test suite: (1) duplicate JSON object keys (`jsonl.py:46`), (2) JSON `NaN`/`Infinity`/`-Infinity` constants (`jsonl.py:51`), (3) `asyncio.LimitOverrunError` (`jsonl.py:86`), (4) cumulative `STDERR_BYTES` across multiple short lines (`jsonl.py:126`), (5) invalid UTF-8 stdout bytes, and (6) all `JsonlProcess.send()` error branches (`FRAME_LIMIT` L73, `TIMEOUT` L75, `EOF` L80).
- `test_C19` (`test_evaluator_isolation.py:108-130`) uses a stub `handle()` answering only `get_document("policy")` rather than real `build_episode`/`PublicTools`.
- `test_C17` uses real `timeout=0.8` + `time.sleep(20)` (~35% of the entire test suite's 2.3 s runtime) and `DockerPolicyTests` uses a real `0.1 s` read deadline (L308), risking CI flakiness under load.  
**Problem:** Core transport security invariants and mid-episode failure scoring are claimed in docs but unverified by tests.  
**Impact:** Regressions in duplicate-key rejection, `NaN` parsing, stderr byte budgeting, or anomalous-book fault scoring can enter `main` undetected.  
**Realistic scenario:** A refactor breaks `_reject_duplicate_keys` in `jsonl.py:46`; `C17` still passes because duplicate keys were never sent in any test.  
**Recommended fix:** Expand `test_evaluator_isolation.py` with unit tests exercising all 6 `jsonl.py` guards, mid-episode faults on both clean and anomalous books, real `build_episode` in `C19`, and mocked/injected clocks instead of `time.sleep(20)`.

---

### Medium Findings

#### F-12: Timezone Split-Brain (`+05:00` vs OS `ZoneInfo("Asia/Almaty")`), Test-Only `tzdata` Misclassification, and Unhashed Pip Lockfile
**Title:** Hardcoded `+05:00` vs OS `ZoneInfo("Asia/Almaty")` corrupts `partial_fx_payment_at_cutoff` & checksums on pre-2024a `tzdb`; `tzdata` is misclassified as test-only; pip lockfile lacks SHA-256 hashes  
**Severity:** Medium *(Disputed High vs Medium in Phase 2: `backend` & `failure-analyst` argued High for cross-host reproducibility; `devops-infra` argued Medium because 2026 hosts/Dockerfile ship `tzdb >= 2024a`)*  
**Confidence:** Confirmed  
**Affected files:** `src/genesis/generation/company.py:102,276,312`, `src/genesis/generation/interactions.py:49,79-80`, `src/genesis/generation/fx_cutoff.py:65,121,141`, `src/genesis/policies/reconcile.py:59-61`, `tests/requirements-schema.in:1-4`, `tests/requirements-schema.lock:1-26`, `Dockerfile:4-6`  
**Evidence:**
- `company.py:102` hardcodes `LOCAL_OFFSET = timezone(timedelta(hours=5))` and slices `timestamp[:10]` (`company.py:276`), while publishing `"timezone": "Asia/Almaty"` (`company.py:312`). Meanwhile `interactions.py:79-80`, `fx_cutoff.py:65,141`, `public_proofs.py:21-22`, and `reconcile.py:59,93` resolve via `ZoneInfo("Asia/Almaty")`, which prefers system `/usr/share/zoneinfo` over pip `tzdata`.
- Empirical simulation of pre-2024a `Asia/Almaty` (`+06:00`, before Kazakhstan's March 2024 UTC+5 transition) on `train:20260106` shifts `2026-01-31T23:45:00+05:00` → `2026-02-01T00:45:00+06:00`, colliding `payment_day == wrong_date`, duplicating `fx:2026-02-01:JPY:KZT` in `acceptable_evidence_sets` (violating `uniqueItems`), failing `verify_public_proofs`, and breaking `checksums.json`.
- Conversely, on hosts/containers *without* OS `/usr/share/zoneinfo` (Windows or slim worker images), `tzdata==2025.2` is listed only in `tests/requirements-schema.in:1`, causing `ZoneInfoNotFoundError` crashes in generation/proofs and **silent 0%-recall abstention** in `reconcile.py:59-61`.
- `tests/requirements-schema.lock` contains version pins only, with zero `--hash=sha256:...` entries for `pip install --require-hashes`.  
**Problem:** Generator timestamps mix a hardcoded UTC+5 offset with host-OS-dependent IANA timezone lookups, while `tzdata` is treated as a test-only dependency and lockfile integrity is unhashed.  
**Impact:** External policy containers or evaluation hosts with older `tzdb < 2024a` or missing OS `tzdata` silently disagree with the judge on cutoff dates or abstain with zero errors.  
**Realistic scenario:** A participant builds a policy container from an older `python:3.11-slim` image (pre-2024a `tzdb` or no `tzdata` package); `reconcile.py` catches `ZoneInfoNotFoundError` and submits `findings: []` on every episode, or miscomputes `2026-01-31T23:45+05:00` as February 1.  
**Recommended fix:** Derive timestamp offsets consistently via `ZoneInfo` and assert the resolved offset matches `+05:00` at startup (or declare `tzdata` as a runtime dependency in `pyproject.toml`), and add `--hash=sha256:...` (`pip install --require-hashes`) to the lockfile.

#### F-13: Unbounded `affected_ids` Length, Unpenalized Evidence Stuffing, Arbitrary-String Trajectory Smuggling, and `>4300`-Digit `claimed_amount_minor` Crash
**Title:** Missing schema/validator bounds on `affected_ids`, ID length, and `claimed_amount_minor` allow 500 KiB trajectory bloat, redaction smuggling, unpenalized evidence stuffing, and in-process `to_jsonl()` crashes  
**Severity:** Medium  
**Confidence:** Confirmed  
**Affected files:** `src/genesis/environment/auditor_env.py:123-160`, `src/genesis/trajectories/records.py:79-94,108`, `src/genesis/evaluation/matcher.py:122`, `schemas/findings.schema.json:18-37`, `schemas/common.schema.json:6-9,35-37`  
**Evidence:**
- `schemas/findings.schema.json:18-24` and `auditor_env.py:124,141` enforce `maxItems: 10` on `evidence_ids`, but place **no `maxItems` or `maxLength`** on `affected_ids` or ID strings. Empirically measured: a valid `submit` with 52,000 `affected_ids` (`457.3 KiB <= 512 KiB` frame budget) passes validation (`invalid_submission=False`) and writes a `468 KiB` single JSONL trajectory row (`3.53 MiB` peak heap).
- Because `records.py:88-89` blanks only `finding["explanation"] = ""` while keeping `affected_ids`/`evidence_ids` verbatim, an untrusted policy can smuggle arbitrary unredacted strings into public trajectory exports.
- `matcher.py:122` checks `evidence_set <= finding.evidence_ids`, so a policy can stuff up to `10` guessed evidence IDs per finding with zero precision penalty as long as the true set is a subset.
- `claimed_amount_minor` accepts negative integers and `10**5000` via the in-process Python API (`auditor_env.py:142` checks only `type(v) is int`), crashing `records.py:108` (`json.dumps`) with `ValueError: Exceeds the limit (4300 digits)` *(safely caught as `BAD_FRAME` over JSONL transport)*.  
**Problem:** Submission validation omits cardinality, string-length, uniqueness (`uniqueItems`), and numeric range bounds.  
**Impact:** Trajectory storage bloat, redaction bypass, evidence-stuffing gaming, and in-process serialization crashes.  
**Realistic scenario:** An agent submits 10 padded evidence IDs per finding (e.g., all ledger rows + `"policy"`) to guarantee `evidence_set <= finding.evidence_ids` without identifying minimal proof, or embeds 400 KiB of arbitrary text inside `affected_ids` that survives public trajectory export.  
**Recommended fix:** Enforce `maxItems` (e.g., 32), `uniqueItems: true`, and `maxLength` (e.g., 128) on `affected_ids`/`evidence_ids`, enforce `0 <= claimed_amount_minor <= 10**18` in both JSON schemas and `auditor_env.py`, and consider validating IDs against known public IDs or penalizing irrelevant evidence bloat.

#### F-14: Published `trajectories/trajectories.jsonl` Originates from a 4-Row Handcrafted Fixture, Omits `reset` ($s_0$), Unconditionally Wipes `explanation`, and Erases Valid Failed Tool Calls
**Title:** `trajectories/trajectories.jsonl` is a 1-episode handcrafted fixture export missing `reset` ($s_0$), wiping `explanation` in `train` mode, and overwriting valid failed tool actions as `{"type": "invalid"}`  
**Severity:** Medium *(Medium on Section 4.5 rubric; High for actual offline RL / behavioral cloning consumers — reconciled in Phase 2)*  
**Confidence:** Confirmed  
**Affected files:** `trajectories/trajectories.jsonl:1-5`, `src/genesis/trajectories/records.py:79-93,192-208`, `src/genesis/evaluation/judge.py:61-63`, `src/genesis/environment/auditor_env.py:354-358`, `tests/test_trajectories.py:21-49,216-225`, `docs/TRAJECTORIES.md:22,24`  
**Evidence:**
- `trajectories/trajectories.jsonl:1-5` records a single 5-step episode (`ep-cf50625d`, 4 ledger rows, USD/KZT) exported from `tests/fixtures/contract/public/` via `python tests/test_trajectories.py --export` (`test_trajectories.py:216-225`), not from `generate_episode()` or `data/train`.
- `records.py:192-204` omits the initial `reset` observation ($s_0$: `task`, `manifest`, `policy`, `currency_exponents`), even though step 5 cites `"policy"` and FX rules that never appear in steps 1–4.
- `records.py:88-89` sets `finding["explanation"] = ""` unconditionally in **both** `train` and `eval` modes, contradicting `docs/TRAJECTORIES.md:22` (which scopes explanation blanking to closed `eval` mode).
- `auditor_env.py:356` + `judge.py:62` + `records.py:80-81` overwrite **schema-valid** tool actions that return `not_found`, `division_by_zero`, or `invalid_cursor` into `{"type": "invalid"}`, destroying action history needed for RL credit assignment. *(Also `SEC-07`: direct `TrajectoryRecorder.record()` without `judge.py`'s `env.last_action_failed` check retains step-100 failed payloads differently.)*  
**Problem:** While schema-valid and leakage-free, the trajectory format omits $s_0$ and corrupts failed tool calls, and the shipped deliverable file comes from a toy unit-test fixture rather than generated benchmark episodes.  
**Impact:** An offline RL/BC trainer cannot reconstruct the state-action-reward sequence $(s_0, a_1, s_1, \dots)$ from `trajectories.jsonl`, and `README.md:60`'s framing of contract fixtures vs published artifacts is misleading.  
**Realistic scenario:** A researcher loads `trajectories/trajectories.jsonl` to train or inspect an auditing policy and discovers there is only 1 four-row USD fixture episode with no `policy`/`manifest` context and wiped tool arguments on error steps.  
**Recommended fix:** Add an optional `step: 0` (or `reset` envelope) containing sanitized `reset` observation, preserve schema-valid tool actions even when returning controlled tool errors (redacting only malformed frames), preserve `explanation` in `train` mode per docs, and generate `trajectories/trajectories.jsonl` from real `data/train` episodes via a `src/genesis/` CLI command.

#### F-15: Post-`submit` Terminal Ack Race Misclassifies Cleanly Submitted Episodes as `TransportCode.EOF`/`TIMEOUT`
**Title:** Sending the post-`submit` terminal `step` ack over the policy pipe creates an EOF/timeout race that marks cleanly submitted episodes as transport failures  
**Severity:** Medium  
**Confidence:** Confirmed (discovered by `failure-analyst`, empirically reproduced by `security` in Phase 2)  
**Affected files:** `src/genesis/evaluation/judge.py:48-71`, `src/genesis/environment/jsonl.py:70-80`, `src/genesis/evaluation/reproducibility.py:155`, `src/genesis/cli/__main__.py:126`  
**Evidence:**
- In `judge.py:49-59`, `env.step(action)` processes `submit` and records `terminated=True, reason="submitted"`, then `judge.py:56` calls `await channel.send({"type": "step", ...}, deadline)`.
- Empirically reproduced: if the policy closes stdin or exits immediately after writing `submit`, or if `monotonic() >= deadline` elapses between line 49 and line 56, `channel.send()` raises `TransportError(EOF | TIMEOUT)`, which `judge.py:66-68` records as `fault = error.code` (producing `JudgeResult(terminated=True, reason="submitted", reward=1.0, transport_error=TransportCode.EOF)`).
- Strict callers (`reproducibility.py:155`, `cli/__main__.py:126`) treat any `transport_error is not None` as a failed benchmark run.  
**Problem:** A policy that exits immediately after emitting its terminal `submit` frame races with the harness's final `step` ack write.  
**Impact:** Valid, complete submissions are spuriously rejected as transport failures depending on OS pipe scheduling or near-deadline submission timing.  
**Realistic scenario:** A fast external policy container writes `{"type":"submit",...}\n`, flushes stdout, and returns from `main()`. The container exits 2 ms before `judge.py:56` writes the final `step` ack, causing `TransportCode.EOF` and disqualifying the episode.  
**Recommended fix:** Once `env.step()` sets `terminated or truncated`, treat `EOF`/`BrokenPipeError` (or tight deadline expiry) during the best-effort final ack send as non-fatal and do not overwrite an already-completed submission with a transport fault.

#### F-16: `DockerPolicy` Bugs: 64 KiB `image inspect` Cap, Missing `--no-healthcheck`/Ulimit/Cgroupns, and Cleanup-Deadline Reuse Masking `create` Errors
**Title:** `DockerPolicy` fails on multi-layer images with `>64 KiB` inspect JSON, omits container hardening flags (`--no-healthcheck`, `ulimit`, `cgroupns`), and masks `create` errors as `PolicyCleanupError`  
**Severity:** Medium  
**Confidence:** Confirmed (verified by `performance`, `devops-infra`, `failure-analyst`, `security`)  
**Affected files:** `src/genesis/environment/docker_policy.py:18,30-37,75-79,90-107,120-140`  
**Evidence:**
- `docker_policy.py:18,75-79,91-93` runs unfiltered `docker image inspect <image>` with a `64 KiB` stdout cap (`_MAX_INSPECT_BYTES = 65536`); multi-layer ML images or label-rich images exceed `64 KiB` and fail with `PROCESS_ERROR` (whereas `--format '{{.Id}}'` returns `<100` bytes).
- `docker_policy.py:30-37,90-107` omits `--no-healthcheck`, `--ulimit nofile`, and `--cgroupns private`, and never checks `Config.Healthcheck` — allowing an image-baked `HEALTHCHECK` to execute probes inside the container and persist probe output in the host Docker daemon despite `--log-driver none`.
- `docker_policy.py:135-140` reuses the same 5 s `cleanup_deadline` across `docker rm -f` and fallback `_container_absent()`; if `docker create` failed (`created=False`) and `rm` times out, `_container_absent()` sees an expired deadline and raises `PolicyCleanupError`, aborting closed evaluation.  
**Problem:** Minor defects in Docker CLI invocation, image inspection, and cleanup error handling reduce container robustness and defense-in-depth.  
**Impact:** Valid ML policy images fail image inspection; untrusted images with `HEALTHCHECK` execute background commands; failed container creation can crash batch evaluation via `PolicyCleanupError`.  
**Realistic scenario:** An entrant's PyTorch image has 25 layers + OCI labels (`78 KiB` inspect JSON). `DockerPolicy._local_image_id()` truncates/fails at `64 KiB` and rejects the image before running step 1.  
**Recommended fix:** Use `docker image inspect --format '{{json .Id}}'` (or inspect only needed fields including `Config.Healthcheck`), add `--no-healthcheck`, `--ulimit nofile=256:256`, and `--cgroupns=private`, and allocate an independent short timeout for `_container_absent()` (skipping cleanup error escalation when `created=False`).

#### F-17: `AuditorEnv` Boundary Bugs & Schema-vs-Code Validation Drift
**Title:** `AuditorEnv` accepts `bool` `timeout_seconds`, negative manifest counts/exponents, bogus policy enums, silent `currency_exponents` drops, unhandled `StopIteration`, stale `last_action_failed` on timeout, and zero validation diagnostics in `train` mode  
**Severity:** Medium  
**Confidence:** Confirmed (reproduced via direct Python probes by `failure-analyst`, `maintainability`, `frontend-ux`)  
**Affected files:** `src/genesis/environment/auditor_env.py:95-108,180-275,344-352,404-427`, `schemas/manifest.schema.json:29-71`, `schemas/policy.schema.json:6-30`, `schemas/observation.schema.json:80-95`, `schemas/config.schema.json:33-39`, `schemas/trajectories.schema.json:34-36`  
**Evidence:**
- `auditor_env.py:253`: `if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:` accepts `timeout_seconds=True` (`bool` is a subclass of `int` → 1 s timeout).
- `auditor_env.py:181-182,265-270`: uses magic-string `_field("document_count", value)` for `currency_exponents`, silently drops non-uppercase currency keys (e.g., `"usd"`), accepts `document_count=-5`, `record_counts.ledger=-1`, `currency_exponents={"KZT": -3}`, and bogus policy strings (`fx_date_rule="BOGUS"`, `rounding_mode="FAKE"`, `timezone=""`).
- `auditor_env.py:213`: `next(key for key in _RECORD_FIELDS if key in value)` raises unhandled `StopIteration` on unrecognized dicts.
- `auditor_env.py:344-345`: timeout branch returns `_finish(...)` without clearing `self._last_action_failed`, leaving stale `True` after a prior invalid step.
- `auditor_env.py:95-108,349-352,421-427`: malformed `submit` returns `status: "submitted"` with `reward=-1.0, info={}` in `train` mode and exposes zero field-level validation reason on trusted `TerminalResult`.
- `schemas/*.json`: zero `"description"` fields across all 14 schemas; `observation.schema.json:80-95` omits the 8-value `error.code` enum enforced by `records.py:24`; `config.schema.json:33-39` omits `interaction_scenarios` enum; `trajectories.schema.json:34-36` permits reset observations rejected by `records.py:43-46`.  
**Problem:** Hand-rolled validation in `AuditorEnv` drifts from JSON Schemas, silently accepts invalid state, and gives poor debugging diagnostics in open `train` mode.  
**Impact:** Subtle bugs in custom episode loaders or callers go undetected until crashing deeper in the stack; RL developers debugging malformed `submit` payloads in `train` mode get no hint which field failed validation.  
**Realistic scenario:** An agent developer in `mode="train"` passes `claimed_amount_minor=10.0` (float instead of int); `env.step()` returns `({"remaining_steps": 99, "status": "submitted"}, -1.0, True, False, {})` with no field diagnostic.  
**Recommended fix:** Reject `bool` for `timeout_seconds`; validate non-negative counts/exponents and known policy enums explicitly; raise `ValueError` instead of `StopIteration` in `_project_record`; clear `_last_action_failed` on timeout; attach structured validation errors to `TerminalResult` (and `info` in `mode="train"` only, preserving neutral `eval` responses); align schemas with `records.py` and add `"description"` annotations.

#### F-18: `apply_allocation_errors()` Crashes with `TypeError` on Schema-Valid `payment["invoice_refs"] = None`
**Title:** Missing `None` guard in `apply_allocation_errors()` raises `TypeError` on schema-valid unlinked payments (`invoice_refs: null`)  
**Severity:** Medium *(downgraded High → Medium in Phase 2: confirmed crash, currently unreachable via `generate_episode()`)*  
**Confidence:** Confirmed (empirically reproduced by `qa-test` and `backend`)  
**Affected files:** `src/genesis/generation/allocation.py:60-65`, `src/genesis/generation/company.py:50`, `schemas/payment.schema.json:28-40`  
**Evidence:**
- `schemas/payment.schema.json:28-40` explicitly permits `"invoice_refs": null` for unlinked payments (matching Hackathon 4.2: missing relationships represented as explicit `null`).
- `allocation.py:62` executes `len(payment["invoice_refs"]) != 1` without a `None` guard, raising `TypeError: object of type 'NoneType' has no len()`. By contrast, `fx_cutoff.py:84` and `reconcile.py:65` safely use `payment["invoice_refs"] or []`.
- `company.py:50` also mis-types `Payment.invoice_refs: list[str]` instead of `list[str] | None`.  
**Problem:** One injector violates the schema contract for nullable `invoice_refs`.  
**Impact:** Safe today only because `company.py:264` and `distractors.py:61` never emit `invoice_refs: None`; any future unlinked-payment distractor or direct injector test immediately crashes.  
**Realistic scenario:** A developer adds a legitimate unlinked-advance distractor payment (`invoice_refs: None`) per `payment.schema.json`; `generate_episode()` immediately crashes in `apply_allocation_errors()`.  
**Recommended fix:** Guard `refs = payment["invoice_refs"] or []` at `allocation.py:62`, update `Payment.invoice_refs: list[str] | None` in `company.py:50`, and add a regression test in `tests/test_allocation_injections.py`.

#### F-19: Missing Python Packaging (`pyproject.toml`), Missing `evaluation/__init__.py`, Cyclic `environment <-> evaluation` Dependency, and Repo-Coupled CLI/Profile Paths
**Title:** No `pyproject.toml`, implicit namespace `evaluation/` package, bidirectional `environment <-> evaluation` cycle, and hardcoded repo-relative paths in CLI & profile loader  
**Severity:** Medium  
**Confidence:** Confirmed  
**Affected files:** Repo root (no `pyproject.toml`), `src/genesis/evaluation/` (missing `__init__.py`), `src/genesis/environment/auditor_env.py:15-17`, `src/genesis/environment/episode.py:6-7`, `src/genesis/cli/__main__.py:13,155-163`, `src/genesis/evaluation/reproducibility.py:48-73,275`  
**Evidence:**
- No `pyproject.toml`/`setup.py` exists; all invocations require manual `PYTHONPATH=src`.
- `src/genesis/evaluation/` has no `__init__.py` (the only subpackage under `src/genesis/` missing one) and mixes 10 unrelated responsibilities (matching, rewards, oracle, proofs, judge harness, permutation, benchmark, reproducibility).
- Bidirectional package cycle: `environment/auditor_env.py:15-17` imports `evaluation.matcher`, `evaluation.reward`, `evaluation.terminal`, while `evaluation/judge.py`, `evaluation/benchmark.py`, and `evaluation/reproducibility.py` import `environment.auditor_env`, `environment.jsonl`, `environment.episode`.
- `cli/__main__.py:13` hardcodes `_REPOSITORY = Path(__file__).resolve().parents[3]` and `reproducibility.py:48-73` hardcodes exact 20-profile category counts (`5 clean`, `5 interacting`, `10 standard`), rejecting custom closed-eval profile files.  
**Problem:** Packaging, module boundaries, and profile loading are tightly coupled to the exact repository layout and local diagnostic profile shape.  
**Impact:** Prevents `pip install -e .` / console-script entrypoints, complicates moving/installing the package, risks circular-import traps during refactoring, and prevents judges from pointing the harness at custom hidden evaluation configs without code edits.  
**Realistic scenario:** A judge installs the package into an evaluation venv or passes a custom 10-episode hidden profile JSON; `pip install .` fails (no `pyproject.toml`) and `load_profiles()` rejects non-`5/5/10` profile mixes.  
**Recommended fix:** Add `pyproject.toml` (Python `>=3.11`, `genesis` console script, runtime `tzdata` dep, optional `[dev]` schema deps), add `src/genesis/evaluation/__init__.py`, split scoring primitives (`scoring/`) from runtime harness (`harness/`) to break the cycle, and allow configurable profile counts/paths via CLI flags.

#### F-20: CI Workflows Never Trigger on `push` to `main` (or `workflow_dispatch`) and Never Exercise `verify --all --require-docker`
**Title:** GitHub Actions workflows trigger only on `pull_request` and never execute `genesis.cli verify --all --require-docker`  
**Severity:** Medium  
**Confidence:** Confirmed  
**Affected files:** `.github/workflows/data-contract.yml:3-8`, `.github/workflows/offline-verify.yml:3-6`, `.github/workflows/policy-isolation.yml:3-6,29`  
**Evidence:**
- All three workflows declare only `on: pull_request:` (no `push: branches: [main]` and no `workflow_dispatch:`).
- `policy-isolation.yml:29` runs only `python -m unittest tests.test_evaluator_isolation`, never `python -m genesis.cli verify --all --require-docker`.  
**Problem:** Post-merge `main` state is never re-verified by CI, manual re-runs via GitHub UI are impossible, and the `--require-docker` CLI path is never CI-tested.  
**Impact:** Direct pushes or merge-commit interactions on `main` go untested, and `--require-docker` regressions are invisible in CI.  
**Realistic scenario:** A PR merges into `main` while another PR changed `cli/__main__.py`; `main` breaks without triggering any CI run.  
**Recommended fix:** Add `push: branches: [main]` and `workflow_dispatch:` to all three workflows, and run `python -m genesis.cli verify --all --require-docker` in `policy-isolation.yml`.

#### F-21: Incomplete Normalization of Empty `interaction_scenarios: []` Breaks Provenance Equivalence and Cross-Split Config Deduplication
**Title:** Omitting `interaction_scenarios` vs specifying `interaction_scenarios: []` produces different `config_hash` / `provenance` and bypasses train/dev config-overlap deduplication  
**Severity:** Medium  
**Confidence:** Confirmed (`backend` `FIN-06`)  
**Affected files:** `src/genesis/generation/company.py:125-134`, `src/genesis/generation/datasets.py:73-82`  
**Evidence:**
- `company.py:125-134` normalizes omitted `interaction_scenarios` vs `[]` differently in the canonicalized config payload hashed into `provenance.config_hash`, and `datasets.py:73-82` checks split overlap using raw config dict equality.  
**Problem:** Two semantically identical configs (`{"error_counts": {}}` and `{"error_counts": {}, "interaction_scenarios": []}`) produce identical episodes but different provenance hashes and pass split-disjointness checks as "distinct."  
**Impact:** Undermines Section 4.4's train/dev vs hidden config separation check and reproducibility metadata consistency.  
**Realistic scenario:** A hidden evaluation profile specifies `interaction_scenarios: []` while a public train episode omits the key; `datasets.py` fails to detect that the exact same episode configuration leaked across splits.  
**Recommended fix:** Normalize empty `interaction_scenarios: []` (and zero-count keys in `error_counts`) canonically in one shared `normalize_config()` function used by both `company.py` and `datasets.py`.

#### F-22: Scaling Bottlenecks Toward Target 2,000-Row Episodes & High-Throughput RL Training (`step()` Copy Overhead & Quadratic `validate_links`/`fx_cutoff`)
**Title:** ~93% redundant copy/JSON round-tripping in `step()` and quadratic $O(L \cdot (I + P))$ loops in `permutation.validate_links` (62×–104× slowdown) and `fx_cutoff.py`  
**Severity:** Medium *(Low for current 40-row v1.0.0 eval; Medium for 2,000-row scaling & 10k-episode RL training — measured & calibrated by `performance`)*  
**Confidence:** Confirmed (empirically benchmarked by `performance`)  
**Affected files:** `src/genesis/environment/tools.py:36-38,50,77,92`, `src/genesis/environment/auditor_env.py:238-241,381,390`, `src/genesis/evaluation/permutation.py:70-78,107,130`, `src/genesis/generation/fx_cutoff.py:79,130-134`, `src/genesis/evaluation/reproducibility.py:238-249`, `src/genesis/generation/datasets.py:142-236`  
**Evidence:**
- At 2,000L / 200D scale, `PublicTools.__init__` spends `12.87 ms` (`77%`) on redundant `_copy_public(_project_record(...))` vs `3.06 ms` for projection alone, and `AuditorEnv.step()` + `TrajectoryRecorder` re-project and deep-copy records 3× (`~93%` of step CPU).
- `permutation.validate_links` rebuilds `invoice_ids` and `payment_ids` sets **inside** the ledger-row loop ($O(L \cdot (I + P))$): measured `11.15 ms` vs `0.18 ms` hoisted (**62.3× slower**) at 2,000 rows, and `35.41 ms` vs `0.34 ms` (**104.0× slower**) at 4,000 rows.
- `fx_cutoff.py` performs quadratic $O(P \cdot L + I \cdot L)$ ledger scans (`9.18 ms` at 2,000 rows); `local_benchmark()` regenerates each episode 5× redundantly (`reproducibility.py:238-249`).  
**Problem:** Unnecessary O(N²) loops and triple deep-copying waste CPU when scaling episodes or running thousands of RL rollouts.  
**Impact:** Non-blocking under the 60 s limit up to 2,000 rows (`~99 ms` subprocess episode), but needlessly slows RL training loops and permutation sweeps by an order of magnitude.  
**Realistic scenario:** Scaling episodes to 2,000–4,000 rows for RL training causes `step()` and `permute_episode()` to spend >90% of CPU copying immutable public dicts and rebuilding identical ID sets.  
**Recommended fix:** Hoist `invoice_ids`/`payment_ids` sets outside the loop in `permutation.py:70-78` (3-line fix → 62× speedup); pre-index ledger rows by `payment_id`/`invoice_id` in `fx_cutoff.py`; project records once on `PublicTools` load and return shallow/single copies on read; reuse generated episodes across policies in `local_benchmark`.

#### F-23: Weak Baseline Policies and Untested Coverage Gaps Across Public Datasets, Policies, Trajectories, and Reward Edge Cases
**Title:** `simple_reconciliation` covers only 1/4 error types (negative return `-0.0315`), `random` never submits findings, and major test paths in datasets/policies/trajectories/rewards remain uncovered  
**Severity:** Medium  
**Confidence:** Confirmed (verified via `verify --all` & `stdlib trace`)  
**Affected files:** `src/genesis/policies/reconcile.py:32-139`, `src/genesis/policies/baseline.py`, `src/genesis/policies/worker.py:14-45`, `tests/test_public_datasets.py:44-156`, `tests/test_allocation_injections.py:151-178`, `tests/test_generator_and_invariants.py:38-90`, `tests/test_terminal_reward.py:47-51`, `tests/test_policy_adapters.py:82-216`, `tests/test_trajectories.py:70-214`  
**Evidence:**
- Measured 80-run benchmark output: `simple_reconciliation`: `P=0.3500, R=0.1875, F1=0.4867, return=-0.0315` (only detects standalone `fx_conversion` on `payment_date`; misses `duplicate_invoice`, `payment_allocation`, `period_cutoff`, and `partial_fx_payment_at_cutoff`); `random`: `P=0.0000, R=0.0000, F1=0.2500, return=-0.5040` (always submits `[]`).
- `test_public_datasets.py` verifies arithmetic provability on only 4/20 committed episodes (`0/10 dev`) and never checks `policy.json`, period bounds, or line-item sums.
- Generator schema tests (`test_allocation_injections.py:151-178`, `test_generator_and_invariants.py:38-90`) bypass strict integer `wire_validator` (accepting floats).
- `test_terminal_reward.py:47-51` never actually triggers the `-1.0` lower clamp; `E17` (`mark_all`) only tests a clean book; `matcher.match()` is untested on empty `acceptable_evidence_sets`; `worker.py` (`5/25` lines covered), `reconcile.py` (`invoice_date`, step cutoff, 30-finding cap), and 15+ `validate_trajectory` rejection paths have zero tests.  
**Problem:** Baseline diagnostic policies provide weak diagnostic signal (negative reconciliation return), and important edge cases in datasets, policies, and validators are untested.  
**Impact:** Weakens measured failure analysis (Section 4.5) and leaves non-trivial policy/validation branches vulnerable to silent regression.  
**Realistic scenario:** An evaluator uses `simple_reconciliation` as a sanity floor for an RL agent and sees it has a negative mean return (`-0.0315`), barely above abstaining (`-0.500`), while regressions in `invoice_date` FX mode or `dev` dataset provability go uncaught.  
**Recommended fix:** Extend `simple_reconciliation` to cover at least basic duplicate/suspense/cutoff checks (or document its single-rule scope more prominently once `F-01` is fixed); test arithmetic provability on all 20/20 `train`+`dev` episodes; use strict `wire_validator` in all generator tests; add coverage for `worker.py`, `validate_trajectory`, and reward lower-clamp edge cases.

#### F-24: Cross-Module Private-Symbol Coupling, Domain-Logic Duplication, Fragile `endswith("_id")` Rewriting, and Inconsistent Clock Injection
**Title:** Extensive cross-package `_private` imports, duplicated FX/evidence helpers, fragile suffix-based ID renaming in `permutation.py`, and non-injected `monotonic()` in `judge.py`/`jsonl.py`  
**Severity:** Medium  
**Confidence:** Confirmed  
**Affected files:** `src/genesis/trajectories/records.py:9-19`, `src/genesis/environment/tools.py:8-10`, `src/genesis/environment/auditor_env.py:75-78,249-344`, `src/genesis/evaluation/judge.py:42`, `src/genesis/environment/jsonl.py:36-74`, `src/genesis/evaluation/permutation.py:11-54,92-104`, `src/genesis/domain/money.py:6-13`, `src/genesis/evaluation/public_proofs.py:18-29`, `src/genesis/policies/reconcile.py:132-139`  
**Evidence:**
- `records.py:9-19` and `tools.py:8-10` import private `_` helpers (`_project_record`, `_validated_submission`, `_is_id`, etc.) directly from `auditor_env.py`; `auditor_env.py:75-78` pollutes `TerminalEvaluator` with test-only union types.
- `convert_minor`, FX lookup key formatting (`fx:<date>:<src>:<dst>`), paired ledger legs, and JSON serialization helpers are duplicated across `domain/money.py`, `public_proofs.py`, `reconcile.py`, and `tools.py`; `Episode` class name collides across modules.
- `permutation.py:11-54` rewrites IDs via generic `key.endswith("_id")` / `key.endswith("_ids")` heuristics plus hardcoded provenance key lists, which silently mis-rewrites or misses future fields.
- `AuditorEnv` supports injectable `clock`, but `judge.py:42` and `jsonl.py:36-74` hardcode `time.monotonic()`, forcing real wall-clock sleeps in isolation tests (`F-11`).  
**Problem:** Tight private coupling and duplicated primitives increase maintenance cost and fragility when extending schemas or adding new error types.  
**Impact:** Any schema field rename (e.g., adding `vendor_tax_id` or changing FX key format) risks subtle divergence across modules or silent permutation corruption.  
**Realistic scenario:** A maintainer adds a `tax_id` string field to `Invoice`; `permutation.py` detects `endswith("_id")` and either crashes or silently rewrites tax numbers as entity IDs.  
**Recommended fix:** Promote shared validation/projection/FX-key helpers to public modules (`domain/` or `environment/contract.py`); replace suffix heuristics in `permutation.py` with an explicit field-rewrite table; inject `clock` into `judge.run_episode()` and `JsonlProcess`.

---

### Low Findings

#### F-25: `export_episode` Writes `evaluator-only/` Files with World-Readable `0o755`/`0o644` Permissions and Has a Dangling-Symlink Check Gap
**Title:** `export_episode` leaves `evaluator-only/truth.json` and `provenance.json` world-readable (`0o755`/`0o644`) on multi-user hosts and checks `target.exists()` instead of `is_symlink()`  
**Severity:** Low *(calibrated Medium → Low in Phase 2: unexploitable from `DockerPolicy` which mounts zero host paths; relevant on shared multi-user hosts)*  
**Confidence:** Confirmed (verified via `stat` by `security` and `failure-analyst`)  
**Affected files:** `src/genesis/generation/export.py:45-70`  
**Evidence:** `staged.rename(target)` moves files out of the `0o700` `TemporaryDirectory` with default umask (`0o755` dir, `0o644` files). `target.exists()` (`export.py:53`) returns `False` on a dangling symlink (failing later with `NotADirectoryError` at `rename`).  
**Problem:** Evaluator-only ground truth lacks restrictive filesystem permissions on shared hosts.  
**Impact:** On a shared Linux host running non-containerized local policies under another UID, `evaluator-only/truth.json` would be readable.  
**Realistic scenario:** An evaluation host stages closed episodes under `/var/tmp/eval/` with default umask `0022` alongside an unprivileged local worker process.  
**Recommended fix:** Apply `os.chmod(evaluator_dir, 0o700)` and `os.chmod(file, 0o600)` before moving, and check `target.exists() or target.is_symlink()`.

#### F-26: `DockerPolicy._host_environment()` Strips `DOCKER_HOST`, `HOME`, and `XDG_RUNTIME_DIR`, Breaking Rootless Docker and macOS Docker Desktop
**Title:** Host `docker` CLI environment scrubbing breaks rootless Linux Docker and macOS Docker Desktop  
**Severity:** Low  
**Confidence:** Confirmed  
**Affected files:** `src/genesis/environment/docker_policy.py:26-27,59`, `docs/ISOLATION_AND_LEAKAGE.md:15,23-29`  
**Evidence:** `_host_environment()` passes only `PATH`, dropping `DOCKER_HOST`, `HOME`, and `XDG_RUNTIME_DIR` needed to locate `~/.docker/run/docker.sock` or `$XDG_RUNTIME_DIR/docker.sock`.  
**Problem:** Over-aggressive host-CLI env scrubbing breaks non-rootful Docker daemons without improving container isolation (container env is already isolated via `docker create`).  
**Impact:** Developers and judges on macOS Docker Desktop or rootless Linux cannot run `DockerPolicy` or `verify --all --require-docker` locally.  
**Realistic scenario:** A judge on macOS with Docker Desktop runs `python -m genesis.cli verify --all --require-docker` and gets connection-refused errors against `/var/run/docker.sock`.  
**Recommended fix:** Preserve `DOCKER_HOST` (and `XDG_RUNTIME_DIR`/`DOCKER_CONTEXT` when explicitly configured) for the host `docker` client while continuing to pass zero host env vars into the container itself.

#### F-27: Shared Non-Cryptographic MT19937 PRNG (`_Ids`) Interleaves Public Record IDs with Private `issue_id`/`change_id`
**Title:** `_Ids` draws both public record IDs and private evaluator-only IDs from a single MT19937 `random.Random` stream  
**Severity:** Low *(latent scaling risk only — unexploitable at current v1.0.0 scale)*  
**Confidence:** Confirmed (verified by `security` and `failure-analyst`)  
**Affected files:** `src/genesis/generation/company.py:174-185,233-345`, `src/genesis/generation/allocation.py:121-132,141,157`, `src/genesis/generation/fx_cutoff.py:173,183`, `src/genesis/generation/interactions.py:89,95,103,120`  
**Evidence:** One `random.Random` generates 96-bit IDs (3 × 32-bit MT19937 words) for both public records and private `issue_id`/`change_id`. At v1.0.0 scale (~87 public IDs = ~261 words `< 624` state words, plus row shuffling), MT19937 state recovery is impossible; it only becomes a theoretical risk if scaled past ~208 sequential unshuffled IDs.  
**Problem:** Public and private ID generators share PRNG state instead of domain-separated streams.  
**Impact:** Zero practical risk today; poor cryptographic hygiene if episodes scale to 2,000 rows without shuffling.  
**Realistic scenario:** A scaled 2,000-row episode leaks >624 consecutive MT19937 words in predictable ID order, enabling state recovery of private IDs.  
**Recommended fix:** Derive a separate seeded `random.Random` (or HMAC-DRBG/SHA-256 counter) for evaluator-only `issue_id`/`change_id`.

#### F-28: Container Image Omits `LICENSE` & `NOTICE` (MIT Redistribution Notice) and Lacks Air-Gapped Wheelhouse Build Option
**Title:** `Dockerfile` / `.dockerignore` excludes `LICENSE` and `NOTICE` from the built container image and requires online PyPI during `docker build`  
**Severity:** Low  
**Confidence:** Confirmed  
**Affected files:** `Dockerfile:13-22`, `.dockerignore:1-14`, `LICENSE`, `NOTICE`  
**Evidence:** `.dockerignore` / `Dockerfile` copy only `src/`, `schemas/`, `data/`, `benchmarks/`, `trajectories/`, and `tests/`, omitting `LICENSE` and `NOTICE`; stage 1 runs `pip wheel` against PyPI without an optional `--find-links` wheelhouse arg.  
**Problem:** Redistributed container images omit the required MIT copyright/license notice and cannot be built in strictly air-gapped environments without editing the Dockerfile.  
**Impact:** Minor license-redistribution omission and build-time offline friction (runtime is 100% offline).  
**Realistic scenario:** An air-gapped evaluation cluster tries `docker build` with a local wheel directory and has to patch `Dockerfile`.  
**Recommended fix:** `COPY LICENSE NOTICE ./` in `Dockerfile` and support an optional `ARG PIP_FIND_LINKS` / `--no-index` build option.

#### F-29: Unenforced ADR 0003 Duplicate-Truth Invariant, Ignored Unknown `tolerances_minor` Keys, Missing Accrual FX Date Policy Field, and `matcher.match()` Direct-Call Recursion Limit
**Title:** Minor evaluator/policy edge cases: unenforced duplicate truth claims, silently ignored unknown tolerance keys, implicit accrual FX date rule, and recursive DFS `RecursionError` above ~1,000 pairs  
**Severity:** Low  
**Confidence:** Confirmed  
**Affected files:** `src/genesis/evaluation/matcher.py:47-65,79-81,96-107`, `src/genesis/generation/company.py:311-317`, `src/genesis/generation/interactions.py:122-127`, `docs/ADR/0003-matching-and-reward.md:21`  
**Evidence:**
- `matcher.py` does not reject duplicate `(issue_type, canonical_affected_ids)` in truth (promised by ADR 0003:21) and silently ignores unknown keys in `tolerances_minor`.
- `policy.json` specifies `fx_date_rule` for payments (`"payment_date"`), but omits the FX date rule used for foreign-currency invoice accruals (`issued_at` is hardcoded in `interactions.py:122-127` and `public_proofs.py:38-40`).
- `matcher.py:47-64` uses recursive augmenting-path DFS: safe inside `AuditorEnv` (capped at `<=30` findings, `0.41 ms`), but hits `RecursionError` if invoked directly on chained `>1,000`-finding inputs.  
**Problem:** Small contract omissions and a direct-call recursion boundary.  
**Impact:** Low at current 30-finding cap; causes subtle surprises on custom truth files or large standalone matcher calls.  
**Realistic scenario:** A caller passes `tolerances_minor={"fx_error": 5}` (typo for `"fx_conversion"`) and gets no error while tolerance silently defaults to `0`.  
**Recommended fix:** Validate `tolerances_minor` keys against `FindingType`, assert unique truth `(issue_type, affected_ids)` keys, declare `accrual_fx_date_rule` in `policy.json`, and convert DFS in `matcher.py` to iterative Hopcroft–Karp or iterative BFS/DFS.

#### F-30: Documentation & ADR Drift, Minor CLI UX Polish, and `load_profiles`/`validate_trajectory`/Test Coupling Fragility
**Title:** Stale docs (`make verify` without `Makefile`, "Proposed" ADRs, mixed RU/EN in `src/`), minor CLI flag ergonomics, and raw error/test-coupling fragility  
**Severity:** Low  
**Confidence:** Confirmed  
**Affected files:** `docs/ARCHITECTURE.md:3,44`, `docs/ADR/0000-context.md`..`0003-matching-and-reward.md`, `docs/DATA_CONTRACT.md:80,97`, `.github/PULL_REQUEST_TEMPLATE.md:18`, `src/genesis/cli/__main__.py:104-105,139,159`, `src/genesis/evaluation/reproducibility.py:39-40`, `src/genesis/trajectories/records.py:130-133`, `tests/test_contract_matrix.py:10-17`, `tests/test_reproducibility_and_permutation.py:46`  
**Evidence:**
- `docs/ARCHITECTURE.md:44` and `.github/PULL_REQUEST_TEMPLATE.md:18` reference `make verify` (no `Makefile` exists); ADRs remain marked `"Предложено"`; `docs/DATA_CONTRACT.md:80,97` contain obsolete pre-implementation notes (`DATA-14`).
- `cli/__main__.py:159` requires `--all` on the only subcommand (`verify`) with missing help text; `datasets.py:243-246` silently ignores `--seed` when checking; `reproducibility.py:39-40` accepts duplicate JSON keys / leaks raw `UnicodeDecodeError`; `records.py:130-133` raises `AttributeError` on non-mapping JSONL rows.
- 12 test modules import and couple tightly to `DataContractTests.setUpClass()`, and `test_reproducibility_and_permutation.py:46` asserts `wall_time_seconds > 0` (flaky on coarse timer ticks).  
**Problem:** Documentation drift and minor defensive/ergonomic rough edges.  
**Impact:** Developer/judge onboarding friction and occasional non-descriptive exceptions on malformed external files.  
**Realistic scenario:** A reviewer follows `.github/PULL_REQUEST_TEMPLATE.md:18` and runs `make verify`, hitting `make: *** No rule to make target 'verify'`.  
**Recommended fix:** Replace `make verify` references with `python -m genesis.cli verify --all`, update ADR statuses to `"Принято"`, extract shared schema test helpers out of `DataContractTests`, check `isinstance(row, dict)` in `validate_trajectory()`, and use `>= 0` for timing assertions.

---

### Informational Findings

#### F-31: Verified Non-Bottlenecks & Resource-Limit Scaling Horizon
**Title:** Core runtime, matcher, calculator, storage, and memory are well within all official hackathon limits at v1.0.0 scale; unfiltered pagination hits the 100-step ceiling around ~4,750 rows  
**Severity:** Informational  
**Confidence:** Confirmed (empirically measured by `performance`)  
**Affected files:** `src/genesis/evaluation/matcher.py:47-137`, `src/genesis/environment/calculator.py:7-108`, `src/genesis/environment/auditor_env.py:20,176`, `src/genesis/policies/reconcile.py:38-45`  
**Evidence:**
- Measured on Apple M1 / CPython (read-only): 80 local benchmark runs = **`78.3 ms`** (`311 KiB` traced heap, `65.9 MiB` RSS); 20 subprocess JSONL episodes = **`1.81 s`** (`78.7 ms/ep`); scaled 2,000L/200D subprocess episode = **`99.4 ms`** (`0.17%` of 60 s limit, `5.8 MiB` heap); repo on disk = **`2.42 MiB`**; CI-reported Docker image = **`50.7 MB`** (`5.1%` of 1 GB limit); dense 30×30 `matcher.match` = **`0.413 ms`**; worst-case 64-op `calculator` = **`0.065 ms`**.
- Scaling horizon: because `list_records` has no filter parameters and `limit <= 50`, full-table scans hit `MAX_STEPS = 100` around `~4,750` total records (measured: `5,000L/500D` needs 112 pages and truncates).  
**Problem:** None at current or 2,000-row spec scale; documented for future scaling roadmap.  
**Impact:** Confirms zero resource-limit blockers for Python 3.11 / 2 CPU / 4 GB RAM / No GPU / 1 GB storage / 60 s / 100 steps (outside `DockerPolicy`'s self-imposed `512m` cap in `F-10`).  
**Realistic scenario:** Scaling future datasets beyond ~4,500 rows without adding table filters (`vendor_id`, `date_range`, `account`) would make exhaustive table inspection impossible within 100 steps.  
**Recommended fix:** When scaling beyond 2,000 rows, add optional indexed filter arguments to `list_records` and gzip-compress archived training trajectories.

---

## 4. Official Hackathon Scorecard (100 Points)

| Criterion | Max | Estimated Score | Status |
| --- | ---: | ---: | --- |
| 4.1 Contract and isolation | 25 | **20** | Partial compliance |
| 4.2 Financial data and ground truth | 25 | **14** | Partial compliance |
| 4.3 Reward function and evaluation reliability | 25 | **18** | Partial compliance |
| 4.4 Reproducibility and generalization | 15 | **9** | Partial compliance |
| 4.5 Engineering and documentation | 10 | **6** | Partial compliance |
| **Total** | **100** | **67 / 100** | **Needs P0 fixes before submission** |

### 4.1 Contract and Isolation — **20 / 25**
- **Confirmed compliance:** `reset(seed, config) -> (obs, info)` & `step(action) -> (obs, reward, terminated, truncated, info)` JSON Gymnasium-style contract (`auditor_env.py`); deterministic transitions; `<=100` step cap; `<=50` page size; opaque scoped pagination cursors; restricted 5-tool set (`list_records`, `get_record`, `get_document`, `calculator`, `submit`); AST-bounded decimal-only calculator; controlled errors consuming a step on invalid actions/unknown IDs; zero unrestricted shell/SQL/filesystem/internet access; zero hidden ground-truth/provenance/seed leakage in observations/info.
- **Partial compliance:** Container isolation (`DockerPolicy` exists and passes `C18` in Linux CI, but is unused in `src/`, lacks a baseline worker Dockerfile, hardcodes `512m`/1CPU/32-PID caps, bills startup to 60 s clock, and `C18` is skipped by default in `verify --all` — `F-09`, `F-10`); transport/truncation handling (`F-02`, `F-03`, `F-15`, `F-16`); `AuditorEnv` validation bounds (`F-17`).
- **Non-compliance:** None of the core isolation blockers fail, though `F-03` (`\ud800` surrogate crash across the policy boundary) is a real robustness breach.
- **Critical gaps & fixes:** Reject lone UTF-16 surrogates (`F-03`), fix clean-episode crash/timeout scoring (`F-02`), wire `DockerPolicy` + worker Dockerfile + configurable caps (`F-10`), and run `verify --all --require-docker` in CI (`F-09`).

### 4.2 Financial Data and Ground Truth — **14 / 25**
- **Confirmed compliance:** All 4 configurable error types (`duplicate_invoice`, `payment_allocation`, `fx_conversion`, `period_cutoff`), 2 interacting scenarios (`partial_fx_payment_at_cutoff`, `duplicate_invoice_with_fx`), clean companies, legitimate distractors, balanced pre-injection histories, causal `provenance.json`, hidden `truth.json`/`provenance.json`/seed, integer minor units, explicit `currency_exponents`, decimal-string FX rates, ISO-8601 timestamps with offsets, no un-converted cross-currency comparison, explicit `null` in schemas.
- **Partial compliance:** Realistic scale & coverage (hardcoded 4-obligation skeleton, max 44 rows vs 200–2,000 target, 0 standalone partial allocations in `train`/`dev`, 0 FX rounding triggers, dead `period_end` branch — `F-04`); sufficient/fair evidence (`partial_fx_payment_at_cutoff` non-identifiable leg attribution & single over-constrained evidence set — `F-05`); `invoice_refs: null` schema support crashes `allocation.py:62` (`F-18`).
- **Non-compliance:** Consistent canonical `affected_ids` contract (`allocation.py` contradicts `interactions.py` and `DATA_CONTRACT.md:60` — **`F-01`, Critical**); non-negative cash realism (FX injection creates unmodeled negative bank overdrafts in 3/10 `train`, 1/10 `dev`, 7/20 local — **`F-06`**).
- **Critical gaps & fixes:** Unify `affected_ids` in `allocation.py` (`F-01`), add alternative minimal `acceptable_evidence_sets` in `interactions.py` (`F-05`), add cash buffer (`F-06`), and parameterize generator scale (`F-04`).

### 4.3 Reward Function and Evaluation Reliability — **18 / 25**
- **Confirmed compliance:** Deterministic 1-to-1 maximum bipartite matching (`matcher.py`), exact baseline reward formula `clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)` (`reward.py`), issue type + canonical `affected_ids` + currency + tolerance + evidence set checks, FP/FN accounting, duplicate resistance, `<=30` findings, `<=10` evidence IDs, `<=1,000` char explanation, invalid structure -> terminated with `-1.0`, zero closed-mode hints, text-independent scoring, all 5 required reward-property tests (`E01–E20`) passing.
- **Partial compliance:** Clean-episode handling & truncation (`F-02`: transport faults/crashes/missing images score `+1.0` on clean episodes; `invalid_submission=True` still records `F1 = 1.0`); evidence/ID bounds (`F-13`: unbounded `affected_ids`, unpenalized evidence stuffing).
- **Non-compliance:** Fair evaluation of contract-compliant auditors is broken by `F-01` + `F-05` (`matcher` assigns `-1.0` to honest auditors due to generator truth contradictions).
- **Critical gaps & fixes:** Fix `F-01` + `F-05` so truth matches public contract; penalize transport faults/truncation on clean episodes (`F-02`); bound `affected_ids` (`F-13`).

### 4.4 Reproducibility and Generalization — **9 / 15**
- **Confirmed compliance:** Bitwise-deterministic episode/document/error generation (`checksums.json`), reproducible transitions & rewards (`PYTHONHASHSEED`-safe), public `train`/`dev` vs local profile separation, `verify_public_proofs` verified on permuted datasets, version/seed/config recording.
- **Partial compliance:** Resistance to overfitting & shortcuts (`F-04` step-0 `manifest.record_counts` bijection + `F-06` negative-cash leak + 5-step rule shortcut); cross-host timezone reproducibility (`F-12` `+05:00` vs OS `ZoneInfo` on pre-2024a `tzdb`); empty `interaction_scenarios: []` normalization (`F-21`).
- **Non-compliance:** Substantive permutation & ID-renaming invariance testing (`F-07`: the 20 paired checks execute **only** zero-step `OraclePolicy` with zero environment/tool calls, and replay runs only empty-submitting `RandomPolicy`).
- **Critical gaps & fixes:** Add `simple_reconciliation` + tool-using replay to the 20 paired permutation checks (`F-07`), eliminate step-0 count/overdraft leaks (`F-04`, `F-06`), and fix timezone offset consistency (`F-12`).

### 4.5 Engineering and Documentation — **6 / 10**
- **Single-command execution (1.5 / 2 — Partial):** `python -m genesis.cli verify --all` runs cleanly offline and in ~50.7 MB Docker, but has 0% test coverage, masks `ModuleNotFoundError`, skips `C18` with exit 0, and lacks `pyproject.toml` (`F-09`, `F-19`).
- **Configuration & version tracking (1.5 / 2 — Partial):** Pinned Docker digest, `requirements-schema.lock`, `checksums.json`, generator/schema versions; lacks `--require-hashes`, misclassifies `tzdata`, and has stale commit SHAs across docs (`F-08`, `F-12`).
- **Trajectory logging (1.0 / 2 — Partial):** Schema-valid, leakage-free `trajectories.jsonl`, but generated from a 4-row handcrafted test fixture, omits `reset` ($s_0$), wipes `explanation` in `train` mode, and erases valid failed actions (`F-14`).
- **Measured failure analysis (1.0 / 2 — Partial):** `docs/DESIGN_NOTE.md` reports real measured numbers for all 4 baselines, but `simple_reconciliation` only handles 1/4 error types (`return = -0.0315`) and misses the `affected_ids` / clean-crash defects (`F-01`, `F-02`, `F-23`).
- **Honest documentation of limitations (1.0 / 2 — Partial):** `docs/LIMITATIONS.md` and `materials/submission/README.md` are genuinely candid about small scale and missing contest artifacts, but `make verify` / ADR / SHA drift (`F-30`) and incomplete submission deliverables (`F-08`: demo video, presentation, Google Drive link) remain open.
- **Unverified items:** Local execution of `C18` Docker smoke test (`docker` daemon unavailable on review host; verified via Linux CI workflow definition and code audit).

---

## 5. Individual Agent Reports

| # | Agent | Main Conclusion | Strongest Finding | Unique Perspective | Important Disagreements |
| --- | --- | --- | --- | --- | --- |
| 1 | **Senior Software Architect** (`architect`) | Clean stdlib core and Gymnasium protocol, undermined by contradictory truth contracts, a 4-obligation ceiling, and unwired `DockerPolicy`. | `ARCH-01` → `F-01` (contradictory `affected_ids` contract) & `ARCH-03` → `F-10` (unused `DockerPolicy`, fragmented CLI). | Mapped the `environment <-> evaluation` package cycle, missing `evaluation/__init__.py`, and clock-injection split (`F-19`, `F-24`). | Initially rated `affected_ids` High (`ARCH-01`); upgraded to **Critical** in Phase 2. Challenged `DA-02` (Critical → High) and `DA-04` (High → Medium on rubric). |
| 2 | **Security Engineer** (`security`) | Strong security/isolation posture with zero Critical/High leakage or blocker breaches; key hardening gaps in clean-episode fault scoring, permissions, and redaction smuggling. | `SEC-01` → `F-02` (transport faults = `+1.0` on clean episodes) & `SEC-03` → `F-13` (redaction smuggling via unbounded `affected_ids`). | Independently reproduced `\ud800` surrogate crash (`F-03`) over real `JsonlProcess`, proved `10**5000` is blocked on JSONL but crashes in-process, and caught `export_episode` `0o644` permissions (`F-25`) & shared MT19937 PRNG (`F-27`). | Initially rated `SEC-01` Medium; upgraded to **High** in Phase 2. Accepted `SEC-02` nuance (Low under Docker, Medium on shared host). |
| 3 | **Backend & Financial Logic Engineer** (`backend`) | Decimal math, bipartite matcher, and reward formula are solid, but truth construction breaks honest auditors (`F-01`, `F-05`) and violates cash invariants (`F-06`). | `FIN-01` → `F-01` (`affected_ids` contradiction), `FIN-02` → `F-05` (non-identifiable 3-leg FX), `FIN-03` → `F-06` (negative bank overdrafts). | Proved `bank` overdrafts in 3/10 `train`, 1/10 `dev`, 7/20 local profiles; caught `interaction_scenarios: []` normalization gap (`F-21`) and missing accrual FX date rule (`F-29`). | Downgraded `QA-04` (`invoice_refs: null`) High → Medium; argued **High** for `ADV-05` (`Asia/Almaty` `+05:00` vs pre-2024a `tzdb`) against `devops-infra`'s Medium. |
| 4 | **Frontend & UX Engineer** (`frontend-ux`) | Confirmed zero frontend/web UI on `main` (appropriate for headless RL env, not penalized); audited CLI, agent protocol/schemas, and docs/demo UX. | `UX-01` → `F-01` (undocumented conflicting `affected_ids`), `UX-02` → `F-09` (opaque `ValueError` in `verify --all`), `UX-05` → `F-17` (silent malformed `submit` in `train` mode). | Identified 0 `"description"` fields across all 14 schemas, missing `fx:...`/`"policy"` guidance in `reset()`, and zero diagnostic reason on malformed `submit` in `train` mode. | Downgraded standalone `UX-02` High → Medium while supporting **High** when merged with `QA-01`; split `UX-03` into High (missing demo video) + Medium (no `demo` CLI). |
| 5 | **Performance Engineer** (`performance`) | Ultra-fast and memory-lean at v1.0.0 (`78.3 ms` / `311 KiB` / `2.4 MiB`), with Docker cap/cold-start risk for ML policies and $O(L \cdot (I+P))$ hotspots at 2,000 rows. | `PERF-01` → `F-10` (`512m`/1CPU/32-PID caps + cold start in 60 s clock) & `PERF-03` → `F-22` (62×–104× quadratic `validate_links` slowdown). | Empirical microbenchmarks across 40–5,000 rows, proving `matcher` (`0.41 ms`) & `calculator` (`0.065 ms`) are non-bottlenecks and measuring the ~4,750-row 100-step ceiling (`F-31`). | Self-corrected `PERF-01` / `DA-06` after checking `ISOLATION_AND_LEAKAGE.md:17` (intentional tight caps: Medium for heuristics, High for ML/RL containers); calibrated `PERF-02`/`PERF-03` to Medium. |
| 6 | **QA & Test Engineer** (`qa-test`) | Fast suite (`145/147` passed in `2.30 s` with lockfile), but `verify --all` has 0% coverage, permutation checks are Oracle-only, and `C17` misses 6 `jsonl.py` guards. | `QA-01` → `F-09` (0% CLI coverage + silent `C18` skip), `QA-02` → `F-07` (Oracle-only pairs), `QA-03` → `F-11` (6 untested `jsonl.py` guards), `QA-04` → `F-18` (`invoice_refs: null` `TypeError`). | Ran live `unittest` + `stdlib trace` statement coverage across the repo, discovering `allocation.py:62` `TypeError` and exact uncovered `jsonl.py` guards. | Accepted `backend`'s downgrade of `QA-04` High → Medium; independently confirmed `FIN-01`, `FIN-02`, `FIN-03`, and `DA-05`/`QA-02` (High). |
| 7 | **DevOps & Infrastructure Engineer** (`devops-infra`) | Strong offline stdlib core and ~50.7 MB digest-pinned container, but contest deliverables are missing and packaging/CI/lockfile hardening gaps remain. | `OPS-01` → `F-08` (missing video, slides, Drive package, stale SHAs), `OPS-03` → `F-12` (`tzdata` misclassified, unhashed lockfile), `OPS-05` → `F-20` (no `push` CI). | Caught `DockerPolicy._host_environment()` stripping `DOCKER_HOST`/`HOME`/`XDG_RUNTIME_DIR` (`F-26`) and missing `LICENSE`/`NOTICE` in Docker image (`F-28`). | Argued **Medium** (not High) for `ADV-05` (`tzdb < 2024a`) because 2026 hosts and pinned Dockerfile ship `tzdb >= 2024a`. |
| 8 | **Skeptical Maintainer** (`maintainability`) | Compact ~4,069 LOC codebase hiding major maintenance traps: conflicting injector contracts, unwired production modules, and brittle CLI slicing. | `MAINT-01` → `F-01` (contradictory `affected_ids`), `MAINT-03` → `F-10` (orphaned `DockerPolicy`/`export_episode`), `MAINT-04` → `F-09` (`test_C`/`test_E` slicing + double execution). | Traced `trajectories/trajectories.jsonl` directly back to `tests/test_trajectories.py --export` (`F-14`) and spotted `permutation.py`'s fragile `endswith("_id")` heuristic (`F-24`). | Upgraded `MAINT-04` Medium → High when merged with `QA-01`/`UX-02` and `MAINT-10` Low → Medium when merged with `UX-06`. |
| 9 | **Adversarial Failure Analyst** (`failure-analyst`) | Calculator sandbox, cursors, and `PYTHONHASHSEED` hold up well, but adversarial probes broke scoring integrity, trajectory serialization, and benchmark anti-shortcutting. | `ADV-01` → `F-02` (clean crash = `+1.0`), `ADV-03` → `F-03` (`\ud800` UTF-16 surrogate crash), `ADV-04` → `F-04` (step-0 `manifest` bijection + 4-step shortcut), `ADV-05` → `F-12` (`Asia/Almaty` `+05:00` split-brain). | Unique discovery of `\ud800` surrogate crash (`F-03`), post-`submit` ack EOF/timeout race (`F-15`), `64 KiB` `image inspect` cap + cleanup deadline reuse (`F-16`), and `timeout_seconds=True` / stale `last_action_failed` (`F-17`). | Downgraded `ADV-01` Critical → High (converging with `security` & `devils-advocate`); challenged `SEC-02` Medium → Low under Docker deployment. |
| 10 | **Independent Devil's Advocate** (`devils-advocate`) | Polished harness around a broken, toy-scale 4-obligation puzzle with checkbox-compliant deliverables; honest domain auditors score `F1 = 0.625` while a 90-line quirk script scores `1.000`. | `DA-01` → `F-01`/`F-05` (empirical `F1=0.625` honest vs `1.000` quirk script), `DA-02` → `F-04` (dead tools/pagination, 5-step rule solve), `DA-05` → `F-07` (Oracle-only permutation tautology). | Built both an honest domain auditor (`F1=0.625`) and a quirk-aware 90-line rule script (`F1=1.000, Return=0.992`) to prove the contract/shortcut gap empirically; proved 0/40 FX rounding & 0/100 `period_end` triggers. | Conceded `DA-02` Critical → High and `DA-04` High → Medium under strict hackathon rubric while preserving High warning for real offline RL training. |

---

## 6. Cross-Agent Disagreements & Resolution

1. **Clean-Episode Crash / Timeout / Truncation Earning `+1.0000` (`SEC-01` Medium vs `DA-03` High vs `ADV-01` Critical) — Resolved: `High` (`F-02`)**
   - *Dispute:* `security` viewed it as Medium (intentional ADR 0003 truncation-as-empty design); `failure-analyst` rated it Critical (reward hacking); `devils-advocate` rated it High.
   - *Evidence & Resolution:* Direct verification of `judge.py:66-70`, `auditor_env.py:408-414`, and `test_evaluator_isolation.py:149-168,205-215` confirmed that even `PROCESS_ERROR` (missing Docker image) and `BAD_FRAME` earn `+1.0` on clean episodes (25% of benchmark), strictly beating honest inspecting policies (`+0.992`). However, crashing still earns `-1.0` on the 75% anomalous episodes (overall return `-0.500`), so it cannot win the benchmark. All three agents converged on **High**.

2. **4-Obligation Micro-Template Generator, Dead Pagination, and 5-Step Shortcut (`DA-02` Critical vs `ARCH-02`/`ADV-04` High vs `FIN-04` Medium) — Resolved: `High` (`F-04`)**
   - *Dispute:* `devils-advocate` rated the toy scale Critical; `backend` scoped distribution gaps as Medium; `architect` and `failure-analyst` rated it High.
   - *Evidence & Resolution:* Empirical verification confirmed `max_ledger = 41 < 50`, identical dicts in `list_records` vs `get_document`, 40/40 step-0 `manifest.record_counts` bijection, 0/40 FX rounding, and a 5-step `F1 = 1.0` rule script. Because `docs/LIMITATIONS.md:6` explicitly discloses the small v1.0.0 scale and pagination works on >50-row fixtures (`C05–C08`), `devils-advocate` conceded Critical → High and `backend`'s scope was upgraded to **High**.

3. **`affected_ids` Contradiction (`ARCH-01`/`UX-01`/`MAINT-01` High vs `FIN-01`/`DA-01`/`ADV-02` Critical) — Resolved: `Critical` (`F-01`)**
   - *Dispute:* Architectural/UX/maintainability agents initially rated it High; backend/adversarial/devil's advocate agents rated it Critical.
   - *Evidence & Resolution:* Live matcher runs proved a spec-compliant auditor following `docs/DATA_CONTRACT.md:60` scores `TP=0, FP=1, FN=1, reward=-1.0` on every standalone `duplicate_invoice` and `payment_allocation` episode — directly corrupting 50/100 rubric points. `architect` upgraded to **Critical**; unanimous consensus.

4. **Trajectory RL Usability (`DA-04` High vs `ARCH-06`/`UX-06` Medium vs `MAINT-10` Low) — Resolved: `Medium` (High for Offline RL) (`F-14`)**
   - *Dispute:* Whether missing `reset` ($s_0$), 4-row test fixture origin, unconditional `explanation` wiping, and valid-action erasure warrant High or Medium.
   - *Evidence & Resolution:* Under Section 4.5 (10 pts), `trajectories.jsonl` mechanically passes schema, replay, and zero-leakage checks (`test_trajectories.py` 7/7). Reconciled to **Medium** for hackathon compliance, with explicit caveat that offline RL/BC usability is **High**-impact blocked.

5. **Hardcoded `+05:00` vs OS `ZoneInfo("Asia/Almaty")` on Pre-2024a `tzdb` (`ADV-05` High vs `devops-infra` Medium) — Principled Nuance Preserved (`F-12`, Medium)**
   - *Dispute:* `backend` and `failure-analyst` proved pre-2024a `tzdb` (`+06:00`) corrupts `partial_fx_payment_at_cutoff`, breaks `uniqueItems`, and invalidates `checksums.json`, arguing **High**. `devops-infra` argued **Medium** because the pinned `Dockerfile` and 2026 hosts ship `tzdb >= 2024a`, restricting exposure to unpatched hosts or older external policy base images.
   - *Resolution:* Recorded as **Medium** with both exposure profiles transparently documented.

6. **Remaining Unverified Item (Environment Limitation):** Live Linux Docker execution of `C18` (`DockerPolicyTests`) could not be run locally because no Docker daemon is installed on the review host; all Docker findings (`F-10`, `F-16`, `F-26`, `F-28`) are verified from source code, `Dockerfile`, and `.github/workflows/policy-isolation.yml`.

---

## 7. Missing Tests

| # | Target Module | Input Conditions | Expected Behavior | Failure Scenario Prevented |
| --- | --- | --- | --- | --- |
| 1 | `allocation.py` + `interactions.py` + `matcher.py` | Run a single spec-compliant auditor (following `docs/DATA_CONTRACT.md:60`) across all 20 `train`+`dev` and 20 local profiles (both standalone and interacting). | `TP == len(truth)`, `FP == 0`, `FN == 0`, `F1 == 1.0` for all 4 `FindingType`s uniformly. | Prevents `F-01` / `F-05` contract drift between standalone and interacting injectors. |
| 2 | `judge.py` + `auditor_env.py` + `reward.py` | Trigger `EOF`, `BAD_FRAME`, `FRAME_LIMIT`, `STDERR_LIMIT`, `TIMEOUT`, and `PROCESS_ERROR` on both `clean=True` and `clean=False` books, at step `0` and after `k=3` valid steps. | Transport/protocol faults score `reward == -1.0, f1 == 0.0` (never `+1.0` on clean books); `PROCESS_ERROR` surfaces as harness error; `invalid_submission` sets `f1 == 0.0`. | Prevents `F-02` clean-episode crash reward hacking and `F-11` coverage gap. |
| 3 | `jsonl.py` + `auditor_env.py` + `records.py` | Send valid-JSON `submit` containing `"\ud800"` in `affected_ids`, `evidence_ids`, and `explanation` over real `JsonlProcess`, followed by `TrajectoryRecorder.to_jsonl()` and `_canonical()`. | Rejected as `TransportCode.BAD_FRAME` or `invalid_submission`; zero `UnicodeEncodeError`. | Prevents `F-03` remote DoS crash in trusted serialization. |
| 4 | `jsonl.py` | Send duplicate JSON keys (`{"type":"submit","type":"calculator"}`), `NaN`/`Infinity`, `LimitOverrunError` (line `> FRAME_BYTES` without `\n`), cumulative `65 KiB` stderr across 1 KiB lines, and invalid UTF-8 `b"\xff\xfe\n"`. | Each rejected cleanly with the exact documented `TransportCode`. | Closes all 6 untested `jsonl.py` safety guards (`F-11`). |
| 5 | `reproducibility.py` + `permutation.py` | Run `simple_reconciliation` and a deterministic multi-tool (`list_records` + `get_document` + `calculator` + `submit`) replay across all 20 original-vs-permuted/ID-renamed pairs. | Identical `(tp, fp, fn, reward, total_steps)` and inverse-mapped findings; baseline ordering `oracle > simple_reconciliation > random > mark_all`. | Eliminates `F-07` Oracle-only permutation tautology. |
| 6 | `cli/__main__.py` | Unit-test `verify()` with missing `jsonschema` (`_FailedTest`), `require_docker=True` vs `False`, broken profile JSON, and a test named `test_CSV_export`. | Surfaces `ModuleNotFoundError` (not `ValueError`), fails when `require_docker=True` & Docker absent, prints safe public diff path, and ignores non-matrix `test_C*` names. | Closes `F-09` (0% CLI test coverage). |
| 7 | `company.py` + `fx_cutoff.py` + `interactions.py` | Assert `running_cash >= 0` across **all** mutated episodes (not only clean books), test `allocation.py` with `payment["invoice_refs"] = None`, and test `ZoneInfo("Asia/Almaty")` offset verification. | `min_cash >= 0` on all 40 episodes; `apply_allocation_errors` ignores `invoice_refs: None` without `TypeError`; offset mismatch caught immediately. | Prevents `F-06`, `F-18`, and `F-12`. |
| 8 | `judge.py` + `jsonl.py` | Policy writes valid `submit`, closes stdin immediately, and exits with code 0 before harness sends final `step` ack. | `JudgeResult(terminated=True, reason="submitted", transport_error=None)`. | Prevents `F-15` post-`submit` ack EOF race. |
| 9 | `docker_policy.py` | Unit-test `_local_image_id` on `>64 KiB` inspect output, verify `--no-healthcheck`/`--ulimit`/`--cgroupns` flags, and test `created=False` + slow `rm` cleanup path. | Parses `--format '{{.Id}}'` cleanly, enforces hardening flags, and does not escalate failed `create` to `PolicyCleanupError`. | Prevents `F-16`. |
| 10 | `auditor_env.py` + `records.py` + `datasets.py` | Pass `timeout_seconds=True`, negative manifest counts, bogus policy enums, `claimed_amount_minor = 10**5000`, `affected_ids` with 500 items, non-dict JSONL rows in `validate_trajectory()`, and verify `verify_public_proofs` on all 20/20 `train`+`dev` episodes. | Strict rejections with `ValueError`/`TypeError` (never `StopIteration`/`AttributeError`); all 20/20 public episodes pass provability & `wire_validator`. | Prevents `F-13`, `F-17`, `F-23`, and `F-30`. |

---

## 8. Improvement Roadmap

### P0 — Immediate Blockers (Fix Before Hackathon Submission)
| Action | Findings | Impact | Difficulty | Dependencies |
| --- | --- | --- | --- | --- |
| Unify `affected_ids` in `allocation.py` with `interactions.py` & `DATA_CONTRACT.md:60`; add minimal alternative `acceptable_evidence_sets` in `interactions.py`; regenerate `data/` + `checksums.json` | `F-01`, `F-05` | **Critical** (unblocks fair scoring across 50/100 rubric pts) | Low (1–2 hrs) | Bump `generator_version`, update `test_public_datasets.py` |
| Penalize transport/protocol faults (`EOF`, `BAD_FRAME`, `FRAME_LIMIT`, `STDERR_LIMIT`, `TIMEOUT`) with `-1.0` on clean episodes, set `f1=0.0` on `invalid_submission`, and fail fast on `PROCESS_ERROR` | `F-02` | **High** (closes `+1.0` crash reward hack) | Low (2–3 hrs) | Update `test_evaluator_isolation.py` & `ADR 0003` |
| Reject lone UTF-16 surrogates (`\ud800`–`\udfff`) in `jsonl.py` and `auditor_env.py` | `F-03` | **High** (closes untrusted → trusted DoS crash) | Low (30 min) | None |
| Complete contest deliverables: `<= 3:00` demo video, presentation, Google Drive master doc, and refresh commit SHAs in `materials/submission/README.md` / `README.md` / `DESIGN_NOTE.md` | `F-08` | **High** (administrative submission blocker) | Medium (half day) | Final `main` commit after P0 code fixes |

### P1 — High Priority (Strongly Recommended Before Official Evaluation)
| Action | Findings | Impact | Difficulty | Dependencies |
| --- | --- | --- | --- | --- |
| Add opening cash buffer in `company.py:228-232` and enforce `cash >= 0` on mutated episodes | `F-06` | **High** (removes overdraft invariant breach & FX leak) | Low (1 hr) | Regenerate `data/` + `checksums.json` alongside `F-01` |
| Run `simple_reconciliation` + tool-using replay on all 20 permutation/ID-renaming pairs and assert baseline ordering | `F-07` | **High** (makes Section 4.4 20-pair check non-tautological) | Low–Medium (2–3 hrs) | None |
| Add `tests/test_cli_verify.py`, surface `ModuleNotFoundError`, eliminate double benchmark execution, replace `test_C`/`test_E` slicing, and run `verify --all --require-docker` + `push` triggers in CI | `F-09`, `F-20` | **High** (hardens primary judge verification gate) | Medium (3–4 hrs) | None |
| Make `DockerPolicy` limits configurable (`2 CPU / 4 GB` option + `--tmpfs`), exclude startup from 60 s clock, fix `64 KiB` inspect/`--no-healthcheck`/cleanup bug, add worker `Dockerfile` & CLI `run-policy`/`demo` | `F-10`, `F-16` | **High** (enables evaluating real containerized ML/RL policies) | Medium (0.5–1 day) | None |
| Close `C17`/`C19` test gaps (6 `jsonl.py` guards, anomalous/multi-step faults, injected clock) & fix post-`submit` ack EOF race | `F-11`, `F-15` | **High** (verifies transport isolation without flaky sleeps) | Medium (3–4 hrs) | Clock injection in `judge.py`/`jsonl.py` |

### P2 — Medium Priority (RL Trainability, Reliability & Technical Debt)
| Action | Findings | Impact | Difficulty | Dependencies |
| --- | --- | --- | --- | --- |
| Parameterize generator scale (`200–2,000` rows, `20–200` docs), add currency/policy diversity, give `get_document` unique detail, remove step-0 `manifest` bijection, enable `period_end` cutoffs | `F-04` | **High for RL training** (turns toy puzzle into genuine sequential RL env) | High (2–4 days) | Schema & dataset v2 bump |
| Add `reset` ($s_0$) to trajectories, preserve valid failed actions & `train`-mode explanations, generate `trajectories.jsonl` from real `data/train` episodes | `F-14` | **Medium–High** (enables offline RL / BC) | Medium (3–4 hrs) | `schemas/trajectories.schema.json` update |
| Fix `AuditorEnv` validation drift (`bool` timeout, negative counts, `StopIteration`, `train` submit diagnostics), bound `affected_ids`/`claimed_amount_minor`, fix `allocation.py:62` `None` crash, normalize `interaction_scenarios: []`, and fix `Asia/Almaty` timezone consistency + lockfile hashes | `F-12`, `F-13`, `F-17`, `F-18`, `F-21` | **Medium** (hardens boundary & cross-host reproducibility) | Medium (1 day) | None |
| Add `pyproject.toml`, `evaluation/__init__.py`, break `environment <-> evaluation` cycle, hoist `validate_links` sets (62× speedup), deduplicate helpers, and replace `endswith("_id")` in `permutation.py` | `F-19`, `F-22`, `F-24` | **Medium** (packaging, maintainability & RL step throughput) | Medium (1 day) | None |
| Strengthen `simple_reconciliation` baseline & close remaining dataset/policy/trajectory test gaps | `F-23` | **Medium** (meaningful diagnostic floor) | Medium (0.5–1 day) | `F-01` |

### P3 — Optional (Polish & Hardening)
| Action | Findings | Impact | Difficulty | Dependencies |
| --- | --- | --- | --- | --- |
| Tighten `export_episode` permissions (`0o700`/`0o600`), preserve `DOCKER_HOST` on host CLI, separate private PRNG stream, copy `LICENSE`/`NOTICE` into Docker image, validate `tolerances_minor`, convert `matcher` DFS to iterative, clean up stale docs/ADRs | `F-25`–`F-30` | **Low** (defense-in-depth & onboarding polish) | Low (3–4 hrs) | None |

---

## 9. Final Verdict

- **Does the repository satisfy the actual hackathon objective?** **Partially.** It implements a real, offline, deterministic financial-audit environment with genuine isolation engineering — not a dashboard or chatbot — and passes all zero-leakage/zero-network/zero-GPU blockers. However, `F-01` (contradictory `affected_ids` scoring honest auditors at `-1.0`), `F-02` (clean-episode crash = `+1.0`), `F-04` (toy 4-obligation scale), and `F-08` (missing contest deliverables) prevent full substantive satisfaction today.
- **Is it a genuine RL environment?** **Mechanically yes, substantively not yet.** It faithfully implements a Gymnasium-style `reset`/`step` JSON MDP with a deterministic reward function, but at max 44 rows (`< 50` page size) with identical `list_records`/`get_document` payloads and 1-to-1 step-0 count leaks, every episode collapses to a 1-shot 4-table dump solved in 5 steps by a 90-line rule script (`F1 = 1.000`).
- **Would you approve it for the official evaluation?** **Not as-is; conditional approval after P0 fixes (`F-01`, `F-02`, `F-03`, `F-08`).** Fixing `F-01`–`F-03` takes less than a day and immediately turns a ~67/100 submission into an ~80–85/100 contender (and ~90+ with P1).
- **Would you approve it for production (real RL model training)?** **No.** Production RL training requires parameterized 200–2,000-row scaling (`F-04`), `s_0` reset states in trajectories (`F-14`), configurable 4 GB Docker policy execution (`F-10`), and proper Python packaging (`F-19`).
- **What is the biggest technical risk?** **`F-01` (`allocation.py` vs `interactions.py` / `DATA_CONTRACT.md:60` `affected_ids` contradiction)** — it silently invalidates fair scoring by giving `F1 = 0, reward = -1.0` to contract-compliant external agents.
- **What is the most underestimated problem?** **The combination of `F-04` (toy 4-obligation skeleton + zero-information `get_document` + step-0 `manifest` bijection) and `F-07` (Oracle-only permutation checks).** Because all 145 tests pass and `verify --all` prints green `20/20`, the team can easily mistake green harness tests for a hard, generalizable RL benchmark.
- **What should be fixed first?** **Fix `F-01` (`allocation.py:143-181`), `F-02` (clean-episode crash scoring), and `F-03` (`\ud800` surrogate crash), then complete `F-08` (demo video, slides, Drive link).**
- **What could prevent the project from passing?** Administratively: submitting without the missing Task #26 deliverables (`F-08`: demo video, presentation, Drive link). Technically: judges running an independent spec-compliant auditor or inspecting truth fairness (`F-01`, `F-05`), testing a crashing container on clean books (`F-02`), or testing containerized ML policies against hardcoded `512m` caps (`F-10`).
- **What would you change as its long-term maintainer?** Add `pyproject.toml`, split `evaluation/` into pure `scoring/` vs `harness/` to eliminate the `environment <-> evaluation` cycle, replace the 4-obligation micro-template with a parameterized multi-currency/multi-policy company simulator (200–2,000 rows) where `get_document` reveals genuine line-item evidence, unify canonical `affected_ids` + multi-set `acceptable_evidence_sets` behind a single truth builder, wire `DockerPolicy` into a first-class CLI with configurable 2 CPU / 4 GB limits, and make `trajectories.jsonl` record complete $(s_0, a_t, s_t, r_t)$ episodes from real generated data.
