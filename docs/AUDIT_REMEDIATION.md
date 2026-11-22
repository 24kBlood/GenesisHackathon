# Genesis FinTech — validated review, hallucination audit, compliance matrix, and remediation

Reviewed revision: `main@e5c3e4430270b538d2293f6e38113d5a68f20444` (current `origin/main`).
Remediation branch: `fix/v2-audit-compliance-and-hardening` (base `e5c3e44`).

This document audits the two pre-existing static review reports (`repository-review.md` and
`repository-review (1).md`, produced outside the repository on 2026-11-22) against the actual
source, records the resulting repository-wide action checklist, and describes the verified fixes
shipped in this pull request. The earlier 10-agent swarm report (PR #52) reviewed the older
`main@72a9473`; its findings were already implemented in PR #55 and are **not** re-audited here
except where they overlap with the current findings.

## 1. Executive summary

The pre-existing review is substantially accurate: 8 of its 9 concrete findings were reproduced
against `e5c3e44`, and its informational timeout observation is a real, unresolved rubric question.
One finding (M5) is a conditional trust-boundary concern, not a demonstrated vulnerability; it is
retained as a documented coordinator responsibility rather than a code defect.

The most serious defects were in the trusted coordinator boundary: an inadmissible closed case could
be scored, and a pre-start infrastructure timeout could be graded as a policy loss. Both are fixed,
together with a reproducible FX generator failure, inconsistent Docker resource ceilings, an
unpinned verifier timezone database, redundant corpus regeneration, and non-atomic trace export.

All 201 tests pass (2 Docker smoke tests skipped locally) and `verify --all` passes end-to-end.
Live Linux Docker execution remains covered by the existing `Policy isolation` CI job, not by this
host. External contest deliverables (video, slides, Drive document, portal entry) are outside git
and are not claimed as complete.

## 2. Hallucination audit of the existing review document

Verification statuses are limited to `Confirmed`, `Partially Confirmed`, `Unverified`,
`False Positive`, and `Already Fixed`.

| Existing claim | Repository evidence | Verification status | Decision | Reason |
|---|---|---|---|---|
| H1: closed judge admits episodes no policy can fully solve | `auditor_env.py:27` caps submissions at 30; `judge.py:35-48` fed generated truth directly into `build_episode`; reproduced `closed_environment().reset()` accepting 31 duplicate-invoice issues with `obligation_count=100` | Confirmed | Retained and fixed | Reproduced on `e5c3e44`; admission now rejects the case before a worker starts (`admission.py:17`, `judge.py:41`) |
| H2: Docker setup deadline recorded as policy loss | `docker_policy.py:141-167` shares `TIMEOUT` between image inspection and interaction; `judge.py:90-94` scored every non-`PROCESS_ERROR` transport fault | Partially Confirmed | Retained and fixed | Code path confirmed; real-daemon incidence unmeasured (as the review itself states). Setup faults now carry `phase="setup"` and are rethrown unscored (`jsonl.py:33-38`, `judge.py:94-99`) |
| M1: schema-valid FX case can have no candidate after rounding | Reproduced `ValueError: not enough independently reconcilable fx_conversion candidates` for seed 50, `base_currency=JPY`, `foreign_currency=KZT`, one FX mutation; `fx_cutoff.py:37-48` rejects non-positive debit legs | Confirmed | Retained and fixed | The exact rate/amount arithmetic in the report was not recomputed, but the failure and its control flow are real. Foreign postings now guarantee a positive base amount (`company.py:277`, `company.py:344`, `company.py:395`) |
| M2: documented Docker ceilings not enforced consistently | `docker_policy.py:36-47` validated only positivity of CPU; `memory` string bypassed the 4096 MiB check applied to `memory_mib` | Confirmed | Retained and fixed | Both ceilings are now normalized and enforced for every input form (`docker_policy.py:39-61`) |
| M3: replay depends on a timezone database that is not consistently pinned | `Dockerfile.policy:23` set `PYTHONTZPATH=""`; `Dockerfile` did not; `pyproject.toml:13` declared `tzdata>=2025.2` while the lock pins `2025.2` | Confirmed | Retained and fixed | Verifier image now sets the same variable and the runtime dependency is pinned; canonical database documented |
| M4: public-run evidence not checked at closed-case admission | `judge.py:35-48` had no `validate_links`/`verify_public_proofs` call, while `reproducibility.py:215-220` does for open profiles | Confirmed | Retained and fixed | `require_admissible_episode` runs link checks, arithmetic proofs, wire-budget validation, and one-to-one Oracle matching before any worker starts |
| M5: closed-case replay API checks shape, not trusted provenance | `company.py:425-432` validates seed range and ID shape only; no coordinator registry exists in the repository | Partially Confirmed | Retained as documentation only | No disclosure or misuse demonstrated; the trust boundary is now explicit in `docs/ISOLATION_AND_LEAKAGE.md:7-8`. Implementing a registry would require an external coordinator outside this repository |
| L1: resolving an episode ID regenerates unrelated episodes | `run_policy.py:18-24` generated every preceding entry; `data/configs/*.json` entries include explicit `episode_id` | Confirmed | Retained and fixed | Explicit IDs are now indexed first; generation is only a legacy fallback (`run_policy.py:21-31`) |
| L2: failed trace export can leave a partial file and obscure the run result | `run_policy.py:45-48` wrote directly to the destination with `xb` before printing the result | Confirmed | Retained and fixed | Trace is staged, flushed, and published no-clobber; status is reported separately from the score (`run_policy.py:40-58`, `run_policy.py:72-80`) |
| Informational: worker timeout on a clean book scores F1=1/reward=1 under the published formula | `terminal.py`/`reward.py` produce the clean-empty result; `auditor_env.py:393-394` deliberately overrides transport/timeout to F1=0, reward=-1 | Partially Confirmed | Deferred with reason | Code behavior is confirmed; the official ruling on forced timeouts is explicitly unverified. Changing it without a jury decision could break the existing scoring contract |

No claim in the existing review was classified as a false positive or as already fixed. The review's
own limitations statement (single reviewer, no tests executed, no Docker daemon) is accurate and was
preserved rather than silently upgraded.

## 3. Official hackathon compliance matrix

| Official requirement | Current implementation evidence | Status | Gap or defect | Proposed PR change | Validation |
|---|---|---|---|---|---|
| Synthetic company data, coherent history, 4 configurable errors, 2 interactions, clean distractors | `src/genesis/generation/{company,allocation,interactions,fx_cutoff,distractors}.py`, `benchmarks/local_eval_profiles.json` | Implemented and verified | None material | M1 positive-amount guarantee prevents a supported config from failing | 201 tests; replay 20/20; pairs 20/20 |
| Hidden truth/provenance isolation | `company.py:614-624`, `docs/ADR/0002-evaluator-isolation.md` | Implemented and verified | Closed replay provenance is trusted-caller only | M5 trust boundary documented | `test_closed_generation.py`; `verify --all` |
| `reset`/`step` JSON contract, 100 steps, 50-row pages, 5 tools | `auditor_env.py:23-32,332-524`, `tools.py` | Implemented and verified | None | — | `test_env_contract_and_isolation.py`, `test_contract_matrix.py` |
| Investigation tools with strict boundaries | `tools.py`, `auditor_env.py:218-232,456-491` | Implemented and verified | None | — | `test_env_tools.py` |
| Step/time limits and termination/truncation | `judge.py:51-99`, `auditor_env.py:370-414` | Implemented and verified | Forced-timeout score interpretation unconfirmed | H2 setup/interaction phase separation | `test_evaluator_isolation.py`, new `SetupFailureTests` |
| Submission schema, validation, one-to-one matching | `auditor_env.py:141-215`, `matcher.py` | Implemented and verified | Inadmissible truth could exceed the cap | H1/M4 admission preflight | new `AdmissionTests`, `test_matcher.py` |
| Evidence-aware scoring, exact reward, clean/FP/duplicate/gaming behavior | `matcher.py`, `metrics.py`, `reward.py`, `terminal.py` | Implemented and verified | None | — | `test_evaluation_matrix.py`, `test_terminal_reward.py` |
| Public train/dev episodes separated from hidden evaluation | `data/configs/*.json`, `datasets.py`, `judge.py:35-48` | Implemented and verified | Explicit-ID lookup regenerated corpus | L1 indexed lookup | `test_public_datasets.py`, new `CliRemediationTests` |
| Four baseline policies | `src/genesis/policies/`, `oracle.py` | Implemented and verified | None | — | `test_policy_adapters.py`, policy diagnostics 80/80 |
| Trajectory logging without truth leakage | `trajectories/records.py`, `docs/TRAJECTORIES.md` | Implemented and verified | Trace export was non-atomic | L2 staged no-clobber publication | `test_trajectories.py`, new trace tests |
| Offline execution, pinned dependencies, license, container, one-command verification | `Dockerfile`, `Dockerfile.policy`, `pyproject.toml`, `tests/requirements-schema.lock`, `LICENSE`, `verify --all` | Implemented and verified | Verifier image did not pin tzdb source | M3 canonical `tzdata==2025.2` + empty `PYTHONTZPATH` in both images | `verify --all` PASS; `test_cli_verify.py` updated |
| Demo and design documentation | `genesis.cli demo`, `docs/DESIGN_NOTE.md`, `README.md` | Implemented and verified | None | — | `python -m genesis.cli demo` PASS |
| Hidden-case admission and infrastructure failure attribution | Previously absent | Implemented and verified | — | H1/H2/M4 | New regression tests |
| Closed replay provenance registry and external operator controls | External coordinator only | Not verifiable | No in-repo registry or coordinator | Documented trust boundary (M5) | Source inspection |
| Video (≤3:00), presentation, Drive master document, portal submission | `materials/submission/` | Missing | External non-git deliverables | Not changed; explicitly out of scope | Not verifiable in repository |

## 4. Repository-wide action checklist

| Finding | Status | Resolution or reason |
|---|---|---|
| H1 — over-budget closed case | **Fixed in PR** | `require_admissible_episode` rejects truth that cannot be submitted fully |
| H2 — setup timeout graded as loss | **Fixed in PR** | Typed `phase="setup"` transport errors are rethrown before scoring |
| M1 — zero-rounded FX posting | **Fixed in PR** | Foreign obligation unit prices are raised until the base posting is positive |
| M2 — Docker ceiling bypass | **Fixed in PR** | CPU and memory ceilings enforced for all accepted representations |
| M3 — timezone database drift | **Fixed in PR** | Canonical tzdb pinned and aligned across verifier and policy images |
| M4 — no closed-case proof check | **Fixed in PR** | Link, arithmetic, budget, and Oracle-match checks in admission |
| M5 — replay provenance trust | **Deferred with reason** | No registry exists inside the repository; trust boundary documented for the external coordinator |
| L1 — redundant corpus generation | **Fixed in PR** | Explicit `episode_id` index checked first, legacy fallback retained |
| L2 — non-atomic trace export | **Fixed in PR** | Staged write + hard-link publish, no-clobber, separate status |
| Informational timeout scoring | **Deferred with reason** | Requires an official ruling; current fail-closed behavior preserved |

Already correct and left unchanged: generator determinism, integer-minor arithmetic, evaluator-only
truth, one-to-one matching, clean-empty metric convention, offline transport, four baselines,
evaluator-only Oracle, and the existing isolation CI workflow.

## 5. File-by-file change summary

| File | Finding | Current behavior | Intended change and expected behavior | Validation |
|---|---|---|---|---|
| `src/genesis/evaluation/admission.py` (new) | H1, M4 | No trusted admission step existed | Validate links, public proofs, submission budgets, and a full-coverage Oracle match; raise `ValueError` for inadmissible cases | `AdmissionTests` |
| `src/genesis/evaluation/judge.py` | H1, M4, H2 | Generated truth went straight to the environment; setup faults were scored | Call admission in the closed factory; rethrow setup-phase transport errors unscored | `AdmissionTests`, `SetupFailureTests` |
| `src/genesis/environment/jsonl.py` | H2 | `TransportError` had only a code | Added `phase` (`setup`/`interaction`, default interaction); interaction behavior unchanged | `SetupFailureTests`, `test_evaluator_isolation.py` |
| `src/genesis/environment/docker_policy.py` | H2, M2 | Setup faults shared the interaction timeout code; CPU/memory ceilings were incomplete | Tag image-inspect/create/start faults as setup; enforce `2` CPU / `4096` MiB across all input formats | `ResourceCeilingTests`, `test_evaluator_isolation.py` |
| `src/genesis/generation/company.py` | M1 | Tiny foreign amounts could round to zero and exhaust FX candidates | Raise foreign unit price to the smallest value producing a positive base posting | `FxAmountTests` for the reproducing seed |
| `src/genesis/cli/run_policy.py` | L1, L2 | Regenerated corpus for ID lookup; wrote trace directly before printing results | Index explicit IDs first; stage/fsync/hard-link traces and report trace status separately | `CliRemediationTests` |
| `Dockerfile`, `pyproject.toml` | M3 | Verifier image used system tzdb; tzdata had a lower bound | Set `PYTHONTZPATH=""` in verifier image; pin `tzdata==2025.2` | `verify --all`, `test_cli_verify.py` |
| `docs/ISOLATION_AND_LEAKAGE.md` | M3, M5, H2 | Trust boundary and resource ceilings partly implicit | Document admission, setup-phase failure handling, enforced ceilings, canonical tzdb, and replay provenance ownership | Documentation review |
| `tests/test_audit_remediation.py` (new) | All | No regressions for these findings | 10 regression tests covering admission, setup/interaction faults, FX, ceilings, ID lookup, and trace publication | 10/10 pass |
| `tests/test_cli_verify.py` | M3 | Expected the old lower-bound tzdata declaration | Expect the pinned declaration | Full suite passes |

Risks and limitations: the setup-phase change relies on `DockerPolicy` tagging its own setup
operations; other trusted runners that raise bare `TransportError` during setup would still be
scored, which is why the phase is part of the transport contract. The FX price adjustment changes
bytes only for configurations whose foreign posting previously rounded to zero; the committed
public corpus regenerates byte-for-byte unchanged. Raising the unit price is bounded and fails
loudly if no positive posting is reachable. Hard-link trace publication requires a local filesystem
that supports links; a documented `os.replace` fallback is used where links are not permitted.

## 6. Tests and checks actually executed

| Command | Result |
|---|---|
| `PYTHONPATH=src python3 -m unittest discover -s tests -p test_audit_remediation.py` | 10 tests, OK |
| `PYTHONPATH=src /tmp/qa_venv/bin/python -m unittest discover -s tests` | 201 tests, OK, 2 skipped (Docker smoke) |
| `PYTHONPATH=src /tmp/qa_venv/bin/python -m genesis.cli verify --all` | PASS: schemas/datasets 19/19, contract 19/19, evaluation 20/20, trajectories/pairs 20/20, runtime 121/121, replay 20/20, pairs 20/20, policy diagnostics 80/80; Docker smoke skipped on this host |
| `PYTHONPATH=src /tmp/qa_venv/bin/python -m genesis.cli demo` | PASS, `F1=1.0000`, `Return=0.9920`, 5 steps |

All commands used `PYTHONDONTWRITEBYTECODE=1` and were run from the repository root. No repository
files were generated as test artifacts. Live Linux Docker `C18` was not run locally; it remains in
the separate `Policy isolation` job.

## 7. Final verdict

The reviewed `e5c3e44` revision is suitable for supervised public demonstration but was not ready
to serve as an authoritative evaluator for closed, ranked submissions. With this PR, the confirmed
admission and failure-attribution defects are closed, the reproducible generator failure is fixed,
and the reproducibility contract is tightened. Remaining gaps are an external coordinator
provenance registry (M5) and an official ruling on forced-timeout scoring; neither is silently
claimed as resolved.
