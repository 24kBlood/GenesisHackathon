---
name: reward-engineering
description: Specification and anti-gaming verification workflow for AuditLab's open 1-to-1 bipartite matcher, tolerance rules, evidence subset validation, and terminal reward function. Use when editing audit_rl/matcher.py or audit_rl/schemas.py validation logic.
---

# Reward Engineering & Anti-Gaming Verification

## When to Use
- Modifying `normalize_finding`, `finding_matches_issue`, `match_findings`, `compute_terminal_reward`, or `evaluate_submission` in `audit_rl/matcher.py`.
- Adjusting payload validation rules in `validate_findings_payload` (`audit_rl/schemas.py`).
- Auditing reward behavior against specification gaming (spam, duplicate findings, false accusations on clean books, malformed payloads).

## Mathematical Contract
1. **Payload Validation**: Rejects wrong `episode_id`, `>30` findings, `>10` `evidence_ids`, `>1000` char `explanation`, non-int or `bool` `claimed_amount_minor`, or unknown `type`. Malformed payloads yield `valid_submission = False` and `reward = -1.0`.
2. **Five-Condition Match (`finding_matches_issue`)**:
   - Normalized `type` exact match.
   - `set(affected_ids)` exact match.
   - Normalized `currency` exact match.
   - `abs(claimed_amount_minor - expected_amount_minor) <= tolerance_minor`.
   - At least one `ev_set` in `issue["acceptable_evidence_sets"]` is a subset of `set(finding["evidence_ids"])`.
3. **Deterministic 1-to-1 Bipartite Matching**:
   - Kuhn's augmenting-path algorithm in deterministic sorted order `(issue_id, finding_index)`.
   - Duplicate findings matching an already-matched issue count as `FP`.
4. **Clean-Book Semantics**:
   - 0 truth issues & 0 findings -> `P = 1.0, R = 1.0, F1 = 1.0`.
   - 0 truth issues & >0 findings -> `P = 0.0, R = 0.0, F1 = 0.0`.
5. **Terminal Reward**:
   - `R = clamp(2.0 * F1 - 1.0 - 0.002 * inspection_steps, -1.0, 1.0)` when valid, else `-1.0`.

## Workflow
1. Review edge cases before changing any matcher or reward logic: clean books, partial evidence, boundary tolerance, ambiguous bipartite ties, and `bool` amounts (`isinstance(True, int)` in Python!).
2. Run the full matcher and reward unit suite:
   ```bash
   python3 -m unittest tests/test_matcher_and_reward.py -v
   ```
3. Run adversarial baseline comparison on 20 held-out eval episodes:
   ```bash
   python3 -m audit_rl run-baselines
   ```

## Verification Checklist
- [ ] Silence on clean books (`R ≈ +1.0`) strictly dominates false accusations (`R = -1.0`).
- [ ] `FlagEverythingAttackPolicy` receives strongly negative return (`< -0.9`).
- [ ] Duplicate findings never increase `TP` or `F1`.
- [ ] Step penalty (`0.002 * inspection_steps`) breaks ties between equal-F1 solutions without overriding audit accuracy.
- [ ] `tests/test_matcher_and_reward.py` passes 100%.
