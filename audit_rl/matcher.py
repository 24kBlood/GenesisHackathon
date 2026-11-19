"""Deterministic finding normalization, 1-to-1 bipartite matching, and reward calculation."""

from __future__ import annotations

from typing import Any

from audit_rl.schemas import validate_findings_payload


def _normalize_id_list(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    cleaned = {str(v).strip() for v in values if isinstance(v, str) and v.strip()}
    return sorted(cleaned)


def normalize_finding(finding: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical normalized representation of a submitted finding."""
    raw_amount = finding.get("claimed_amount_minor")
    if isinstance(raw_amount, bool) or not isinstance(raw_amount, int):
        claimed_amount: int | None = None
    else:
        claimed_amount = int(raw_amount)

    raw_type = finding.get("type")
    norm_type = raw_type.strip().lower() if isinstance(raw_type, str) else ""

    raw_currency = finding.get("currency")
    norm_currency = raw_currency.strip().upper() if isinstance(raw_currency, str) else ""

    raw_explanation = finding.get("explanation")
    norm_explanation = raw_explanation.strip() if isinstance(raw_explanation, str) else ""

    return {
        "type": norm_type,
        "affected_ids": _normalize_id_list(finding.get("affected_ids")),
        "claimed_amount_minor": claimed_amount,
        "currency": norm_currency,
        "evidence_ids": _normalize_id_list(finding.get("evidence_ids")),
        "explanation": norm_explanation,
    }


def finding_matches_issue(finding: dict[str, Any], issue: dict[str, Any]) -> bool:
    """Check whether a finding satisfies all matching criteria for a truth issue."""
    if not isinstance(finding, dict) or not isinstance(issue, dict):
        return False

    norm = normalize_finding(finding)

    issue_type = issue.get("type")
    if not isinstance(issue_type, str) or norm["type"] != issue_type.strip().lower():
        return False

    issue_affected = _normalize_id_list(issue.get("affected_ids"))
    if not issue_affected or set(norm["affected_ids"]) != set(issue_affected):
        return False

    issue_currency = issue.get("currency")
    if not isinstance(issue_currency, str) or norm["currency"] != issue_currency.strip().upper():
        return False

    claimed_amount = norm["claimed_amount_minor"]
    expected_amount = issue.get("expected_amount_minor")
    if claimed_amount is None or isinstance(expected_amount, bool) or not isinstance(expected_amount, int):
        return False

    raw_tol = issue.get("tolerance_minor", 0)
    if raw_tol is None:
        tolerance = 0
    elif isinstance(raw_tol, bool) or not isinstance(raw_tol, int) or raw_tol < 0:
        return False
    else:
        tolerance = int(raw_tol)

    if abs(claimed_amount - expected_amount) > tolerance:
        return False

    acceptable_sets = issue.get("acceptable_evidence_sets")
    if not isinstance(acceptable_sets, list) or not acceptable_sets:
        return False

    finding_evidence = set(norm["evidence_ids"])
    for ev_set in acceptable_sets:
        if not isinstance(ev_set, (list, tuple, set)):
            continue
        required = {str(x).strip() for x in ev_set if isinstance(x, str) and x.strip()}
        if required.issubset(finding_evidence):
            return True

    return False


def match_findings(
    findings: list[dict[str, Any]],
    truth_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute deterministic maximum-cardinality 1-to-1 matching and P/R/F1 metrics.

    Uses Kuhn's augmenting-path algorithm over truth issues sorted by ``issue_id``
    and candidate findings ordered by ascending ``finding_index``.
    """
    sorted_issues = sorted(
        truth_issues,
        key=lambda iss: str(iss.get("issue_id", "")),
    )

    num_issues = len(sorted_issues)
    num_findings = len(findings)

    # Build deterministic adjacency list: issue_idx -> ascending finding indices
    adj: list[list[int]] = []
    for issue in sorted_issues:
        candidates = [
            f_idx
            for f_idx, finding in enumerate(findings)
            if finding_matches_issue(finding, issue)
        ]
        adj.append(candidates)

    # match_finding_to_issue[f_idx] = matched issue_idx (or -1 if unmatched)
    match_finding_to_issue: list[int] = [-1] * num_findings

    def _augment(issue_idx: int, visited_findings: set[int]) -> bool:
        for f_idx in adj[issue_idx]:
            if f_idx in visited_findings:
                continue
            visited_findings.add(f_idx)
            prev_issue = match_finding_to_issue[f_idx]
            if prev_issue == -1 or _augment(prev_issue, visited_findings):
                match_finding_to_issue[f_idx] = issue_idx
                return True
        return False

    for issue_idx in range(num_issues):
        _augment(issue_idx, set())

    matched_pairs: list[dict[str, Any]] = []
    matched_issue_indices: set[int] = set()
    matched_finding_indices: set[int] = set()

    for f_idx, issue_idx in enumerate(match_finding_to_issue):
        if issue_idx != -1:
            matched_issue_indices.add(issue_idx)
            matched_finding_indices.add(f_idx)
            matched_pairs.append(
                {
                    "issue_id": str(sorted_issues[issue_idx].get("issue_id", "")),
                    "finding_index": f_idx,
                }
            )

    matched_pairs.sort(key=lambda pair: (pair["issue_id"], pair["finding_index"]))

    unmatched_finding_indices = [
        f_idx for f_idx in range(num_findings) if f_idx not in matched_finding_indices
    ]
    missed_issue_ids = [
        str(sorted_issues[i].get("issue_id", ""))
        for i in range(num_issues)
        if i not in matched_issue_indices
    ]

    tp = len(matched_pairs)
    fp = num_findings - tp
    fn = num_issues - tp

    if num_issues == 0 and num_findings == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    elif num_issues == 0 and num_findings > 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = 2 * tp + fp + fn
        f1 = (2 * tp) / denom if denom > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_pairs": matched_pairs,
        "unmatched_finding_indices": unmatched_finding_indices,
        "missed_issue_ids": missed_issue_ids,
    }


def compute_terminal_reward(
    f1: float,
    inspection_steps: int,
    valid_submission: bool = True,
) -> float:
    """Compute bounded terminal reward in [-1.0, 1.0]."""
    if not valid_submission:
        return -1.0
    steps = max(0, int(inspection_steps))
    raw = 2.0 * float(f1) - 1.0 - 0.002 * steps
    return max(-1.0, min(1.0, raw))


def evaluate_submission(
    payload: dict[str, Any] | None,
    truth: dict[str, Any],
    inspection_steps: int,
    expected_episode_id: str,
) -> dict[str, Any]:
    """Validate and score a findings submission against ground-truth issues."""
    truth_issues = list(truth.get("issues", [])) if isinstance(truth, dict) else []
    steps = max(0, int(inspection_steps))

    if payload is None:
        metrics = match_findings([], truth_issues)
        reward = compute_terminal_reward(
            metrics["f1"],
            steps,
            valid_submission=True,
        )
        return {
            **metrics,
            "valid_submission": True,
            "validation_errors": [],
            "reward": reward,
            "inspection_steps": steps,
            "submitted_count": 0,
            "truth_count": len(truth_issues),
        }

    errors = validate_findings_payload(payload, expected_episode_id)
    if errors:
        missed_ids = sorted(
            str(issue.get("issue_id", "")) for issue in truth_issues
        )
        reward = compute_terminal_reward(0.0, steps, valid_submission=False)
        return {
            "tp": 0,
            "fp": 0,
            "fn": len(truth_issues),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "matched_pairs": [],
            "unmatched_finding_indices": [],
            "missed_issue_ids": missed_ids,
            "valid_submission": False,
            "validation_errors": list(errors),
            "reward": reward,
            "inspection_steps": steps,
            "submitted_count": len(payload.get("findings", []))
            if isinstance(payload, dict) and isinstance(payload.get("findings"), list)
            else 0,
            "truth_count": len(truth_issues),
        }

    findings = list(payload.get("findings", []))
    metrics = match_findings(findings, truth_issues)
    reward = compute_terminal_reward(
        metrics["f1"],
        steps,
        valid_submission=True,
    )
    return {
        **metrics,
        "valid_submission": True,
        "validation_errors": [],
        "reward": reward,
        "inspection_steps": steps,
        "submitted_count": len(findings),
        "truth_count": len(truth_issues),
    }
