"""Trusted admission checks run before a closed case reaches any policy."""

from genesis.environment.auditor_env import (
    MAX_AFFECTED_IDS,
    MAX_EVIDENCE_IDS,
    MAX_FINDINGS,
    _validate_submission,
)
from genesis.generation.company import Episode

from .matcher import match
from .oracle import OraclePolicy
from .permutation import evidence_catalog, validate_links
from .public_proofs import verify_public_proofs


def require_admissible_episode(episode: Episode) -> None:
    """Reject cases whose truth cannot be fully submitted under the public contract."""
    validate_links(episode)
    verify_public_proofs(episode)
    truth = episode.evaluator.truth
    issues = truth["issues"]
    evidence = evidence_catalog(episode.public)
    if len(issues) > MAX_FINDINGS:
        raise ValueError(
            f"case has {len(issues)} truth issues, exceeding the {MAX_FINDINGS}-finding submission budget"
        )
    for issue in issues:
        if not 1 <= len(issue["affected_ids"]) <= MAX_AFFECTED_IDS:
            raise ValueError("truth affected_ids exceed the submission budget")
        if not any(0 < len(set(ids)) <= MAX_EVIDENCE_IDS for ids in issue["acceptable_evidence_sets"]):
            raise ValueError("no acceptable evidence set fits the submission budget")
    oracle = OraclePolicy(truth, evidence)
    action = oracle.act({"manifest": {"episode_id": truth["episode_id"]}}, None)
    submission, error = _validate_submission(action, truth["episode_id"])
    if submission is None:
        raise ValueError(f"oracle submission is not wire-valid: {error}")
    result = match(submission["findings"], issues, evidence)
    if result.tp != len(issues) or result.fp or result.fn:
        raise ValueError("truth cannot be matched one-to-one within the public submission contract")
