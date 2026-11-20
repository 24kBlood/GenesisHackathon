"""The oracle is an evaluator-only upper-bound check."""

from collections.abc import Collection
from copy import deepcopy

from genesis.domain.models import EpisodeTruth, Finding
from genesis.policies.protocol import Action, Observation, submit


class OraclePolicy:
    """Use trusted truth to produce a valid submission for the same episode."""

    def __init__(self, truth: EpisodeTruth, public_evidence_ids: Collection[str]):
        self._episode_id = truth["episode_id"]
        self._findings: list[Finding] = []
        if len(truth["issues"]) > 30:
            raise ValueError("oracle cannot submit more than 30 issues")
        available = set(public_evidence_ids)
        for issue in sorted(truth["issues"], key=lambda entry: entry["issue_id"]):
            eligible = [
                tuple(sorted(ids)) for ids in issue["acceptable_evidence_sets"]
                if 0 < len(ids) <= 10 and set(ids) <= available
            ]
            if not eligible:
                raise ValueError("oracle requires a complete public evidence set per issue")
            evidence = min(eligible, key=lambda ids: (len(ids), ids))
            self._findings.append({
                "type": issue["type"],
                "affected_ids": issue["affected_ids"].copy(),
                "claimed_amount_minor": issue["expected_amount_minor"],
                "currency": issue["currency"],
                "evidence_ids": list(evidence),
                "explanation": "Evaluator upper-bound check.",
            })

    def act(self, observation: Observation, previous_action: Action | None) -> Action:
        if previous_action is not None or observation["manifest"]["episode_id"] != self._episode_id:
            raise ValueError("oracle received an unexpected episode")
        return submit(self._episode_id, deepcopy(self._findings))
