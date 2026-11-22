"""Trusted terminal scoring for one episode."""

from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass

from genesis.domain.models import FindingsSubmission, TruthIssue

from .matcher import MatchResult, match
from .reward import TerminalScore, score_terminal


@dataclass(frozen=True)
class TerminalEvaluation:
    matching: MatchResult
    score: TerminalScore
    validation_error: str | None = None


class EpisodeEvaluator:
    """Bind validated truth, public evidence IDs and official tolerances."""

    def __init__(
        self,
        issues: Sequence[TruthIssue],
        public_evidence_ids: Collection[str],
        *,
        tolerances_minor: Mapping[str, int] | None = None,
    ):
        self._issues = deepcopy(tuple(issues))
        self._public_evidence_ids = frozenset(public_evidence_ids)
        self._tolerances_minor = dict(tolerances_minor or {})

    def __call__(
        self,
        submission: FindingsSubmission,
        inspection_steps: int,
        *,
        invalid_submission: bool,
        validation_error: str | None = None,
    ) -> TerminalEvaluation:
        findings = [] if invalid_submission else submission["findings"]
        matching = match(
            findings, self._issues, self._public_evidence_ids,
            tolerances_minor=self._tolerances_minor,
        )
        score = score_terminal(
            matching.tp, matching.fp, matching.fn, inspection_steps,
            invalid_submission=invalid_submission,
        )
        return TerminalEvaluation(matching, score, validation_error=validation_error)
