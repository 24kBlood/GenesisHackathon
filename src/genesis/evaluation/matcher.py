"""Deterministic matching for structurally validated episode findings."""

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Collection, Mapping, Sequence

from genesis.domain.enums import FindingType
from genesis.domain.models import Finding, TruthIssue


DEFAULT_TOLERANCE_MINOR = 0
_FINDING_TYPES = frozenset(kind.value for kind in FindingType)

_Claim = tuple[str, tuple[str, ...], int, str]
_Evidence = frozenset[str]


@dataclass(frozen=True)
class MatchedFinding:
    finding_index: int
    issue_id: str


@dataclass(frozen=True)
class MatchResult:
    """Evaluator-only diagnostics. Indices refer to the original submission."""

    matches: tuple[MatchedFinding, ...]
    unmatched_finding_indices: tuple[int, ...]
    unmatched_issue_ids: tuple[str, ...]

    @property
    def tp(self) -> int:
        return len(self.matches)

    @property
    def fp(self) -> int:
        return len(self.unmatched_finding_indices)

    @property
    def fn(self) -> int:
        return len(self.unmatched_issue_ids)


def _canonical_ids(ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(ids)))


def _maximum_matching(edges: Sequence[Mapping[str, int]]) -> dict[str, int]:
    """Return issue ID -> claim group using deterministic iterative augmenting paths."""
    issue_to_group: dict[str, int] = {}
    for start_group in range(len(edges)):
        visited: set[str] = set()
        stack: list[tuple[int, Iterator[str], str | None]] = [
            (start_group, iter(edges[start_group]), None)
        ]
        while stack:
            group, neighbors, _ = stack[-1]
            try:
                issue_id = next(neighbors)
            except StopIteration:
                stack.pop()
                continue
            if issue_id in visited:
                continue
            visited.add(issue_id)
            current = issue_to_group.get(issue_id)
            if current is None:
                issue_to_group[issue_id] = group
                for index in range(len(stack) - 1, 0, -1):
                    parent_group = stack[index - 1][0]
                    via_issue = stack[index][2]
                    assert via_issue is not None
                    issue_to_group[via_issue] = parent_group
                break
            stack.append((current, iter(edges[current]), issue_id))
    return issue_to_group


def match(
    findings: Sequence[Finding],
    truth: Sequence[TruthIssue],
    public_evidence_ids: Collection[str],
    *,
    tolerances_minor: Mapping[str, int] | None = None,
) -> MatchResult:
    """Match one episode using its public record catalog and published tolerances.

    The caller validates the submission and truth schema and selects the episode.
    A missing tolerance entry means an exact amount match in that currency.
    """
    tolerances = tolerances_minor or {}
    if any(not isinstance(kind, str) or kind not in _FINDING_TYPES for kind in tolerances):
        raise ValueError("Tolerances must use known finding types")
    if any(type(amount) is not int or amount < 0 for amount in tolerances.values()):
        raise ValueError("Tolerances must be non-negative integers in minor units")

    public_ids = frozenset(public_evidence_ids)
    groups: dict[_Claim, list[tuple[_Evidence, int]]] = defaultdict(list)
    for index, finding in enumerate(findings):
        claim = (
            finding["type"],
            _canonical_ids(finding["affected_ids"]),
            finding["claimed_amount_minor"],
            finding["currency"],
        )
        evidence = frozenset(finding["evidence_ids"])
        if evidence <= public_ids:
            groups[claim].append((evidence, index))

    ordered_truth = sorted(truth, key=lambda issue: issue["issue_id"])
    if len({issue["issue_id"] for issue in ordered_truth}) != len(ordered_truth):
        raise ValueError("Truth issue IDs must be unique within an episode")
    truth_by_scope: dict[
        tuple[str, tuple[str, ...], str], list[tuple[str, int, tuple[_Evidence, ...]]]
    ] = defaultdict(list)
    seen_truth_claims: set[_Claim] = set()
    for issue in ordered_truth:
        canonical_affected = _canonical_ids(issue["affected_ids"])
        truth_claim = (issue["type"], canonical_affected, issue["expected_amount_minor"], issue["currency"])
        if truth_claim in seen_truth_claims:
            raise ValueError("Truth claims must be unique within an episode")
        seen_truth_claims.add(truth_claim)
        scope = (issue["type"], canonical_affected, issue["currency"])
        acceptable = tuple(frozenset(ids) for ids in issue["acceptable_evidence_sets"])
        truth_by_scope[scope].append(
            (issue["issue_id"], issue["expected_amount_minor"], acceptable)
        )

    edges: list[dict[str, int]] = []
    for finding_type, affected_ids, amount, currency in sorted(groups):
        candidates = groups[finding_type, affected_ids, amount, currency]
        ordered_candidates = sorted(
            candidates, key=lambda item: (tuple(sorted(item[0])), item[1])
        )
        tolerance = tolerances.get(finding_type, DEFAULT_TOLERANCE_MINOR)
        group_edges: dict[str, int] = {}
        scope = (finding_type, affected_ids, currency)
        for issue_id, expected_amount, acceptable in truth_by_scope.get(scope, ()):
            if abs(amount - expected_amount) > tolerance:
                continue
            for evidence, index in ordered_candidates:
                if any(required <= evidence for required in acceptable):
                    group_edges[issue_id] = index
                    break
        edges.append(group_edges)

    assignment = _maximum_matching(edges)
    matches = tuple(
        MatchedFinding(edges[group][issue_id], issue_id)
        for issue_id, group in sorted(assignment.items())
    )
    matched_indices = {pair.finding_index for pair in matches}
    return MatchResult(
        matches,
        tuple(index for index in range(len(findings)) if index not in matched_indices),
        tuple(issue["issue_id"] for issue in ordered_truth if issue["issue_id"] not in assignment),
    )
