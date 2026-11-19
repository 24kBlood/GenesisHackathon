"""Validated JSON mappings at the matcher boundary.

These shared types describe the wire payloads. Constructing a dict does not
validate it. The caller validates external values before the matcher runs.
"""

from typing import TypedDict


class Finding(TypedDict):
    type: str
    affected_ids: list[str]
    claimed_amount_minor: int
    currency: str
    evidence_ids: list[str]
    explanation: str


class FindingsSubmission(TypedDict):
    episode_id: str
    findings: list[Finding]


class TruthIssue(TypedDict):
    issue_id: str
    type: str
    affected_ids: list[str]
    expected_amount_minor: int
    currency: str
    acceptable_evidence_sets: list[list[str]]


class EpisodeTruth(TypedDict):
    episode_id: str
    issues: list[TruthIssue]
