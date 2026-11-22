"""Acceptance tests for the one-to-one matching contract."""

from itertools import permutations
from unittest import TestCase

from genesis.domain.models import Finding, TruthIssue
from genesis.evaluation.matcher import MatchResult, MatchedFinding, match


PUBLIC_IDS = ("ev-a", "ev-b", "ev-c", "ep-payment")


def finding(
    amount: int = 100,
    *,
    evidence: tuple[str, ...] = ("ev-a", "ev-b"),
    affected: tuple[str, ...] = ("invoice-b", "invoice-a"),
) -> Finding:
    return {
        "type": "fx_conversion",
        "affected_ids": list(affected),
        "claimed_amount_minor": amount,
        "currency": "KZT",
        "evidence_ids": list(evidence),
        "explanation": "Сверка по открытому курсу.",
    }


def issue(
    issue_id: str = "issue-a",
    amount: int = 100,
    *,
    acceptable: tuple[tuple[str, ...], ...] = (("ev-a", "ev-b"), ("ev-c",)),
) -> TruthIssue:
    return {
        "issue_id": issue_id,
        "type": "fx_conversion",
        "affected_ids": ["invoice-a", "invoice-b"],
        "expected_amount_minor": amount,
        "currency": "KZT",
        "acceptable_evidence_sets": [list(ids) for ids in acceptable],
    }


class MatcherTests(TestCase):
    def assert_counts(self, result: MatchResult, expected: tuple[int, int, int]) -> None:
        self.assertEqual((result.tp, result.fp, result.fn), expected)

    def test_augmenting_path_recovers_a_match_greedy_would_miss(self) -> None:
        claims = [finding(100), finding(101)]
        truth = [issue("issue-a", 100), issue("issue-b", 99)]
        result = match(claims, truth, PUBLIC_IDS, tolerances_minor={"fx_conversion": 1})

        self.assert_counts(result, (2, 0, 0))
        self.assertEqual(
            result.matches,
            (MatchedFinding(1, "issue-a"), MatchedFinding(0, "issue-b")),
        )

    def test_ambiguous_pairs_are_deterministic_across_permutations(self) -> None:
        claims = [finding(100), finding(101)]
        truth = [issue("issue-a", 100), issue("issue-b", 101)]
        assignments = set()
        for claim_order in permutations(claims):
            for truth_order in permutations(truth):
                result = match(
                    claim_order, truth_order, tuple(reversed(PUBLIC_IDS)),
                    tolerances_minor={"fx_conversion": 1},
                )
                self.assert_counts(result, (2, 0, 0))
                assignments.add(
                    tuple(
                        (pair.issue_id, claim_order[pair.finding_index]["claimed_amount_minor"])
                        for pair in result.matches
                    )
                )

        self.assertEqual(len(assignments), 1)

    def test_evidence_from_separate_duplicate_claims_is_never_combined(self) -> None:
        claims = [finding(evidence=("ev-a",)), finding(evidence=("ev-b",))]
        result = match(claims, [issue()], PUBLIC_IDS)

        self.assert_counts(result, (0, 2, 1))
        self.assertEqual(result.unmatched_finding_indices, (0, 1))
        self.assertEqual(result.unmatched_issue_ids, ("issue-a",))

    def test_canonical_duplicate_group_earns_one_tp_and_selects_evidence_in_order(self) -> None:
        claims = [
            finding(evidence=("ev-c",)),
            finding(evidence=("ev-b", "ev-a"), affected=("invoice-a", "invoice-b")),
            finding(evidence=("ev-a", "ev-b"), affected=("invoice-a", "invoice-a", "invoice-b")),
        ]
        claims[2]["explanation"] = "Та же находка с другим пояснением."
        result = match(claims, [issue()], PUBLIC_IDS)

        self.assert_counts(result, (1, 2, 0))
        self.assertEqual(result.matches, (MatchedFinding(1, "issue-a"),))
        self.assertEqual(result.unmatched_finding_indices, (0, 2))

    def test_duplicate_group_cannot_credit_two_candidates_under_tolerance(self) -> None:
        claims = [finding(evidence=("ev-c",)), finding(evidence=("ev-a", "ev-b"))]
        truth = [
            issue("issue-a", 99, acceptable=(("ev-c",),)),
            issue("issue-b", 101, acceptable=(("ev-a", "ev-b"),)),
        ]
        result = match(claims, truth, PUBLIC_IDS, tolerances_minor={"fx_conversion": 1})

        self.assert_counts(result, (1, 1, 1))
        self.assertEqual(result.matches, (MatchedFinding(0, "issue-a"),))

    def test_full_alternative_evidence_set_and_known_extra_record_are_allowed(self) -> None:
        result = match([finding(evidence=("ep-payment", "ev-c"))], [issue()], PUBLIC_IDS)

        self.assert_counts(result, (1, 0, 0))

    def test_unknown_evidence_id_disqualifies_even_with_complete_set(self) -> None:
        result = match([finding(evidence=("ev-c", "unknown"))], [issue()], PUBLIC_IDS)

        self.assert_counts(result, (0, 1, 1))

    def test_wrong_type_currency_and_case_sensitive_affected_id_do_not_match(self) -> None:
        wrong_type = finding()
        wrong_type["type"] = "period_cutoff"
        wrong_currency = finding()
        wrong_currency["currency"] = "USD"
        wrong_id = finding(affected=("Invoice-a", "invoice-b"))
        for claim in (wrong_type, wrong_currency, wrong_id):
            with self.subTest(claim=claim):
                self.assert_counts(match([claim], [issue()], PUBLIC_IDS), (0, 1, 1))

    def test_exact_amount_by_default_and_inclusive_configured_tolerance(self) -> None:
        baseline = match([finding(101)], [issue()], PUBLIC_IDS)
        at_limit = match([finding(101)], [issue()], PUBLIC_IDS, tolerances_minor={"fx_conversion": 1})
        outside = match([finding(102)], [issue()], PUBLIC_IDS, tolerances_minor={"fx_conversion": 1})
        unrelated = match([finding(101)], [issue()], PUBLIC_IDS, tolerances_minor={"period_cutoff": 1})

        for result, expected in (
            (baseline, (0, 1, 1)), (at_limit, (1, 0, 0)),
            (outside, (0, 1, 1)), (unrelated, (0, 1, 1)),
        ):
            self.assert_counts(result, expected)

    def test_invalid_tolerance_configuration_is_rejected(self) -> None:
        for value in (-1, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "non-negative integers"):
                    match([], [], PUBLIC_IDS, tolerances_minor={"fx_conversion": value})
        for bad_key in ("unknown_type", 123):
            with self.subTest(bad_key=bad_key):
                with self.assertRaisesRegex(ValueError, "known finding types"):
                    match([], [], PUBLIC_IDS, tolerances_minor={bad_key: 0})

    def test_duplicate_truth_ids_and_claims_are_rejected_before_matching(self) -> None:
        with self.assertRaisesRegex(ValueError, "IDs must be unique"):
            match([finding()], [issue(), issue(amount=101)], PUBLIC_IDS)
        with self.assertRaisesRegex(ValueError, "Truth claims must be unique"):
            match([finding()], [issue("issue-a", 100), issue("issue-b", 100)], PUBLIC_IDS)

    def test_iterative_maximum_matching_scales_past_recursion_limit(self) -> None:
        n = 1200
        claims = [finding(amount=i, affected=("invoice-a",), evidence=("ev-c",)) for i in range(n - 1, -1, -1)]
        truth = [issue(f"issue-{i:04d}", amount=i, acceptable=(("ev-c",),)) for i in range(n - 1, -1, -1)]
        for row in truth:
            row["affected_ids"] = ["invoice-a"]
        result = match(claims, truth, PUBLIC_IDS, tolerances_minor={"fx_conversion": 1})
        self.assert_counts(result, (n, 0, 0))

    def test_empty_truth_or_submissions_produce_expected_counts(self) -> None:
        self.assert_counts(match([], [], PUBLIC_IDS), (0, 0, 0))
        self.assert_counts(match([finding()], [], PUBLIC_IDS), (0, 1, 0))
        self.assert_counts(match([], [issue()], PUBLIC_IDS), (0, 0, 1))
