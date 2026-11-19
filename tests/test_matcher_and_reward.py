"""Unit tests for finding normalization, 1-to-1 bipartite matching, validation, and reward shaping."""

from __future__ import annotations

import unittest
from typing import Any

from audit_rl.matcher import (
    compute_terminal_reward,
    evaluate_submission,
    finding_matches_issue,
    match_findings,
    normalize_finding,
)
from audit_rl.schemas import MAX_EVIDENCE_PER_FINDING, MAX_EXPLANATION_CHARS, MAX_FINDINGS


def _make_issue(
    issue_id: str = "ISS-001",
    itype: str = "fx_conversion_error",
    affected_ids: list[str] | None = None,
    expected_amount_minor: int = 5000,
    currency: str = "USD",
    tolerance_minor: int = 1,
    acceptable_evidence_sets: list[list[str]] | None = None,
) -> dict[str, Any]:
    return {
        "issue_id": issue_id,
        "type": itype,
        "affected_ids": affected_ids if affected_ids is not None else ["INV-00001"],
        "expected_amount_minor": expected_amount_minor,
        "currency": currency,
        "tolerance_minor": tolerance_minor,
        "acceptable_evidence_sets": acceptable_evidence_sets
        if acceptable_evidence_sets is not None
        else [["DOC-000001", "ROW-000001"]],
    }


def _make_finding(
    ftype: str = "fx_conversion_error",
    affected_ids: list[str] | None = None,
    claimed_amount_minor: int = 5000,
    currency: str = "USD",
    evidence_ids: list[str] | None = None,
    explanation: str = "Verified discrepancy.",
) -> dict[str, Any]:
    return {
        "type": ftype,
        "affected_ids": affected_ids if affected_ids is not None else ["INV-00001"],
        "claimed_amount_minor": claimed_amount_minor,
        "currency": currency,
        "evidence_ids": evidence_ids if evidence_ids is not None else ["DOC-000001", "ROW-000001"],
        "explanation": explanation,
    }


class TestMatcherAndReward(unittest.TestCase):
    def test_01_exact_tp_matching(self) -> None:
        issue = _make_issue()
        finding = _make_finding()
        self.assertTrue(finding_matches_issue(finding, issue))
        res = match_findings([finding], [issue])
        self.assertEqual(res["tp"], 1)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 0)
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["recall"], 1.0)
        self.assertEqual(res["f1"], 1.0)

    def test_02_normalization_case_whitespace_and_id_order(self) -> None:
        issue = _make_issue(
            itype="duplicate_invoice",
            affected_ids=["INV-00002", "INV-00001"],
            currency="EUR",
            acceptable_evidence_sets=[["DOC-1", "DOC-2"]],
        )
        finding = _make_finding(
            ftype="  DUPLICATE_INVOICE ",
            affected_ids=["INV-00001", "INV-00002"],
            currency=" eur ",
            evidence_ids=["DOC-2", "DOC-1"],
        )
        norm = normalize_finding(finding)
        self.assertEqual(norm["type"], "duplicate_invoice")
        self.assertEqual(norm["currency"], "EUR")
        self.assertEqual(norm["affected_ids"], ["INV-00001", "INV-00002"])
        self.assertTrue(finding_matches_issue(finding, issue))

    def test_03_tolerance_boundary_inclusive_and_exclusive(self) -> None:
        issue = _make_issue(expected_amount_minor=1000, tolerance_minor=2)
        self.assertTrue(finding_matches_issue(_make_finding(claimed_amount_minor=998), issue))
        self.assertTrue(finding_matches_issue(_make_finding(claimed_amount_minor=1002), issue))
        self.assertFalse(finding_matches_issue(_make_finding(claimed_amount_minor=997), issue))
        self.assertFalse(finding_matches_issue(_make_finding(claimed_amount_minor=1003), issue))

    def test_04_evidence_subset_matching_allows_superset_and_alternative_sets(self) -> None:
        issue = _make_issue(
            acceptable_evidence_sets=[
                ["DOC-1", "ROW-1"],
                ["INV-00001", "ROW-2"],
            ]
        )
        # Superset of first acceptable set matches
        self.assertTrue(
            finding_matches_issue(_make_finding(evidence_ids=["DOC-1", "ROW-1", "FX-1"]), issue)
        )
        # Exact second acceptable set matches
        self.assertTrue(
            finding_matches_issue(_make_finding(evidence_ids=["INV-00001", "ROW-2"]), issue)
        )

    def test_05_unsupported_or_partial_evidence_rejected(self) -> None:
        issue = _make_issue(acceptable_evidence_sets=[["DOC-1", "ROW-1"]])
        self.assertFalse(finding_matches_issue(_make_finding(evidence_ids=["DOC-1"]), issue))
        self.assertFalse(finding_matches_issue(_make_finding(evidence_ids=[]), issue))
        self.assertFalse(finding_matches_issue(_make_finding(evidence_ids=["DOC-999", "ROW-999"]), issue))

    def test_06_duplicate_findings_count_as_fp_no_benefit(self) -> None:
        issue = _make_issue()
        f1 = _make_finding()
        f2 = _make_finding()
        res = match_findings([f1, f2], [issue])
        self.assertEqual(res["tp"], 1)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 0)
        self.assertAlmostEqual(res["precision"], 0.5, places=6)
        self.assertAlmostEqual(res["recall"], 1.0, places=6)
        self.assertLess(res["f1"], 1.0)

        r_single = compute_terminal_reward(1.0, inspection_steps=5)
        r_dup = compute_terminal_reward(res["f1"], inspection_steps=5)
        self.assertLess(r_dup, r_single)

    def test_07_flag_everything_attack_yields_poor_reward(self) -> None:
        issue = _make_issue()
        spam = [_make_finding()] + [
            _make_finding(affected_ids=[f"INV-{i:05d}"], evidence_ids=[f"DOC-{i:06d}"])
            for i in range(2, 31)
        ]
        res = match_findings(spam, [issue])
        self.assertEqual(res["tp"], 1)
        self.assertEqual(res["fp"], 29)
        self.assertLess(res["f1"], 0.1)
        reward = compute_terminal_reward(res["f1"], inspection_steps=2)
        self.assertLess(reward, -0.8)

    def test_08_silence_on_clean_books_beats_false_accusations(self) -> None:
        truth = {"episode_id": "ep-clean", "issues": []}
        silent = evaluate_submission(
            {"episode_id": "ep-clean", "findings": []},
            truth=truth,
            inspection_steps=0,
            expected_episode_id="ep-clean",
        )
        false_acc = evaluate_submission(
            {"episode_id": "ep-clean", "findings": [_make_finding()]},
            truth=truth,
            inspection_steps=0,
            expected_episode_id="ep-clean",
        )
        self.assertEqual(silent["f1"], 1.0)
        self.assertEqual(silent["reward"], 1.0)
        self.assertEqual(false_acc["f1"], 0.0)
        self.assertEqual(false_acc["reward"], -1.0)
        self.assertGreater(silent["reward"], false_acc["reward"])

    def test_09_complete_correct_answer_beats_silence_on_erroneous_books(self) -> None:
        issue = _make_issue()
        truth = {"episode_id": "ep-err", "issues": [issue]}
        correct = evaluate_submission(
            {"episode_id": "ep-err", "findings": [_make_finding()]},
            truth=truth,
            inspection_steps=10,
            expected_episode_id="ep-err",
        )
        silent = evaluate_submission(
            {"episode_id": "ep-err", "findings": []},
            truth=truth,
            inspection_steps=0,
            expected_episode_id="ep-err",
        )
        self.assertEqual(correct["f1"], 1.0)
        self.assertEqual(silent["f1"], 0.0)
        self.assertGreater(correct["reward"], silent["reward"])

    def test_10_step_economy_slightly_differentiates_equal_answers(self) -> None:
        r_fast = compute_terminal_reward(1.0, inspection_steps=5)
        r_slow = compute_terminal_reward(1.0, inspection_steps=25)
        self.assertAlmostEqual(r_fast - r_slow, 0.002 * 20, places=6)
        self.assertGreater(r_fast, r_slow)
        # Still bounded in [-1.0, 1.0]
        self.assertEqual(compute_terminal_reward(0.0, inspection_steps=100), -1.0)

    def test_11_guessed_ids_fail_matching(self) -> None:
        issue = _make_issue(affected_ids=["INV-00001"])
        guessed = _make_finding(affected_ids=["INV-99999"])
        self.assertFalse(finding_matches_issue(guessed, issue))

    def test_12_wrong_currency_fails_matching(self) -> None:
        issue = _make_issue(currency="USD")
        wrong_cur = _make_finding(currency="EUR")
        self.assertFalse(finding_matches_issue(wrong_cur, issue))

    def test_13_wrong_error_type_fails_matching(self) -> None:
        issue = _make_issue(itype="fx_conversion_error")
        wrong_type = _make_finding(ftype="period_cutoff_error")
        self.assertFalse(finding_matches_issue(wrong_type, issue))

    def test_14_invalid_payload_too_many_findings_gives_minus_one(self) -> None:
        truth = {"episode_id": "ep-1", "issues": []}
        payload = {"episode_id": "ep-1", "findings": [_make_finding() for _ in range(MAX_FINDINGS + 1)]}
        res = evaluate_submission(payload, truth, inspection_steps=0, expected_episode_id="ep-1")
        self.assertFalse(res["valid_submission"])
        self.assertEqual(res["reward"], -1.0)
        self.assertTrue(res["validation_errors"])

    def test_15_invalid_payload_too_many_evidence_ids_gives_minus_one(self) -> None:
        truth = {"episode_id": "ep-1", "issues": []}
        bad_finding = _make_finding(evidence_ids=[f"DOC-{i}" for i in range(MAX_EVIDENCE_PER_FINDING + 1)])
        res = evaluate_submission(
            {"episode_id": "ep-1", "findings": [bad_finding]},
            truth,
            inspection_steps=0,
            expected_episode_id="ep-1",
        )
        self.assertFalse(res["valid_submission"])
        self.assertEqual(res["reward"], -1.0)

    def test_16_invalid_payload_explanation_too_long_gives_minus_one(self) -> None:
        truth = {"episode_id": "ep-1", "issues": []}
        bad_finding = _make_finding(explanation="x" * (MAX_EXPLANATION_CHARS + 1))
        res = evaluate_submission(
            {"episode_id": "ep-1", "findings": [bad_finding]},
            truth,
            inspection_steps=0,
            expected_episode_id="ep-1",
        )
        self.assertFalse(res["valid_submission"])
        self.assertEqual(res["reward"], -1.0)

    def test_17_invalid_payload_wrong_episode_id_gives_minus_one(self) -> None:
        truth = {"episode_id": "ep-expected", "issues": []}
        res = evaluate_submission(
            {"episode_id": "ep-wrong", "findings": []},
            truth,
            inspection_steps=0,
            expected_episode_id="ep-expected",
        )
        self.assertFalse(res["valid_submission"])
        self.assertEqual(res["reward"], -1.0)

    def test_18_invalid_payload_bool_amount_gives_minus_one(self) -> None:
        truth = {"episode_id": "ep-1", "issues": []}
        bad_finding = _make_finding(claimed_amount_minor=True)  # type: ignore[arg-type]
        res = evaluate_submission(
            {"episode_id": "ep-1", "findings": [bad_finding]},
            truth,
            inspection_steps=0,
            expected_episode_id="ep-1",
        )
        self.assertFalse(res["valid_submission"])
        self.assertEqual(res["reward"], -1.0)

    def test_19_none_payload_treated_as_valid_empty_submission(self) -> None:
        truth = {"episode_id": "ep-clean", "issues": []}
        res = evaluate_submission(None, truth, inspection_steps=50, expected_episode_id="ep-clean")
        self.assertTrue(res["valid_submission"])
        self.assertEqual(res["f1"], 1.0)
        self.assertAlmostEqual(res["reward"], 1.0 - 0.002 * 50, places=6)

    def test_20_deterministic_tie_breaking_on_ambiguous_bipartite_matching(self) -> None:
        # Two issues that can both match finding 0, while issue 2 can also match finding 1.
        # Maximum cardinality matching must match BOTH issues (TP=2) deterministically!
        iss_a = _make_issue(
            issue_id="ISS-001",
            expected_amount_minor=1000,
            tolerance_minor=5,
            acceptable_evidence_sets=[["DOC-A"]],
        )
        iss_b = _make_issue(
            issue_id="ISS-002",
            expected_amount_minor=1004,
            tolerance_minor=5,
            acceptable_evidence_sets=[["DOC-A"], ["DOC-B"]],
        )
        f0 = _make_finding(claimed_amount_minor=1001, evidence_ids=["DOC-A"])  # matches both
        f1 = _make_finding(claimed_amount_minor=1004, evidence_ids=["DOC-B"])  # matches only ISS-002

        res1 = match_findings([f0, f1], [iss_b, iss_a])
        res2 = match_findings([f0, f1], [iss_a, iss_b])
        self.assertEqual(res1["tp"], 2)
        self.assertEqual(res1["fp"], 0)
        self.assertEqual(res1["fn"], 0)
        self.assertEqual(res1["matched_pairs"], res2["matched_pairs"])
        self.assertEqual(
            res1["matched_pairs"],
            [
                {"issue_id": "ISS-001", "finding_index": 0},
                {"issue_id": "ISS-002", "finding_index": 1},
            ],
        )

    def test_21_ambiguous_single_issue_multiple_matching_findings_picks_lowest_index(self) -> None:
        iss = _make_issue(issue_id="ISS-001")
        f0 = _make_finding(explanation="first")
        f1 = _make_finding(explanation="second")
        res = match_findings([f0, f1], [iss])
        self.assertEqual(res["matched_pairs"], [{"issue_id": "ISS-001", "finding_index": 0}])
        self.assertEqual(res["unmatched_finding_indices"], [1])


if __name__ == "__main__":
    unittest.main()
