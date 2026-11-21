"""Fixed evaluator checks E01–E20 on fictitious findings and books."""

from itertools import permutations

from genesis.evaluation.matcher import MatchResult, match
from genesis.evaluation.reward import score_terminal

from test_contract_matrix import FixtureCase
from test_matcher import PUBLIC_IDS, finding, issue


class EvaluationMatrix(FixtureCase):
    def counts(self, result: MatchResult, expected):
        self.assertEqual((result.tp, result.fp, result.fn), expected)

    def score(self, findings, *, truth=None, tolerance=None, steps=0):
        truth = [issue()] if truth is None else truth
        tolerances = None if tolerance is None else {"fx_conversion": tolerance}
        result = match(findings, truth, PUBLIC_IDS, tolerances_minor=tolerances)
        return result, score_terminal(result.tp, result.fp, result.fn, steps)

    def test_E01_complete_finding_matches_exactly_once(self):
        result, score = self.score([finding()])
        self.counts(result, (1, 0, 0))
        self.assertEqual((score.metrics.f1, score.reward), (1, 1))

    def test_E02_empty_anomalous_submission_misses_the_finding(self):
        result, score = self.score([])
        self.counts(result, (0, 0, 1))
        self.assertLess(score.reward, self.score([finding()])[1].reward)

    def test_E03_empty_clean_book_metrics(self):
        result, score = self.score([], truth=[])
        self.counts(result, (0, 0, 0))
        self.assertEqual((score.metrics.precision, score.metrics.recall, score.metrics.f1), (0, 0, 1))

    def test_E04_false_clean_book_finding_loses_to_empty(self):
        result, score = self.score([finding()], truth=[])
        self.counts(result, (0, 1, 0))
        self.assertEqual(score.metrics.f1, 0)
        self.assertLess(score.reward, self.score([], truth=[])[1].reward)

    def test_E05_wrong_error_type_is_unmatched(self):
        result, _ = self.score([{**finding(), "type": "period_cutoff"}])
        self.counts(result, (0, 1, 1))

    def test_E06_inexact_affected_id_sets_are_unmatched(self):
        for affected in (
            ("invoice-a", "guessed-id"),
            ("invoice-a", "invoice-b", "extra-id"),
            ("invoice-a",),
        ):
            with self.subTest(affected=affected):
                self.counts(self.score([finding(affected=affected)])[0], (0, 1, 1))

    def test_E07_id_permutation_preserves_canonical_match(self):
        normal, _ = self.score([finding(affected=("invoice-a", "invoice-b"))])
        reverse, _ = self.score([finding(affected=("invoice-b", "invoice-a"))])
        self.counts(normal, (1, 0, 0))
        self.assertEqual(normal, reverse)

    def test_E08_wrong_currency_is_unmatched_at_same_amount(self):
        result, _ = self.score([{**finding(), "currency": "USD"}])
        self.counts(result, (0, 1, 1))

    def test_E09_declared_tolerance_includes_both_boundaries(self):
        for amount in (98, 102):
            self.counts(self.score([finding(amount)], tolerance=2)[0], (1, 0, 0))

    def test_E10_boundary_excess_and_zero_tolerance_are_rejected(self):
        for amount, tolerance in ((97, 2), (103, 2), (101, 0), (101, None)):
            self.counts(self.score([finding(amount)], tolerance=tolerance)[0], (0, 1, 1))

    def test_E11_alternative_evidence_sets_work_individually(self):
        for evidence in (("ev-a", "ev-b"), ("ev-c",)):
            self.counts(self.score([finding(evidence=evidence)])[0], (1, 0, 0))

    def test_E12_incomplete_known_evidence_is_unmatched(self):
        self.counts(self.score([finding(evidence=("ev-a",))])[0], (0, 1, 1))

    def test_E13_unknown_evidence_invalidates_an_otherwise_complete_set(self):
        self.counts(self.score([finding(evidence=("ev-a", "ev-b", "unknown"))])[0], (0, 1, 1))

    def test_E14_duplicate_and_paraphrase_add_false_positives(self):
        claim = finding()
        repeated, score = self.score([claim, claim, {**claim, "explanation": "Same amounts and proof"}])
        self.counts(repeated, (1, 2, 0))
        self.assertLess(score.reward, self.score([claim])[1].reward)

    def test_E15_partial_evidence_is_not_merged_across_duplicates(self):
        claims = [finding(evidence=("ev-a",)), finding(evidence=("ev-b",))]
        self.counts(self.score(claims)[0], (0, 2, 1))

    def test_E16_competing_candidates_get_deterministic_maximum_matching(self):
        claims, truth = [finding(100), finding(101)], [issue("issue-a", 100), issue("issue-b", 99)]
        assignments = set()
        for candidates in permutations(claims):
            for references in permutations(truth):
                result, _ = self.score(candidates, truth=references, tolerance=1)
                self.counts(result, (2, 0, 0))
                assignments.add(frozenset(
                    (item.issue_id, candidates[item.finding_index]["claimed_amount_minor"]) for item in result.matches
                ))
        self.assertEqual(assignments, {frozenset({("issue-a", 101), ("issue-b", 100)})})

    def test_E18_structural_limits_and_inclusive_controls(self):
        baseline = self.fixture.findings["findings"][0]
        for field, accepted, rejected in (
            ("findings", [baseline] * 30, [baseline] * 31),
            ("evidence_ids", ["policy"] * 10, ["policy"] * 11),
            ("explanation", "a" * 1000, "a" * 1001),
        ):
            for value, valid in ((accepted, True), (rejected, False)):
                with self.subTest(field=field, valid=valid):
                    claims = value if field == "findings" else [{**baseline, field: value}]
                    self.reset()
                    _, reward, terminated, truncated, _ = self.submit(claims)
                    self.assertEqual((terminated, truncated), (True, False))
                    self.assertEqual(self.env.terminal_result.invalid_submission, not valid)
                    if not valid:
                        self.assertEqual(reward, -1)

    def test_E19_equal_results_differ_by_one_inspection_cost(self):
        before, after = (self.score([finding()], steps=steps)[1] for steps in (10, 11))
        self.assertEqual(before.metrics, after.metrics)
        self.assertAlmostEqual(before.reward - after.reward, 0.002)

    def test_E20_step_exhaustion_scores_empty_at_actual_step_count(self):
        for clean in (True, False):
            with self.subTest(clean=clean):
                self.reset(clean=clean)
                self.steps(100)
                terminal = self.env.terminal_result
                self.assertEqual(terminal.submission["findings"], [])
                self.assertEqual((terminal.truncated, terminal.score.inspection_steps), (True, 100))
                self.counts(terminal.matching, (0, 0, 0 if clean else 1))
                self.assertEqual(terminal.score.reward, 0.8 if clean else -1.0)
