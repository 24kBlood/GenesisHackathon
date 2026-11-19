import unittest

from genesis.evaluation.metrics import Metrics, calculate_metrics
from genesis.evaluation.reward import score_terminal


class MetricsTests(unittest.TestCase):
    def test_precision_recall_and_f1_from_counts(self) -> None:
        self.assertEqual(calculate_metrics(tp=2, fp=1, fn=2), Metrics(2 / 3, 1 / 2, 4 / 7))

    def test_zero_denominators_and_clean_episode(self) -> None:
        self.assertEqual(calculate_metrics(tp=0, fp=0, fn=0), Metrics(0.0, 0.0, 1.0))
        self.assertEqual(calculate_metrics(tp=0, fp=0, fn=2), Metrics(0.0, 0.0, 0.0))
        self.assertEqual(calculate_metrics(tp=0, fp=3, fn=0), Metrics(0.0, 0.0, 0.0))

    def test_counts_must_be_non_negative_integers(self) -> None:
        for counts in ((-1, 0, 0), (0, -1, 0), (0, 0, -1)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                calculate_metrics(*counts)
        for counts in ((True, 0, 0), (0, 1.5, 0), (0, 0, "2")):
            with self.subTest(counts=counts), self.assertRaises(TypeError):
                calculate_metrics(*counts)


class TerminalRewardTests(unittest.TestCase):
    def test_correct_answer_beats_silence_on_anomalous_books(self) -> None:
        correct = score_terminal(tp=2, fp=0, fn=0, inspection_steps=0)
        silence = score_terminal(tp=0, fp=0, fn=2, inspection_steps=0)
        self.assertEqual(correct.reward, 1.0)
        self.assertEqual(silence.reward, -1.0)

    def test_silence_beats_false_allegations_on_clean_books(self) -> None:
        silence = score_terminal(tp=0, fp=0, fn=0, inspection_steps=5)
        false_allegations = score_terminal(tp=0, fp=2, fn=0, inspection_steps=5)
        self.assertAlmostEqual(silence.reward, 0.99)
        self.assertEqual(false_allegations.reward, -1.0)

    def test_duplicate_and_label_all_reduce_score(self) -> None:
        correct = score_terminal(tp=1, fp=0, fn=0, inspection_steps=10)
        duplicate = score_terminal(tp=1, fp=1, fn=0, inspection_steps=10)
        label_all = score_terminal(tp=1, fp=29, fn=1, inspection_steps=10)
        self.assertAlmostEqual(correct.reward, 0.98)
        self.assertAlmostEqual(duplicate.reward, 2 * (2 / 3) - 1 - 0.02)
        self.assertLess(duplicate.reward, correct.reward)
        self.assertLess(label_all.reward, -0.8)

    def test_equal_answers_differ_by_step_cost_until_clipped(self) -> None:
        earlier = score_terminal(tp=1, fp=1, fn=0, inspection_steps=98)
        later = score_terminal(tp=1, fp=1, fn=0, inspection_steps=99)
        self.assertAlmostEqual(earlier.reward - later.reward, 0.002)
        self.assertEqual(later.inspection_steps, 99)

    def test_truncation_scores_empty_answer_at_actual_steps(self) -> None:
        clean = score_terminal(tp=0, fp=0, fn=0, inspection_steps=100)
        anomalous = score_terminal(tp=0, fp=0, fn=2, inspection_steps=100)
        self.assertEqual(clean.inspection_steps, 100)
        self.assertEqual(clean.reward, 0.8)
        self.assertEqual(anomalous.reward, -1.0)

    def test_invalid_structure_overrides_f1(self) -> None:
        clean = score_terminal(tp=0, fp=0, fn=0, inspection_steps=3, invalid_submission=True)
        anomalous = score_terminal(tp=0, fp=0, fn=2, inspection_steps=3, invalid_submission=True)
        self.assertEqual(clean.metrics, Metrics(0.0, 0.0, 1.0))
        self.assertEqual(anomalous.metrics, Metrics(0.0, 0.0, 0.0))
        self.assertEqual(clean.reward, -1.0)
        self.assertEqual(anomalous.reward, -1.0)

    def test_invalid_structure_requires_empty_answer_counts(self) -> None:
        for tp, fp in ((1, 0), (0, 1)):
            with self.subTest(tp=tp, fp=fp), self.assertRaises(ValueError):
                score_terminal(tp, fp, fn=0, inspection_steps=3, invalid_submission=True)

    def test_inspection_steps_must_be_integers_within_budget(self) -> None:
        for steps in (-1, 101):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                score_terminal(tp=0, fp=0, fn=0, inspection_steps=steps)
        for steps in (True, 1.5, "3"):
            with self.subTest(steps=steps), self.assertRaises(TypeError):
                score_terminal(tp=0, fp=0, fn=0, inspection_steps=steps)
