"""Точность, полнота и F1 по результатам сопоставления."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    f1: float


def calculate_metrics(tp: int, fp: int, fn: int) -> Metrics:
    """Вычислить метрики по целочисленным неотрицательным счётчикам."""
    for name, count in (("tp", tp), ("fp", fp), ("fn", fn)):
        if type(count) is not int:
            raise TypeError(f"{name} must be an integer")
        if count < 0:
            raise ValueError(f"{name} must be non-negative")

    submitted = tp + fp
    expected = tp + fn
    denominator = submitted + expected
    return Metrics(
        precision=tp / submitted if submitted else 0.0,
        recall=tp / expected if expected else 0.0,
        f1=2 * tp / denominator if denominator else 1.0,
    )
