"""Детерминированная терминальная оценка FinTech."""

from dataclasses import dataclass

from .metrics import Metrics, calculate_metrics


@dataclass(frozen=True)
class TerminalScore:
    metrics: Metrics
    reward: float
    inspection_steps: int


def score_terminal(
    tp: int,
    fp: int,
    fn: int,
    inspection_steps: int,
    *,
    invalid_submission: bool = False,
) -> TerminalScore:
    """Вычислить итоговую награду, равную возврату эпизода.

    При усечении и неверной структуре ответ пуст: tp=fp=0, fn — размер
    эталона. Неверная структура всегда даёт F1=0 и награду -1.
    Среда передаёт число фактических шагов исследования:
    до 99 при submit и до 100 при усечении.
    """
    if type(inspection_steps) is not int:
        raise TypeError("inspection_steps must be an integer")
    if not 0 <= inspection_steps <= 100:
        raise ValueError("inspection_steps must be between 0 and 100")

    metrics = calculate_metrics(tp, fp, fn)
    if invalid_submission:
        if tp or fp:
            raise ValueError("invalid submission must be scored as an empty answer")
        metrics = Metrics(0.0, 0.0, 0.0)
        reward = -1.0
    else:
        reward = max(-1.0, min(1.0, 2 * metrics.f1 - 1 - 0.002 * inspection_steps))

    return TerminalScore(metrics, reward, inspection_steps)
