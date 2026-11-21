"""Standard policies that use only the public action and observation protocol."""

from .baseline import MarkAllPolicy, RandomPolicy
from .protocol import Policy
from .reconcile import FxReconciliationPolicy


def make_policy(name: str, *, seed: int = 0) -> Policy:
    if name == "random":
        return RandomPolicy(seed)
    if name == "simple_reconciliation":
        return FxReconciliationPolicy()
    if name == "mark_all":
        return MarkAllPolicy()
    raise ValueError("unknown public policy")
