"""Baseline and oracle policies for the Audit RL environment.

Implements four mandatory policies:
1. RandomExplorerPolicy: deterministic random exploration within budget.
2. SimpleReconciliationPolicy: honest rule-based auditor using only the narrow tool API.
3. FlagEverythingAttackPolicy: adversarial spam policy testing reward robustness.
4. OraclePolicy: evaluator-only oracle using out-of-band truth to verify reward ceiling (1.0).
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from audit_rl.schemas import (
    CURRENCY_EXPONENTS,
    ERROR_TYPES,
    MAX_EVIDENCE_PER_FINDING,
    MAX_EXPLANATION_CHARS,
    MAX_FINDINGS,
    MAX_PAGE_LIMIT,
)

FORBIDDEN_TRAJECTORY_KEYS = frozenset(
    {
        "truth",
        "provenance",
        "seed",
        "acceptable_evidence_sets",
        "pre_injection_checks",
        "injected_errors",
        "expected_amount_minor",
        "truth_issues",
    }
)


def _stable_seed_offset(policy_seed: int, episode_id: str) -> int:
    digest = hashlib.sha256(f"{policy_seed}:{episode_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _extract_items(obs: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract paginated record items from a list_records observation."""
    for key in ("records", "items", "rows", "data"):
        val = obs.get(key)
        if isinstance(val, list):
            return val
    result = obs.get("result")
    if isinstance(result, dict):
        for key in ("records", "items", "rows", "data"):
            val = result.get(key)
            if isinstance(val, list):
                return val
    return []


def _extract_next_cursor(obs: dict[str, Any]) -> str | None:
    if "next_cursor" in obs:
        return obs["next_cursor"]
    result = obs.get("result")
    if isinstance(result, dict) and "next_cursor" in result:
        return result["next_cursor"]
    return None


def _parse_iso_dt(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _convert_minor(
    amount_minor: int,
    source_currency: str,
    target_currency: str,
    rate_str: str,
    rounding_rule: str = "ROUND_HALF_EVEN",
) -> int:
    if source_currency == target_currency:
        return amount_minor
    src_exp = CURRENCY_EXPONENTS.get(source_currency, 2)
    tgt_exp = CURRENCY_EXPONENTS.get(target_currency, 2)
    rounding = ROUND_HALF_UP if rounding_rule == "ROUND_HALF_UP" else ROUND_HALF_EVEN
    major = Decimal(amount_minor) / (Decimal(10) ** src_exp)
    converted = major * Decimal(rate_str)
    minor = (converted * (Decimal(10) ** tgt_exp)).quantize(Decimal("1"), rounding=rounding)
    return int(minor)


class BasePolicy:
    """Base interface for Audit RL policies."""

    name: str = "base"

    def reset(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> None:
        """Reset per-episode internal state from the initial observation."""
        self.episode_id: str = str(obs.get("manifest", {}).get("episode_id", ""))
        self.manifest: dict[str, Any] = dict(obs.get("manifest", {}))
        self.policy_doc: dict[str, Any] = dict(obs.get("policy", {}))

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Choose the next JSON-compatible tool action."""
        raise NotImplementedError

    def run_episode(
        self,
        env: Any,
        obs: dict[str, Any],
        info: dict[str, Any] | None = None,
        record_trajectory: bool = False,
    ) -> dict[str, Any]:
        """Run this policy to completion on an already-reset environment."""
        start_time = time.perf_counter()
        self.reset(obs, info or {})
        episode_id = self.episode_id
        trajectory: list[dict[str, Any]] = []
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            action = self.act(obs)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            steps += 1
            total_reward += float(reward)
            if record_trajectory:
                trajectory.append(
                    {
                        "episode_id": episode_id,
                        "policy": self.name,
                        "step": steps,
                        "action": action,
                        "observation": next_obs,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
            obs = next_obs
            if steps > 500:
                break

        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        report = env.get_evaluator_report() if hasattr(env, "get_evaluator_report") else {}
        return {
            "episode_id": episode_id,
            "policy": self.name,
            "steps": steps,
            "inspection_steps": int(report.get("inspection_steps", max(0, steps - 1))),
            "terminated": terminated,
            "truncated": truncated,
            "reward": float(report.get("reward", total_reward)),
            "metrics": dict(report.get("metrics", {})),
            "report": report,
            "runtime_ms": runtime_ms,
            "trajectory": trajectory,
        }


class RandomExplorerPolicy(BasePolicy):
    """Deterministic random exploration policy that uses all tool types and submits within budget."""

    name: str = "random_explorer"

    def __init__(self, seed: int = 0, min_explore_steps: int = 6, max_explore_steps: int = 16) -> None:
        self.seed = int(seed)
        self.min_explore_steps = min_explore_steps
        self.max_explore_steps = max_explore_steps
        self._rng = random.Random(self.seed)
        self._target_steps = 8
        self._steps_taken = 0
        self._seen_invoices: list[dict[str, Any]] = []
        self._seen_payments: list[dict[str, Any]] = []
        self._seen_ledger: list[dict[str, Any]] = []
        self._seen_doc_ids: list[str] = []
        self._last_table: str | None = None

    def reset(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> None:
        super().reset(obs, info)
        self._rng = random.Random(_stable_seed_offset(self.seed, self.episode_id))
        budget = int(obs.get("budget_remaining", 100))
        hi = max(1, min(self.max_explore_steps, budget - 1))
        lo = max(1, min(self.min_explore_steps, hi))
        self._target_steps = self._rng.randint(lo, hi)
        self._steps_taken = 0
        self._seen_invoices = []
        self._seen_payments = []
        self._seen_ledger = []
        self._seen_doc_ids = []
        self._last_table = None

    def _ingest_obs(self, obs: dict[str, Any]) -> None:
        items = _extract_items(obs)
        if not items or not self._last_table:
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            if self._last_table == "invoices":
                self._seen_invoices.append(item)
                if item.get("doc_id"):
                    self._seen_doc_ids.append(str(item["doc_id"]))
            elif self._last_table == "payments":
                self._seen_payments.append(item)
                if item.get("doc_id"):
                    self._seen_doc_ids.append(str(item["doc_id"]))
            elif self._last_table == "ledger":
                self._seen_ledger.append(item)

    def _build_random_submission(self) -> dict[str, Any]:
        findings: list[dict[str, Any]] = []
        num_guesses = self._rng.choice([0, 0, 1, 2])
        for _ in range(num_guesses):
            err_type = self._rng.choice(ERROR_TYPES)
            if err_type == "payment_allocation_mismatch" and self._seen_payments:
                pay = self._rng.choice(self._seen_payments)
                findings.append(
                    {
                        "type": err_type,
                        "affected_ids": [str(pay.get("payment_id", "PAY-UNKNOWN"))],
                        "claimed_amount_minor": int(abs(pay.get("amount_minor", 100))),
                        "currency": str(pay.get("currency", self.manifest.get("base_currency", "USD"))),
                        "evidence_ids": [str(pay["doc_id"])] if pay.get("doc_id") else [],
                        "explanation": "Random explorer guess on payment allocation.",
                    }
                )
            elif self._seen_invoices:
                inv = self._rng.choice(self._seen_invoices)
                affected = [str(inv.get("invoice_id", "INV-UNKNOWN"))]
                if err_type == "duplicate_invoice" and len(self._seen_invoices) >= 2:
                    inv2 = self._rng.choice(self._seen_invoices)
                    if inv2.get("invoice_id") != inv.get("invoice_id"):
                        affected = sorted([str(inv["invoice_id"]), str(inv2["invoice_id"])])
                findings.append(
                    {
                        "type": err_type,
                        "affected_ids": affected,
                        "claimed_amount_minor": int(abs(inv.get("total_minor", 100))),
                        "currency": str(inv.get("currency", self.manifest.get("base_currency", "USD"))),
                        "evidence_ids": [str(inv["doc_id"])] if inv.get("doc_id") else [],
                        "explanation": f"Random explorer guess for {err_type}.",
                    }
                )
        return {
            "action": "submit",
            "findings": {
                "episode_id": self.episode_id,
                "findings": findings[:MAX_FINDINGS],
            },
        }

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._ingest_obs(obs)
        budget = int(obs.get("budget_remaining", 1))
        if self._steps_taken >= self._target_steps or budget <= 1:
            self._last_table = None
            return self._build_random_submission()

        self._steps_taken += 1
        choices = ["list_records", "calculator"]
        if self._seen_invoices or self._seen_payments or self._seen_ledger:
            choices.append("get_record")
        if self._seen_doc_ids:
            choices.append("get_document")

        kind = self._rng.choice(choices)
        if kind == "list_records":
            table = self._rng.choice(["invoices", "payments", "ledger", "fx"])
            self._last_table = table
            return {
                "action": "list_records",
                "table": table,
                "cursor": "0",
                "limit": self._rng.randint(10, MAX_PAGE_LIMIT),
            }
        if kind == "get_document" and self._seen_doc_ids:
            self._last_table = None
            return {
                "action": "get_document",
                "doc_id": self._rng.choice(self._seen_doc_ids),
            }
        if kind == "get_record":
            self._last_table = None
            if self._seen_invoices:
                inv = self._rng.choice(self._seen_invoices)
                return {"action": "get_record", "table": "invoices", "id": inv["invoice_id"]}
            if self._seen_payments:
                pay = self._rng.choice(self._seen_payments)
                return {"action": "get_record", "table": "payments", "id": pay["payment_id"]}
            if self._seen_ledger:
                row = self._rng.choice(self._seen_ledger)
                return {"action": "get_record", "table": "ledger", "id": row["row_id"]}

        self._last_table = None
        a = self._rng.randint(100, 9999)
        b = self._rng.randint(1, 500)
        return {"action": "calculator", "expression": f"round(({a} - {b}) / 100, 2)"}


class FlagEverythingAttackPolicy(BasePolicy):
    """Adversarial policy that flags up to 30 records across error types to test reward gaming resistance."""

    name: str = "flag_everything_attack"

    def __init__(self) -> None:
        self._phase = 0
        self._invoices: list[dict[str, Any]] = []
        self._payments: list[dict[str, Any]] = []
        self._ledger: list[dict[str, Any]] = []

    def reset(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> None:
        super().reset(obs, info)
        self._phase = 0
        self._invoices = []
        self._payments = []
        self._ledger = []

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        budget = int(obs.get("budget_remaining", 1))
        if self._phase == 1:
            self._invoices.extend(_extract_items(obs))
        elif self._phase == 2:
            self._payments.extend(_extract_items(obs))
        elif self._phase == 3:
            self._ledger.extend(_extract_items(obs))

        if budget <= 1 or self._phase >= 3:
            return self._build_spam_submission()

        self._phase += 1
        table = ["invoices", "payments", "ledger"][self._phase - 1]
        return {
            "action": "list_records",
            "table": table,
            "cursor": "0",
            "limit": MAX_PAGE_LIMIT,
        }

    def _build_spam_submission(self) -> dict[str, Any]:
        findings: list[dict[str, Any]] = []
        base_currency = str(self.manifest.get("base_currency", "USD"))
        ledger_by_inv: dict[str, list[str]] = {}
        ledger_by_pay: dict[str, list[str]] = {}
        for row in self._ledger:
            if row.get("invoice_id") and row.get("row_id"):
                ledger_by_inv.setdefault(str(row["invoice_id"]), []).append(str(row["row_id"]))
            if row.get("payment_id") and row.get("row_id"):
                ledger_by_pay.setdefault(str(row["payment_id"]), []).append(str(row["row_id"]))

        for inv in self._invoices:
            if len(findings) >= MAX_FINDINGS:
                break
            inv_id = str(inv.get("invoice_id", ""))
            doc_id = str(inv.get("doc_id", ""))
            ev = [x for x in [doc_id] + ledger_by_inv.get(inv_id, [])[:2] if x]
            for err_type in ("fx_conversion_error", "period_cutoff_error"):
                if len(findings) >= MAX_FINDINGS:
                    break
                findings.append(
                    {
                        "type": err_type,
                        "affected_ids": [inv_id],
                        "claimed_amount_minor": int(abs(inv.get("total_minor", 100))),
                        "currency": base_currency if err_type == "fx_conversion_error" else str(inv.get("currency", base_currency)),
                        "evidence_ids": ev[:MAX_EVIDENCE_PER_FINDING],
                        "explanation": f"Adversarial blanket flag for {err_type} on {inv_id}.",
                    }
                )

        for pay in self._payments:
            if len(findings) >= MAX_FINDINGS:
                break
            pay_id = str(pay.get("payment_id", ""))
            doc_id = str(pay.get("doc_id", ""))
            ev = [x for x in [doc_id] + ledger_by_pay.get(pay_id, [])[:2] if x]
            findings.append(
                {
                    "type": "payment_allocation_mismatch",
                    "affected_ids": [pay_id],
                    "claimed_amount_minor": int(abs(pay.get("amount_minor", 100))),
                    "currency": str(pay.get("currency", base_currency)),
                    "evidence_ids": ev[:MAX_EVIDENCE_PER_FINDING],
                    "explanation": f"Adversarial blanket flag on payment {pay_id}.",
                }
            )

        return {
            "action": "submit",
            "findings": {
                "episode_id": self.episode_id,
                "findings": findings[:MAX_FINDINGS],
            },
        }


class SimpleReconciliationPolicy(BasePolicy):
    """Honest rule-based auditor using ONLY the narrow tool interface.

    Paginates invoices, payments, ledger, and fx within the 100-step budget,
    reconciles all four error classes, and submits structured findings with evidence.
    Never accesses truth, provenance, or internal environment state.
    """

    name: str = "simple_reconciliation"

    def __init__(self, use_calculator: bool = True) -> None:
        self.use_calculator = use_calculator
        self._tables_order = ["invoices", "payments", "ledger", "fx"]
        self._table_idx = 0
        self._cursor: str | None = "0"
        self._active_table: str | None = None
        self._data: dict[str, list[dict[str, Any]]] = {}
        self._calc_done = False
        self._pending_findings: list[dict[str, Any]] | None = None

    def reset(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> None:
        super().reset(obs, info)
        self._table_idx = 0
        self._cursor = "0"
        self._active_table = None
        self._data = {t: [] for t in self._tables_order}
        self._calc_done = False
        self._pending_findings = None

    def _ingest_page(self, obs: dict[str, Any]) -> None:
        if self._active_table is None:
            return
        items = _extract_items(obs)
        self._data[self._active_table].extend(items)
        next_cursor = _extract_next_cursor(obs)
        if next_cursor is not None and len(items) > 0:
            self._cursor = str(next_cursor)
        else:
            self._table_idx += 1
            self._cursor = "0"
            self._active_table = None

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._active_table is not None:
            self._ingest_page(obs)

        budget = int(obs.get("budget_remaining", 1))
        reserve = 2 if self.use_calculator and not self._calc_done else 1

        if self._table_idx < len(self._tables_order) and budget > reserve:
            table = self._tables_order[self._table_idx]
            self._active_table = table
            return {
                "action": "list_records",
                "table": table,
                "cursor": self._cursor or "0",
                "limit": MAX_PAGE_LIMIT,
            }

        self._active_table = None
        if self._pending_findings is None:
            self._pending_findings = self._reconcile()

        if self.use_calculator and not self._calc_done and budget > 1 and self._pending_findings:
            self._calc_done = True
            sample = self._pending_findings[0]
            amt = int(sample["claimed_amount_minor"])
            return {
                "action": "calculator",
                "expression": f"abs({amt})",
            }

        return {
            "action": "submit",
            "findings": {
                "episode_id": self.episode_id,
                "findings": (self._pending_findings or [])[:MAX_FINDINGS],
            },
        }

    def _reconcile(self) -> list[dict[str, Any]]:
        invoices = self._data["invoices"]
        payments = self._data["payments"]
        ledger = self._data["ledger"]
        fx_rows = self._data["fx"]

        base_currency = str(self.manifest.get("base_currency") or self.policy_doc.get("base_currency", "USD"))
        fx_date_policy = str(self.policy_doc.get("fx_date_policy", "invoice_issue_date"))
        rounding_rule = str(self.policy_doc.get("rounding_rule", "ROUND_HALF_EVEN"))
        period_start_str = str(self.manifest.get("period_start", ""))
        period_end_str = str(self.manifest.get("period_end", ""))
        period_start_dt = _parse_iso_dt(period_start_str) if period_start_str else None
        period_end_dt = _parse_iso_dt(period_end_str) if period_end_str else None

        inv_by_id: dict[str, dict[str, Any]] = {
            str(inv["invoice_id"]): inv for inv in invoices if inv.get("invoice_id")
        }

        inv_ledger_rows: dict[str, list[dict[str, Any]]] = {}
        pay_ledger_rows: dict[str, list[dict[str, Any]]] = {}
        for row in ledger:
            inv_id = row.get("invoice_id")
            pay_id = row.get("payment_id")
            if pay_id:
                pay_ledger_rows.setdefault(str(pay_id), []).append(row)
            elif inv_id:
                inv_ledger_rows.setdefault(str(inv_id), []).append(row)

        # Build FX lookup: (date, source, target) -> (rate_str, fx_key)
        fx_map: dict[tuple[str, str, str], tuple[str, str]] = {}
        fx_dates_by_pair: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
        for row in fx_rows:
            d = str(row.get("date", ""))
            src = str(row.get("source_currency", ""))
            tgt = str(row.get("target_currency", ""))
            rate = str(row.get("rate", ""))
            if d and src and tgt and rate:
                fx_key = f"{d}:{src}:{tgt}"
                fx_map[(d, src, tgt)] = (rate, fx_key)
                fx_dates_by_pair.setdefault((src, tgt), []).append((d, rate, fx_key))
        for pair in fx_dates_by_pair:
            fx_dates_by_pair[pair].sort(key=lambda x: x[0])

        def lookup_fx(dt_str: str, src: str, tgt: str) -> tuple[str, str] | None:
            if (dt_str, src, tgt) in fx_map:
                return fx_map[(dt_str, src, tgt)]
            candidates = fx_dates_by_pair.get((src, tgt), [])
            best: tuple[str, str] | None = None
            for d, rate, key in candidates:
                if d <= dt_str:
                    best = (rate, key)
                else:
                    break
            return best

        # Map invoice_id -> earliest payment timestamp (for payment_date FX policy)
        inv_payment_dates: dict[str, str] = {}
        for pay in payments:
            refs = pay.get("invoice_refs")
            if isinstance(refs, list):
                p_date = str(pay.get("timestamp", ""))[:10]
                for ref in refs:
                    iid = ref.get("invoice_id")
                    if iid and p_date:
                        if str(iid) not in inv_payment_dates or p_date < inv_payment_dates[str(iid)]:
                            inv_payment_dates[str(iid)] = p_date

        findings: list[dict[str, Any]] = []

        # 1. Duplicate invoice detection
        # Group by (entity_id, currency, total_minor, normalized line_items signature including service_date).
        # Legitimate recurring invoices have different service_date values so they will NOT collide.
        dup_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for inv in invoices:
            total_minor = int(inv.get("total_minor", 0))
            if total_minor <= 0:
                continue
            items = inv.get("line_items") or []
            item_sig = tuple(
                (
                    str(it.get("service_date", "")),
                    str(it.get("description", "")),
                    int(it.get("amount_minor", 0)),
                )
                for it in items
            )
            key = (
                str(inv.get("entity_id", "")),
                str(inv.get("currency", "")),
                total_minor,
                item_sig,
            )
            dup_groups.setdefault(key, []).append(inv)

        for group in dup_groups.values():
            if len(group) < 2:
                continue
            group_sorted = sorted(group, key=lambda x: (str(x.get("issued_at", "")), str(x.get("invoice_id", ""))))
            orig = group_sorted[0]
            for dup in group_sorted[1:]:
                orig_id = str(orig["invoice_id"])
                dup_id = str(dup["invoice_id"])
                ev_ids: list[str] = []
                for did in (orig.get("doc_id"), dup.get("doc_id")):
                    if did and str(did) not in ev_ids:
                        ev_ids.append(str(did))
                for r in inv_ledger_rows.get(orig_id, [])[:2]:
                    if r.get("row_id") and str(r["row_id"]) not in ev_ids:
                        ev_ids.append(str(r["row_id"]))
                for r in inv_ledger_rows.get(dup_id, [])[:2]:
                    if r.get("row_id") and str(r["row_id"]) not in ev_ids:
                        ev_ids.append(str(r["row_id"]))
                findings.append(
                    {
                        "type": "duplicate_invoice",
                        "affected_ids": sorted([orig_id, dup_id]),
                        "claimed_amount_minor": int(dup["total_minor"]),
                        "currency": str(dup["currency"]),
                        "evidence_ids": ev_ids[:MAX_EVIDENCE_PER_FINDING],
                        "explanation": (
                            f"Observed: {orig_id} ({orig.get('doc_id')}) and {dup_id} ({dup.get('doc_id')}) "
                            f"share entity {dup.get('entity_id')}, line items, and total {dup['total_minor']} {dup['currency']}. "
                            f"Calculated: duplicate overstatement = {int(dup['total_minor'])} {dup['currency']}. "
                            f"Rule: single-posting uniqueness per invoice service event. Strength: confirmed."
                        )[:MAX_EXPLANATION_CHARS],
                    }
                )

        # 2. Payment allocation mismatch detection
        for pay in payments:
            pay_id = str(pay.get("payment_id", ""))
            pay_doc_id = str(pay.get("doc_id", ""))
            amount_minor = int(pay.get("amount_minor", 0))
            currency = str(pay.get("currency", base_currency))
            refs = pay.get("invoice_refs")
            pay_rows = pay_ledger_rows.get(pay_id, [])

            if refs is None or (isinstance(refs, list) and len(refs) == 0):
                if amount_minor > 0:
                    ev_ids = [pay_doc_id] if pay_doc_id else []
                    for r in pay_rows[:4]:
                        if r.get("row_id") and str(r["row_id"]) not in ev_ids:
                            ev_ids.append(str(r["row_id"]))
                    findings.append(
                        {
                            "type": "payment_allocation_mismatch",
                            "affected_ids": [pay_id],
                            "claimed_amount_minor": abs(amount_minor),
                            "currency": currency,
                            "evidence_ids": ev_ids[:MAX_EVIDENCE_PER_FINDING],
                            "explanation": (
                                f"Observed: payment {pay_id} ({pay_doc_id}) amount={amount_minor} {currency} has null/empty invoice_refs. "
                                f"Calculated: unallocated = |{amount_minor} - 0| = {abs(amount_minor)} {currency}. "
                                f"Rule: every cash settlement must reconcile to allocated invoice references. Strength: confirmed."
                            )[:MAX_EXPLANATION_CHARS],
                        }
                    )
            elif isinstance(refs, list):
                alloc_sum = sum(int(ref.get("allocated_minor", 0)) for ref in refs)
                if alloc_sum != amount_minor:
                    diff = abs(amount_minor - alloc_sum)
                    ref_inv_ids = [str(ref["invoice_id"]) for ref in refs if ref.get("invoice_id")]
                    affected = sorted(set([pay_id] + ref_inv_ids))
                    ev_ids = [pay_doc_id] if pay_doc_id else []
                    for iid in ref_inv_ids:
                        inv_obj = inv_by_id.get(iid)
                        if inv_obj and inv_obj.get("doc_id") and str(inv_obj["doc_id"]) not in ev_ids:
                            ev_ids.append(str(inv_obj["doc_id"]))
                    for r in pay_rows[:4]:
                        if r.get("row_id") and str(r["row_id"]) not in ev_ids:
                            ev_ids.append(str(r["row_id"]))
                    findings.append(
                        {
                            "type": "payment_allocation_mismatch",
                            "affected_ids": affected,
                            "claimed_amount_minor": diff,
                            "currency": currency,
                            "evidence_ids": ev_ids[:MAX_EVIDENCE_PER_FINDING],
                            "explanation": (
                                f"Observed: payment {pay_id} ({pay_doc_id}) amount={amount_minor} {currency}, "
                                f"allocated refs={ref_inv_ids} sum={alloc_sum}. "
                                f"Calculated: |{amount_minor} - {alloc_sum}| = {diff} {currency}. "
                                f"Rule: sum(invoice_refs.allocated_minor) must equal payment amount_minor. Strength: confirmed."
                            )[:MAX_EXPLANATION_CHARS],
                        }
                    )

        # 3. FX conversion error detection
        for inv in invoices:
            inv_id = str(inv.get("invoice_id", ""))
            inv_cur = str(inv.get("currency", base_currency))
            total_minor = int(inv.get("total_minor", 0))
            if inv_cur == base_currency or total_minor <= 0:
                continue
            rows = inv_ledger_rows.get(inv_id, [])
            if not rows:
                continue
            # Identify posted base-currency debit amount for the invoice entry
            base_debits = [
                (r, int(r.get("debit_minor", 0)))
                for r in rows
                if str(r.get("currency", "")) == base_currency and int(r.get("debit_minor", 0)) > 0
            ]
            if not base_debits:
                continue
            primary_row, posted_base_minor = max(base_debits, key=lambda pair: pair[1])

            if fx_date_policy == "posting_date":
                fx_date = str(primary_row.get("timestamp", ""))[:10]
            elif fx_date_policy == "payment_date":
                fx_date = inv_payment_dates.get(inv_id) or str(inv.get("issued_at", ""))[:10]
            else:
                fx_date = str(inv.get("issued_at", ""))[:10]

            fx_hit = lookup_fx(fx_date, inv_cur, base_currency)
            if fx_hit is None:
                continue
            rate_str, fx_key = fx_hit
            expected_base_minor = _convert_minor(
                total_minor, inv_cur, base_currency, rate_str, rounding_rule
            )
            diff = abs(posted_base_minor - expected_base_minor)
            if diff > 1:
                ev_ids = []
                if inv.get("doc_id"):
                    ev_ids.append(str(inv["doc_id"]))
                if primary_row.get("row_id"):
                    ev_ids.append(str(primary_row["row_id"]))
                ev_ids.append(fx_key)
                findings.append(
                    {
                        "type": "fx_conversion_error",
                        "affected_ids": [inv_id],
                        "claimed_amount_minor": diff,
                        "currency": base_currency,
                        "evidence_ids": ev_ids[:MAX_EVIDENCE_PER_FINDING],
                        "explanation": (
                            f"Observed: invoice {inv_id} ({total_minor} {inv_cur}) posted in {primary_row.get('row_id')} as "
                            f"{posted_base_minor} {base_currency}; FX {fx_key} rate={rate_str} ({rounding_rule}). "
                            f"Calculated: expected={expected_base_minor}, |{posted_base_minor} - {expected_base_minor}| = {diff} {base_currency}. "
                            f"Rule: foreign invoices must convert at policy fx_date_policy={fx_date_policy}. Strength: confirmed."
                        )[:MAX_EXPLANATION_CHARS],
                    }
                )

        # 4. Period cutoff error detection
        if period_start_dt is not None and period_end_dt is not None:
            p_start_date = period_start_dt.date()
            p_end_date = period_end_dt.date()
            for inv in invoices:
                inv_id = str(inv.get("invoice_id", ""))
                total_minor = int(inv.get("total_minor", 0))
                if total_minor <= 0:
                    continue
                items = inv.get("line_items") or []
                if not items:
                    continue
                service_dates: list[date] = []
                for it in items:
                    sd = it.get("service_date")
                    if sd:
                        try:
                            service_dates.append(date.fromisoformat(str(sd)[:10]))
                        except ValueError:
                            pass
                if not service_dates:
                    continue
                service_outside = any(sd < p_start_date or sd > p_end_date for sd in service_dates)
                rows = inv_ledger_rows.get(inv_id, [])
                if not rows:
                    continue
                in_period_rows = []
                out_period_rows = []
                for r in rows:
                    ts = r.get("timestamp")
                    if not ts:
                        continue
                    try:
                        rdt = _parse_iso_dt(str(ts))
                    except ValueError:
                        continue
                    if period_start_dt <= rdt <= period_end_dt:
                        in_period_rows.append(r)
                    else:
                        out_period_rows.append(r)

                cutoff_row: dict[str, Any] | None = None
                if service_outside and in_period_rows:
                    cutoff_row = in_period_rows[0]
                elif (not service_outside) and out_period_rows and not in_period_rows:
                    cutoff_row = out_period_rows[0]

                if cutoff_row is not None:
                    ev_ids = []
                    if inv.get("doc_id"):
                        ev_ids.append(str(inv["doc_id"]))
                    if cutoff_row.get("row_id"):
                        ev_ids.append(str(cutoff_row["row_id"]))
                    for r in rows[:2]:
                        if r.get("row_id") and str(r["row_id"]) not in ev_ids:
                            ev_ids.append(str(r["row_id"]))
                    findings.append(
                        {
                            "type": "period_cutoff_error",
                            "affected_ids": [inv_id],
                            "claimed_amount_minor": total_minor,
                            "currency": str(inv.get("currency", base_currency)),
                            "evidence_ids": ev_ids[:MAX_EVIDENCE_PER_FINDING],
                            "explanation": (
                                f"Observed: invoice {inv_id} ({inv.get('doc_id')}) service_date={service_dates[0].isoformat()} "
                                f"posted in {cutoff_row.get('row_id')} at {cutoff_row.get('timestamp')}. "
                                f"Calculated: period window [{p_start_date.isoformat()}, {p_end_date.isoformat()}], misstated={total_minor} {inv.get('currency', base_currency)}. "
                                f"Rule: accrual cutoff requires recognition in the service period. Strength: confirmed."
                            )[:MAX_EXPLANATION_CHARS],
                        }
                    )

        # Deduplicate identical findings deterministically
        seen_keys: set[tuple[Any, ...]] = set()
        unique_findings: list[dict[str, Any]] = []
        for f in findings:
            fkey = (
                f["type"],
                tuple(sorted(f["affected_ids"])),
                int(f["claimed_amount_minor"]),
                f["currency"],
            )
            if fkey not in seen_keys:
                seen_keys.add(fkey)
                unique_findings.append(f)

        return unique_findings[:MAX_FINDINGS]


DEFAULT_UNTRAINED_PARAMS: dict[str, Any] = {
    "check_duplicate": False,
    "check_payment": False,
    "check_fx": False,
    "check_cutoff": False,
    "fx_tolerance_threshold": 50000,
    "submit_confidence_threshold": 0.95,
    "max_pages_per_table": 1,
    "use_calculator": False,
}


class LearnableAuditPolicy(SimpleReconciliationPolicy):
    """Deterministic parameterized RL audit policy optimized strictly from terminal rewards.

    Starts with uncalibrated conservative parameters (low untrained F1/return) and is tuned
    exclusively on training episodes using terminal reward signals without accessing truth.
    """

    name: str = "learnable_audit"

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        version: str = "untrained-0.1.0",
    ) -> None:
        merged = dict(DEFAULT_UNTRAINED_PARAMS)
        if params:
            merged.update(params)
        self.params: dict[str, Any] = merged
        self.version: str = str(version)
        super().__init__(use_calculator=bool(self.params.get("use_calculator", False)))
        self._pages_fetched: dict[str, int] = {}

    def get_params(self) -> dict[str, Any]:
        return dict(self.params)

    def set_params(self, params: dict[str, Any], version: str | None = None) -> None:
        self.params = dict(params)
        self.use_calculator = bool(self.params.get("use_calculator", False))
        if version is not None:
            self.version = str(version)

    def reset(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> None:
        self.use_calculator = bool(self.params.get("use_calculator", False))
        super().reset(obs, info)
        self._pages_fetched = {t: 0 for t in self._tables_order}

    def _ingest_page(self, obs: dict[str, Any]) -> None:
        if self._active_table is None:
            return
        table = self._active_table
        items = _extract_items(obs)
        self._data[table].extend(items)
        self._pages_fetched[table] = self._pages_fetched.get(table, 0) + 1
        max_pages = max(1, int(self.params.get("max_pages_per_table", 1)))
        next_cursor = _extract_next_cursor(obs)
        if next_cursor is not None and len(items) > 0 and self._pages_fetched[table] < max_pages:
            self._cursor = str(next_cursor)
        else:
            self._table_idx += 1
            self._cursor = "0"
            self._active_table = None

    def _reconcile(self) -> list[dict[str, Any]]:
        candidates = super()._reconcile()
        conf_threshold = float(self.params.get("submit_confidence_threshold", 0.95))
        fx_tol = int(self.params.get("fx_tolerance_threshold", 50000))
        enabled_map = {
            "duplicate_invoice": bool(self.params.get("check_duplicate", False)),
            "payment_allocation_mismatch": bool(self.params.get("check_payment", False)),
            "fx_conversion_error": bool(self.params.get("check_fx", False)),
            "period_cutoff_error": bool(self.params.get("check_cutoff", False)),
        }
        filtered: list[dict[str, Any]] = []
        for cand in candidates:
            ftype = str(cand.get("type", ""))
            if not enabled_map.get(ftype, False):
                continue
            if ftype == "fx_conversion_error" and int(cand.get("claimed_amount_minor", 0)) <= fx_tol:
                continue
            # Confirmed deterministic open-record detections carry 0.85 confidence
            confidence = 0.85
            if confidence < conf_threshold:
                continue
            filtered.append(cand)
        return filtered[:MAX_FINDINGS]


class OraclePolicy(BasePolicy):
    """Evaluator-only oracle that receives out-of-band truth to verify the 1.0 reward ceiling."""

    name: str = "oracle"

    def __init__(self, truth: dict[str, Any]) -> None:
        self._truth = truth

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        episode_id = str(obs.get("manifest", {}).get("episode_id", self.episode_id))
        canonical_findings: list[dict[str, Any]] = []
        for issue in self._truth.get("issues", [])[:MAX_FINDINGS]:
            ev_sets = issue.get("acceptable_evidence_sets") or []
            evidence = list(ev_sets[0])[:MAX_EVIDENCE_PER_FINDING] if ev_sets else []
            canonical_findings.append(
                {
                    "type": str(issue["type"]),
                    "affected_ids": list(issue["affected_ids"]),
                    "claimed_amount_minor": int(issue["expected_amount_minor"]),
                    "currency": str(issue["currency"]),
                    "evidence_ids": evidence,
                    "explanation": str(
                        issue.get("description") or f"Oracle canonical finding for {issue.get('issue_id', 'issue')}."
                    )[:MAX_EXPLANATION_CHARS],
                }
            )
        return {
            "action": "submit",
            "findings": {
                "episode_id": episode_id,
                "findings": canonical_findings,
            },
        }


def run_policy_on_env(
    policy: BasePolicy,
    env: Any,
    seed: int,
    config: Any = None,
    episode: Any = None,
    record_trajectory: bool = True,
) -> dict[str, Any]:
    """Reset env on (seed, config[, episode]) and run policy to completion."""
    if episode is not None:
        obs, info = env.reset(seed=seed, config=config, episode=episode)
    else:
        obs, info = env.reset(seed=seed, config=config)
    return policy.run_episode(env, obs, info, record_trajectory=record_trajectory)


def _find_leak_paths(obj: Any, prefix: str = "") -> list[str]:
    leaks: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key_str = str(k)
            path = f"{prefix}.{key_str}" if prefix else key_str
            if key_str in FORBIDDEN_TRAJECTORY_KEYS:
                leaks.append(path)
            leaks.extend(_find_leak_paths(v, path))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            leaks.extend(_find_leak_paths(item, f"{prefix}[{idx}]"))
    return leaks


def scan_trajectories_for_leaks(path: str | Path) -> dict[str, Any]:
    """Verify every JSONL line in trajectories has required keys and zero truth/seed leaks."""
    required_keys = {
        "episode_id",
        "policy",
        "step",
        "action",
        "observation",
        "reward",
        "terminated",
        "truncated",
    }
    p = Path(path)
    if not p.exists():
        return {"passed": False, "lines": 0, "errors": [f"File not found: {p}"]}

    errors: list[str] = []
    line_count = 0
    with p.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw:
                continue
            line_count += 1
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {idx}: invalid JSON ({exc})")
                continue
            missing = required_keys - set(rec.keys())
            if missing:
                errors.append(f"Line {idx}: missing keys {sorted(missing)}")
            leaks = _find_leak_paths(rec)
            if leaks:
                errors.append(f"Line {idx}: forbidden keys leaked: {leaks[:5]}")
            if len(errors) >= 20:
                break

    if line_count == 0:
        errors.append("Trajectory file is empty.")

    return {"passed": len(errors) == 0, "lines": line_count, "errors": errors}


def record_trajectories(
    episodes_spec: list[dict[str, Any]],
    policies: list[BasePolicy],
    output_path: str | Path,
    env_factory: Any = None,
) -> dict[str, Any]:
    """Run policies on episodes_spec and write sanitized JSONL trajectories."""
    from audit_rl.env import AuditEnv

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    total_steps = 0
    total_runs = 0

    with out.open("w", encoding="utf-8") as fh:
        for spec in episodes_spec:
            seed = int(spec["seed"])
            cfg = spec.get("config")
            for policy in policies:
                env = env_factory() if env_factory is not None else AuditEnv(mode="train")
                try:
                    result = run_policy_on_env(policy, env, seed=seed, config=cfg, record_trajectory=True)
                finally:
                    if hasattr(env, "close"):
                        env.close()
                total_runs += 1
                for step_rec in result["trajectory"]:
                    fh.write(json.dumps(step_rec, ensure_ascii=False) + "\n")
                    total_steps += 1

    return {"output_path": str(out), "runs": total_runs, "steps": total_steps}
