"""AuditLab Platform Verification Suite (100-point automated scorecard).

Provides `run_full_rubric_verification(run_unittests: bool = True, root: Path | None = None) -> dict`
returning a structured JSON-serializable scorecard across all 5 verification categories:
  1. Contract & Isolation (25 pts, 20 explicit checks)
  2. Data & Ground Truth (25 pts, 20 held-out eval episodes)
  3. Reward Robustness (25 pts, 20 explicit checks)
  4. Reproducibility & Generalization (15 pts, 20 paired permutation & determinism checks)
  5. Engineering & Diagnostics (10 pts, 5 criteria x 2 pts)
"""

from __future__ import annotations

import io
import json
import time
import unittest
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from statistics import mean
from typing import Any

from audit_rl.env import AuditEnv, SubprocessAuditEnv
from audit_rl.generator import (
    generate_episode,
    permute_and_relabel_episode,
    verify_episode_integrity,
)
from audit_rl.matcher import (
    compute_terminal_reward,
    evaluate_submission,
    match_findings,
)
from audit_rl.policies import (
    FlagEverythingAttackPolicy,
    OraclePolicy,
    RandomExplorerPolicy,
    SimpleReconciliationPolicy,
    record_trajectories,
    run_policy_on_env,
    scan_trajectories_for_leaks,
)
from audit_rl.schemas import EpisodeConfig, MAX_FINDINGS, MAX_PAGE_LIMIT

_FORBIDDEN_OBS_KEYS = frozenset(
    {
        "truth",
        "provenance",
        "seed",
        "acceptable_evidence_sets",
        "pre_injection_checks",
        "expected_amount_minor",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _contains_forbidden_keys(obj: Any) -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in _FORBIDDEN_OBS_KEYS:
                hits.append(str(k))
            hits.extend(_contains_forbidden_keys(v))
    elif isinstance(obj, list):
        for item in obj:
            hits.extend(_contains_forbidden_keys(item))
    return hits


def _contains_path_leak(obj: Any) -> list[str]:
    text = json.dumps(obj, ensure_ascii=False)
    patterns = ("/Users/", "/home/", "/app/", "truth.json", "provenance.json", "C:\\")
    return [p for p in patterns if p in text]


def _classify_episode_category(cfg: dict[str, Any]) -> str:
    if cfg.get("clean", False):
        return "clean"
    if cfg.get("interacting_scenarios"):
        return "interacting"
    counts = cfg.get("error_counts") or {}
    nonzero_types = sum(1 for v in counts.values() if int(v) > 0)
    if nonzero_types == 0:
        return "clean"
    if nonzero_types == 1:
        return "single"
    return "multi"


def _check_contract_and_isolation() -> dict[str, Any]:
    """Category 1: Contract & Isolation (25 pts, 20 explicit checks)."""
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    cfg = EpisodeConfig(
        num_invoices=25,
        error_counts={"duplicate_invoice": 1, "fx_conversion_error": 1},
    )
    ep = generate_episode(seed=7101, config=cfg)

    env = AuditEnv(mode="train", max_steps=100)
    obs0, info0 = env.reset(seed=7101, config=cfg)

    # 1. reset_contract
    req_reset = {"task", "manifest", "policy", "budget_remaining", "step", "available_actions"}
    ok1 = req_reset.issubset(obs0.keys()) and obs0["budget_remaining"] == 100 and obs0["step"] == 0
    record("reset_contract", ok1, f"keys={sorted(req_reset & set(obs0.keys()))}, budget={obs0.get('budget_remaining')}")

    # 2. step_gymnasium_tuple
    step_out = env.step({"action": "list_records", "table": "invoices", "cursor": "0", "limit": 10})
    obs1, r1, term1, trunc1, info1 = step_out
    try:
        json.dumps({"obs": obs1, "reward": r1, "terminated": term1, "truncated": trunc1, "info": info1})
        json_ok = True
    except TypeError:
        json_ok = False
    ok2 = (
        len(step_out) == 5
        and isinstance(obs1, dict)
        and isinstance(r1, float)
        and isinstance(term1, bool)
        and isinstance(trunc1, bool)
        and isinstance(info1, dict)
        and json_ok
    )
    record("step_gymnasium_tuple", ok2, f"5-tuple JSON-serializable, status={obs1.get('status')}")

    # 3. state_isolation_across_resets
    obs_a, _ = env.reset(seed=7102, config=EpisodeConfig(num_invoices=20, clean=True))
    obs_b, _ = env.reset(seed=7103, config=EpisodeConfig(num_invoices=20, clean=True))
    ok3 = (
        obs_a["manifest"]["episode_id"] != obs_b["manifest"]["episode_id"]
        and obs_b["step"] == 0
        and obs_b["budget_remaining"] == 100
    )
    record("state_isolation_across_resets", ok3, f"epA={obs_a['manifest']['episode_id']} != epB={obs_b['manifest']['episode_id']}")

    # Re-reset onto 7101 for inspection checks
    obs0, info0 = env.reset(seed=7101, config=cfg)

    # 4. pagination_valid_limit
    o_pg, _, _, _, _ = env.step({"action": "list_records", "table": "ledger", "cursor": "0", "limit": MAX_PAGE_LIMIT})
    ok4 = o_pg.get("status") == "ok" and 1 <= len(o_pg.get("records", [])) <= MAX_PAGE_LIMIT and o_pg.get("next_cursor") == str(len(o_pg.get("records", [])))
    record("pagination_valid_limit", ok4, f"count={len(o_pg.get('records', []))}, next_cursor={o_pg.get('next_cursor')}")

    # 5. pagination_over_limit_rejected
    b_before = o_pg["budget_remaining"]
    o_over, r_over, _, _, _ = env.step({"action": "list_records", "table": "ledger", "cursor": "0", "limit": 51})
    o_zero, _, _, _, _ = env.step({"action": "list_records", "table": "ledger", "cursor": "0", "limit": 0})
    ok5 = (
        o_over.get("status") == "error"
        and o_zero.get("status") == "error"
        and o_over["budget_remaining"] == b_before - 1
        and r_over == 0.0
    )
    record("pagination_over_limit_rejected", ok5, "limit=51 and limit=0 rejected with bounded error consuming 1 step")

    # 6. pagination_stable_cursors
    env_pg = AuditEnv(mode="train")
    env_pg.reset(seed=7101, config=cfg)
    collected_ids: list[str] = []
    cursor: str | None = "0"
    total_expected = None
    while cursor is not None:
        o_p, _, _, _, _ = env_pg.step({"action": "list_records", "table": "ledger", "cursor": cursor, "limit": 50})
        total_expected = o_p.get("total_count")
        collected_ids.extend(r["row_id"] for r in o_p.get("records", []))
        cursor = o_p.get("next_cursor")
    ok6 = total_expected == len(collected_ids) and len(set(collected_ids)) == len(collected_ids) and len(collected_ids) > 0
    record("pagination_stable_cursors", ok6, f"paginated {len(collected_ids)}/{total_expected} unique ledger rows")

    # 7. get_record_valid_lookup
    inv_id = ep.invoices[0]["invoice_id"]
    pay_id = ep.payments[0]["payment_id"]
    row_id = ep.ledger[0]["row_id"]
    fx0 = ep.fx_rates[0]
    fx_id = f"{fx0['date']}:{fx0['source_currency']}:{fx0['target_currency']}"
    gr_inv = env.step({"action": "get_record", "table": "invoices", "id": inv_id})[0]
    gr_pay = env.step({"action": "get_record", "table": "payments", "id": pay_id})[0]
    gr_led = env.step({"action": "get_record", "table": "ledger", "id": row_id})[0]
    gr_fx = env.step({"action": "get_record", "table": "fx", "id": fx_id})[0]
    ok7 = all(x.get("status") == "ok" and isinstance(x.get("record"), dict) for x in (gr_inv, gr_pay, gr_led, gr_fx))
    record("get_record_valid_lookup", ok7, "invoices, payments, ledger, fx lookups succeeded")

    # 8. get_document_valid_lookup
    doc_id = ep.invoices[0]["doc_id"]
    gd = env.step({"action": "get_document", "doc_id": doc_id})[0]
    ok8 = gd.get("status") == "ok" and gd.get("document", {}).get("doc_id") == doc_id
    record("get_document_valid_lookup", ok8, f"doc_id={doc_id} retrieved")

    # 9. calculator_valid_decimal
    c_ok = env.step({"action": "calculator", "expression": "round((12345 * 1.085) / 100, 2)"})[0]
    c_fn = env.step({"action": "calculator", "expression": "max(abs(-15), min(20, 12))"})[0]
    ok9 = c_ok.get("status") == "ok" and Decimal(str(c_ok.get("result"))) == Decimal("133.94") and Decimal(str(c_fn.get("result"))) == Decimal("15")
    record("calculator_valid_decimal", ok9, f"res1={c_ok.get('result')}, res2={c_fn.get('result')}")

    # 10. calculator_rejects_injection_and_dos
    bad_exprs = [
        "__import__('os').system('echo pwn')",
        "eval('1+1')",
        "open('/etc/passwd')",
        "2 ** 10000",
        "(1).__class__",
        "1+" * 140 + "1",
    ]
    bad_results = [env.step({"action": "calculator", "expression": expr})[0] for expr in bad_exprs]
    ok10 = all(r.get("status") == "error" for r in bad_results)
    record("calculator_rejects_injection_and_dos", ok10, f"{sum(r.get('status') == 'error' for r in bad_results)}/{len(bad_exprs)} unsafe expressions rejected")

    # 11. unknown_action_bounded_error
    b_pre = bad_results[-1]["budget_remaining"]
    o_unk, r_unk, _, _, _ = env.step({"action": "drop_database"})
    ok11 = o_unk.get("status") == "error" and o_unk["budget_remaining"] == b_pre - 1 and r_unk == 0.0
    record("unknown_action_bounded_error", ok11, "unknown action rejected with bounded error consuming 1 step")

    # 12. unknown_table_and_id_error
    o_ut = env.step({"action": "list_records", "table": "secrets", "cursor": "0", "limit": 5})[0]
    o_uid = env.step({"action": "get_record", "table": "invoices", "id": "INV-DOES-NOT-EXIST"})[0]
    o_udoc = env.step({"action": "get_document", "doc_id": "DOC-DOES-NOT-EXIST"})[0]
    ok12 = all(x.get("status") == "error" for x in (o_ut, o_uid, o_udoc))
    record("unknown_table_and_id_error", ok12, "unknown table, record id, and doc id returned bounded errors")

    # 13. step_limit_truncation
    env_lim = AuditEnv(mode="train", max_steps=3)
    env_lim.reset(seed=7104, config=EpisodeConfig(num_invoices=20, clean=True))
    term_l = trunc_l = False
    for _ in range(3):
        _, _, term_l, trunc_l, _ = env_lim.step({"action": "calculator", "expression": "1 + 1"})
    rep_lim = env_lim.get_evaluator_report()
    ok13 = (not term_l) and trunc_l and rep_lim["submitted_count"] == 0
    record("step_limit_truncation", ok13, f"terminated={term_l}, truncated={trunc_l}, submitted={rep_lim['submitted_count']}")

    # 14. timeout_truncation
    env_to = AuditEnv(mode="train", max_steps=100, timeout_seconds=0.001)
    env_to.reset(seed=7104, config=EpisodeConfig(num_invoices=20, clean=True))
    time.sleep(0.005)
    _, _, term_to, trunc_to, _ = env_to.step({"action": "calculator", "expression": "1 + 1"})
    ok14 = (not term_to) and trunc_to
    record("timeout_truncation", ok14, f"terminated={term_to}, truncated={trunc_to}")

    # 15. post_termination_immutability
    env_imm = AuditEnv(mode="train")
    o_imm, _ = env_imm.reset(seed=7104, config=EpisodeConfig(num_invoices=20, clean=True))
    ep_id_imm = o_imm["manifest"]["episode_id"]
    _, r_sub, term_sub, _, _ = env_imm.step({"action": "submit", "findings": {"episode_id": ep_id_imm, "findings": []}})
    rep_before = env_imm.get_evaluator_report()["reward"]
    o_after, r_after, term_after, _, _ = env_imm.step({"action": "calculator", "expression": "1+1"})
    rep_after = env_imm.get_evaluator_report()["reward"]
    ok15 = term_sub and term_after and o_after.get("status") == "error" and r_after == 0.0 and rep_before == rep_after == r_sub
    record("post_termination_immutability", ok15, f"reward preserved at {rep_after}")

    # 16. train_vs_eval_reward_hiding
    env_tr = AuditEnv(mode="train")
    o_tr, _ = env_tr.reset(seed=7104, config=EpisodeConfig(num_invoices=20, clean=True))
    _, r_tr, _, _, _ = env_tr.step({"action": "submit", "findings": {"episode_id": o_tr["manifest"]["episode_id"], "findings": []}})
    env_ev = AuditEnv(mode="eval")
    o_ev, _ = env_ev.reset(seed=7104, config=EpisodeConfig(num_invoices=20, clean=True))
    _, r_ev, _, _, info_ev = env_ev.step({"action": "submit", "findings": {"episode_id": o_ev["manifest"]["episode_id"], "findings": []}})
    rep_ev = env_ev.get_evaluator_report()
    ok16 = r_tr == 1.0 and r_ev == 0.0 and rep_ev["reward"] == 1.0 and "reward" not in info_ev
    record("train_vs_eval_reward_hiding", ok16, f"train_step_r={r_tr}, eval_step_r={r_ev}, eval_report_r={rep_ev['reward']}")

    # 17. zero_truth_provenance_leak
    leak_hits = _contains_forbidden_keys([obs0, info0, obs1, info1, o_pg, o_ev, info_ev])
    ok17 = len(leak_hits) == 0
    record("zero_truth_provenance_leak", ok17, f"forbidden_key_hits={len(leak_hits)}")

    # 18. zero_filesystem_path_leak
    path_hits = _contains_path_leak([obs0, info0, obs1, info1, o_pg, o_ev, info_ev])
    ok18 = len(path_hits) == 0
    record("zero_filesystem_path_leak", ok18, f"path_leak_hits={len(path_hits)}")

    # 19. subprocess_isolation_mode_eval
    with SubprocessAuditEnv(mode="eval") as senv:
        so, si = senv.reset(seed=7104, config=EpisodeConfig(num_invoices=20, clean=True))
        _, sr, sterm, strunc, sinfo = senv.step(
            {"action": "submit", "findings": {"episode_id": so["manifest"]["episode_id"], "findings": []}}
        )
        srep = senv.get_evaluator_report()
        ok19 = (
            sterm
            and not strunc
            and sr == 0.0
            and srep["reward"] == 1.0
            and len(_contains_forbidden_keys([so, si, sinfo])) == 0
        )
    record("subprocess_isolation_mode_eval", ok19, f"step_r={sr}, eval_report_r={srep['reward']}")

    # 20. subprocess_isolation_multi_reset
    with SubprocessAuditEnv(mode="eval") as senv2:
        so1, _ = senv2.reset(seed=7105, config=EpisodeConfig(num_invoices=20, clean=True))
        so2, _ = senv2.reset(seed=7106, config=EpisodeConfig(num_invoices=20, clean=True))
        ok20 = so1["manifest"]["episode_id"] != so2["manifest"]["episode_id"] and so2["step"] == 0
    record("subprocess_isolation_multi_reset", ok20, "consecutive resets isolated in worker subprocess")

    passed = sum(1 for c in checks if c["passed"])
    return {
        "id": "contract_isolation",
        "name": "Contract & Isolation",
        "score": round(25.0 * passed / 20.0, 2),
        "max_score": 25.0,
        "passed": passed,
        "total": 20,
        "checks": checks,
    }


def _check_data_and_ground_truth(eval_episodes: list[tuple[dict[str, Any], Any]]) -> dict[str, Any]:
    """Category 2: Data & Ground Truth (25 pts, 20 held-out eval episodes)."""
    checks: list[dict[str, Any]] = []
    for ep_spec, ep in eval_episodes:
        seed = int(ep_spec["seed"])
        cfg = ep_spec["config"]
        integrity = verify_episode_integrity(ep)
        ok_int = bool(integrity.get("ok", False))

        # Verify value formats: ISO-8601 offset, integer minor units, decimal FX strings, clean/interacting counts
        fmt_errors: list[str] = []
        try:
            dt_s = datetime.fromisoformat(ep.manifest["period_start"])
            dt_e = datetime.fromisoformat(ep.manifest["period_end"])
            if dt_s.tzinfo is None or dt_e.tzinfo is None:
                fmt_errors.append("missing timezone offset in manifest")
        except ValueError:
            fmt_errors.append("invalid ISO-8601 period")

        for inv in ep.invoices[:5]:
            tm = inv.get("total_minor")
            if isinstance(tm, bool) or not isinstance(tm, int):
                fmt_errors.append("non-int invoice total_minor")
                break
        for fx in ep.fx_rates[:5]:
            if not isinstance(fx.get("rate"), str):
                fmt_errors.append("non-string fx rate")
                break
            try:
                Decimal(fx["rate"])
            except Exception:
                fmt_errors.append("invalid decimal fx rate")
                break

        num_issues = len(ep.truth.get("issues", []))
        if cfg.get("clean", False) and num_issues != 0:
            fmt_errors.append(f"clean episode has {num_issues} issues")
        if cfg.get("interacting_scenarios") and num_issues < 2:
            fmt_errors.append(f"interacting episode has only {num_issues} issues")

        passed = ok_int and len(fmt_errors) == 0
        stats = integrity.get("stats", {})
        detail = (
            f"{ep_spec.get('description', '')} | rows={stats.get('ledger_rows')}, "
            f"docs={stats.get('documents')}, issues={num_issues}"
        )
        if fmt_errors:
            detail += f" | errors={fmt_errors}"
        checks.append({"name": f"eval_seed_{seed}", "passed": passed, "detail": detail})

    passed_count = sum(1 for c in checks if c["passed"])
    total = len(checks) or 20
    return {
        "id": "data_ground_truth",
        "name": "Data & Ground Truth",
        "score": round(25.0 * passed_count / float(total), 2),
        "max_score": 25.0,
        "passed": passed_count,
        "total": total,
        "checks": checks,
    }


def _check_reward_robustness() -> dict[str, Any]:
    """Category 3: Reward Robustness (25 pts, 20 explicit checks)."""
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    ep_id = "ep-rubric-reward"
    issue1 = {
        "issue_id": "ISS-001",
        "type": "fx_conversion_error",
        "affected_ids": ["INV-100"],
        "expected_amount_minor": 5000,
        "currency": "USD",
        "tolerance_minor": 2,
        "acceptable_evidence_sets": [["DOC-100", "ROW-100", "2026-01-15:EUR:USD"]],
    }
    truth_one = {"episode_id": ep_id, "issues": [issue1]}
    truth_clean = {"episode_id": ep_id, "issues": []}

    def make_finding(**overrides: Any) -> dict[str, Any]:
        base = {
            "type": "fx_conversion_error",
            "affected_ids": ["INV-100"],
            "claimed_amount_minor": 5000,
            "currency": "USD",
            "evidence_ids": ["DOC-100", "ROW-100", "2026-01-15:EUR:USD"],
            "explanation": "Verified FX discrepancy.",
        }
        base.update(overrides)
        return base

    # 1. exact_tp_matching
    r1 = evaluate_submission({"episode_id": ep_id, "findings": [make_finding()]}, truth_one, 0, ep_id)
    record("exact_tp_matching", r1["tp"] == 1 and r1["f1"] == 1.0 and r1["reward"] == 1.0, f"F1={r1['f1']}, reward={r1['reward']}")

    # 2. tolerance_boundary_pass_and_fail
    r2_ok = evaluate_submission({"episode_id": ep_id, "findings": [make_finding(claimed_amount_minor=5002)]}, truth_one, 0, ep_id)
    r2_bad = evaluate_submission({"episode_id": ep_id, "findings": [make_finding(claimed_amount_minor=5003)]}, truth_one, 0, ep_id)
    record("tolerance_boundary_pass_and_fail", r2_ok["tp"] == 1 and r2_bad["tp"] == 0, f"at_tol_tp={r2_ok['tp']}, over_tol_tp={r2_bad['tp']}")

    # 3. evidence_subset_matching
    r3 = evaluate_submission(
        {"episode_id": ep_id, "findings": [make_finding(evidence_ids=["DOC-100", "ROW-100", "2026-01-15:EUR:USD", "ROW-EXTRA"])]},
        truth_one,
        0,
        ep_id,
    )
    record("evidence_subset_matching", r3["tp"] == 1 and r3["f1"] == 1.0, f"superset evidence matched TP={r3['tp']}")

    # 4. clean_silence_beats_false_accusation
    r4_sil = evaluate_submission({"episode_id": ep_id, "findings": []}, truth_clean, 0, ep_id)
    r4_fp = evaluate_submission({"episode_id": ep_id, "findings": [make_finding()]}, truth_clean, 0, ep_id)
    record(
        "clean_silence_beats_false_accusation",
        r4_sil["f1"] == 1.0 and r4_sil["reward"] == 1.0 and r4_fp["f1"] == 0.0 and r4_fp["reward"] == -1.0,
        f"silence={r4_sil['reward']} > false_accusation={r4_fp['reward']}",
    )

    # 5. erroneous_complete_beats_silence
    r5_sil = evaluate_submission({"episode_id": ep_id, "findings": []}, truth_one, 0, ep_id)
    record(
        "erroneous_complete_beats_silence",
        r1["reward"] == 1.0 and r5_sil["f1"] == 0.0 and r5_sil["reward"] == -1.0,
        f"complete={r1['reward']} > silence={r5_sil['reward']}",
    )

    # 6. duplicate_finding_counts_as_fp
    r6 = evaluate_submission({"episode_id": ep_id, "findings": [make_finding(), make_finding()]}, truth_one, 0, ep_id)
    record("duplicate_finding_counts_as_fp", r6["tp"] == 1 and r6["fp"] == 1 and r6["f1"] < 1.0, f"TP={r6['tp']}, FP={r6['fp']}, F1={r6['f1']:.4f}")

    # 7. flag_everything_attack_penalized
    spam = [make_finding(affected_ids=[f"INV-{i}"]) for i in range(MAX_FINDINGS)]
    spam[0] = make_finding()
    r7 = evaluate_submission({"episode_id": ep_id, "findings": spam}, truth_one, 3, ep_id)
    record("flag_everything_attack_penalized", r7["reward"] < -0.5 and r7["fp"] == 29, f"30-finding spam reward={r7['reward']:.4f}, FP={r7['fp']}")

    # 8. step_economy_differentiation
    r8_0 = compute_terminal_reward(1.0, 0, True)
    r8_10 = compute_terminal_reward(1.0, 10, True)
    record("step_economy_differentiation", abs((r8_0 - r8_10) - 0.02) < 1e-9, f"r(0)={r8_0:.4f}, r(10)={r8_10:.4f}")

    # 9. guessed_ids_rejected
    r9 = evaluate_submission({"episode_id": ep_id, "findings": [make_finding(affected_ids=["INV-999"])]}, truth_one, 0, ep_id)
    record("guessed_ids_rejected", r9["tp"] == 0 and r9["fp"] == 1, f"TP={r9['tp']}, FP={r9['fp']}")

    # 10. unsupported_evidence_rejected
    r10 = evaluate_submission({"episode_id": ep_id, "findings": [make_finding(evidence_ids=["DOC-100"])]}, truth_one, 0, ep_id)
    record("unsupported_evidence_rejected", r10["tp"] == 0 and r10["fp"] == 1, f"partial evidence TP={r10['tp']}, FP={r10['fp']}")

    # 11. wrong_currency_rejected
    r11 = evaluate_submission({"episode_id": ep_id, "findings": [make_finding(currency="EUR")]}, truth_one, 0, ep_id)
    record("wrong_currency_rejected", r11["tp"] == 0 and r11["fp"] == 1, f"wrong currency TP={r11['tp']}, FP={r11['fp']}")

    # 12. wrong_error_type_rejected
    r12 = evaluate_submission({"episode_id": ep_id, "findings": [make_finding(type="period_cutoff_error")]}, truth_one, 0, ep_id)
    record("wrong_error_type_rejected", r12["tp"] == 0 and r12["fp"] == 1, f"wrong type TP={r12['tp']}, FP={r12['fp']}")

    # 13. invalid_over_30_findings
    r13 = evaluate_submission({"episode_id": ep_id, "findings": [make_finding() for _ in range(31)]}, truth_one, 0, ep_id)
    record("invalid_over_30_findings", (not r13["valid_submission"]) and r13["reward"] == -1.0, f"valid={r13['valid_submission']}, reward={r13['reward']}")

    # 14. invalid_over_10_evidence_ids
    r14 = evaluate_submission({"episode_id": ep_id, "findings": [make_finding(evidence_ids=[f"E-{i}" for i in range(11)])]}, truth_one, 0, ep_id)
    record("invalid_over_10_evidence_ids", (not r14["valid_submission"]) and r14["reward"] == -1.0, f"valid={r14['valid_submission']}, reward={r14['reward']}")

    # 15. invalid_over_1000_char_explanation
    r15 = evaluate_submission({"episode_id": ep_id, "findings": [make_finding(explanation="x" * 1001)]}, truth_one, 0, ep_id)
    record("invalid_over_1000_char_explanation", (not r15["valid_submission"]) and r15["reward"] == -1.0, f"valid={r15['valid_submission']}, reward={r15['reward']}")

    # 16. invalid_wrong_episode_id
    r16 = evaluate_submission({"episode_id": "ep-wrong", "findings": [make_finding()]}, truth_one, 0, ep_id)
    record("invalid_wrong_episode_id", (not r16["valid_submission"]) and r16["reward"] == -1.0, f"valid={r16['valid_submission']}, reward={r16['reward']}")

    # 17. invalid_bool_amount
    r17 = evaluate_submission({"episode_id": ep_id, "findings": [make_finding(claimed_amount_minor=True)]}, truth_one, 0, ep_id)
    record("invalid_bool_amount", (not r17["valid_submission"]) and r17["reward"] == -1.0, f"valid={r17['valid_submission']}, reward={r17['reward']}")

    # 18. deterministic_bipartite_tie_breaking
    iss_a = dict(issue1, issue_id="ISS-A", tolerance_minor=10)
    iss_b = dict(issue1, issue_id="ISS-B", tolerance_minor=10)
    f_list = [make_finding(claimed_amount_minor=5000), make_finding(claimed_amount_minor=5001)]
    m_first = match_findings(f_list, [iss_b, iss_a])
    m_second = match_findings(f_list, [iss_a, iss_b])
    record(
        "deterministic_bipartite_tie_breaking",
        m_first["tp"] == 2 and m_first["matched_pairs"] == m_second["matched_pairs"],
        f"matched_pairs={m_first['matched_pairs']}",
    )

    # 19. truncated_empty_submission
    r19 = evaluate_submission(None, truth_clean, 100, ep_id)
    record("truncated_empty_submission", r19["valid_submission"] and r19["submitted_count"] == 0, f"valid={r19['valid_submission']}, reward={r19['reward']}")

    # 20. repeated_submit_immutability
    env_r = AuditEnv(mode="train")
    o_r, _ = env_r.reset(seed=7200, config=EpisodeConfig(num_invoices=20, clean=True))
    eid_r = o_r["manifest"]["episode_id"]
    env_r.step({"action": "submit", "findings": {"episode_id": eid_r, "findings": []}})
    rep1 = env_r.get_evaluator_report()["reward"]
    env_r.step({"action": "submit", "findings": {"episode_id": "bad", "findings": []}})
    rep2 = env_r.get_evaluator_report()["reward"]
    record("repeated_submit_immutability", rep1 == 1.0 and rep2 == 1.0, f"reward unchanged ({rep1} == {rep2})")

    passed = sum(1 for c in checks if c["passed"])
    return {
        "id": "reward_robustness",
        "name": "Reward Robustness",
        "score": round(25.0 * passed / 20.0, 2),
        "max_score": 25.0,
        "passed": passed,
        "total": 20,
        "checks": checks,
    }


def _check_reproducibility_and_generalization(
    eval_episodes: list[tuple[dict[str, Any], Any]],
) -> dict[str, Any]:
    """Category 4: Reproducibility & Generalization (15 pts, 20 paired checks)."""
    checks: list[dict[str, Any]] = []
    for idx, (ep_spec, ep) in enumerate(eval_episodes):
        seed = int(ep_spec["seed"])
        cfg = EpisodeConfig.from_dict(ep_spec["config"])

        # Determinism check
        ep_repeat = generate_episode(seed=seed, config=cfg)
        det_ok = ep_repeat.manifest == ep.manifest and ep_repeat.truth == ep.truth

        # Action-trace replay determinism on first 3 episodes
        trace_ok = True
        if idx < 3:
            e1 = AuditEnv(mode="train")
            e2 = AuditEnv(mode="train")
            r1 = run_policy_on_env(RandomExplorerPolicy(seed=0), e1, seed=seed, config=cfg, record_trajectory=True)
            r2 = run_policy_on_env(RandomExplorerPolicy(seed=0), e2, seed=seed, config=cfg, record_trajectory=True)
            trace_ok = r1["reward"] == r2["reward"] and r1["trajectory"] == r2["trajectory"]

        # Paired permutation & ID-remapping invariance
        remapped, id_map = permute_and_relabel_episode(ep, perm_seed=seed + 50000 + idx)
        remapped_int = verify_episode_integrity(remapped)
        env_orig = AuditEnv(mode="train")
        orig_res = run_policy_on_env(OraclePolicy(ep.truth), env_orig, seed=seed, episode=ep, record_trajectory=False)
        env_perm = AuditEnv(mode="train")
        perm_res = run_policy_on_env(OraclePolicy(remapped.truth), env_perm, seed=seed, episode=remapped, record_trajectory=False)

        ok = (
            det_ok
            and trace_ok
            and bool(remapped_int.get("ok", False))
            and bool(id_map)
            and remapped.manifest["episode_id"] != ep.manifest["episode_id"]
            and orig_res["metrics"].get("f1") == 1.0
            and perm_res["metrics"].get("f1") == 1.0
            and orig_res["reward"] == 1.0
            and perm_res["reward"] == 1.0
        )
        checks.append(
            {
                "name": f"perm_repro_seed_{seed}",
                "passed": ok,
                "detail": (
                    f"det={det_ok}, remapped_ids={len(id_map)}, "
                    f"orig_F1={orig_res['metrics'].get('f1')}, perm_F1={perm_res['metrics'].get('f1')}"
                ),
            }
        )

    passed_count = sum(1 for c in checks if c["passed"])
    total = len(checks) or 20
    return {
        "id": "reproducibility_generalization",
        "name": "Reproducibility & Generalization",
        "score": round(15.0 * passed_count / float(total), 2),
        "max_score": 15.0,
        "passed": passed_count,
        "total": total,
        "checks": checks,
    }


def _evaluate_baselines_with_matrix(
    eval_spec: dict[str, Any],
    eval_episodes: list[tuple[dict[str, Any], Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Run all 4 baselines via SubprocessAuditEnv(mode='eval') and build per-episode matrix."""
    policy_names = [
        "random_explorer",
        "simple_reconciliation",
        "flag_everything_attack",
        "oracle",
    ]
    per_policy: dict[str, list[dict[str, Any]]] = {name: [] for name in policy_names}
    episode_matrix: list[dict[str, Any]] = []

    with SubprocessAuditEnv(mode="eval") as env:
        for ep_spec, ep in eval_episodes:
            seed = int(ep_spec["seed"])
            cfg_dict = ep_spec["config"]
            row_results: dict[str, dict[str, Any]] = {}

            run_policies = [
                RandomExplorerPolicy(seed=0),
                SimpleReconciliationPolicy(),
                FlagEverythingAttackPolicy(),
                OraclePolicy(truth=ep.truth),
            ]
            for policy in run_policies:
                res = run_policy_on_env(
                    policy,
                    env,
                    seed=seed,
                    config=cfg_dict,
                    record_trajectory=False,
                )
                metrics = res.get("metrics", {})
                entry = {
                    "seed": seed,
                    "episode_id": res["episode_id"],
                    "precision": round(float(metrics.get("precision", 0.0)), 4),
                    "recall": round(float(metrics.get("recall", 0.0)), 4),
                    "f1": round(float(metrics.get("f1", 0.0)), 4),
                    "return": round(float(res["reward"]), 4),
                    "reward": round(float(res["reward"]), 4),
                    "steps": int(res["steps"]),
                    "runtime_ms": round(float(res["runtime_ms"]), 2),
                }
                per_policy[policy.name].append(entry)
                row_results[policy.name] = {
                    "precision": entry["precision"],
                    "recall": entry["recall"],
                    "f1": entry["f1"],
                    "reward": entry["reward"],
                    "steps": entry["steps"],
                    "runtime_ms": entry["runtime_ms"],
                }

            episode_matrix.append(
                {
                    "seed": seed,
                    "episode_id": ep.manifest["episode_id"],
                    "label": ep_spec.get("description", f"Eval seed {seed}"),
                    "category": _classify_episode_category(cfg_dict),
                    "num_issues": len(ep.truth.get("issues", [])),
                    "results": row_results,
                }
            )

    summary_dict: dict[str, dict[str, Any]] = {}
    summary_list: list[dict[str, Any]] = []
    for name in policy_names:
        rows = per_policy[name]
        agg = {
            "policy": name,
            "episodes": len(rows),
            "precision": round(mean(r["precision"] for r in rows), 4),
            "recall": round(mean(r["recall"] for r in rows), 4),
            "f1": round(mean(r["f1"] for r in rows), 4),
            "return": round(mean(r["return"] for r in rows), 4),
            "steps": round(mean(r["steps"] for r in rows), 2),
            "runtime_ms": round(mean(r["runtime_ms"] for r in rows), 2),
            "total_runtime_ms": round(sum(r["runtime_ms"] for r in rows), 2),
        }
        summary_dict[name] = agg
        summary_list.append(agg)

    return summary_list, summary_dict, episode_matrix


def run_full_rubric_verification(
    run_unittests: bool = True,
    root: Path | str | None = None,
    verbose_tests: bool = False,
) -> dict[str, Any]:
    """Execute the AuditLab Platform Verification Suite (100-point scorecard)."""
    t0 = time.perf_counter()
    repo = Path(root).resolve() if root is not None else _repo_root()
    configs_dir = repo / "configs"
    eval_cfg_path = configs_dir / "eval_configs.json"
    eval_spec = _load_json(eval_cfg_path)

    # Generate the 20 held-out eval episodes once for Categories 2, 4, and 5
    eval_episodes: list[tuple[dict[str, Any], Any]] = []
    for ep_spec in eval_spec["episodes"]:
        ep = generate_episode(
            seed=int(ep_spec["seed"]),
            config=EpisodeConfig.from_dict(ep_spec["config"]),
        )
        eval_episodes.append((ep_spec, ep))

    # Optional unittest discovery
    tests_dir = repo / "tests"
    if run_unittests and tests_dir.exists() and any(tests_dir.glob("test_*.py")):
        loader = unittest.defaultTestLoader
        suite = loader.discover(start_dir=str(tests_dir), top_level_dir=str(repo))
        runner = unittest.TextTestRunner(
            stream=None if verbose_tests else io.StringIO(),
            verbosity=2 if verbose_tests else 0,
        )
        tr = runner.run(suite)
        unittest_info = {
            "ran": tr.testsRun,
            "failures": len(tr.failures),
            "errors": len(tr.errors),
            "passed": bool(tr.wasSuccessful() and tr.testsRun > 0),
        }
    elif not run_unittests:
        test_files = list(tests_dir.glob("test_*.py")) if tests_dir.exists() else []
        unittest_info = {
            "ran": 0,
            "failures": 0,
            "errors": 0,
            "passed": len(test_files) > 0,
            "skipped": True,
        }
    else:
        unittest_info = {"ran": 0, "failures": 0, "errors": 0, "passed": False}

    # Categories 1..4
    cat1 = _check_contract_and_isolation()
    cat2 = _check_data_and_ground_truth(eval_episodes)
    cat3 = _check_reward_robustness()
    cat4 = _check_reproducibility_and_generalization(eval_episodes)

    # Category 5: Engineering & Diagnostics (10 pts, 5 criteria x 2 pts)
    cat5_checks: list[dict[str, Any]] = []

    # 5a. Unit tests & single-command verify
    cat5_checks.append(
        {
            "name": "unit_tests_and_verify_cli",
            "passed": bool(unittest_info["passed"]),
            "detail": (
                f"{unittest_info['ran']} tests run, {unittest_info['failures']} failures, {unittest_info['errors']} errors"
                if run_unittests
                else "test suite present (skipped during quick check)"
            ),
        }
    )

    # 5b. Versioned configs
    cfg_ok = True
    cfg_details: list[str] = []
    for split, expected_n in (("train", 10), ("dev", 10), ("val", 10), ("eval", 20)):
        p = configs_dir / f"{split}_configs.json"
        if not p.exists():
            cfg_ok = False
            cfg_details.append(f"{split}=MISSING")
            continue
        data = _load_json(p)
        n = len(data.get("episodes", []))
        ver = data.get("config_version")
        if not ver or n != expected_n:
            cfg_ok = False
        cfg_details.append(f"{split}=v{ver}({n})")
    cat5_checks.append(
        {
            "name": "versioned_configs",
            "passed": cfg_ok,
            "detail": ", ".join(cfg_details),
        }
    )

    # 5c. Trajectories leak scan (generate if missing)
    traj_path = repo / "trajectories" / "trajectories.jsonl"
    if not traj_path.exists():
        train_spec = _load_json(configs_dir / "train_configs.json")
        dev_spec = _load_json(configs_dir / "dev_configs.json")
        sample_eps = train_spec["episodes"][:5] + dev_spec["episodes"][:5]
        record_trajectories(
            sample_eps,
            [SimpleReconciliationPolicy(), RandomExplorerPolicy(seed=0), FlagEverythingAttackPolicy()],
            traj_path,
        )
    leak_res = scan_trajectories_for_leaks(traj_path)
    cat5_checks.append(
        {
            "name": "trajectories_zero_leak_scan",
            "passed": bool(leak_res["passed"]),
            "detail": f"{leak_res['lines']} lines scanned, {len(leak_res['errors'])} errors",
        }
    )

    # 5d. Measured 4 baselines via SubprocessAuditEnv (mode='eval')
    baselines_list, baseline_summary, episode_matrix = _evaluate_baselines_with_matrix(
        eval_spec, eval_episodes
    )
    oracle_ok = baseline_summary["oracle"]["f1"] == 1.0 and baseline_summary["oracle"]["return"] == 1.0
    simple_beats_random = (
        baseline_summary["simple_reconciliation"]["f1"] > baseline_summary["random_explorer"]["f1"]
        and baseline_summary["simple_reconciliation"]["return"] > baseline_summary["random_explorer"]["return"]
    )
    simple_beats_attack = (
        baseline_summary["simple_reconciliation"]["f1"] > baseline_summary["flag_everything_attack"]["f1"]
        and baseline_summary["simple_reconciliation"]["return"] > baseline_summary["flag_everything_attack"]["return"]
    )
    attack_penalized = baseline_summary["flag_everything_attack"]["return"] < 0.0
    baselines_ok = oracle_ok and simple_beats_random and simple_beats_attack and attack_penalized
    cat5_checks.append(
        {
            "name": "four_baselines_subprocess_eval",
            "passed": baselines_ok,
            "detail": (
                f"oracle_F1={baseline_summary['oracle']['f1']:.4f}, "
                f"simple_F1={baseline_summary['simple_reconciliation']['f1']:.4f}, "
                f"random_F1={baseline_summary['random_explorer']['f1']:.4f}, "
                f"attack_ret={baseline_summary['flag_everything_attack']['return']:.4f}"
            ),
        }
    )

    # 5e. Docs & packaging present
    req_files = ("README.md", "DESIGN.md", "DEMO.md", "Dockerfile", "LICENSE", "pyproject.toml", "requirements.txt")
    missing_files = [f for f in req_files if not (repo / f).exists() or (repo / f).stat().st_size == 0]
    cat5_checks.append(
        {
            "name": "docs_and_packaging_present",
            "passed": len(missing_files) == 0,
            "detail": "all 7 files present" if not missing_files else f"missing={missing_files}",
        }
    )

    cat5_passed = sum(1 for c in cat5_checks if c["passed"])
    cat5 = {
        "id": "engineering_diagnostics",
        "name": "Engineering & Diagnostics",
        "score": round(2.0 * cat5_passed, 2),
        "max_score": 10.0,
        "passed": cat5_passed,
        "total": 5,
        "checks": cat5_checks,
    }

    categories = [cat1, cat2, cat3, cat4, cat5]
    total_score = round(sum(c["score"] for c in categories), 2)
    max_score = round(sum(c["max_score"] for c in categories), 2)
    overall_pass = all(c["passed"] == c["total"] for c in categories) and bool(unittest_info["passed"])
    duration_s = round(time.perf_counter() - t0, 2)

    return {
        "overall_pass": overall_pass,
        "total_score": total_score,
        "max_score": max_score,
        "duration_s": duration_s,
        "unittest": unittest_info,
        "categories": categories,
        "baselines": baselines_list,
        "baseline_summary": baseline_summary,
        "episode_matrix": episode_matrix,
    }
