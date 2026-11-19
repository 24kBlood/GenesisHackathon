"""Zero-dependency offline local Web Application for AuditLab.

AuditLab — An autonomous financial audit simulation and AI training environment.

Serves an 8-page FinTech research & RL training dashboard over Python stdlib
``http.server.ThreadingHTTPServer`` with JSON API endpoints backed by
``AuditEnv``, ``SubprocessAuditEnv``, ``safe_calculate``, ``audit_rl.rubric``,
``audit_rl.storage``, and ``audit_rl.trainer``.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from audit_rl.calculator import CalculatorError, safe_calculate
from audit_rl.env import AuditEnv, SubprocessAuditEnv
from audit_rl.generator import generate_episode
from audit_rl.policies import (
    FlagEverythingAttackPolicy,
    OraclePolicy,
    RandomExplorerPolicy,
    SimpleReconciliationPolicy,
)
from audit_rl.schemas import (
    CURRENCY_EXPONENTS,
    DEFAULT_MAX_STEPS,
    ERROR_TYPES,
    EpisodeConfig,
    INTERACTING_SCENARIOS,
    MAX_EVIDENCE_PER_FINDING,
    MAX_EXPLANATION_CHARS,
    MAX_FINDINGS,
    MAX_PAGE_LIMIT,
    SCHEMA_VERSION,
)

_VALID_SPLITS = ("train", "dev", "val", "eval")
_VALID_POLICIES = (
    "simple_reconciliation",
    "random_explorer",
    "flag_everything_attack",
    "oracle",
)
_VALID_TABLES = ("invoices", "payments", "ledger", "fx")
_MAX_BODY_BYTES = 2 * 1024 * 1024


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _classify_episode_category(cfg: dict[str, Any]) -> str:
    if cfg.get("clean"):
        return "clean"
    if cfg.get("interacting_scenarios"):
        return "interacting"
    err_counts = cfg.get("error_counts") or {}
    active_types = [k for k, v in err_counts.items() if int(v or 0) > 0]
    if len(active_types) == 0:
        return "clean"
    if len(active_types) == 1:
        return "single"
    return "multi"


def _load_all_configs(root: Path) -> dict[str, Any]:
    configs_dir = root / "configs"
    result: dict[str, Any] = {}
    for split in _VALID_SPLITS:
        cfg_path = configs_dir / f"{split}_configs.json"
        if not cfg_path.exists():
            result[split] = []
            continue
        spec = _load_json(cfg_path)
        episodes = []
        for idx, ep in enumerate(spec.get("episodes", [])):
            cfg = ep.get("config", {})
            episodes.append(
                {
                    "index": idx,
                    "seed": int(ep["seed"]),
                    "description": str(ep.get("description", "")),
                    "category": _classify_episode_category(cfg),
                    "base_currency": cfg.get("base_currency", "USD"),
                    "num_invoices": cfg.get("num_invoices", 0),
                    "clean": bool(cfg.get("clean", False)),
                    "error_counts": cfg.get("error_counts", {}),
                    "interacting_scenarios": cfg.get("interacting_scenarios", []),
                    "config": cfg,
                }
            )
        result[split] = episodes
    return result


def _resolve_episode_spec(root: Path, payload: dict[str, Any]) -> tuple[int, dict[str, Any], str]:
    """Resolve (seed, config_dict, description) from either {split, index} or {seed, config}."""
    if "seed" in payload and "config" in payload and payload["config"] is not None:
        seed = int(payload["seed"])
        cfg = payload["config"]
        if not isinstance(cfg, dict):
            raise ValueError("'config' must be a JSON object")
        desc = str(payload.get("description", f"Custom episode (seed={seed})"))
        return seed, cfg, desc

    split = str(payload.get("split", "dev"))
    if split not in _VALID_SPLITS:
        raise ValueError(f"Invalid split {split!r}; expected one of {_VALID_SPLITS}")

    cfg_path = root / "configs" / f"{split}_configs.json"
    if not cfg_path.exists():
        raise ValueError(f"Config file for split {split!r} not found")

    raw_index = payload.get("index", 0)
    if isinstance(raw_index, bool) or not isinstance(raw_index, int):
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid episode 'index'; must be an integer") from exc
    else:
        index = raw_index

    spec = _load_json(cfg_path)
    episodes = spec.get("episodes", [])
    if index < 0 or index >= len(episodes):
        raise ValueError(f"Episode index {index} out of range for split {split!r} (0..{len(episodes) - 1})")

    ep_spec = episodes[index]
    return int(ep_spec["seed"]), dict(ep_spec["config"]), str(ep_spec.get("description", ""))


def _summarize_observation(action: dict[str, Any], obs: dict[str, Any]) -> str:
    act_type = action.get("action")
    if obs.get("status") == "error":
        return f"Error: {obs.get('error', 'unknown error')}"
    if act_type == "list_records":
        table = obs.get("table", action.get("table", "?"))
        count = obs.get("count", len(obs.get("records", [])))
        total = obs.get("total_count", "?")
        cursor = obs.get("cursor", "0")
        nxt = obs.get("next_cursor")
        nxt_label = f"next={nxt}" if nxt is not None else "end"
        return f"Returned {count}/{total} rows from {table} (cursor={cursor}, {nxt_label})"
    if act_type == "get_record":
        return f"Fetched {obs.get('table', action.get('table', '?'))} record {obs.get('id', action.get('id', '?'))}"
    if act_type == "get_document":
        doc = obs.get("document", {})
        doc_type = doc.get("doc_type", "document") if isinstance(doc, dict) else "document"
        return f"Fetched {doc_type} {obs.get('doc_id', action.get('doc_id', '?'))}"
    if act_type == "calculator":
        return f"{action.get('expression', '')} = {obs.get('result', '?')}"
    if act_type == "submit":
        findings = action.get("findings", {})
        n_findings = len(findings.get("findings", [])) if isinstance(findings, dict) else 0
        return f"Submitted {n_findings} finding(s) for {obs.get('episode_id', '')}"
    return f"Status: {obs.get('status', 'ok')}"


def _make_policy(policy_name: str, seed: int, cfg_dict: dict[str, Any]) -> Any:
    if policy_name == "simple_reconciliation":
        return SimpleReconciliationPolicy(use_calculator=True)
    if policy_name == "random_explorer":
        return RandomExplorerPolicy(seed=0)
    if policy_name == "flag_everything_attack":
        return FlagEverythingAttackPolicy()
    if policy_name == "oracle":
        ep = generate_episode(seed=seed, config=EpisodeConfig.from_dict(cfg_dict))
        return OraclePolicy(truth=ep.truth)
    if policy_name == "learnable_untrained":
        from audit_rl.policies import LearnableAuditPolicy

        return LearnableAuditPolicy()
    raise ValueError(f"Unknown policy {policy_name!r}; expected one of {_VALID_POLICIES}")


def _get_store(root: Path) -> Any | None:
    try:
        from audit_rl.storage import PlatformStore

        return PlatformStore(db_path=root / "data" / "platform.db")
    except Exception:
        return None


def _overview_api(root: Path) -> dict[str, Any]:
    configs = _load_all_configs(root)
    split_counts = {k: len(v) for k, v in configs.items()}
    total_episodes = sum(split_counts.values())

    store = _get_store(root)
    store_stats: dict[str, Any] = {}
    recent_experiments: list[dict[str, Any]] = []
    policy_versions: list[dict[str, Any]] = []
    if store is not None:
        try:
            store_stats = store.get_overview_stats()
            recent_experiments = store.list_experiments(limit=15)
            policy_versions = store.list_policy_versions(limit=10)
        except Exception:
            pass

    return {
        "product": "AuditLab",
        "tagline": "An autonomous financial audit simulation and AI training environment.",
        "splits": split_counts,
        "total_episodes": total_episodes,
        "evaluated_agents_count": len(_VALID_POLICIES) + len(policy_versions),
        "store_stats": store_stats,
        "recent_experiments": recent_experiments,
        "policy_versions": policy_versions,
        "system": {
            "offline": True,
            "stdlib_only": True,
            "isolation": "SubprocessAuditEnv (mode=eval)",
            "schema_version": SCHEMA_VERSION,
        },
    }


def _settings_api(root: Path) -> dict[str, Any]:
    configs = _load_all_configs(root)
    return {
        "product": "AuditLab",
        "tagline": "An autonomous financial audit simulation and AI training environment.",
        "environment": {
            "schema_version": SCHEMA_VERSION,
            "default_max_steps": DEFAULT_MAX_STEPS,
            "max_page_limit": MAX_PAGE_LIMIT,
            "max_findings": MAX_FINDINGS,
            "max_evidence_per_finding": MAX_EVIDENCE_PER_FINDING,
            "max_explanation_chars": MAX_EXPLANATION_CHARS,
            "error_types": list(ERROR_TYPES),
            "interacting_scenarios": list(INTERACTING_SCENARIOS),
            "currencies": dict(CURRENCY_EXPONENTS),
        },
        "datasets": {k: len(v) for k, v in configs.items()},
        "isolation": {
            "evaluator_wrapper": "SubprocessAuditEnv",
            "eval_mode": "eval",
            "train_mode": "train",
            "note": "Out-of-process evaluator isolation hides private truth/provenance and terminal rewards during eval.",
        },
        "training_defaults": {
            "seed": 42,
            "epochs": 2,
            "train_split": "train",
            "val_split": "val",
            "eval_split": "eval",
            "storage_db": "data/platform.db",
        },
    }


def _run_episode_api(server: AuditGUIServer, payload: dict[str, Any]) -> dict[str, Any]:
    root = server.root
    seed, cfg_dict, description = _resolve_episode_spec(root, payload)
    policy_name = str(payload.get("policy", "simple_reconciliation"))
    policy = _make_policy(policy_name, seed, cfg_dict)

    t0 = time.perf_counter()
    trajectory: list[dict[str, Any]] = []
    submitted_findings: list[dict[str, Any]] = []

    with SubprocessAuditEnv(mode="eval") as env:
        obs, info = env.reset(seed=seed, config=cfg_dict)
        manifest = dict(obs.get("manifest", {}))
        policy_doc = dict(obs.get("policy", {}))
        max_steps = int(info.get("max_steps", obs.get("budget_remaining", DEFAULT_MAX_STEPS)))
        policy.reset(obs, info)

        step_num = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = policy.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            step_num += 1
            if action.get("action") == "submit":
                raw_findings = action.get("findings")
                if isinstance(raw_findings, dict) and isinstance(raw_findings.get("findings"), list):
                    submitted_findings = list(raw_findings["findings"])
            trajectory.append(
                {
                    "step": step_num,
                    "action": action,
                    "action_type": str(action.get("action", "unknown")),
                    "budget_remaining": next_obs.get("budget_remaining", max(0, max_steps - step_num)),
                    "observation_summary": _summarize_observation(action, next_obs),
                    "observation": next_obs,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            obs = next_obs
            if step_num > 500:
                break

        report = env.get_evaluator_report()

    wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    metrics = dict(report.get("metrics", {}))
    evaluator_report = {
        "episode_id": report.get("episode_id", manifest.get("episode_id", "")),
        "tp": int(metrics.get("tp", report.get("tp", 0))),
        "fp": int(metrics.get("fp", report.get("fp", 0))),
        "fn": int(metrics.get("fn", report.get("fn", 0))),
        "precision": round(float(metrics.get("precision", report.get("precision", 0.0))), 4),
        "recall": round(float(metrics.get("recall", report.get("recall", 0.0))), 4),
        "f1": round(float(metrics.get("f1", report.get("f1", 0.0))), 4),
        "reward": round(float(report.get("reward", 0.0)), 4),
        "steps": int(report.get("steps", step_num)),
        "inspection_steps": int(report.get("inspection_steps", max(0, step_num - 1))),
        "max_steps": max_steps,
        "runtime_ms": round(float(report.get("runtime_ms", wall_ms)), 2),
        "wall_runtime_ms": wall_ms,
        "valid_submission": bool(report.get("valid_submission", True)),
        "validation_errors": list(report.get("validation_errors", [])),
        "submitted_count": int(report.get("submitted_count", len(submitted_findings))),
        "truth_count": int(report.get("truth_count", 0)),
        "matched_pairs": list(metrics.get("matched_pairs", [])),
        "unmatched_finding_indices": list(metrics.get("unmatched_finding_indices", [])),
        "missed_issue_ids": list(metrics.get("missed_issue_ids", [])),
    }

    result = {
        "seed": seed,
        "split": str(payload.get("split", "custom")),
        "index": payload.get("index", 0),
        "description": description,
        "policy_name": policy_name,
        "manifest": manifest,
        "policy": policy_doc,
        "trajectory": trajectory,
        "submitted_findings": submitted_findings,
        "report": evaluator_report,
    }
    server.latest_run = result
    return result


def _inspect_episode_api(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    seed, cfg_dict, description = _resolve_episode_spec(root, payload)
    table = str(payload.get("table", "invoices"))
    if table not in _VALID_TABLES:
        raise ValueError(f"Unknown table {table!r}; expected one of {_VALID_TABLES}")

    raw_limit = payload.get("limit", 25)
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid 'limit'; must be an integer") from exc
    if limit < 1 or limit > MAX_PAGE_LIMIT:
        raise ValueError(f"Limit must be between 1 and {MAX_PAGE_LIMIT}, got {limit}")

    cursor = payload.get("cursor", "0")
    if cursor is None or cursor == "":
        cursor = "0"
    cursor_str = str(cursor)

    env = AuditEnv(mode="eval")
    reset_obs, _ = env.reset(seed=seed, config=cfg_dict)
    page_obs, _, _, _, _ = env.step(
        {
            "action": "list_records",
            "table": table,
            "cursor": cursor_str,
            "limit": limit,
        }
    )
    if page_obs.get("status") == "error":
        raise ValueError(str(page_obs.get("error", "Failed to list records")))

    return {
        "seed": seed,
        "description": description,
        "episode_id": reset_obs["manifest"].get("episode_id", ""),
        "manifest": reset_obs["manifest"],
        "policy": reset_obs["policy"],
        "table": page_obs.get("table", table),
        "records": page_obs.get("records", []),
        "count": int(page_obs.get("count", 0)),
        "total_count": int(page_obs.get("total_count", 0)),
        "cursor": str(page_obs.get("cursor", cursor_str)),
        "next_cursor": page_obs.get("next_cursor"),
        "limit": limit,
    }


def _lookup_record_api(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    seed, cfg_dict, _ = _resolve_episode_spec(root, payload)
    raw_id = payload.get("id")
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise ValueError("Missing or empty 'id' to look up")
    rec_id = raw_id.strip()

    env = AuditEnv(mode="eval")
    reset_obs, _ = env.reset(seed=seed, config=cfg_dict)

    attempts: list[tuple[str, dict[str, Any]]] = []
    if rec_id.startswith("DOC-"):
        attempts.append(("document", {"action": "get_document", "doc_id": rec_id}))
    elif rec_id.startswith("ROW-"):
        attempts.append(("ledger", {"action": "get_record", "table": "ledger", "id": rec_id}))
    elif rec_id.startswith("INV-"):
        attempts.append(("invoices", {"action": "get_record", "table": "invoices", "id": rec_id}))
    elif rec_id.startswith("PAY-"):
        attempts.append(("payments", {"action": "get_record", "table": "payments", "id": rec_id}))
    elif ":" in rec_id:
        attempts.append(("fx", {"action": "get_record", "table": "fx", "id": rec_id}))
    else:
        attempts.extend(
            [
                ("document", {"action": "get_document", "doc_id": rec_id}),
                ("invoices", {"action": "get_record", "table": "invoices", "id": rec_id}),
                ("payments", {"action": "get_record", "table": "payments", "id": rec_id}),
                ("ledger", {"action": "get_record", "table": "ledger", "id": rec_id}),
                ("fx", {"action": "get_record", "table": "fx", "id": rec_id}),
            ]
        )

    last_error = f"Record {rec_id!r} not found"
    for source, action in attempts:
        obs, _, _, _, _ = env.step(action)
        if obs.get("status") == "ok":
            data = obs.get("document") if source == "document" else obs.get("record")
            return {
                "ok": True,
                "id": rec_id,
                "episode_id": reset_obs["manifest"].get("episode_id", ""),
                "action": action["action"],
                "source": source,
                "data": data,
            }
        last_error = str(obs.get("error", last_error))

    return {
        "ok": False,
        "id": rec_id,
        "episode_id": reset_obs["manifest"].get("episode_id", ""),
        "error": last_error,
    }


def _workspace_reset_api(server: AuditGUIServer, payload: dict[str, Any]) -> dict[str, Any]:
    session_id = str(payload.get("session_id", "default"))[:64]
    seed, cfg_dict, description = _resolve_episode_spec(server.root, payload)
    env = AuditEnv(mode="eval")
    obs, info = env.reset(seed=seed, config=cfg_dict)
    max_steps = int(info.get("max_steps", obs.get("budget_remaining", DEFAULT_MAX_STEPS)))

    with server.workspace_lock:
        # Keep bounded session map
        if len(server.workspace_sessions) > 32 and session_id not in server.workspace_sessions:
            oldest_key = next(iter(server.workspace_sessions))
            del server.workspace_sessions[oldest_key]
        server.workspace_sessions[session_id] = {
            "env": env,
            "seed": seed,
            "split": str(payload.get("split", "custom")),
            "index": payload.get("index", 0),
            "description": description,
            "manifest": obs["manifest"],
            "policy": obs["policy"],
            "max_steps": max_steps,
            "history": [],
            "terminated": False,
            "truncated": False,
            "report": None,
        }

    return {
        "session_id": session_id,
        "seed": seed,
        "description": description,
        "episode_id": obs["manifest"].get("episode_id", ""),
        "manifest": obs["manifest"],
        "policy": obs["policy"],
        "step": 0,
        "budget_remaining": max_steps,
        "max_steps": max_steps,
        "available_actions": obs.get("available_actions", []),
        "terminated": False,
        "truncated": False,
    }


def _workspace_step_api(server: AuditGUIServer, payload: dict[str, Any]) -> dict[str, Any]:
    session_id = str(payload.get("session_id", "default"))[:64]

    raw_action = payload.get("action")
    if isinstance(raw_action, dict):
        action = raw_action
    elif isinstance(raw_action, str):
        action = {k: v for k, v in payload.items() if k != "session_id"}
    else:
        raise ValueError("Missing 'action' in workspace step payload")

    with server.workspace_lock:
        sess = server.workspace_sessions.get(session_id)
        if sess is None:
            raise ValueError("No active workspace session; call POST /api/workspace/reset first")

        env: AuditEnv = sess["env"]
        obs, reward, terminated, truncated, info = env.step(action)
        step_num = int(info.get("step", len(sess["history"]) + 1))
        summary = _summarize_observation(action, obs)
        entry = {
            "step": step_num,
            "action": action,
            "action_type": str(action.get("action", "unknown")),
            "observation_summary": summary,
            "observation": obs,
            "budget_remaining": obs.get("budget_remaining", max(0, sess["max_steps"] - step_num)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        sess["history"].append(entry)
        sess["terminated"] = bool(terminated)
        sess["truncated"] = bool(truncated)
        report = None
        if terminated or truncated:
            report = env.get_evaluator_report()
            sess["report"] = report

    return {
        "session_id": session_id,
        "episode_id": sess["manifest"].get("episode_id", ""),
        "step": step_num,
        "action": action,
        "observation": obs,
        "observation_summary": summary,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "budget_remaining": entry["budget_remaining"],
        "max_steps": sess["max_steps"],
        "report": report,
        "history_count": len(sess["history"]),
    }


def _calculator_api(payload: dict[str, Any]) -> dict[str, Any]:
    expression = payload.get("expression", "")
    try:
        result = safe_calculate(expression)
        return {"ok": True, "expression": expression, "result": result}
    except CalculatorError as exc:
        return {"ok": False, "expression": expression, "error": str(exc)}


def _build_episode_matrix(root: Path, details: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    spec = _load_json(root / "configs" / "eval_configs.json")
    episodes = spec.get("episodes", [])
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for pol_name, rows in details.items():
        for row in rows:
            s = int(row["seed"])
            by_seed.setdefault(s, {})[pol_name] = {
                "f1": round(float(row.get("f1", 0.0)), 4),
                "reward": round(float(row.get("return", row.get("reward", 0.0))), 4),
                "precision": round(float(row.get("precision", 0.0)), 4),
                "recall": round(float(row.get("recall", 0.0)), 4),
                "steps": int(row.get("steps", 0)),
            }

    matrix: list[dict[str, Any]] = []
    for ep_spec in episodes:
        seed = int(ep_spec["seed"])
        cfg = ep_spec.get("config", {})
        err_counts = cfg.get("error_counts") or {}
        num_issues = sum(int(v or 0) for v in err_counts.values()) + len(cfg.get("interacting_scenarios") or [])
        matrix.append(
            {
                "seed": seed,
                "label": str(ep_spec.get("description", f"seed {seed}")),
                "category": _classify_episode_category(cfg),
                "num_issues": num_issues,
                "results": by_seed.get(seed, {}),
            }
        )
    return matrix


def _compute_category_breakdown(matrix: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    buckets: dict[str, dict[str, list[float]]] = {}
    for row in matrix:
        cat = str(row.get("category", "other"))
        for pol, metrics in (row.get("results") or {}).items():
            buckets.setdefault(cat, {}).setdefault(pol, []).append(float(metrics.get("f1", 0.0)))
    out: dict[str, dict[str, dict[str, float]]] = {}
    for cat, pol_map in buckets.items():
        out[cat] = {}
        for pol, vals in pol_map.items():
            out[cat][pol] = {"mean_f1": round(sum(vals) / len(vals), 4) if vals else 0.0, "count": len(vals)}
    return out


def _baselines_api(root: Path) -> dict[str, Any]:
    from audit_rl.cli import evaluate_baselines

    eval_cfg = root / "configs" / "eval_configs.json"
    result = evaluate_baselines(eval_cfg)
    if "episode_matrix" not in result:
        result["episode_matrix"] = _build_episode_matrix(root, result.get("details", {}))
    result["category_breakdown"] = _compute_category_breakdown(result["episode_matrix"])

    # Persist benchmark run in PlatformStore when available
    store = _get_store(root)
    if store is not None:
        try:
            store.record_experiment(
                kind="benchmark",
                policy_name="all_baselines",
                policy_version="1.0.0",
                config_version=str(result.get("config_version", "1.0.0")),
                split="eval",
                summary=result.get("summary", {}),
                details={"num_episodes": result.get("num_episodes", 20)},
            )
        except Exception:
            pass
    return result


def _verify_api(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    from audit_rl.rubric import run_full_rubric_verification

    run_unittests = bool(payload.get("run_unittests", True))
    try:
        res = run_full_rubric_verification(run_unittests=run_unittests, root=root)
    except TypeError:
        res = run_full_rubric_verification(run_unittests=run_unittests)
    if "episode_matrix" in res and "category_breakdown" not in res:
        res["category_breakdown"] = _compute_category_breakdown(res["episode_matrix"])
    return res


def _train_api(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    from audit_rl.trainer import run_training_experiment

    seed = int(payload.get("seed", 42))
    epochs = int(payload.get("epochs", 2))
    if epochs < 1 or epochs > 10:
        raise ValueError("epochs must be between 1 and 10")
    train_split = str(payload.get("train_split", "train"))
    eval_split = str(payload.get("eval_split", "eval"))
    save_to_store = bool(payload.get("save_to_store", True))

    try:
        return run_training_experiment(
            seed=seed,
            epochs=epochs,
            train_split=train_split,
            eval_split=eval_split,
            save_to_store=save_to_store,
            root=root,
        )
    except TypeError:
        return run_training_experiment(
            seed=seed,
            epochs=epochs,
            train_split=train_split,
            eval_split=eval_split,
            save_to_store=save_to_store,
        )


def _training_history_api(root: Path) -> dict[str, Any]:
    store = _get_store(root)
    if store is None:
        return {"policy_versions": [], "experiments": []}
    return {
        "policy_versions": store.list_policy_versions(limit=20),
        "experiments": store.list_experiments(limit=20, kind="train"),
    }


def _trajectories_api(server: AuditGUIServer, limit: int = 50) -> dict[str, Any]:
    traj_path = server.root / "trajectories" / "trajectories.jsonl"
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    order: list[tuple[str, str]] = []

    if traj_path.exists():
        with traj_path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (str(rec.get("episode_id", "unknown")), str(rec.get("policy", "unknown")))
                if key not in grouped:
                    grouped[key] = []
                    order.append(key)
                action = rec.get("action") or {}
                obs = rec.get("observation") or {}
                grouped[key].append(
                    {
                        "step": int(rec.get("step", len(grouped[key]) + 1)),
                        "action": action,
                        "action_type": str(action.get("action", "unknown")),
                        "observation_summary": _summarize_observation(action, obs),
                        "observation": obs,
                        "reward": float(rec.get("reward", 0.0)),
                        "terminated": bool(rec.get("terminated", False)),
                        "truncated": bool(rec.get("truncated", False)),
                    }
                )

    episodes: list[dict[str, Any]] = []
    if server.latest_run is not None:
        lr = server.latest_run
        episodes.append(
            {
                "source": "live_run",
                "episode_id": str(lr.get("manifest", {}).get("episode_id", "live")),
                "policy": str(lr.get("policy_name", "live")),
                "step_count": len(lr.get("trajectory", [])),
                "steps": lr.get("trajectory", []),
                "report": lr.get("report"),
            }
        )

    for key in order[:limit]:
        steps = grouped[key]
        episodes.append(
            {
                "source": "trajectories.jsonl",
                "episode_id": key[0],
                "policy": key[1],
                "step_count": len(steps),
                "steps": steps,
                "report": None,
            }
        )

    return {
        "file_exists": traj_path.exists(),
        "total_episodes": len(episodes),
        "episodes": episodes,
    }


def _trajectories_export_bytes(server: AuditGUIServer) -> bytes:
    traj_path = server.root / "trajectories" / "trajectories.jsonl"
    if traj_path.exists():
        return traj_path.read_bytes()
    if server.latest_run is not None:
        lines = [
            json.dumps(
                {
                    "episode_id": server.latest_run.get("manifest", {}).get("episode_id", ""),
                    "policy": server.latest_run.get("policy_name", ""),
                    **step,
                },
                ensure_ascii=False,
            )
            for step in server.latest_run.get("trajectory", [])
        ]
        return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    return b""


class AuditGUIServer(ThreadingHTTPServer):
    """Threaded HTTP server carrying repository root and workspace session state."""

    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        root: Path,
        quiet: bool = False,
    ) -> None:
        self.root = root
        self.quiet = quiet
        self.workspace_lock = threading.Lock()
        self.workspace_sessions: dict[str, dict[str, Any]] = {}
        self.latest_run: dict[str, Any] | None = None
        super().__init__(server_address, handler_cls)


class AuditGUIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the AuditLab 8-page Web Application and JSON API."""

    server: AuditGUIServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if not getattr(self.server, "quiet", False):
            super().log_message(format, *args)

    def _send_bytes(
        self,
        status: int,
        content_type: str,
        body: bytes,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, data: Any) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", raw)

    def _validate_path(self) -> str | None:
        parsed = urlparse(self.path)
        decoded = unquote(parsed.path)
        if ".." in decoded or "\x00" in decoded or "\\" in decoded:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid request path"})
            return None
        return parsed.path

    def _read_json_body(self) -> dict[str, Any]:
        raw_len = self.headers.get("Content-Length", "0") or "0"
        try:
            length = int(raw_len)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header") from exc
        if length < 0:
            raise ValueError("Negative Content-Length header")
        if length == 0:
            return {}
        if length > _MAX_BODY_BYTES:
            raise ValueError(f"Request body exceeds maximum size of {_MAX_BODY_BYTES} bytes")
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("JSON request body must be an object")
        return parsed

    def do_GET(self) -> None:  # noqa: N802
        path = self._validate_path()
        if path is None:
            return
        qs = parse_qs(urlparse(self.path).query)
        try:
            if path in ("/", "/index.html"):
                self._send_bytes(
                    HTTPStatus.OK,
                    "text/html; charset=utf-8",
                    _DASHBOARD_HTML.encode("utf-8"),
                )
                return
            if path == "/api/health":
                self._send_json(HTTPStatus.OK, {"status": "ok", "product": "AuditLab"})
                return
            if path == "/api/overview":
                self._send_json(HTTPStatus.OK, _overview_api(self.server.root))
                return
            if path == "/api/configs":
                self._send_json(HTTPStatus.OK, _load_all_configs(self.server.root))
                return
            if path == "/api/settings":
                self._send_json(HTTPStatus.OK, _settings_api(self.server.root))
                return
            if path == "/api/training/history":
                self._send_json(HTTPStatus.OK, _training_history_api(self.server.root))
                return
            if path == "/api/trajectories":
                raw_lim = qs.get("limit", ["50"])[0]
                lim = max(1, min(200, int(raw_lim))) if raw_lim.isdigit() else 50
                self._send_json(HTTPStatus.OK, _trajectories_api(self.server, limit=lim))
                return
            if path == "/api/trajectories/export":
                body = _trajectories_export_bytes(self.server)
                self._send_bytes(
                    HTTPStatus.OK,
                    "application/x-ndjson; charset=utf-8",
                    body,
                    extra_headers={"Content-Disposition": 'attachment; filename="auditlab_trajectories.jsonl"'},
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})
        except Exception as exc:  # pragma: no cover - defensive
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        path = self._validate_path()
        if path is None:
            return
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        try:
            if path == "/api/verify":
                self._send_json(HTTPStatus.OK, _verify_api(self.server.root, payload))
                return
            if path == "/api/episode/run":
                self._send_json(HTTPStatus.OK, _run_episode_api(self.server, payload))
                return
            if path == "/api/episode/inspect":
                self._send_json(HTTPStatus.OK, _inspect_episode_api(self.server.root, payload))
                return
            if path == "/api/record/lookup":
                self._send_json(HTTPStatus.OK, _lookup_record_api(self.server.root, payload))
                return
            if path == "/api/workspace/reset":
                self._send_json(HTTPStatus.OK, _workspace_reset_api(self.server, payload))
                return
            if path == "/api/workspace/step":
                self._send_json(HTTPStatus.OK, _workspace_step_api(self.server, payload))
                return
            if path == "/api/calculator":
                self._send_json(HTTPStatus.OK, _calculator_api(payload))
                return
            if path == "/api/baselines":
                self._send_json(HTTPStatus.OK, _baselines_api(self.server.root))
                return
            if path == "/api/train":
                self._send_json(HTTPStatus.OK, _train_api(self.server.root, payload))
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    root: Path | None = None,
    quiet: bool = False,
) -> AuditGUIServer:
    """Create and bind an ``AuditGUIServer`` instance."""
    resolved_root = Path(root).resolve() if root is not None else _repo_root()
    return AuditGUIServer((host, port), AuditGUIHandler, root=resolved_root, quiet=quiet)


def run_gui(
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    root: Path | None = None,
) -> None:
    """Start the local offline AuditLab Web Application and block until interrupted."""
    server = create_server(host=host, port=port, root=root, quiet=False)
    actual_host, actual_port = server.server_address[:2]
    url = f"http://{actual_host}:{actual_port}"
    print(f"AuditLab Web Application running at {url} (100% offline, Press Ctrl+C to stop)")

    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down AuditLab Web Application...")
    finally:
        server.server_close()


_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AuditLab — Autonomous Financial Audit Simulation &amp; AI Training</title>
<style>
  :root, html[data-theme="dark"] {
    --bg: #0b0f17;
    --bg-sidebar: #0e1420;
    --bg-surface: #121926;
    --bg-elev: #172030;
    --bg-code: #090d14;
    --bg-hover: #1d283b;
    --border: rgba(148, 163, 184, 0.16);
    --border-strong: rgba(148, 163, 184, 0.30);
    --text: #e2e8f0;
    --text-bright: #f8fafc;
    --text-muted: #94a3b8;
    --text-dim: #64748b;

    --emerald: #10b981;
    --emerald-soft: rgba(16, 185, 129, 0.14);
    --cyan: #06b6d4;
    --cyan-soft: rgba(6, 182, 212, 0.14);
    --sky: #38bdf8;
    --sky-soft: rgba(56, 189, 248, 0.14);
    --amber: #f59e0b;
    --amber-soft: rgba(245, 158, 11, 0.15);
    --rose: #f43f5e;
    --rose-soft: rgba(244, 63, 94, 0.15);
    --violet: #a855f7;
    --violet-soft: rgba(168, 85, 247, 0.15);
    --btn-primary-bg: #0ea5e9;
    --btn-primary-fg: #031119;
    --th-bg: #0f1623;
  }

  html[data-theme="light"] {
    --bg: #f4f6fa;
    --bg-sidebar: #ffffff;
    --bg-surface: #ffffff;
    --bg-elev: #f8fafc;
    --bg-code: #f1f5f9;
    --bg-hover: #eef2f7;
    --border: rgba(15, 23, 42, 0.11);
    --border-strong: rgba(15, 23, 42, 0.22);
    --text: #1e293b;
    --text-bright: #0f172a;
    --text-muted: #475569;
    --text-dim: #64748b;

    --emerald: #059669;
    --emerald-soft: rgba(5, 150, 105, 0.10);
    --cyan: #0891b2;
    --cyan-soft: rgba(8, 145, 178, 0.10);
    --sky: #0284c7;
    --sky-soft: rgba(2, 132, 199, 0.10);
    --amber: #d97706;
    --amber-soft: rgba(217, 119, 6, 0.11);
    --rose: #e11d48;
    --rose-soft: rgba(225, 29, 72, 0.10);
    --violet: #7c3aed;
    --violet-soft: rgba(124, 58, 237, 0.10);
    --btn-primary-bg: #0284c7;
    --btn-primary-fg: #ffffff;
    --th-bg: #f1f5f9;
  }

  :root {
    --font-sans: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    --radius: 10px;
  }

  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    padding: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    font-size: 13.5px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    min-height: 100vh;
  }

  /* App Layout */
  .layout {
    display: flex;
    min-height: 100vh;
  }
  .sidebar {
    width: 248px;
    flex-shrink: 0;
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
  }
  .sidebar-head {
    padding: 18px 16px 14px;
    border-bottom: 1px solid var(--border);
  }
  .brand { display: flex; align-items: center; gap: 10px; }
  .brand-mark {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: var(--sky);
    color: var(--btn-primary-fg);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono);
    font-weight: 800;
    font-size: 13px;
    letter-spacing: -0.5px;
  }
  .brand-title { font-size: 16px; font-weight: 700; color: var(--text-bright); letter-spacing: -0.02em; }
  .brand-tagline { font-size: 11px; color: var(--text-muted); margin-top: 6px; line-height: 1.35; }

  .nav-list {
    padding: 12px 10px;
    display: flex;
    flex-direction: column;
    gap: 3px;
    flex: 1;
  }
  .nav-section-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-dim);
    padding: 10px 10px 4px;
  }
  .nav-btn {
    width: 100%;
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    color: var(--text-muted);
    padding: 8px 11px;
    border-radius: 7px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: var(--font-sans);
    transition: background 0.12s ease, color 0.12s ease;
  }
  .nav-btn:hover { background: var(--bg-hover); color: var(--text-bright); }
  .nav-btn.active {
    background: var(--sky-soft);
    border-color: rgba(56, 189, 248, 0.28);
    color: var(--sky);
    font-weight: 600;
  }
  .nav-idx {
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--text-dim);
  }
  .nav-btn.active .nav-idx { color: var(--sky); }

  .sidebar-foot {
    padding: 12px 14px;
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .theme-toggle {
    width: 100%;
    padding: 7px 10px;
    border-radius: 7px;
    border: 1px solid var(--border-strong);
    background: var(--bg-elev);
    color: var(--text);
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: var(--font-sans);
  }
  .theme-toggle:hover { background: var(--bg-hover); }

  .main {
    flex: 1;
    min-width: 0;
    padding: 20px 26px 48px;
    max-width: 1320px;
  }

  .page-header {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
  }
  .page-title { font-size: 20px; font-weight: 700; color: var(--text-bright); margin: 0; letter-spacing: -0.02em; }
  .page-sub { font-size: 12.5px; color: var(--text-muted); margin: 3px 0 0; }

  .page { display: none; }
  .page.active { display: block; }

  /* Pills */
  .pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 11px;
    font-family: var(--font-mono);
    font-weight: 600;
    border: 1px solid var(--border);
    background: var(--bg-elev);
    color: var(--text-muted);
    white-space: nowrap;
  }
  .pill-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--emerald); }
  .pill-emerald { color: var(--emerald); background: var(--emerald-soft); border-color: rgba(16, 185, 129, 0.32); }
  .pill-cyan { color: var(--cyan); background: var(--cyan-soft); border-color: rgba(6, 182, 212, 0.32); }
  .pill-sky { color: var(--sky); background: var(--sky-soft); border-color: rgba(56, 189, 248, 0.32); }
  .pill-amber { color: var(--amber); background: var(--amber-soft); border-color: rgba(245, 158, 11, 0.32); }
  .pill-rose { color: var(--rose); background: var(--rose-soft); border-color: rgba(244, 63, 94, 0.32); }
  .pill-violet { color: var(--violet); background: var(--violet-soft); border-color: rgba(168, 85, 247, 0.32); }

  /* Cards & Controls */
  .card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    margin-bottom: 16px;
  }
  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }
  .card-title {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin: 0;
  }

  .controls-grid { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; }
  .field { display: flex; flex-direction: column; gap: 4px; min-width: 130px; }
  .field.grow { flex: 1 1 260px; }
  label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
  }
  select, input[type="text"], input[type="number"], input[type="search"], textarea {
    background: var(--bg-elev);
    border: 1px solid var(--border-strong);
    color: var(--text);
    border-radius: 7px;
    padding: 7px 10px;
    font-size: 13px;
    font-family: var(--font-sans);
    outline: none;
  }
  select:focus, input:focus, textarea:focus { border-color: var(--sky); }
  .mono-input { font-family: var(--font-mono); }

  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    padding: 7px 14px;
    border-radius: 7px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid var(--border-strong);
    background: var(--bg-elev);
    color: var(--text);
    font-family: var(--font-sans);
    white-space: nowrap;
    text-decoration: none;
  }
  .btn:hover:not(:disabled) { background: var(--bg-hover); border-color: var(--sky); }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-primary {
    background: var(--btn-primary-bg);
    border-color: var(--btn-primary-bg);
    color: var(--btn-primary-fg);
    font-weight: 700;
  }
  .btn-primary:hover:not(:disabled) { filter: brightness(1.08); color: var(--btn-primary-fg); }
  .btn-sm { padding: 5px 10px; font-size: 12px; }

  /* Grids & KPIs */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }
  .kpi {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--border-strong);
    border-radius: 9px;
    padding: 13px 15px;
  }
  .kpi-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 4px; }
  .kpi-val { font-family: var(--font-mono); font-size: 22px; font-weight: 700; color: var(--text-bright); line-height: 1.2; }
  .kpi-sub { font-size: 11px; font-family: var(--font-mono); color: var(--text-dim); margin-top: 4px; }
  .val-good { color: var(--emerald); }
  .val-warn { color: var(--amber); }
  .val-bad { color: var(--rose); }
  .val-sky { color: var(--sky); }

  .two-col { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }
  .meta-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    padding: 10px 14px;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 16px;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-muted);
  }
  .meta-strip strong { color: var(--text-bright); }

  /* Tables & Formatted Cells */
  .table-wrap {
    overflow-x: auto;
    max-width: 100%;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg-surface);
  }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th, td {
    text-align: left;
    padding: 8px 11px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }
  th {
    background: var(--th-bg);
    color: var(--text-muted);
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700;
    position: sticky;
    top: 0;
    white-space: nowrap;
  }
  tr:last-child td { border-bottom: none; }
  tbody tr.clickable-row { cursor: pointer; }
  tbody tr:hover td { background: var(--bg-hover); }
  tbody tr.selected-row td { background: var(--sky-soft); }
  .mono { font-family: var(--font-mono); }
  .num { font-family: var(--font-mono); text-align: right; white-space: nowrap; }
  .col-nowrap { white-space: nowrap; }

  .tag-list { display: flex; flex-wrap: wrap; gap: 4px; }
  .tag {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 5px;
    font-family: var(--font-mono);
    font-size: 11px;
    background: var(--bg-elev);
    color: var(--text);
    border: 1px solid var(--border);
  }
  .null-badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 5px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-dim);
    background: var(--bg-elev);
    border: 1px dashed var(--border-strong);
  }

  /* Line Items & Invoice Refs */
  .col-line-items { white-space: normal; min-width: 260px; max-width: 420px; }
  .line-items-list { display: flex; flex-direction: column; gap: 6px; }
  .line-item-card {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 9px;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .line-item-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
  .line-item-desc { font-weight: 600; font-size: 12px; color: var(--text-bright); word-break: break-word; }
  .line-item-date {
    padding: 1px 6px;
    border-radius: 999px;
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--cyan);
    background: var(--cyan-soft);
    border: 1px solid rgba(6, 182, 212, 0.25);
    white-space: nowrap;
  }
  .line-item-math { font-family: var(--font-mono); font-size: 11.5px; color: var(--text-muted); white-space: nowrap; }
  .line-item-math strong { color: var(--emerald); }

  .col-refs { white-space: normal; min-width: 180px; max-width: 340px; }
  .ref-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 2px 8px;
    border-radius: 5px;
    font-family: var(--font-mono);
    font-size: 11.5px;
    background: var(--cyan-soft);
    border: 1px solid rgba(6, 182, 212, 0.26);
    color: var(--text);
    white-space: nowrap;
  }
  .ref-chip-id { color: var(--cyan); font-weight: 600; }
  .ref-chip-amt { color: var(--emerald); }

  /* Subtabs */
  .subtabs { display: flex; gap: 6px; flex-wrap: wrap; }
  .subtab {
    padding: 5px 12px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-elev);
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 12px;
    cursor: pointer;
  }
  .subtab.active { background: var(--sky); border-color: var(--sky); color: var(--btn-primary-fg); font-weight: 700; }

  /* Code / JSON */
  pre.json-box {
    margin: 0;
    padding: 10px 12px;
    background: var(--bg-code);
    border: 1px solid var(--border);
    border-radius: 7px;
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--text);
    overflow: auto;
    max-height: 290px;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* Drawers & Findings */
  .findings-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; }
  .finding-card {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-left: 4px solid var(--amber);
    border-radius: 8px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .finding-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
  .finding-expl { font-size: 12px; color: var(--text-muted); line-height: 1.45; }
  .evidence-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 5px;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    background: var(--cyan-soft);
    color: var(--cyan);
    border: 1px solid rgba(6, 182, 212, 0.34);
  }
  .evidence-chip:hover { filter: brightness(1.15); }
  .drawer {
    margin-top: 12px;
    padding: 12px 14px;
    border-radius: 8px;
    background: var(--bg-elev);
    border: 1px solid var(--sky);
    display: none;
  }
  .drawer.open { display: block; }

  /* Rubric Score Ring & Category Cards */
  .rubric-hero { display: flex; flex-wrap: wrap; align-items: center; gap: 24px; }
  .score-ring-wrap { position: relative; width: 136px; height: 136px; flex-shrink: 0; }
  .score-ring-center { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
  .score-ring-val { font-family: var(--font-mono); font-size: 28px; font-weight: 800; color: var(--text-bright); line-height: 1; }
  .score-ring-max { font-family: var(--font-mono); font-size: 11px; color: var(--text-muted); margin-top: 4px; }
  .rubric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; }
  .cat-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--sky);
    border-radius: var(--radius);
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 9px;
  }
  .prog-track { height: 7px; border-radius: 999px; background: var(--bg-code); border: 1px solid var(--border); overflow: hidden; }
  .prog-fill { height: 100%; border-radius: 999px; transition: width 0.35s ease; }
  .check-list { margin-top: 8px; display: flex; flex-direction: column; gap: 5px; max-height: 280px; overflow-y: auto; }
  .check-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 5px 8px;
    border-radius: 6px;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    font-size: 12px;
  }
  .check-badge { font-family: var(--font-mono); font-size: 10px; font-weight: 700; padding: 2px 5px; border-radius: 4px; flex-shrink: 0; }
  .check-pass { background: var(--emerald-soft); color: var(--emerald); }
  .check-fail { background: var(--rose-soft); color: var(--rose); }

  /* Trajectory Scrubber */
  .scrubber-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 12px;
  }
  .scrubber-slider { flex: 1 1 180px; accent-color: var(--sky); cursor: pointer; }
  .budget-wrap { display: flex; align-items: center; gap: 8px; min-width: 160px; font-family: var(--font-mono); font-size: 11.5px; color: var(--text-muted); }
  .budget-track { flex: 1; height: 7px; border-radius: 999px; background: var(--bg-code); border: 1px solid var(--border); overflow: hidden; }
  .budget-fill { height: 100%; background: var(--emerald); transition: width 0.2s ease; }
  .timeline-strip { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 12px; }
  .step-dot-btn {
    padding: 3px 8px;
    border-radius: 5px;
    border: 1px solid var(--border);
    background: var(--bg-elev);
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 11px;
    cursor: pointer;
  }
  .step-dot-btn.active { background: var(--sky-soft); border-color: var(--sky); color: var(--sky); font-weight: 700; }

  /* Calculator */
  .calc-row { display: flex; gap: 8px; margin-bottom: 8px; }
  .calc-row input { flex: 1; }
  .calc-presets { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
  .preset-chip {
    background: var(--bg-elev);
    border: 1px solid var(--border);
    color: var(--text-muted);
    border-radius: 999px;
    padding: 3px 9px;
    font-size: 11px;
    font-family: var(--font-mono);
    cursor: pointer;
  }
  .preset-chip:hover { border-color: var(--sky); color: var(--text-bright); }
  .calc-output {
    padding: 10px 12px;
    border-radius: 7px;
    background: var(--bg-code);
    border: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 13px;
    min-height: 44px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  /* Heatmap */
  .hm-cell { font-family: var(--font-mono); font-size: 11px; text-align: center; border-radius: 5px; padding: 4px 7px; font-weight: 700; white-space: nowrap; }
  .hm-sub { display: block; font-size: 9.5px; font-weight: 500; opacity: 0.85; }

  .empty-state { padding: 26px 14px; text-align: center; color: var(--text-muted); font-size: 13px; }
  .error-banner {
    padding: 10px 14px;
    border-radius: 8px;
    background: var(--rose-soft);
    border: 1px solid rgba(244, 63, 94, 0.4);
    color: var(--rose);
    font-size: 13px;
    margin-bottom: 14px;
    display: none;
  }
  .spinner {
    display: inline-block;
    width: 12px;
    height: 12px;
    border: 2px solid rgba(128, 128, 128, 0.3);
    border-top-color: currentColor;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  @media (max-width: 900px) {
    .layout { flex-direction: column; }
    .sidebar { width: 100%; height: auto; position: static; border-right: none; border-bottom: 1px solid var(--border); }
    .nav-list { flex-direction: row; flex-wrap: wrap; }
    .nav-btn { width: auto; }
    .nav-section-label { display: none; }
    .main { padding: 16px; }
  }
</style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="sidebar-head">
      <div class="brand">
        <div class="brand-mark">AL</div>
        <div class="brand-title">AuditLab</div>
      </div>
      <div class="brand-tagline">An autonomous financial audit simulation and AI training environment.</div>
    </div>

    <nav class="nav-list" role="navigation">
      <div class="nav-section-label">Platform</div>
      <button class="nav-btn active" data-page="overview"><span>Overview</span><span class="nav-idx">01</span></button>
      <button class="nav-btn" data-page="explorer"><span>Company Explorer</span><span class="nav-idx">02</span></button>
      <button class="nav-btn" data-page="workspace"><span>Agent Workspace</span><span class="nav-idx">03</span></button>
      <button class="nav-btn" data-page="evaluation"><span>Evaluation Results</span><span class="nav-idx">04</span></button>
      <div class="nav-section-label">AI Training &amp; Analysis</div>
      <button class="nav-btn" data-page="training"><span>Training</span><span class="nav-idx">05</span></button>
      <button class="nav-btn" data-page="benchmarks"><span>Benchmarks</span><span class="nav-idx">06</span></button>
      <button class="nav-btn" data-page="trajectories"><span>Trajectory Explorer</span><span class="nav-idx">07</span></button>
      <button class="nav-btn" data-page="settings"><span>Settings</span><span class="nav-idx">08</span></button>
    </nav>

    <div class="sidebar-foot">
      <button class="theme-toggle" id="btn-theme-toggle">
        <span id="theme-label">Dark Theme</span>
        <span class="mono" style="font-size:11px; color:var(--text-muted);">Toggle</span>
      </button>
      <div class="pill pill-emerald" style="justify-content:center;"><span class="pill-dot"></span>Offline · Stdlib Only</div>
    </div>
  </aside>

  <main class="main">
    <!-- PAGE 1: OVERVIEW -->
    <section id="page-overview" class="page active">
      <div class="page-header">
        <div>
          <h1 class="page-title">Overview</h1>
          <p class="page-sub">Live platform metrics, synthetic company splits, persisted experiments, and evaluator status.</p>
        </div>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
          <span class="pill pill-sky" id="ov-isolation-pill">SubprocessAuditEnv · mode=eval</span>
          <button class="btn btn-sm" id="btn-ov-refresh">Refresh</button>
        </div>
      </div>

      <div id="ov-error" class="error-banner"></div>
      <div class="kpi-grid" id="ov-kpis"></div>

      <div class="two-col">
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">Recent Persisted Experiments (SQLite)</h2>
            <span class="pill" id="ov-exp-count">0 runs</span>
          </div>
          <div class="table-wrap" id="ov-exp-table">
            <div class="empty-state">No persisted experiments yet. Run Training, Benchmarks, or Verification to log results.</div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <h2 class="card-title">Saved Policy Versions</h2>
            <span class="pill pill-violet" id="ov-ver-count">0 versions</span>
          </div>
          <div class="table-wrap" id="ov-ver-table">
            <div class="empty-state">No trained policy versions saved yet. Run a training experiment on Page 05.</div>
          </div>
        </div>
      </div>
    </section>

    <!-- PAGE 2: COMPANY EXPLORER -->
    <section id="page-explorer" class="page">
      <div class="page-header">
        <div>
          <h1 class="page-title">Company Explorer</h1>
          <p class="page-sub">Inspect synthetic company profiles, accounting policies, and paginated financial records via narrow tool APIs.</p>
        </div>
        <span class="pill pill-sky" id="exp-ep-pill">—</span>
      </div>

      <div class="card">
        <div class="controls-grid">
          <div class="field">
            <label for="ins-split">Split</label>
            <select id="ins-split">
              <option value="dev" selected>dev</option>
              <option value="train">train</option>
              <option value="val">val</option>
              <option value="eval">eval</option>
            </select>
          </div>
          <div class="field grow">
            <label for="ins-episode">Company / Episode</label>
            <select id="ins-episode"></select>
          </div>
          <div class="field" style="min-width:96px;">
            <label for="ins-limit">Page (&le;50)</label>
            <select id="ins-limit">
              <option value="10">10</option>
              <option value="25" selected>25</option>
              <option value="50">50</option>
            </select>
          </div>
          <div class="field" style="min-width:190px;">
            <label for="ins-filter">Filter Current Page</label>
            <input type="search" id="ins-filter" placeholder="Search ID, currency, item…" />
          </div>
          <button id="btn-ins-load" class="btn btn-primary">Inspect</button>
        </div>
      </div>

      <div id="ins-error" class="error-banner"></div>
      <div class="meta-strip" id="exp-profile-strip">Select an episode to inspect company profile and reporting policy.</div>

      <div class="card">
        <div class="card-header">
          <div class="subtabs" id="ins-table-tabs">
            <button class="subtab active" data-table="invoices">invoices</button>
            <button class="subtab" data-table="payments">payments</button>
            <button class="subtab" data-table="ledger">ledger</button>
            <button class="subtab" data-table="fx">fx</button>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="pill" id="ins-page-info">—</span>
            <button class="btn btn-sm" id="btn-ins-prev" disabled>&larr; Prev</button>
            <button class="btn btn-sm" id="btn-ins-next" disabled>Next &rarr;</button>
          </div>
        </div>
        <div class="table-wrap" id="ins-table-wrap">
          <div class="empty-state">Loading records…</div>
        </div>
        <div id="row-detail-drawer" class="drawer">
          <div class="card-header" style="margin-bottom:8px;">
            <strong class="mono" id="row-detail-title"></strong>
            <button class="btn btn-sm" id="btn-close-row-detail">Close &times;</button>
          </div>
          <pre class="json-box" id="row-detail-json"></pre>
        </div>
      </div>

      <div class="two-col">
        <div class="card">
          <div class="card-header"><h2 class="card-title">manifest.json</h2></div>
          <pre class="json-box" id="ins-manifest-json">{}</pre>
        </div>
        <div class="card">
          <div class="card-header"><h2 class="card-title">policy.json</h2></div>
          <pre class="json-box" id="ins-policy-json">{}</pre>
        </div>
      </div>
    </section>

    <!-- PAGE 3: AGENT WORKSPACE -->
    <section id="page-workspace" class="page">
      <div class="page-header">
        <div>
          <h1 class="page-title">Agent Workspace</h1>
          <p class="page-sub">Run an autonomous agent or step manually through narrow investigation tools with live step budget tracking.</p>
        </div>
        <span class="pill pill-emerald" id="ws-status-pill">No active session</span>
      </div>

      <div class="card">
        <div class="controls-grid">
          <div class="field">
            <label for="ws-split">Split</label>
            <select id="ws-split">
              <option value="dev" selected>dev</option>
              <option value="train">train</option>
              <option value="val">val</option>
              <option value="eval">eval</option>
            </select>
          </div>
          <div class="field grow">
            <label for="ws-episode">Episode</label>
            <select id="ws-episode"></select>
          </div>
          <div class="field">
            <label for="ws-policy">Autonomous Policy</label>
            <select id="ws-policy">
              <option value="simple_reconciliation">Rule-Based Auditor</option>
              <option value="random_explorer">Random Explorer</option>
              <option value="flag_everything_attack">Mark-All Attack</option>
              <option value="oracle">Oracle Evaluator Adapter</option>
            </select>
          </div>
          <button id="btn-ws-reset" class="btn">Start Manual Session</button>
          <button id="btn-ws-autorun" class="btn btn-primary">Run Autonomous Agent</button>
        </div>
      </div>

      <div id="ws-error" class="error-banner"></div>

      <div class="card">
        <div class="card-header">
          <h2 class="card-title">Active Session State &amp; Step Budget</h2>
          <div class="budget-wrap" style="min-width:220px;">
            <span>Budget:</span>
            <div class="budget-track"><div class="budget-fill" id="ws-budget-fill" style="width:100%;"></div></div>
            <strong id="ws-budget-text">100/100</strong>
          </div>
        </div>
        <div class="two-col">
          <div>
            <label style="display:block; margin-bottom:6px;">Manual Tool Invocation (AuditEnv)</label>
            <div class="controls-grid" style="margin-bottom:10px;">
              <div class="field" style="min-width:140px;">
                <label for="ws-tool-action">Action</label>
                <select id="ws-tool-action">
                  <option value="list_records">list_records</option>
                  <option value="get_record">get_record</option>
                  <option value="get_document">get_document</option>
                  <option value="calculator">calculator</option>
                  <option value="submit">submit (empty clean)</option>
                </select>
              </div>
              <div class="field" id="ws-arg-table-wrap" style="min-width:110px;">
                <label for="ws-arg-table">Table</label>
                <select id="ws-arg-table">
                  <option value="invoices">invoices</option>
                  <option value="payments">payments</option>
                  <option value="ledger">ledger</option>
                  <option value="fx">fx</option>
                </select>
              </div>
              <div class="field grow" id="ws-arg-param-wrap">
                <label for="ws-arg-param" id="ws-arg-param-label">Cursor / Limit (e.g. 0,10)</label>
                <input type="text" id="ws-arg-param" class="mono-input" value="0,10" />
              </div>
              <button id="btn-ws-step" class="btn btn-primary">Execute Step</button>
            </div>
            <pre class="json-box" id="ws-last-obs">Start a manual session or run an autonomous agent to inspect observations here.</pre>
          </div>

          <div>
            <label style="display:block; margin-bottom:6px;">Safe AST Decimal Calculator (Standalone Tool)</label>
            <div class="calc-row">
              <input type="text" id="calc-input" class="mono-input" value="round(123450 * 1.0875, 2)" />
              <button id="btn-calc-eval" class="btn btn-primary">Evaluate</button>
            </div>
            <div class="calc-presets">
              <button class="preset-chip" data-expr="round(123450 * 1.0875, 2)">round(123450 * 1.0875, 2)</button>
              <button class="preset-chip" data-expr="abs(45200 - 41950)">abs(45200 - 41950)</button>
              <button class="preset-chip" data-expr="__import__('os').system('id')">Injection Block Demo</button>
              <button class="preset-chip" data-expr="2 ** 100000">Power-DoS Block Demo</button>
            </div>
            <div class="calc-output" id="calc-output">
              <span style="color:var(--text-muted);">Evaluates via AST Decimal parser (never eval())</span>
            </div>
            <div style="margin-top:12px;">
              <label style="display:block; margin-bottom:6px;">Actions Performed (<span id="ws-actions-count">0</span>)</label>
              <div class="table-wrap" style="max-height:170px; overflow-y:auto;">
                <table>
                  <thead><tr><th>Step</th><th>Action</th><th>Summary</th></tr></thead>
                  <tbody id="ws-actions-tbody"><tr><td colspan="3" class="empty-state">No actions yet.</td></tr></tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- PAGE 4: EVALUATION RESULTS -->
    <section id="page-evaluation" class="page">
      <div class="page-header">
        <div>
          <h1 class="page-title">Evaluation Results &amp; Platform Verification</h1>
          <p class="page-sub">Submitted findings, evidence-first lookup drawer, and full 100-point Platform Verification Suite.</p>
        </div>
        <button id="btn-run-verify" class="btn btn-primary">Run 100-Point Verification Suite</button>
      </div>

      <div id="eval-error" class="error-banner"></div>

      <div class="card">
        <div class="card-header">
          <h2 class="card-title">Latest Episode Evaluation</h2>
          <span class="pill pill-sky" id="eval-ep-pill">Run an agent on Page 03</span>
        </div>
        <div class="kpi-grid" id="eval-kpis"></div>
        <div id="eval-findings-container">
          <div class="empty-state">Run an autonomous agent in <strong>Agent Workspace</strong> to inspect findings, TP/FP/FN, and clickable evidence proofs.</div>
        </div>
        <div id="evidence-drawer" class="drawer">
          <div class="card-header" style="margin-bottom:8px;">
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="pill pill-cyan" id="ev-source-pill">lookup</span>
              <strong class="mono" id="ev-id-title"></strong>
            </div>
            <button class="btn btn-sm" id="btn-close-evidence">Close &times;</button>
          </div>
          <div id="ev-formatted-preview" style="margin-bottom:8px;"></div>
          <pre class="json-box" id="ev-raw-json"></pre>
        </div>
      </div>

      <div class="card rubric-hero">
        <div class="score-ring-wrap">
          <svg viewBox="0 0 140 140" width="136" height="136">
            <circle cx="70" cy="70" r="58" fill="none" stroke="rgba(148,163,184,0.16)" stroke-width="11"></circle>
            <circle id="rubric-ring-arc" cx="70" cy="70" r="58" fill="none" stroke="#10b981" stroke-width="11"
              stroke-linecap="round" stroke-dasharray="364.42" stroke-dashoffset="364.42"
              transform="rotate(-90 70 70)" style="transition: stroke-dashoffset 0.6s ease;"></circle>
          </svg>
          <div class="score-ring-center">
            <div class="score-ring-val" id="rubric-score-val">—</div>
            <div class="score-ring-max">/ 100 pts</div>
          </div>
        </div>
        <div style="flex:1 1 300px;">
          <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:6px;">
            <strong style="font-size:16px; color:var(--text-bright);">AuditLab Platform Verification Suite</strong>
            <span class="pill" id="rubric-pass-pill">Not Run</span>
          </div>
          <p style="margin:0 0 10px; color:var(--text-muted); font-size:12.5px;">
            Verifies contract &amp; isolation (25 pts), data &amp; ground truth (25 pts), reward robustness (25 pts),
            reproducibility &amp; generalization (15 pts), and engineering diagnostics (10 pts).
          </p>
          <div style="display:flex; gap:8px; flex-wrap:wrap;" id="rubric-stats-pills"></div>
        </div>
      </div>

      <div id="rubric-categories" class="rubric-grid"></div>
    </section>

    <!-- PAGE 5: TRAINING -->
    <section id="page-training" class="page">
      <div class="page-header">
        <div>
          <h1 class="page-title">RL Training System</h1>
          <p class="page-sub">Train <code>LearnableAuditPolicy</code> strictly from terminal rewards on <code>train</code> episodes and verify generalization on unseen <code>val</code>/<code>eval</code> splits.</p>
        </div>
        <button id="btn-train-history" class="btn btn-sm">Refresh History</button>
      </div>

      <div class="card">
        <div class="controls-grid">
          <div class="field" style="min-width:110px;">
            <label for="tr-seed">Seed</label>
            <input type="number" id="tr-seed" value="42" />
          </div>
          <div class="field" style="min-width:110px;">
            <label for="tr-epochs">Epochs (1–10)</label>
            <input type="number" id="tr-epochs" value="2" min="1" max="10" />
          </div>
          <div class="field">
            <label for="tr-train-split">Train Split</label>
            <select id="tr-train-split"><option value="train" selected>train (10)</option><option value="dev">dev (10)</option></select>
          </div>
          <div class="field">
            <label for="tr-eval-split">Held-Out Eval Split</label>
            <select id="tr-eval-split"><option value="eval" selected>eval (20)</option><option value="val">val (10)</option></select>
          </div>
          <button id="btn-run-train" class="btn btn-primary">Run Training Experiment</button>
        </div>
      </div>

      <div id="tr-error" class="error-banner"></div>
      <div id="tr-results" style="display:none;">
        <div class="kpi-grid" id="tr-kpis"></div>

        <div class="two-col">
          <div class="card">
            <div class="card-header">
              <h2 class="card-title">Training Reward Curve (SVG)</h2>
              <span class="pill pill-emerald" id="tr-version-pill">—</span>
            </div>
            <div id="tr-reward-svg"></div>
          </div>

          <div class="card">
            <div class="card-header"><h2 class="card-title">Learned Parameters (Before &rarr; After)</h2></div>
            <div class="two-col" style="grid-template-columns:1fr 1fr; gap:10px;">
              <div><label style="display:block; margin-bottom:5px;">Untrained Params</label><pre class="json-box" id="tr-params-before">{}</pre></div>
              <div><label style="display:block; margin-bottom:5px;">Trained Params</label><pre class="json-box" id="tr-params-after">{}</pre></div>
            </div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <h2 class="card-title">Persisted Policy Versions (SQLite)</h2>
          <span class="pill" id="tr-hist-pill">0 saved</span>
        </div>
        <div class="table-wrap" id="tr-history-table">
          <div class="empty-state">Click <strong>Run Training Experiment</strong> to train and persist a new policy version.</div>
        </div>
      </div>
    </section>

    <!-- PAGE 6: BENCHMARKS -->
    <section id="page-benchmarks" class="page">
      <div class="page-header">
        <div>
          <h1 class="page-title">Held-Out Benchmarks &amp; Heatmap</h1>
          <p class="page-sub">Compare policies across 20 held-out <code>eval</code> episodes in isolated <code>SubprocessAuditEnv(mode="eval")</code>.</p>
        </div>
        <button id="btn-run-baselines" class="btn btn-primary">Run 20-Episode Benchmark</button>
      </div>

      <div id="bench-error" class="error-banner"></div>
      <div id="bench-results" style="display:none;">
        <div class="two-col">
          <div class="card">
            <div class="card-header">
              <h2 class="card-title">Comparative Metrics</h2>
              <span class="pill pill-emerald" id="bench-ep-pill">20 eval episodes</span>
            </div>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr><th>Policy</th><th class="num">P</th><th class="num">R</th><th class="num">F1</th><th class="num">Return</th><th class="num">Steps</th><th class="num">Avg ms</th></tr>
                </thead>
                <tbody id="bench-tbody"></tbody>
              </table>
            </div>
          </div>

          <div class="card">
            <div class="card-header"><h2 class="card-title">F1 vs Mean Return (SVG)</h2></div>
            <div id="bench-svg-chart"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header"><h2 class="card-title">Per-Category F1 Breakdown</h2></div>
          <div class="table-wrap" id="bench-cat-table"></div>
        </div>

        <div class="card">
          <div class="card-header"><h2 class="card-title">20-Episode &times; Policy Heatmap Matrix</h2></div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Seed</th><th>Category</th><th>Description</th><th class="num">Issues</th>
                  <th style="text-align:center;">Oracle</th>
                  <th style="text-align:center;">Rule-Based</th>
                  <th style="text-align:center;">Random</th>
                  <th style="text-align:center;">Mark-All</th>
                </tr>
              </thead>
              <tbody id="bench-heatmap-tbody"></tbody>
            </table>
          </div>
        </div>
      </div>
      <div id="bench-empty" class="card empty-state">
        Click <strong>Run 20-Episode Benchmark</strong> (or run Verification on Page 04) to populate charts and the heatmap matrix.
      </div>
    </section>

    <!-- PAGE 7: TRAJECTORY EXPLORER -->
    <section id="page-trajectories" class="page">
      <div class="page-header">
        <div>
          <h1 class="page-title">Trajectory Explorer</h1>
          <p class="page-sub">Replay recorded trajectories from <code>trajectories/trajectories.jsonl</code> and live agent runs step by step.</p>
        </div>
        <div style="display:flex; gap:8px;">
          <button id="btn-traj-refresh" class="btn btn-sm">Refresh</button>
          <a href="/api/trajectories/export" class="btn btn-sm btn-primary" download="auditlab_trajectories.jsonl">Export JSONL</a>
        </div>
      </div>

      <div id="traj-error" class="error-banner"></div>
      <div class="card">
        <div class="controls-grid">
          <div class="field grow">
            <label for="traj-select">Recorded Episode</label>
            <select id="traj-select"></select>
          </div>
          <span class="pill pill-sky" id="traj-step-pill">Step 0 / 0</span>
        </div>
      </div>

      <div class="card">
        <div class="scrubber-bar">
          <button class="btn btn-sm" id="btn-step-prev">&larr; Prev</button>
          <button class="btn btn-sm btn-primary" id="btn-step-play">&#9654; Play</button>
          <button class="btn btn-sm" id="btn-step-next">Next &rarr;</button>
          <input type="range" id="step-slider" class="scrubber-slider" min="0" max="0" value="0" />
        </div>
        <div class="timeline-strip" id="timeline-strip"></div>
        <div id="step-stage"><div class="empty-state">Loading recorded trajectories…</div></div>
      </div>
    </section>

    <!-- PAGE 8: SETTINGS -->
    <section id="page-settings" class="page">
      <div class="page-header">
        <div>
          <h1 class="page-title">Settings &amp; Environment Contract</h1>
          <p class="page-sub">Read-only view of schema constants, dataset splits, evaluator isolation mode, and training defaults.</p>
        </div>
        <span class="pill pill-emerald">Pure Python Stdlib</span>
      </div>
      <div class="two-col" id="settings-grid"></div>
    </section>
  </main>
</div>

<script>
(function() {
  const state = {
    configs: { train: [], dev: [], val: [], eval: [] },
    inspector: { table: "invoices", cursor: "0", nextCursor: null, cursorStack: [], records: [] },
    workspace: { active: false, split: "dev", index: 7, maxSteps: 100, actions: [] },
    latestRun: null,
    trajectories: [],
    replay: { steps: [], idx: 0, timer: null },
  };

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

  const ERROR_STYLES = {
    duplicate_invoice: { color: "#f59e0b", pill: "pill-amber" },
    payment_allocation_mismatch: { color: "#f43f5e", pill: "pill-rose" },
    fx_conversion_error: { color: "#06b6d4", pill: "pill-cyan" },
    period_cutoff_error: { color: "#a855f7", pill: "pill-violet" },
  };

  async function api(url, opts = {}) {
    const res = await fetch(url, { headers: { "Content-Type": "application/json" }, ...opts });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  function showError(id, msg) {
    const el = $(id);
    if (!el) return;
    el.style.display = msg ? "block" : "none";
    el.textContent = msg || "";
  }

  // Theme Toggle
  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    $("theme-label").textContent = theme === "light" ? "Light Theme" : "Dark Theme";
    try { localStorage.setItem("auditlab_theme", theme); } catch (_) {}
  }
  try {
    const saved = localStorage.getItem("auditlab_theme");
    if (saved === "light" || saved === "dark") applyTheme(saved);
  } catch (_) {}
  $("btn-theme-toggle").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") || "dark";
    applyTheme(cur === "dark" ? "light" : "dark");
  });

  // Sidebar Navigation
  function goToPage(page) {
    document.querySelectorAll(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.page === page));
    document.querySelectorAll(".page").forEach((p) => p.classList.toggle("active", p.id === "page-" + page));
    if (page === "overview") loadOverview();
    if (page === "trajectories") loadTrajectories();
    if (page === "training") loadTrainingHistory();
    if (page === "settings") loadSettings();
  }
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => goToPage(btn.dataset.page));
  });

  function populateEpisodes(selectId, split) {
    const list = state.configs[split] || [];
    const sel = $(selectId);
    sel.innerHTML = list.map((ep) =>
      `<option value="${ep.index}">#${ep.index} · seed ${ep.seed} · [${esc(ep.category)}] ${esc(ep.description)} (${esc(ep.base_currency)})</option>`
    ).join("");
    if (split === "dev" && list.length > 7) sel.value = "7";
  }

  // =========================================================================
  // PAGE 1: OVERVIEW
  // =========================================================================
  async function loadOverview() {
    showError("ov-error", "");
    try {
      const data = await api("/api/overview");
      const s = data.splits || {};
      const st = data.store_stats || {};
      $("ov-kpis").innerHTML = `
        <div class="kpi" style="border-left-color:var(--sky);">
          <div class="kpi-label">Synthetic Companies</div>
          <div class="kpi-val">${data.total_episodes}</div>
          <div class="kpi-sub">train:${s.train || 0} · dev:${s.dev || 0} · val:${s.val || 0} · eval:${s.eval || 0}</div>
        </div>
        <div class="kpi" style="border-left-color:var(--emerald);">
          <div class="kpi-label">Evaluated Agents</div>
          <div class="kpi-val">${data.evaluated_agents_count}</div>
          <div class="kpi-sub">baselines + saved policies</div>
        </div>
        <div class="kpi" style="border-left-color:var(--violet);">
          <div class="kpi-label">Persisted Experiments</div>
          <div class="kpi-val">${st.total_experiments ?? (data.recent_experiments || []).length}</div>
          <div class="kpi-sub">SQLite data/platform.db</div>
        </div>
        <div class="kpi" style="border-left-color:var(--cyan);">
          <div class="kpi-label">Schema Version</div>
          <div class="kpi-val">${esc(data.system?.schema_version || "1.0.0")}</div>
          <div class="kpi-sub">100% offline stdlib</div>
        </div>
      `;

      const exps = data.recent_experiments || [];
      $("ov-exp-count").textContent = `${exps.length} recent`;
      $("ov-exp-table").innerHTML = exps.length ? `
        <table>
          <thead><tr><th>ID</th><th>Kind</th><th>Policy</th><th>Split</th><th>Created</th></tr></thead>
          <tbody>${exps.map((e) => `
            <tr>
              <td class="mono">#${esc(e.id)}</td>
              <td><span class="pill pill-sky">${esc(e.kind)}</span></td>
              <td class="mono">${esc(e.policy_name)}</td>
              <td class="mono">${esc(e.split)}</td>
              <td class="mono col-nowrap">${esc((e.created_at || "").slice(0, 19))}</td>
            </tr>
          `).join("")}</tbody>
        </table>
      ` : `<div class="empty-state">No persisted experiments yet.</div>`;

      const vers = data.policy_versions || [];
      $("ov-ver-count").textContent = `${vers.length} saved`;
      $("ov-ver-table").innerHTML = vers.length ? `
        <table>
          <thead><tr><th>Version</th><th>Policy</th><th class="num">Train R</th><th class="num">Val F1</th><th class="num">Eval F1</th></tr></thead>
          <tbody>${vers.map((v) => `
            <tr>
              <td class="mono">${esc(v.version_id)}</td>
              <td class="mono">${esc(v.policy_name)}</td>
              <td class="num">${Number(v.train_mean_reward ?? 0).toFixed(3)}</td>
              <td class="num val-good">${Number(v.val_f1 ?? 0).toFixed(3)}</td>
              <td class="num val-good">${Number(v.eval_f1 ?? 0).toFixed(3)}</td>
            </tr>
          `).join("")}</tbody>
        </table>
      ` : `<div class="empty-state">No trained policy versions saved yet.</div>`;
    } catch (err) {
      showError("ov-error", err.message);
    }
  }
  $("btn-ov-refresh").addEventListener("click", loadOverview);

  // =========================================================================
  // PAGE 2: COMPANY EXPLORER
  // =========================================================================
  $("ins-split").addEventListener("change", (e) => { populateEpisodes("ins-episode", e.target.value); loadInspector(true); });
  $("ins-episode").addEventListener("change", () => loadInspector(true));
  $("ins-limit").addEventListener("change", () => loadInspector(true));
  $("btn-ins-load").addEventListener("click", () => loadInspector(true));
  $("ins-filter").addEventListener("input", () => renderFilteredInspectorTable());
  $("btn-close-row-detail").addEventListener("click", () => $("row-detail-drawer").classList.remove("open"));

  document.querySelectorAll("#ins-table-tabs .subtab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#ins-table-tabs .subtab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.inspector.table = btn.dataset.table;
      loadInspector(true);
    });
  });

  $("btn-ins-next").addEventListener("click", () => {
    if (state.inspector.nextCursor !== null) {
      state.inspector.cursorStack.push(state.inspector.cursor);
      state.inspector.cursor = state.inspector.nextCursor;
      loadInspector(false);
    }
  });
  $("btn-ins-prev").addEventListener("click", () => {
    if (state.inspector.cursorStack.length > 0) {
      state.inspector.cursor = state.inspector.cursorStack.pop();
      loadInspector(false);
    }
  });

  async function loadInspector(resetCursor) {
    if (resetCursor) {
      state.inspector.cursor = "0";
      state.inspector.cursorStack = [];
    }
    showError("ins-error", "");
    $("row-detail-drawer").classList.remove("open");
    try {
      const data = await api("/api/episode/inspect", {
        method: "POST",
        body: JSON.stringify({
          split: $("ins-split").value,
          index: parseInt($("ins-episode").value || "0", 10),
          table: state.inspector.table,
          cursor: state.inspector.cursor,
          limit: parseInt($("ins-limit").value, 10),
        }),
      });
      state.inspector.nextCursor = data.next_cursor;
      state.inspector.records = data.records || [];
      $("exp-ep-pill").textContent = data.episode_id || "—";
      $("ins-manifest-json").textContent = JSON.stringify(data.manifest, null, 2);
      $("ins-policy-json").textContent = JSON.stringify(data.policy, null, 2);

      const m = data.manifest || {};
      const p = data.policy || {};
      $("exp-profile-strip").innerHTML = `
        <span>company: <strong>${esc(m.company_name || "N/A")}</strong></span>
        <span>base_currency: <strong>${esc(m.base_currency)}</strong></span>
        <span>period: <strong>${esc((m.period_start || "").slice(0, 10))} &rarr; ${esc((m.period_end || "").slice(0, 10))}</strong></span>
        <span>fx_policy: <strong>${esc(p.fx_date_policy)}</strong> (${esc(p.rounding_rule)})</span>
      `;

      const start = parseInt(data.cursor, 10) + (data.count > 0 ? 1 : 0);
      const end = parseInt(data.cursor, 10) + data.count;
      $("ins-page-info").textContent = `${data.table}: ${start}–${end} of ${data.total_count}`;
      $("btn-ins-prev").disabled = state.inspector.cursorStack.length === 0;
      $("btn-ins-next").disabled = data.next_cursor === null;
      renderFilteredInspectorTable();
    } catch (err) {
      showError("ins-error", err.message);
    }
  }

  const NOWRAP_COLS = new Set([
    "doc_id", "invoice_id", "payment_id", "row_id", "entity_id",
    "issued_at", "due_at", "timestamp", "date",
    "currency", "source_currency", "target_currency",
    "total_minor", "amount_minor", "debit_minor", "credit_minor", "rate"
  ]);

  function renderLineItemsCell(items) {
    if (!Array.isArray(items) || items.length === 0) {
      return `<td class="col-line-items"><span class="null-badge">none</span></td>`;
    }
    const cards = items.map((it) => `
      <div class="line-item-card">
        <div class="line-item-top">
          <span class="line-item-desc">${esc(it.description ?? "Item")}</span>
          ${it.service_date ? `<span class="line-item-date">${esc(it.service_date)}</span>` : ""}
        </div>
        <div class="line-item-math">${esc(it.quantity ?? 1)} &times; ${esc(it.unit_price_minor ?? 0)} = <strong>${esc(it.amount_minor ?? 0)}</strong></div>
      </div>
    `).join("");
    return `<td class="col-line-items"><div class="line-items-list">${cards}</div></td>`;
  }

  function renderInvoiceRefsCell(refs) {
    if (!Array.isArray(refs) || refs.length === 0) {
      return `<td class="col-refs"><span class="null-badge">null</span></td>`;
    }
    const chips = refs.map((ref) => {
      if (!ref) return `<span class="null-badge">null</span>`;
      if (typeof ref === "string") return `<span class="ref-chip"><span class="ref-chip-id">${esc(ref)}</span></span>`;
      const invId = ref.invoice_id;
      const alloc = ref.allocated_minor ?? ref.amount_minor;
      if (!invId) return `<span class="null-badge">null${alloc !== undefined ? `: ${esc(alloc)}` : ""}</span>`;
      return `<span class="ref-chip"><span class="ref-chip-id">${esc(invId)}</span>${alloc !== undefined ? `: <span class="ref-chip-amt">${esc(alloc)}</span>` : ""}</span>`;
    }).join("");
    return `<td class="col-refs"><div class="tag-list">${chips}</div></td>`;
  }

  function renderFilteredInspectorTable() {
    const q = ($("ins-filter").value || "").trim().toLowerCase();
    const all = state.inspector.records || [];
    const filtered = q ? all.filter((r) => JSON.stringify(r).toLowerCase().includes(q)) : all;
    const wrap = $("ins-table-wrap");
    if (!filtered.length) {
      wrap.innerHTML = `<div class="empty-state">No matching records in this page.</div>`;
      return;
    }
    const cols = [];
    filtered.forEach((r) => Object.keys(r).forEach((k) => { if (!cols.includes(k)) cols.push(k); }));
    const thead = `<tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>`;
    const tbody = filtered.map((r, rIdx) => `<tr class="clickable-row" data-row-idx="${rIdx}">${cols.map((c) => {
      const v = r[c];
      if (c === "line_items") return renderLineItemsCell(v);
      if (c === "invoice_refs") return renderInvoiceRefsCell(v);
      const nowrap = NOWRAP_COLS.has(c) ? " col-nowrap" : "";
      if (v === null || v === undefined) return `<td class="mono${nowrap}"><span class="null-badge">null</span></td>`;
      if (typeof v === "object") return `<td class="mono" style="white-space:normal; word-break:break-word;">${esc(JSON.stringify(v))}</td>`;
      if (typeof v === "number") return `<td class="num${nowrap}">${esc(v)}</td>`;
      return `<td class="mono${nowrap}">${esc(v)}</td>`;
    }).join("")}</tr>`).join("");
    wrap.innerHTML = `<table><thead>${thead}</thead><tbody>${tbody}</tbody></table>`;

    wrap.querySelectorAll("tbody tr.clickable-row").forEach((tr) => {
      tr.addEventListener("click", () => {
        wrap.querySelectorAll("tbody tr").forEach((row) => row.classList.remove("selected-row"));
        tr.classList.add("selected-row");
        const rec = filtered[parseInt(tr.dataset.rowIdx, 10)];
        $("row-detail-title").textContent = rec.invoice_id || rec.payment_id || rec.row_id || rec.doc_id || "Record";
        $("row-detail-json").textContent = JSON.stringify(rec, null, 2);
        $("row-detail-drawer").classList.add("open");
      });
    });
  }

  // =========================================================================
  // PAGE 3: AGENT WORKSPACE
  // =========================================================================
  $("ws-split").addEventListener("change", (e) => populateEpisodes("ws-episode", e.target.value));
  $("ws-tool-action").addEventListener("change", (e) => {
    const act = e.target.value;
    $("ws-arg-table-wrap").style.display = (act === "list_records" || act === "get_record") ? "flex" : "none";
    $("ws-arg-param-wrap").style.display = act === "submit" ? "none" : "flex";
    if (act === "list_records") { $("ws-arg-param-label").textContent = "Cursor,Limit"; $("ws-arg-param").value = "0,10"; }
    if (act === "get_record") { $("ws-arg-param-label").textContent = "Record ID"; $("ws-arg-param").value = "INV-00001"; }
    if (act === "get_document") { $("ws-arg-param-label").textContent = "Document ID"; $("ws-arg-param").value = "DOC-00001"; }
    if (act === "calculator") { $("ws-arg-param-label").textContent = "Expression"; $("ws-arg-param").value = "round(12345 * 1.08, 2)"; }
  });

  function updateWsBudget(rem, maxSteps) {
    const pct = Math.max(0, Math.min(100, (rem / Math.max(1, maxSteps)) * 100));
    $("ws-budget-fill").style.width = `${pct}%`;
    $("ws-budget-text").textContent = `${rem}/${maxSteps}`;
  }

  function renderWsActions() {
    const list = state.workspace.actions;
    $("ws-actions-count").textContent = String(list.length);
    $("ws-actions-tbody").innerHTML = list.length ? list.map((a) => `
      <tr>
        <td class="mono">${a.step}</td>
        <td><span class="pill pill-sky">${esc(a.action_type)}</span></td>
        <td class="mono" style="font-size:11.5px;">${esc(a.summary)}</td>
      </tr>
    `).join("") : `<tr><td colspan="3" class="empty-state">No actions yet.</td></tr>`;
  }

  $("btn-ws-reset").addEventListener("click", async () => {
    showError("ws-error", "");
    try {
      const split = $("ws-split").value;
      const index = parseInt($("ws-episode").value || "0", 10);
      const data = await api("/api/workspace/reset", {
        method: "POST",
        body: JSON.stringify({ split, index, session_id: "ui" }),
      });
      state.workspace = { active: true, split, index, maxSteps: data.max_steps, actions: [] };
      $("ws-status-pill").textContent = `Active: ${data.episode_id}`;
      updateWsBudget(data.budget_remaining, data.max_steps);
      renderWsActions();
      $("ws-last-obs").textContent = JSON.stringify({ manifest: data.manifest, policy: data.policy }, null, 2);
    } catch (err) {
      showError("ws-error", err.message);
    }
  });

  $("btn-ws-step").addEventListener("click", async () => {
    showError("ws-error", "");
    try {
      const actType = $("ws-tool-action").value;
      const param = $("ws-arg-param").value.trim();
      let action = { action: actType };
      if (actType === "list_records") {
        const parts = param.split(",");
        action.table = $("ws-arg-table").value;
        action.cursor = (parts[0] || "0").trim();
        action.limit = parseInt((parts[1] || "10").trim(), 10) || 10;
      } else if (actType === "get_record") {
        action.table = $("ws-arg-table").value;
        action.id = param;
      } else if (actType === "get_document") {
        action.doc_id = param;
      } else if (actType === "calculator") {
        action.expression = param;
      } else if (actType === "submit") {
        action.findings = { episode_id: ($("ws-status-pill").textContent.replace("Active: ", "")), findings: [] };
      }
      const res = await api("/api/workspace/step", {
        method: "POST",
        body: JSON.stringify({ session_id: "ui", action }),
      });
      updateWsBudget(res.budget_remaining, res.max_steps);
      state.workspace.actions.push({ step: res.step, action_type: actType, summary: res.observation_summary });
      renderWsActions();
      $("ws-last-obs").textContent = JSON.stringify(res.report ? { observation: res.observation, report: res.report } : res.observation, null, 2);
      if (res.terminated || res.truncated) {
        $("ws-status-pill").textContent = `Finished (${res.episode_id})`;
      }
    } catch (err) {
      showError("ws-error", err.message);
    }
  });

  $("btn-ws-autorun").addEventListener("click", async () => {
    const btn = $("btn-ws-autorun");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Running…`;
    showError("ws-error", "");
    try {
      const split = $("ws-split").value;
      const index = parseInt($("ws-episode").value || "0", 10);
      const data = await api("/api/episode/run", {
        method: "POST",
        body: JSON.stringify({ split, index, policy: $("ws-policy").value }),
      });
      state.latestRun = data;
      state.workspace.split = split;
      state.workspace.index = index;
      state.workspace.actions = (data.trajectory || []).map((t) => ({
        step: t.step, action_type: t.action_type, summary: t.observation_summary,
      }));
      const r = data.report || {};
      $("ws-status-pill").textContent = `Done: ${r.episode_id} (F1=${Number(r.f1).toFixed(2)})`;
      updateWsBudget(Math.max(0, (r.max_steps || 100) - (r.steps || 0)), r.max_steps || 100);
      renderWsActions();
      $("ws-last-obs").textContent = JSON.stringify(r, null, 2);
      renderEvaluationResults(data);
    } catch (err) {
      showError("ws-error", err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run Autonomous Agent";
    }
  });

  // Standalone Calculator
  async function evalCalc() {
    try {
      const data = await api("/api/calculator", {
        method: "POST",
        body: JSON.stringify({ expression: $("calc-input").value }),
      });
      $("calc-output").innerHTML = data.ok
        ? `<span style="color:var(--text-muted);">${esc(data.expression)} =</span><strong class="val-good" style="font-size:16px;">${esc(data.result)}</strong>`
        : `<span class="val-bad">Blocked: ${esc(data.error)}</span>`;
    } catch (err) {
      $("calc-output").innerHTML = `<span class="val-bad">${esc(err.message)}</span>`;
    }
  }
  $("btn-calc-eval").addEventListener("click", evalCalc);
  $("calc-input").addEventListener("keydown", (e) => { if (e.key === "Enter") evalCalc(); });
  document.querySelectorAll(".preset-chip").forEach((c) => c.addEventListener("click", () => { $("calc-input").value = c.dataset.expr; evalCalc(); }));

  // =========================================================================
  // PAGE 4: EVALUATION RESULTS & VERIFICATION
  // =========================================================================
  function renderEvaluationResults(data) {
    const r = data.report || {};
    $("eval-ep-pill").textContent = `${r.episode_id || ""} · ${data.policy_name}`;
    $("eval-kpis").innerHTML = `
      <div class="kpi" style="border-left-color:var(--emerald);"><div class="kpi-label">F1 Score</div><div class="kpi-val ${r.f1 >= 0.85 ? "val-good" : "val-bad"}">${Number(r.f1).toFixed(4)}</div></div>
      <div class="kpi" style="border-left-color:var(--sky);"><div class="kpi-label">Terminal Reward</div><div class="kpi-val ${r.reward > 0 ? "val-good" : "val-bad"}">${Number(r.reward).toFixed(4)}</div></div>
      <div class="kpi" style="border-left-color:var(--cyan);"><div class="kpi-label">Precision / Recall</div><div class="kpi-val">${Number(r.precision).toFixed(2)} / ${Number(r.recall).toFixed(2)}</div></div>
      <div class="kpi" style="border-left-color:var(--amber);"><div class="kpi-label">TP / FP / FN</div><div class="kpi-val">${r.tp} / ${r.fp} / ${r.fn}</div></div>
      <div class="kpi" style="border-left-color:var(--violet);"><div class="kpi-label">Steps</div><div class="kpi-val">${r.steps}</div><div class="kpi-sub">${r.runtime_ms} ms</div></div>
    `;
    const findings = data.submitted_findings || [];
    if (!findings.length) {
      $("eval-findings-container").innerHTML = `<div class="empty-state">No findings submitted (clean books or empty submission).</div>`;
      return;
    }
    $("eval-findings-container").innerHTML = `<div class="findings-grid">` + findings.map((f, i) => {
      const st = ERROR_STYLES[f.type] || { color: "#38bdf8", pill: "pill-sky" };
      const evChips = (f.evidence_ids || []).concat(f.affected_ids || []).map((id) =>
        `<button class="evidence-chip" data-lookup-id="${esc(id)}">${esc(id)}</button>`
      ).join("");
      return `
        <div class="finding-card" style="border-left-color:${st.color};">
          <div class="finding-top">
            <span class="pill ${st.pill}">#${i + 1} ${esc(f.type)}</span>
            <strong class="mono">${esc(f.claimed_amount_minor)} ${esc(f.currency)}</strong>
          </div>
          <div class="finding-expl">${esc(f.explanation || "")}</div>
          <div class="tag-list">${evChips}</div>
        </div>
      `;
    }).join("") + `</div>`;

    document.querySelectorAll("#eval-findings-container .evidence-chip").forEach((chip) => {
      chip.addEventListener("click", () => lookupEvidence(chip.dataset.lookupId));
    });
  }

  async function lookupEvidence(id) {
    const drawer = $("evidence-drawer");
    drawer.classList.add("open");
    $("ev-id-title").textContent = id;
    $("ev-source-pill").textContent = "loading…";
    $("ev-formatted-preview").innerHTML = "";
    $("ev-raw-json").textContent = "Querying AuditEnv…";
    try {
      const data = await api("/api/record/lookup", {
        method: "POST",
        body: JSON.stringify({ split: state.workspace.split, index: state.workspace.index, id }),
      });
      if (!data.ok) {
        $("ev-source-pill").textContent = "not found";
        $("ev-raw-json").textContent = data.error || "Not found";
        return;
      }
      $("ev-source-pill").textContent = `${data.action} · ${data.source}`;
      $("ev-raw-json").textContent = JSON.stringify(data.data, null, 2);
    } catch (err) {
      $("ev-raw-json").textContent = err.message;
    }
  }
  $("btn-close-evidence").addEventListener("click", () => $("evidence-drawer").classList.remove("open"));

  $("btn-run-verify").addEventListener("click", async () => {
    const btn = $("btn-run-verify");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Verifying…`;
    showError("eval-error", "");
    try {
      const data = await api("/api/verify", { method: "POST", body: JSON.stringify({ run_unittests: true }) });
      const score = Number(data.total_score ?? 0);
      const maxScore = Number(data.max_score ?? 100);
      const circ = 364.42;
      $("rubric-ring-arc").style.strokeDashoffset = String(circ * (1 - Math.min(1, score / Math.max(1, maxScore))));
      $("rubric-score-val").textContent = score.toFixed(1);
      $("rubric-pass-pill").className = "pill " + (data.overall_pass ? "pill-emerald" : "pill-rose");
      $("rubric-pass-pill").textContent = data.overall_pass ? "PASS" : "FAIL";
      const ut = data.unittest || {};
      $("rubric-stats-pills").innerHTML = `
        <span class="pill pill-emerald">Unittests: ${ut.ran ?? 0} run, ${ut.failures ?? 0} fail</span>
        <span class="pill pill-sky">Duration: ${Number(data.duration_s || 0).toFixed(2)}s</span>
      `;
      $("rubric-categories").innerHTML = (data.categories || []).map((cat) => {
        const pct = cat.max_score > 0 ? Math.min(100, (cat.score / cat.max_score) * 100) : 0;
        const checksHtml = (cat.checks || []).map((ch) => `
          <div class="check-row">
            <span class="check-badge ${ch.passed ? "check-pass" : "check-fail"}">${ch.passed ? "PASS" : "FAIL"}</span>
            <div><strong>${esc(ch.name)}</strong><div class="mono" style="font-size:11px; color:var(--text-muted);">${esc(ch.detail || "")}</div></div>
          </div>
        `).join("");
        return `
          <div class="cat-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <strong>${esc(cat.name)}</strong>
              <strong class="mono val-good">${Number(cat.score).toFixed(1)}/${Number(cat.max_score).toFixed(0)}</strong>
            </div>
            <div class="prog-track"><div class="prog-fill" style="width:${pct}%; background:var(--emerald);"></div></div>
            <details><summary class="mono" style="font-size:11.5px; color:var(--text-muted); cursor:pointer;">${cat.passed}/${cat.total} checks passed</summary><div class="check-list">${checksHtml}</div></details>
          </div>
        `;
      }).join("");
      if (data.baselines || data.episode_matrix) renderBenchmarkFromAny(data);
    } catch (err) {
      showError("eval-error", err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run 100-Point Verification Suite";
    }
  });

  // =========================================================================
  // PAGE 5: TRAINING
  // =========================================================================
  async function loadTrainingHistory() {
    try {
      const data = await api("/api/training/history");
      const vers = data.policy_versions || [];
      $("tr-hist-pill").textContent = `${vers.length} saved`;
      $("tr-history-table").innerHTML = vers.length ? `
        <table>
          <thead><tr><th>Version ID</th><th>Policy</th><th>Seed</th><th class="num">Train Mean R</th><th class="num">Val F1</th><th class="num">Eval F1</th><th>Created</th></tr></thead>
          <tbody>${vers.map((v) => `
            <tr>
              <td class="mono">${esc(v.version_id)}</td>
              <td class="mono">${esc(v.policy_name)}</td>
              <td class="mono">${esc(v.seed)}</td>
              <td class="num">${Number(v.train_mean_reward ?? 0).toFixed(4)}</td>
              <td class="num val-good">${Number(v.val_f1 ?? 0).toFixed(4)}</td>
              <td class="num val-good">${Number(v.eval_f1 ?? 0).toFixed(4)}</td>
              <td class="mono col-nowrap">${esc((v.created_at || "").slice(0, 19))}</td>
            </tr>
          `).join("")}</tbody>
        </table>
      ` : `<div class="empty-state">No trained policy versions persisted yet.</div>`;
    } catch (_) {}
  }
  $("btn-train-history").addEventListener("click", loadTrainingHistory);

  $("btn-run-train").addEventListener("click", async () => {
    const btn = $("btn-run-train");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Training on mode=train…`;
    showError("tr-error", "");
    try {
      const data = await api("/api/train", {
        method: "POST",
        body: JSON.stringify({
          seed: parseInt($("tr-seed").value || "42", 10),
          epochs: parseInt($("tr-epochs").value || "2", 10),
          train_split: $("tr-train-split").value,
          eval_split: $("tr-eval-split").value,
          save_to_store: true,
        }),
      });
      $("tr-results").style.display = "block";
      $("tr-version-pill").textContent = data.policy_version || "trained";
      const u = data.untrained_eval || {};
      const t = data.trained_eval || {};
      const rb = data.rule_based_eval || {};
      const delta = Number((t.f1 ?? 0) - (u.f1 ?? 0));
      $("tr-kpis").innerHTML = `
        <div class="kpi" style="border-left-color:var(--amber);"><div class="kpi-label">Untrained Policy (Held-Out)</div><div class="kpi-val">${Number(u.f1 ?? 0).toFixed(4)}</div><div class="kpi-sub">Return: ${Number(u.return ?? u.reward ?? 0).toFixed(4)}</div></div>
        <div class="kpi" style="border-left-color:var(--emerald);"><div class="kpi-label">Trained Policy (Held-Out)</div><div class="kpi-val val-good">${Number(t.f1 ?? 0).toFixed(4)}</div><div class="kpi-sub">Return: ${Number(t.return ?? t.reward ?? 0).toFixed(4)}</div></div>
        <div class="kpi" style="border-left-color:var(--cyan);"><div class="kpi-label">F1 Improvement Delta</div><div class="kpi-val ${delta > 0 ? "val-good" : "val-warn"}">${delta >= 0 ? "+" : ""}${delta.toFixed(4)}</div><div class="kpi-sub">strictly terminal-reward trained</div></div>
        <div class="kpi" style="border-left-color:var(--violet);"><div class="kpi-label">Rule-Based Baseline</div><div class="kpi-val">${Number(rb.f1 ?? 0).toFixed(4)}</div><div class="kpi-sub">Return: ${Number(rb.return ?? rb.reward ?? 0).toFixed(4)}</div></div>
      `;
      $("tr-params-before").textContent = JSON.stringify(data.params_before || {}, null, 2);
      $("tr-params-after").textContent = JSON.stringify(data.params_after || {}, null, 2);
      renderRewardCurveSvg(data.history || []);
      loadTrainingHistory();
    } catch (err) {
      showError("tr-error", err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run Training Experiment";
    }
  });

  function renderRewardCurveSvg(history) {
    if (!history.length) { $("tr-reward-svg").innerHTML = `<div class="empty-state">No reward history points.</div>`; return; }
    const W = 520, H = 200, padL = 42, padR = 14, padT = 14, padB = 28;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const xOf = (i) => padL + (history.length === 1 ? plotW / 2 : (i / (history.length - 1)) * plotW);
    const yOf = (v) => padT + ((1.0 - Math.max(-1.0, Math.min(1.0, v))) / 2.0) * plotH;
    const pts = history.map((h, i) => `${xOf(i).toFixed(1)},${yOf(Number(h.reward ?? 0)).toFixed(1)}`).join(" ");
    const grid = [1, 0.5, 0, -0.5, -1].map((v) => {
      const y = yOf(v).toFixed(1);
      return `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="rgba(148,163,184,0.16)" /><text x="${padL - 6}" y="${Number(y) + 3}" text-anchor="end" fill="#64748b" font-size="9.5" font-family="monospace">${v.toFixed(1)}</text>`;
    }).join("");
    const dots = history.map((h, i) => `<circle cx="${xOf(i).toFixed(1)}" cy="${yOf(Number(h.reward ?? 0)).toFixed(1)}" r="3" fill="#10b981"><title>Ep ${i + 1}: R=${Number(h.reward ?? 0).toFixed(3)}</title></circle>`).join("");
    $("tr-reward-svg").innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-height:210px; display:block;">${grid}<polyline fill="none" stroke="#0ea5e9" stroke-width="2" points="${pts}" />${dots}</svg>`;
  }

  // =========================================================================
  // PAGE 6: BENCHMARKS
  // =========================================================================
  $("btn-run-baselines").addEventListener("click", async () => {
    const btn = $("btn-run-baselines");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Running 20 Eval Episodes…`;
    showError("bench-error", "");
    try {
      const data = await api("/api/baselines", { method: "POST", body: "{}" });
      renderBenchmarkFromAny(data);
    } catch (err) {
      showError("bench-error", err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run 20-Episode Benchmark";
    }
  });

  function normalizeSummary(data) {
    if (data.summary && typeof data.summary === "object") return data.summary;
    if (data.baseline_summary && typeof data.baseline_summary === "object") return data.baseline_summary;
    const out = {};
    if (Array.isArray(data.baselines)) {
      data.baselines.forEach((b) => {
        const name = b.policy || b.name;
        if (name) out[name] = { precision: b.precision, recall: b.recall, f1: b.f1, return: b.return, steps: b.steps, runtime_ms: b.runtime_ms };
      });
    }
    return out;
  }

  function renderBenchmarkFromAny(data) {
    const summary = normalizeSummary(data);
    const names = Object.keys(summary);
    if (!names.length) return;
    $("bench-empty").style.display = "none";
    $("bench-results").style.display = "block";
    const matrix = data.episode_matrix || [];
    $("bench-ep-pill").textContent = `${data.num_episodes || matrix.length || 20} eval episodes`;

    $("bench-tbody").innerHTML = names.map((name) => {
      const s = summary[name];
      return `<tr><td class="mono" style="font-weight:700;">${esc(name)}</td><td class="num">${Number(s.precision).toFixed(4)}</td><td class="num">${Number(s.recall).toFixed(4)}</td><td class="num" style="font-weight:700;">${Number(s.f1).toFixed(4)}</td><td class="num ${s.return > 0 ? "val-good" : "val-bad"}" style="font-weight:700;">${Number(s.return).toFixed(4)}</td><td class="num">${Number(s.steps).toFixed(2)}</td><td class="num">${Number(s.runtime_ms).toFixed(2)}</td></tr>`;
    }).join("");

    // SVG Grouped Bar Chart
    const W = 520, H = 210, padL = 40, padR = 14, padT = 16, padB = 40;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const yOf = (v) => padT + ((1.0 - Math.max(-1.0, Math.min(1.0, v))) / 2.0) * plotH;
    const zeroY = yOf(0), groupW = plotW / names.length, barW = Math.min(30, groupW * 0.28);
    const grid = [1, 0.5, 0, -0.5, -1].map((v) => `<line x1="${padL}" y1="${yOf(v).toFixed(1)}" x2="${W - padR}" y2="${yOf(v).toFixed(1)}" stroke="${v === 0 ? "rgba(148,163,184,0.45)" : "rgba(148,163,184,0.14)"}" /><text x="${padL - 6}" y="${(yOf(v) + 3).toFixed(1)}" text-anchor="end" fill="#64748b" font-size="9.5" font-family="monospace">${v.toFixed(1)}</text>`).join("");
    const bars = names.map((name, i) => {
      const s = summary[name], f1 = Number(s.f1 || 0), ret = Number(s.return || 0), cx = padL + groupW * i + groupW / 2;
      const f1Top = yOf(f1), f1H = Math.max(1, zeroY - f1Top);
      const retTop = ret >= 0 ? yOf(ret) : zeroY, retH = Math.max(1, Math.abs(yOf(ret) - zeroY));
      const shortN = name.replace("_reconciliation", "_recon").replace("_everything_attack", "_attack").replace("_explorer", "_expl");
      return `<rect x="${(cx - barW - 2).toFixed(1)}" y="${f1Top.toFixed(1)}" width="${barW.toFixed(1)}" height="${f1H.toFixed(1)}" rx="3" fill="#06b6d4" /><rect x="${(cx + 2).toFixed(1)}" y="${retTop.toFixed(1)}" width="${barW.toFixed(1)}" height="${retH.toFixed(1)}" rx="3" fill="${ret >= 0 ? "#10b981" : "#f43f5e"}" /><text x="${cx.toFixed(1)}" y="${H - 12}" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="monospace">${esc(shortN)}</text>`;
    }).join("");
    $("bench-svg-chart").innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-height:220px; display:block;">${grid}${bars}</svg>`;

    // Category Breakdown
    const cb = data.category_breakdown || {};
    const cats = Object.keys(cb);
    $("bench-cat-table").innerHTML = cats.length ? `
      <table>
        <thead><tr><th>Category</th>${names.map((n) => `<th class="num">${esc(n)}</th>`).join("")}</tr></thead>
        <tbody>${cats.map((c) => `<tr><td class="mono">${esc(c)}</td>${names.map((n) => `<td class="num">${cb[c][n] ? Number(cb[c][n].mean_f1).toFixed(4) : "—"}</td>`).join("")}</tr>`).join("")}</tbody>
      </table>
    ` : `<div class="empty-state">No category breakdown available.</div>`;

    // Heatmap
    const cols = ["oracle", "simple_reconciliation", "random_explorer", "flag_everything_attack"];
    $("bench-heatmap-tbody").innerHTML = matrix.map((ep) => `
      <tr>
        <td class="mono col-nowrap">${esc(ep.seed)}</td>
        <td class="col-nowrap"><span class="pill pill-sky">${esc(ep.category)}</span></td>
        <td style="color:var(--text-muted);">${esc(ep.label)}</td>
        <td class="num">${esc(ep.num_issues)}</td>
        ${cols.map((c) => {
          const r = (ep.results || {})[c];
          if (!r) return `<td style="text-align:center;">—</td>`;
          const f1 = Number(r.f1 ?? 0), rew = Number(r.reward ?? 0);
          const bg = f1 >= 0.99 && rew >= 0.8 ? "var(--emerald-soft)" : (f1 >= 0.5 ? "var(--amber-soft)" : "var(--rose-soft)");
          const fg = f1 >= 0.99 && rew >= 0.8 ? "var(--emerald)" : (f1 >= 0.5 ? "var(--amber)" : "var(--rose)");
          return `<td><div class="hm-cell" style="background:${bg}; color:${fg};">F1 ${f1.toFixed(2)}<span class="hm-sub">R ${rew >= 0 ? "+" : ""}${rew.toFixed(2)}</span></div></td>`;
        }).join("")}
      </tr>
    `).join("");
  }

  // =========================================================================
  // PAGE 7: TRAJECTORY EXPLORER
  // =========================================================================
  function stopReplay() {
    if (state.replay.timer) { clearInterval(state.replay.timer); state.replay.timer = null; }
    $("btn-step-play").innerHTML = "&#9654; Play";
  }

  async function loadTrajectories() {
    stopReplay();
    showError("traj-error", "");
    try {
      const data = await api("/api/trajectories");
      state.trajectories = data.episodes || [];
      const sel = $("traj-select");
      if (!state.trajectories.length) {
        sel.innerHTML = `<option value="">No recorded trajectories found</option>`;
        $("step-stage").innerHTML = `<div class="empty-state">Run an agent or <code>python3 -m audit_rl generate-datasets</code> to populate trajectories.</div>`;
        return;
      }
      sel.innerHTML = state.trajectories.map((ep, i) =>
        `<option value="${i}">[${esc(ep.source)}] ${esc(ep.episode_id)} · ${esc(ep.policy)} (${ep.step_count} steps)</option>`
      ).join("");
      selectTrajectory(0);
    } catch (err) {
      showError("traj-error", err.message);
    }
  }
  $("btn-traj-refresh").addEventListener("click", loadTrajectories);
  $("traj-select").addEventListener("change", (e) => { stopReplay(); selectTrajectory(parseInt(e.target.value, 10)); });

  function selectTrajectory(idx) {
    const ep = state.trajectories[idx];
    if (!ep) return;
    state.replay.steps = ep.steps || [];
    state.replay.idx = 0;
    const slider = $("step-slider");
    slider.min = "0";
    slider.max = String(Math.max(0, state.replay.steps.length - 1));
    slider.value = "0";
    $("timeline-strip").innerHTML = state.replay.steps.map((s, i) =>
      `<button class="step-dot-btn" data-sidx="${i}">${String(s.step).padStart(2, "0")}·${esc(s.action_type)}</button>`
    ).join("");
    document.querySelectorAll("#timeline-strip .step-dot-btn").forEach((b) => {
      b.addEventListener("click", () => { stopReplay(); renderReplayStep(parseInt(b.dataset.sidx, 10)); });
    });
    renderReplayStep(0);
  }

  function renderReplayStep(idx) {
    const steps = state.replay.steps;
    if (!steps.length) return;
    const c = Math.max(0, Math.min(steps.length - 1, idx));
    state.replay.idx = c;
    $("step-slider").value = String(c);
    const s = steps[c];
    $("traj-step-pill").textContent = `Step ${s.step} / ${steps.length}`;
    $("btn-step-prev").disabled = c === 0;
    $("btn-step-next").disabled = c === steps.length - 1;
    document.querySelectorAll("#timeline-strip .step-dot-btn").forEach((b, i) => b.classList.toggle("active", i === c));
    $("step-stage").innerHTML = `
      <div class="meta-strip" style="margin-bottom:10px;">
        <span>step: <strong>${s.step}</strong></span>
        <span>action: <strong>${esc(s.action_type)}</strong></span>
        <span>summary: <strong>${esc(s.observation_summary)}</strong></span>
      </div>
      <div class="two-col">
        <div><label style="display:block; margin-bottom:5px;">Action</label><pre class="json-box">${esc(JSON.stringify(s.action, null, 2))}</pre></div>
        <div><label style="display:block; margin-bottom:5px;">Observation</label><pre class="json-box">${esc(JSON.stringify(s.observation, null, 2))}</pre></div>
      </div>
    `;
  }
  $("btn-step-prev").addEventListener("click", () => { stopReplay(); renderReplayStep(state.replay.idx - 1); });
  $("btn-step-next").addEventListener("click", () => { stopReplay(); renderReplayStep(state.replay.idx + 1); });
  $("step-slider").addEventListener("input", (e) => { stopReplay(); renderReplayStep(parseInt(e.target.value, 10)); });
  $("btn-step-play").addEventListener("click", () => {
    if (state.replay.timer) { stopReplay(); return; }
    if (state.replay.idx >= state.replay.steps.length - 1) renderReplayStep(0);
    $("btn-step-play").innerHTML = "&#10074;&#10074; Pause";
    state.replay.timer = setInterval(() => {
      if (state.replay.idx >= state.replay.steps.length - 1) stopReplay();
      else renderReplayStep(state.replay.idx + 1);
    }, 650);
  });

  // =========================================================================
  // PAGE 8: SETTINGS
  // =========================================================================
  async function loadSettings() {
    try {
      const data = await api("/api/settings");
      $("settings-grid").innerHTML = `
        <div class="card"><div class="card-header"><h2 class="card-title">Environment Contract</h2></div><pre class="json-box">${esc(JSON.stringify(data.environment, null, 2))}</pre></div>
        <div class="card"><div class="card-header"><h2 class="card-title">Dataset Splits &amp; Isolation</h2></div><pre class="json-box">${esc(JSON.stringify({ datasets: data.datasets, isolation: data.isolation, training_defaults: data.training_defaults }, null, 2))}</pre></div>
      `;
    } catch (_) {}
  }

  // Init
  async function init() {
    try {
      const cfgs = await api("/api/configs");
      state.configs = cfgs;
      populateEpisodes("ins-episode", $("ins-split").value);
      populateEpisodes("ws-episode", $("ws-split").value);
      loadOverview();
      loadInspector(true);
    } catch (err) {
      showError("ov-error", "Failed to initialize: " + err.message);
    }
  }
  init();
})();
</script>
</body>
</html>
"""
