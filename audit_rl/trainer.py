"""Deterministic stdlib-only RL training loop for AuditLab.

Trains `LearnableAuditPolicy` strictly on training episodes using terminal rewards
from `AuditEnv(mode="train")`, without ever accessing validation/evaluation splits
or private ground truth during training, then evaluates on unseen `val` and `eval`
splits via `SubprocessAuditEnv(mode="eval")`.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any

from audit_rl.env import AuditEnv, SubprocessAuditEnv
from audit_rl.policies import (
    BasePolicy,
    DEFAULT_UNTRAINED_PARAMS,
    LearnableAuditPolicy,
    SimpleReconciliationPolicy,
    run_policy_on_env,
)
from audit_rl.storage import PlatformStore


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_split_configs(root: Path, split: str) -> dict[str, Any]:
    path = root / "configs" / f"{split}_configs.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _classify_category(cfg: dict[str, Any]) -> str:
    if cfg.get("clean", False):
        return "clean"
    if cfg.get("interacting_scenarios"):
        return "interacting"
    counts = cfg.get("error_counts") or {}
    nonzero = sum(1 for v in counts.values() if int(v) > 0)
    if nonzero == 0:
        return "clean"
    if nonzero == 1:
        return "single"
    return "multi"


def _evaluate_policy_on_split(
    policy: BasePolicy,
    episodes_spec: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate a policy on a held-out split via SubprocessAuditEnv(mode='eval')."""
    rows: list[dict[str, Any]] = []
    with SubprocessAuditEnv(mode="eval") as env:
        for ep_spec in episodes_spec:
            seed = int(ep_spec["seed"])
            cfg = ep_spec["config"]
            res = run_policy_on_env(policy, env, seed=seed, config=cfg, record_trajectory=False)
            metrics = res.get("metrics", {})
            rows.append(
                {
                    "seed": seed,
                    "episode_id": res["episode_id"],
                    "category": _classify_category(cfg),
                    "precision": round(float(metrics.get("precision", 0.0)), 4),
                    "recall": round(float(metrics.get("recall", 0.0)), 4),
                    "f1": round(float(metrics.get("f1", 0.0)), 4),
                    "return": round(float(res["reward"]), 4),
                    "steps": int(res["steps"]),
                    "runtime_ms": round(float(res["runtime_ms"]), 2),
                }
            )

    return {
        "episodes": len(rows),
        "precision": round(mean(r["precision"] for r in rows), 4) if rows else 0.0,
        "recall": round(mean(r["recall"] for r in rows), 4) if rows else 0.0,
        "f1": round(mean(r["f1"] for r in rows), 4) if rows else 0.0,
        "return": round(mean(r["return"] for r in rows), 4) if rows else 0.0,
        "steps": round(mean(r["steps"] for r in rows), 2) if rows else 0.0,
        "runtime_ms": round(mean(r["runtime_ms"] for r in rows), 2) if rows else 0.0,
        "per_episode": rows,
    }


def _rollout_train_split(
    params: dict[str, Any],
    train_episodes: list[dict[str, Any]],
    version: str,
) -> tuple[float, list[dict[str, Any]]]:
    """Run candidate params on training episodes using AuditEnv(mode='train') terminal rewards ONLY."""
    env = AuditEnv(mode="train")
    policy = LearnableAuditPolicy(params=params, version=version)
    rewards: list[float] = []
    traces: list[dict[str, Any]] = []

    for ep_spec in train_episodes:
        seed = int(ep_spec["seed"])
        cfg = ep_spec["config"]
        obs, info = env.reset(seed=seed, config=cfg)
        policy.reset(obs, info)
        terminated = False
        truncated = False
        terminal_reward = 0.0
        steps = 0
        while not (terminated or truncated):
            action = policy.act(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                terminal_reward = float(r)
        # Approximate F1 strictly for logging from evaluator report (optimization uses terminal_reward only)
        rep = env.get_evaluator_report()
        f1 = float(rep.get("metrics", {}).get("f1", 0.0))
        rewards.append(terminal_reward)
        traces.append(
            {
                "seed": seed,
                "episode_id": str(info.get("episode_id", obs.get("manifest", {}).get("episode_id", ""))),
                "reward": round(terminal_reward, 4),
                "f1": round(f1, 4),
                "steps": steps,
                "policy_version": version,
            }
        )

    return mean(rewards) if rewards else -1.0, traces


def _build_category_breakdown(
    untrained_eval: dict[str, Any],
    trained_eval: dict[str, Any],
    rule_based_eval: dict[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    categories = ("clean", "single", "multi", "interacting")
    breakdown: dict[str, dict[str, dict[str, float]]] = {}
    for cat in categories:
        cat_entry: dict[str, dict[str, float]] = {}
        for label, eval_res in (
            ("untrained", untrained_eval),
            ("trained", trained_eval),
            ("rule_based", rule_based_eval),
        ):
            subset = [r for r in eval_res["per_episode"] if r["category"] == cat]
            if subset:
                cat_entry[label] = {
                    "episodes": len(subset),
                    "f1": round(mean(r["f1"] for r in subset), 4),
                    "return": round(mean(r["return"] for r in subset), 4),
                    "steps": round(mean(r["steps"] for r in subset), 2),
                }
            else:
                cat_entry[label] = {"episodes": 0, "f1": 0.0, "return": 0.0, "steps": 0.0}
        breakdown[cat] = cat_entry
    return breakdown


def run_training_experiment(
    seed: int = 42,
    epochs: int = 2,
    train_split: str = "train",
    eval_split: str = "eval",
    save_to_store: bool = True,
    root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Train LearnableAuditPolicy on train_split via terminal rewards and evaluate on val & eval."""
    t0 = time.perf_counter()
    repo = Path(root).resolve() if root is not None else _repo_root()
    epochs = max(1, int(epochs))

    train_spec = _load_split_configs(repo, train_split)
    val_spec = _load_split_configs(repo, "val")
    eval_spec = _load_split_configs(repo, eval_split)

    train_episodes = train_spec["episodes"]
    val_episodes = val_spec["episodes"]
    eval_episodes = eval_spec["episodes"]

    params_before = dict(DEFAULT_UNTRAINED_PARAMS)
    untrained_policy = LearnableAuditPolicy(params=params_before, version="untrained-0.1.0")

    # Step 1: Evaluate untrained policy on unseen val and eval splits (SubprocessAuditEnv mode="eval")
    untrained_val = _evaluate_policy_on_split(untrained_policy, val_episodes)
    untrained_eval = _evaluate_policy_on_split(untrained_policy, eval_episodes)

    # Step 2: Deterministic evolutionary / coordinate policy-search RL on train_split ONLY
    rng = random.Random(int(seed))
    best_params = dict(params_before)
    best_reward, init_traces = _rollout_train_split(best_params, train_episodes, version="ep0-init")
    train_mean_reward_before = round(best_reward, 4)

    history: list[dict[str, Any]] = []
    ep_counter = 0
    for tr in init_traces:
        ep_counter += 1
        history.append({"episode_idx": ep_counter, "epoch": 0, **tr})

    # Structured parameter search space (evaluated strictly by scalar terminal reward)
    search_space: list[tuple[str, list[Any]]] = [
        ("submit_confidence_threshold", [0.95, 0.75, 0.50]),
        ("max_pages_per_table", [1, 5, 25]),
        ("check_duplicate", [False, True]),
        ("check_payment", [False, True]),
        ("check_fx", [False, True]),
        ("check_cutoff", [False, True]),
        ("fx_tolerance_threshold", [50000, 100, 1]),
        ("use_calculator", [False, True]),
    ]

    for epoch in range(1, epochs + 1):
        # Seeded evolutionary mutation step (helps escape flat plateaus where multiple params must move together)
        if best_params["submit_confidence_threshold"] > 0.85:
            mutated = dict(best_params)
            mutated["submit_confidence_threshold"] = 0.75
            mutated["max_pages_per_table"] = 5
            mutated[rng.choice(["check_duplicate", "check_payment", "check_fx", "check_cutoff"])] = True
            cand_reward, _ = _rollout_train_split(mutated, train_episodes, version=f"ep{epoch}-mut")
            if cand_reward > best_reward + 1e-6:
                best_reward = cand_reward
                best_params = mutated

        # Coordinate-wise policy search across parameters
        order = list(search_space)
        if epoch > 1:
            rng.shuffle(order)

        for param_name, candidates in order:
            current_val = best_params[param_name]
            for val in candidates:
                if val == current_val:
                    continue
                trial = dict(best_params)
                trial[param_name] = val
                cand_reward, _ = _rollout_train_split(trial, train_episodes, version=f"ep{epoch}-{param_name}")
                if cand_reward > best_reward + 1e-6:
                    best_reward = cand_reward
                    best_params = trial
                    current_val = val

        # Record full epoch rollout with the updated policy
        epoch_version = f"learnable-s{seed}-e{epoch}"
        epoch_reward, epoch_traces = _rollout_train_split(best_params, train_episodes, version=epoch_version)
        best_reward = max(best_reward, epoch_reward)
        for tr in epoch_traces:
            ep_counter += 1
            history.append({"episode_idx": ep_counter, "epoch": epoch, **tr})

    policy_version = f"learnable-s{seed}-e{epochs}"
    train_mean_reward_after = round(best_reward, 4)

    # Step 3: Evaluate trained policy and rule-based baseline on unseen val and eval splits
    trained_policy = LearnableAuditPolicy(params=best_params, version=policy_version)
    trained_val = _evaluate_policy_on_split(trained_policy, val_episodes)
    trained_eval = _evaluate_policy_on_split(trained_policy, eval_episodes)
    rule_based_eval = _evaluate_policy_on_split(SimpleReconciliationPolicy(), eval_episodes)

    category_breakdown = _build_category_breakdown(untrained_eval, trained_eval, rule_based_eval)
    improvement_delta = {
        "train_reward": round(train_mean_reward_after - train_mean_reward_before, 4),
        "val_f1": round(trained_val["f1"] - untrained_val["f1"], 4),
        "val_return": round(trained_val["return"] - untrained_val["return"], 4),
        "eval_f1": round(trained_eval["f1"] - untrained_eval["f1"], 4),
        "eval_return": round(trained_eval["return"] - untrained_eval["return"], 4),
    }

    duration_s = round(time.perf_counter() - t0, 2)

    # Strip per_episode lists from compact summary while keeping them in full report
    def _compact(ev: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in ev.items() if k != "per_episode"}

    experiment_id: int | None = None
    if save_to_store:
        store = PlatformStore(db_path=db_path if db_path is not None else repo / "data" / "platform.db")
        store.save_policy_version(
            policy_name=trained_policy.name,
            seed=int(seed),
            params=best_params,
            train_mean_reward=train_mean_reward_after,
            val_f1=trained_val["f1"],
            eval_f1=trained_eval["f1"],
            version_id=policy_version,
        )
        experiment_id = store.record_experiment(
            kind="train",
            policy_name=trained_policy.name,
            policy_version=policy_version,
            config_version=str(train_spec.get("config_version", "1.0.0")),
            split=f"{train_split}->{eval_split}",
            summary={
                "seed": int(seed),
                "epochs": epochs,
                "train_mean_reward_before": train_mean_reward_before,
                "train_mean_reward_after": train_mean_reward_after,
                "untrained_val": _compact(untrained_val),
                "trained_val": _compact(trained_val),
                "untrained_eval": _compact(untrained_eval),
                "trained_eval": _compact(trained_eval),
                "rule_based_eval": _compact(rule_based_eval),
                "improvement_delta": improvement_delta,
                "duration_s": duration_s,
            },
            details={
                "params_before": params_before,
                "params_after": best_params,
                "category_breakdown": category_breakdown,
                "history": history,
            },
        )

    return {
        "policy_name": trained_policy.name,
        "policy_version": policy_version,
        "seed": int(seed),
        "epochs": epochs,
        "train_split": train_split,
        "eval_split": eval_split,
        "duration_s": duration_s,
        "experiment_id": experiment_id,
        "params_before": params_before,
        "params_after": best_params,
        "train_mean_reward_before": train_mean_reward_before,
        "train_mean_reward_after": train_mean_reward_after,
        "untrained_val": _compact(untrained_val),
        "trained_val": _compact(trained_val),
        "untrained_eval": _compact(untrained_eval),
        "trained_eval": _compact(trained_eval),
        "rule_based_eval": _compact(rule_based_eval),
        "improvement_delta": improvement_delta,
        "category_breakdown": category_breakdown,
        "history": history,
    }
