"""Out-of-process JSON-lines worker for isolated AuditEnv execution.

Invoked via ``python3 -m audit_rl.eval_worker`` by ``SubprocessAuditEnv``.
Reads one JSON command per line from stdin and writes one JSON response per line to stdout.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from audit_rl.env import AuditEnv
from audit_rl.schemas import DEFAULT_MAX_STEPS, GeneratedEpisode


def _write_response(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _deserialize_episode(ep_dict: dict[str, Any]) -> GeneratedEpisode:
    return GeneratedEpisode(
        manifest=ep_dict["manifest"],
        policy=ep_dict["policy"],
        ledger=ep_dict["ledger"],
        invoices=ep_dict["invoices"],
        payments=ep_dict["payments"],
        fx_rates=ep_dict["fx_rates"],
        truth=ep_dict["truth"],
        provenance=ep_dict["provenance"],
    )


def main() -> int:
    env: AuditEnv | None = None

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("Request must be a JSON object")
            cmd = req.get("cmd")

            if cmd == "init":
                mode = req.get("mode", "eval")
                max_steps = req.get("max_steps", DEFAULT_MAX_STEPS)
                timeout_seconds = req.get("timeout_seconds", 60.0)
                env = AuditEnv(
                    mode=mode,
                    max_steps=max_steps,
                    timeout_seconds=timeout_seconds,
                )
                _write_response({"ok": True})

            elif cmd == "reset":
                if env is None:
                    raise RuntimeError("Worker not initialized; send 'init' first")
                seed = req.get("seed", 0)
                config = req.get("config")
                ep_dict = req.get("episode")
                episode = _deserialize_episode(ep_dict) if isinstance(ep_dict, dict) else None
                obs, info = env.reset(seed=seed, config=config, episode=episode)
                _write_response({"ok": True, "observation": obs, "info": info})

            elif cmd == "step":
                if env is None:
                    raise RuntimeError("Worker not initialized; send 'init' first")
                action = req.get("action")
                obs, reward, terminated, truncated, info = env.step(action)
                _write_response(
                    {
                        "ok": True,
                        "observation": obs,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": info,
                    }
                )

            elif cmd == "get_evaluator_report":
                if env is None:
                    raise RuntimeError("Worker not initialized; send 'init' first")
                report = env.get_evaluator_report()
                _write_response({"ok": True, "report": report})

            elif cmd == "close":
                _write_response({"ok": True})
                return 0

            else:
                raise ValueError(f"Unknown worker command: {cmd!r}")

        except Exception as exc:
            _write_response(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
