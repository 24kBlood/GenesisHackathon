"""Start a public baseline policy over the ENV-10 JSONL streams."""

import argparse
import json
import sys
from typing import BinaryIO

from genesis.environment.jsonl import FRAME_BYTES

from . import make_policy
from .protocol import Policy, PolicySession


def run(policy: Policy, incoming: BinaryIO, outgoing: BinaryIO) -> None:
    session = PolicySession(policy)
    while frame := incoming.readline(FRAME_BYTES + 1):
        if len(frame) > FRAME_BYTES or not frame.endswith(b"\n"):
            raise ValueError("invalid policy frame")
        message = json.loads(frame.decode("utf-8"))
        if type(message) is not dict:
            raise ValueError("invalid policy frame")
        action = session.respond(message)
        if action is None:
            return
        payload = (json.dumps(action, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        if len(payload) > FRAME_BYTES:
            raise ValueError("policy action exceeds frame limit")
        outgoing.write(payload)
        outgoing.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=("random", "simple_reconciliation", "mark_all"))
    parser.add_argument("--seed", type=int, default=0, help="seed for the random policy only")
    args = parser.parse_args(argv)
    try:
        run(make_policy(args.name, seed=args.seed), sys.stdin.buffer, sys.stdout.buffer)
    except (OSError, ValueError, RecursionError):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
