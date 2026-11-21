"""Run a public diagnostic profile with a baseline or a local policy image."""

import asyncio
import json
from time import monotonic

from genesis.environment.docker_policy import DockerPolicy, PolicyCleanupError
from genesis.environment.jsonl import TransportError
from genesis.evaluation.benchmark import LocalPolicyRunner
from genesis.evaluation.judge import run_episode
from genesis.evaluation.reproducibility import _diagnostics, _env, load_profiles
from genesis.policies import make_policy
from genesis.trajectories import TrajectoryRecorder


def run_policy(root, args):
    try:
        profiles = load_profiles(root / "benchmarks" / "local_eval_profiles.json")
        profile = next((profile for profile in profiles if profile.name == args.profile), None)
        if profile is None:
            print("Unknown profile. Available:", ", ".join(profile.name for profile in profiles))
            return 1
        if args.image:
            runner = DockerPolicy(args.image, cpus=args.cpus, memory_mib=args.memory_mib, tmpfs_mib=args.tmpfs_mib)
        else:
            runner = LocalPolicyRunner(lambda: make_policy(args.policy))
        trace = TrajectoryRecorder() if args.trace else None
        outcome = asyncio.run(run_episode(
            _env(clock=monotonic), seed=profile.seed, config=profile.config, runner=runner,
            on_step=trace.record if trace is not None else None,
        ))
        if trace is not None:
            contents = trace.to_jsonl()
            with args.trace.open("xb") as output:
                output.write(contents)
        result = {"profile": profile.name, **_diagnostics(outcome.terminal), "transport_error": outcome.transport_error}
    except TransportError as error:
        print(f"FAIL policy infrastructure: {error.code}")
        return 1
    except (OSError, ValueError, PolicyCleanupError) as error:
        print(f"FAIL policy run: {error}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return int(outcome.transport_error is not None)
