"""Run a public episode with a baseline or a local policy image."""

import asyncio
import errno
import json
import os
from time import monotonic
from uuid import uuid4

from genesis.environment.docker_policy import DockerPolicy, PolicyCleanupError
from genesis.environment.jsonl import TransportError
from genesis.evaluation.benchmark import BenchmarkEpisode, LocalPolicyRunner
from genesis.evaluation.judge import run_episode
from genesis.evaluation.reproducibility import _diagnostics, _env, load_profiles
from genesis.generation.company import generate_episode
from genesis.generation.datasets import load_dataset_entries
from genesis.policies import make_policy
from genesis.trajectories import TrajectoryRecorder


def _resolve_profile(root, name):
    if name.startswith("episode-"):
        entries = load_dataset_entries(root / "data" / "configs")
        for entry in entries:
            if entry.config.get("episode_id") == name:
                return BenchmarkEpisode(name, entry.seed, entry.config)
        for entry in entries:
            episode = generate_episode(entry.seed, entry.config)
            if episode.public.manifest["episode_id"] == name:
                return BenchmarkEpisode(name, entry.seed, entry.config)
        raise ValueError(f"unknown public train/dev episode: {name}")

    profiles = load_profiles(root / "benchmarks" / "local_eval_profiles.json")
    for profile in profiles:
        if profile.name == name:
            return profile
    raise ValueError("unknown profile. Available: " + ", ".join(profile.name for profile in profiles))


def _publish_trace(destination, contents):
    staged = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with staged.open("xb") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(staged, destination)
        except OSError as error:
            if error.errno not in (errno.EPERM, errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)):
                raise
            if destination.exists():
                raise FileExistsError(f"trace already exists: {destination}") from None
            os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)


def run_policy(root, args):
    try:
        profile = _resolve_profile(root, args.profile)
        if args.image:
            runner = DockerPolicy(args.image, cpus=args.cpus, memory_mib=args.memory_mib, tmpfs_mib=args.tmpfs_mib)
        else:
            runner = LocalPolicyRunner(lambda: make_policy(args.policy))
        trace = TrajectoryRecorder() if args.trace else None
        outcome = asyncio.run(run_episode(
            _env(clock=monotonic), seed=profile.seed, config=profile.config, runner=runner,
            on_step=trace.record if trace is not None else None,
        ))
        result = {"profile": profile.name, **_diagnostics(outcome.terminal), "transport_error": outcome.transport_error}
        if trace is not None:
            try:
                _publish_trace(args.trace, trace.to_jsonl())
            except (OSError, ValueError) as error:
                result["trace"] = f"failed: {type(error).__name__}: {error}"
                print(f"trace publication failed: {error}")
            else:
                result["trace"] = "written"
    except TransportError as error:
        print(f"FAIL policy infrastructure: {error.code}")
        return 1
    except (OSError, ValueError, PolicyCleanupError) as error:
        print(f"FAIL policy run: {error}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return int(outcome.transport_error is not None)
