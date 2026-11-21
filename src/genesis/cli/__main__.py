"""Verify Genesis on the team's public, synthetic regression inputs."""

import argparse
import asyncio
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from io import TextIOBase
from pathlib import Path
from time import monotonic
from unittest import TestResult, TestLoader, TestSuite


_REPOSITORY = Path(__file__).resolve().parents[3]
_REQUIRED_MODULES = {
    "test_env_contract_and_isolation", "test_public_datasets", "test_trajectories",
    "test_reproducibility_and_permutation",
}
_GROUPS = ("schemas-and-datasets", "contract-matrix", "evaluation-matrix", "trajectories-and-pairs",
           "runtime-regressions", "docker-policy-smoke")


class _Discard(TextIOBase):
    def write(self, text):
        return len(text)


class _Summary(TestResult):
    """Track tested IDs and exception classes without printing diagnostic values."""

    def addError(self, test, error):
        self.errors.append((test, error[0].__name__))

    def addFailure(self, test, error):
        self.failures.append((test, error[0].__name__))

    def addSubTest(self, test, subtest, error):
        if error is not None:
            register = self.addFailure if issubclass(error[0], test.failureException) else self.addError
            register(test, error)


def _cases(suite):
    for item in suite:
        if isinstance(item, TestSuite):
            yield from _cases(item)
        else:
            yield item


def _test_groups(root):
    tests = root / "tests"
    suite = TestLoader().discover(str(tests), top_level_dir=str(tests))
    groups = {name: [] for name in _GROUPS}
    modules = set()
    matrix_ids = Counter()
    for case in _cases(suite):
        identifier = case.id()
        module = identifier.split(".", 1)[0]
        method = identifier.rsplit(".", 1)[-1]
        modules.add(module)
        if method.startswith(("test_C", "test_E")):
            matrix_ids[method[5:8]] += 1
        if ".DockerPolicyTests." in identifier:
            group = "docker-policy-smoke"
        elif method.startswith("test_C"):
            group = "contract-matrix"
        elif method.startswith("test_E"):
            group = "evaluation-matrix"
        elif module in ("test_env_contract_and_isolation", "test_public_datasets"):
            group = "schemas-and-datasets"
        elif module in ("test_reproducibility_and_permutation", "test_trajectories"):
            group = "trajectories-and-pairs"
        else:
            group = "runtime-regressions"
        groups[group].append(case)
    expected_ids = {f"{category}{number:02}" for category in "CE" for number in range(1, 21)}
    if set(matrix_ids) != expected_ids or any(count != 1 for count in matrix_ids.values()):
        raise ValueError("expected one test for each C01–C20 and E01–E20")
    if not _REQUIRED_MODULES <= modules or any(not members for members in groups.values()):
        raise ValueError("required verification groups are missing")
    return groups


def _run_group(name, cases, require_docker):
    started = monotonic()
    result = _Summary()
    with redirect_stdout(_Discard()), redirect_stderr(_Discard()):
        TestSuite(cases).run(result)
    adverse = result.failures + result.errors + result.unexpectedSuccesses
    skipped_ids = {test.id() for test, _ in result.skipped}
    failed_ids = {item[0].id() if isinstance(item, tuple) else item.id() for item in adverse}
    passed = len(cases) - len(skipped_ids | failed_ids)
    if result.testsRun != len(cases):  # Setup errors can prevent tests from running.
        passed = max(0, min(passed, result.testsRun - len(skipped_ids)))
    optional_skip = name == "docker-policy-smoke" and not require_docker
    ok = not adverse and (not skipped_ids or optional_skip) and result.testsRun == len(cases)
    status = "FAIL" if not ok else "SKIP" if skipped_ids else "PASS"
    print(f"{status} {name}: {passed}/{len(cases)} passed, {len(skipped_ids)} skipped "
          f"({monotonic() - started:.3f}s)")
    if adverse:
        first = adverse[0]
        case, reason = first if isinstance(first, tuple) else (first, "unexpected success")
        print(f"  first diagnostic: {case.id()} [{reason}]")
    elif skipped_ids and name == "docker-policy-smoke":
        print("  Linux Docker integration runs in the separate Policy isolation job.")
    return ok


def _verify_profiles(root):
    started = monotonic()
    group = "replay-and-permutation"
    try:
        from genesis.evaluation.reproducibility import benchmark_summary, local_benchmark, verify_reproducibility

        profiles = root / "benchmarks" / "local_eval_profiles.json"
        checks = verify_reproducibility(profiles)
        ok = checks.replay_passed == checks.pair_passed == checks.total == 20
        status = "PASS" if ok else "FAIL"
        print(f"{status} {group}: replay {checks.replay_passed}/{checks.total}, "
              f"pairs {checks.pair_passed}/{checks.total} ({monotonic() - started:.3f}s)")
        group = "policy-diagnostics"
        started = monotonic()
        benchmark = asyncio.run(local_benchmark(profiles))
        policy_counts = Counter(row.policy for row in benchmark.diagnostics)
        expected = dict.fromkeys(("random", "simple_reconciliation", "mark_all", "oracle"), 20)
        if policy_counts != expected or any(row.transport_error is not None for row in benchmark.diagnostics):
            raise ValueError("incomplete policy diagnostics")
        summary = benchmark_summary(benchmark)
    except Exception as error:
        print(f"FAIL {group} ({monotonic() - started:.3f}s): {type(error).__name__}")
        return False

    print(f"PASS {group}: {len(benchmark.diagnostics)}/80 local runs ({monotonic() - started:.3f}s)")
    for name in expected:
        metrics = summary["diagnostics"][name]
        elapsed = summary["wall_time_seconds"][name]
        print(f"  {name}: P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
              f"F1={metrics['f1']:.4f} return={metrics['return']:.4f} "
              f"steps={metrics['steps']:.2f} time={elapsed:.4f}s")
    return ok


def verify(root, *, require_docker=False):
    try:
        with redirect_stdout(_Discard()), redirect_stderr(_Discard()):
            groups = _test_groups(root)
    except (OSError, ImportError, ValueError, AssertionError) as error:
        print(f"FAIL verification setup: {type(error).__name__} (test discovery or required groups)")
        return 1
    statuses = [_run_group(name, members, require_docker) for name, members in groups.items()]
    statuses.append(_verify_profiles(root))
    return 0 if all(statuses) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify_parser = commands.add_parser("verify", help="run local regression and diagnostic checks")
    verify_parser.add_argument("--all", action="store_true", required=True)
    verify_parser.add_argument("--require-docker", action="store_true", help="fail if the Docker integration is skipped")
    args = parser.parse_args()
    return verify(_REPOSITORY, require_docker=args.require_docker)


if __name__ == "__main__":
    raise SystemExit(main())
