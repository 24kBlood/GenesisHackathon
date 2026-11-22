"""Install the data and fixtures used by the command-line checks."""

from collections import defaultdict
from pathlib import Path

from setuptools import setup


patterns = [
    "pyproject.toml",
    "benchmarks/local_eval_profiles.json",
    "schemas/*.schema.json",
    "tests/test_*.py",
    "tests/requirements-schema.lock",
    "tests/fixtures/contract/public/*",
    "tests/fixtures/contract/evaluator-only/truth.json",
    "tests/fixtures/contract/evaluator-only/provenance.json",
    "data/README.md",
    "data/checksums.json",
    "data/configs/*_configs.json",
    "trajectories/trajectories.jsonl",
]
for split in ("train", "dev"):
    for name in ("manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "policy.json"):
        patterns.append(f"data/{split}/episode-*/{name}")

data_files = defaultdict(list)
for pattern in patterns:
    for path in sorted(Path().glob(pattern)):
        if path.is_file():
            destination = Path("share/genesis-fintech-env") / path.parent
            data_files[destination.as_posix()].append(path.as_posix())

setup(data_files=sorted(data_files.items()))
