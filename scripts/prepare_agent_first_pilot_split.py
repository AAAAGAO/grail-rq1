#!/usr/bin/env python3
"""Create the deterministic Official/Resources Agent pilot split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_DATASETS = ("official", "resources")
DEFAULT_SALT = "agent-first-pilot-v1"


def build_split(datasets: list[str], query_count: int, salt: str) -> dict[str, object]:
    if query_count < 3:
        raise ValueError("query_count must be at least 3")
    dev_count = query_count // 3
    payload: dict[str, object] = {
        "schema_version": 1,
        "salt": salt,
        "query_count_per_dataset": query_count,
        "dev_count_per_dataset": dev_count,
        "test_count_per_dataset": query_count - dev_count,
        "datasets": {},
    }
    split_datasets = payload["datasets"]
    assert isinstance(split_datasets, dict)
    for dataset in datasets:
        ranked = sorted(
            range(1, query_count + 1),
            key=lambda query_id: hashlib.sha256(
                f"{salt}|{dataset}|{query_id}".encode("utf-8")
            ).hexdigest(),
        )
        split_datasets[dataset] = {
            "dev": sorted(ranked[:dev_count]),
            "test": sorted(ranked[dev_count:]),
        }
    return payload


def load_split(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_count = int(payload["query_count_per_dataset"])
    for dataset, parts in payload["datasets"].items():
        dev = [int(item) for item in parts["dev"]]
        test = [int(item) for item in parts["test"]]
        if set(dev).intersection(test):
            raise ValueError(f"split overlap for {dataset}")
        if set(dev).union(test) != set(range(1, expected_count + 1)):
            raise ValueError(f"split does not cover all queries for {dataset}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--query-count", type=int, default=30)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    args = parser.parse_args()
    payload = build_split(
        [item.strip() for item in args.datasets.split(",") if item.strip()],
        args.query_count,
        args.salt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
