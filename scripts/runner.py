"""Orchestration helpers for the graph retrieval release."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from urllib.error import HTTPError, URLError

from scripts.llm import CachedChat, TOKYO_ENDPOINT
from scripts.retrieval import DATASETS
from scripts.support import read_queries
from scripts.graph import sha256


class TransportRetryChat(CachedChat):
    """Retry transient transport failures without changing the request."""

    def call(self, system, user):
        errors = []
        for attempt in range(3):
            try:
                record = super().call(system, user)
                if errors:
                    record = {**record, "transport_retry_errors": errors}
                return record
            except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
                if isinstance(error, HTTPError) and error.code not in {
                    408, 429, 500, 502, 503, 504,
                }:
                    raise
                if attempt == 2:
                    raise
                errors.append(type(error).__name__)
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable retry state")


def job_args(args, dataset: str) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=dataset,
        configuration="adaptive_agent",
        controller_version="v12",
        candidate_retention="online",
        phase="full",
        continue_on_error=True,
        terminal_ranking="independent-score",
        relation_mask="full",
        verification_policy="online",
        graph_root=args.graph_root.resolve(),
        queries_root=args.queries_root.resolve(),
        output_dir=(args.output / dataset / "adaptive_agent").resolve(),
        model=args.model,
        endpoint=args.endpoint,
        api_key_env="DASHSCOPE_API_KEY",
    )


def preflight(args) -> dict[str, str]:
    inputs = {}
    for dataset in DATASETS:
        query_path = args.queries_root / f"{dataset}.csv"
        queries = read_queries(query_path)
        if len(queries) != 30 or {
            int(query["query_id"]) for query in queries
        } != set(range(1, 31)):
            raise ValueError(f"{dataset} must contain query IDs 1..30")
        files = [
            args.graph_root / dataset / name
            for name in (
                "knowledge_pairs.csv",
                "nodes.csv",
                "edges.csv",
                "edge_evidence.csv",
            )
        ]
        for path in [*files, query_path]:
            inputs[str(path.resolve())] = sha256(path)
    return inputs
