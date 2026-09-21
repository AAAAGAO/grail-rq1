"""Frozen nine-dataset evaluation, with explicit failures and same-model controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time
from urllib.error import HTTPError, URLError

from scripts.run_agent_first_graph_pilot import (
    DATASETS, CachedChat, TOKYO_ENDPOINT, _read_queries, run_experiment, sha256,
)

METHODS = ("pair_bm25", "fixed_graph", "adaptive_agent")


class TransportRetryChat(CachedChat):
    """Retry transport failures only, without changing model or prompt."""
    def call(self, system, user):
        errors = []
        for attempt in range(3):
            try:
                record = super().call(system, user)
                if errors:
                    record = {**record, "transport_retry_errors": errors}
                return record
            except (URLError, TimeoutError, ConnectionError) as error:
                if isinstance(error, HTTPError) and error.code not in {408, 429, 500, 502, 503, 504}:
                    raise
                if attempt == 2:
                    raise
                errors.append(type(error).__name__)
                time.sleep(2 ** attempt)


def job_args(args, dataset, method):
    return argparse.Namespace(
        dataset=dataset, configuration=method, controller_version="v12",
        candidate_retention="online", phase="full", continue_on_error=True,
        split_manifest=Path("config/agent_first_graph_pilot_split.json"),
        graph_root=args.graph_root.resolve(), queries_root=args.queries_root.resolve(),
        output_dir=args.output / dataset / method, repetition=1, workers=1,
        model=args.model, endpoint=args.endpoint, api_key_env="DASHSCOPE_API_KEY",
        deepseek_fallback=False, deepseek_model="deepseek-v4-flash",
        deepseek_endpoint="https://api.deepseek.com/chat/completions",
        deepseek_key_env="DEEPSEEK_API_KEY",
    )


def preflight(args):
    inputs = {}
    for dataset in DATASETS:
        query_path = args.queries_root / f"{dataset}.csv"
        queries = _read_queries(query_path)
        assert len(queries) == 30 and {q["query_id"] for q in queries} == set(range(1, 31)), dataset
        files = [args.graph_root / dataset / name for name in
                 ("knowledge_pairs.csv", "nodes.csv", "edges.csv", "edge_evidence.csv")]
        for path in [*files, query_path]:
            inputs[str(path.resolve())] = sha256(path)
    return inputs


def run(args):
    inputs = preflight(args)
    args.output.mkdir(parents=True, exist_ok=True)
    from scripts import feedback_pair_agent, run_agent_first_graph_pilot
    from scripts import run_authoritative_api_graph_agent, evaluate_authoritative_graph_retrieval
    from scripts import evaluate_authoritative_graph_opportunity, graph_reference_overrides
    from scripts import graph_knowledge_retention, run_grail_rq_suite
    from baselines.api_evidence_llm import baseline
    modules = [feedback_pair_agent, run_agent_first_graph_pilot, run_authoritative_api_graph_agent,
               evaluate_authoritative_graph_retrieval, evaluate_authoritative_graph_opportunity,
               graph_reference_overrides, graph_knowledge_retention, run_grail_rq_suite, baseline]
    sources = [Path(m.__file__) for m in modules] + [Path(__file__)]
    if graph_reference_overrides.REGISTRY.exists():
        sources.append(graph_reference_overrides.REGISTRY)
    frozen = {str(p.resolve()): sha256(p) for p in sources}
    methods = tuple(args.methods.split(","))
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
        raise ValueError("invalid methods")
    identity = {"model": args.model, "endpoint": args.endpoint, "datasets": list(DATASETS),
                "methods": list(methods), "expected_queries_per_run": 30,
                "expected_query_method_runs": 270 * len(methods), "inputs": inputs, "source_hashes": frozen,
                "controller_version": "v12", "candidate_retention": "online",
                "max_transport_attempts": 3,
                "evaluation_scope": "full existing nine-dataset queries, not an untouched holdout"}
    manifest_path = args.output / "suite_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert all(previous[k] == value for k, value in identity.items()), "suite identity changed"
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in sources:
        shutil.copy2(path, snapshot / path.name)
    manifest = {**identity, "status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "concurrent_jobs": args.jobs, "runs": {}}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    futures = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for dataset in DATASETS:
            for method in methods:
                local = job_args(args, dataset, method)
                futures[pool.submit(run_experiment, local, chat=TransportRetryChat(local))] = (dataset, method)
        for future in as_completed(futures):
            dataset, method = futures[future]
            try:
                result = future.result()
                manifest["runs"][f"{dataset}/{method}"] = result
            except Exception as error:
                manifest["runs"][f"{dataset}/{method}"] = {"status": "failed", "error": str(error)}
            manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"RUN FINISHED {dataset}/{method} {manifest['runs'][f'{dataset}/{method}']['status']}", flush=True)
    changed = [p for p, digest in {**inputs, **frozen}.items() if sha256(Path(p)) != digest]
    manifest["integrity_changed_files"] = changed
    manifest["status"] = ("complete" if not changed and all(r["status"] == "complete"
                          for r in manifest["runs"].values()) else "partial")
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "runs": len(manifest["runs"]),
                      "changed_files": changed}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--queries-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=9)
    parser.add_argument("--methods", default="adaptive_agent",
                        help="Only explicitly requested methods; default runs Agent alone")
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--endpoint", default=TOKYO_ENDPOINT)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 12:
        parser.error("--jobs must be between 1 and 12")
    run(args)
