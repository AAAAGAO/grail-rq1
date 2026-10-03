import argparse
import json
from pathlib import Path

from rakg import DATASETS
from rakg.common import load_queries, validate_release, write_json
from rakg.llm import Client
from rakg.retrieval import Retriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--query")
    parser.add_argument("--query-id", type=int)
    parser.add_argument("--queries", type=Path, default=Path("queries.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--model")
    parser.add_argument("--controller", choices=("heuristic", "llm"), default="llm")
    parser.add_argument("--verifier", choices=("similarity", "llm"), default="llm")
    parser.add_argument("--endpoint")
    parser.add_argument("--max-actions", type=int, default=6)
    parser.add_argument("--max-expansions", type=int, default=3)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    queries = validate_release(args.data_root, args.queries)
    selected = queries
    if not args.all:
        if not args.dataset:
            parser.error("--dataset is required unless --all is used")
        selected = [row for row in queries if row["dataset"] == args.dataset]
        if args.query_id is not None:
            selected = [row for row in selected if row["query_id"] == args.query_id]
        if args.query:
            selected = [{
                "dataset": args.dataset,
                "query_id": 0,
                "query": args.query,
                "ground_truth_apis": [],
            }]
    client = Client(endpoint=args.endpoint) if args.controller == "llm" or args.verifier == "llm" else None
    report = []
    for row in selected:
        retriever = Retriever(
            args.data_root,
            row["dataset"],
            model_name=args.model,
            controller=args.controller,
            verifier=args.verifier,
            client=client,
            max_actions=args.max_actions,
            max_expansions=args.max_expansions,
        )
        result = retriever.retrieve(row["query"])
        result["query_id"] = row["query_id"]
        result["ground_truth_apis"] = row["ground_truth_apis"]
        report.append(result)
    output = args.output / ("results.json" if args.all else f"{args.dataset}.json")
    write_json(output, report if args.all else report[0])
    print(json.dumps({"written": str(output), "queries": len(report)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
