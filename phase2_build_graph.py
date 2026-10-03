import argparse
import json
from pathlib import Path

from rakg.llm import Client
from rakg.phase2 import build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--declarations")
    parser.add_argument("--relations")
    parser.add_argument("--model")
    parser.add_argument("--endpoint")
    parser.add_argument("--same-api-k", type=int, default=3)
    parser.add_argument("--cross-api-k", type=int, default=3)
    parser.add_argument("--run-llm", action="store_true")
    args = parser.parse_args()
    client = Client(endpoint=args.endpoint, model=args.model) if args.run_llm else None
    result = build(
        Path(args.pairs),
        Path(args.output),
        declarations_path=Path(args.declarations) if args.declarations else None,
        relations_path=Path(args.relations) if args.relations else None,
        client=client,
        same_api_k=args.same_api_k,
        cross_api_k=args.cross_api_k,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
