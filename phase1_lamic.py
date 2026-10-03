import argparse

from rakg.llm import Client
from rakg.phase1 import identify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--demonstrations")
    parser.add_argument("--model")
    parser.add_argument("--endpoint")
    parser.add_argument("--run-llm", action="store_true")
    args = parser.parse_args()
    client = Client(endpoint=args.endpoint, model=args.model) if args.run_llm else None
    identify(
        args.input,
        args.output,
        demonstrations_path=args.demonstrations,
        model_name=args.model,
        client=client,
    )


if __name__ == "__main__":
    main()
