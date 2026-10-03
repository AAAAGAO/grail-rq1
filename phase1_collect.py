import argparse

from rakg.phase1 import collect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tutorial", required=True)
    parser.add_argument("--stackoverflow", required=True)
    parser.add_argument("--specification", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-output")
    parser.add_argument("--dataset", default="")
    args = parser.parse_args()
    collect(
        args.tutorial,
        args.stackoverflow,
        args.specification,
        args.output,
        dataset=args.dataset,
        reference_output_path=args.reference_output,
    )


if __name__ == "__main__":
    main()
