import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

from scripts.llm import TOKYO_ENDPOINT, DEFAULT_MODEL, normalize_endpoint
from scripts.retrieval import DATASETS, run_experiment
from scripts.runner import job_args, TransportRetryChat, preflight

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=os.environ.get('ALIYUN_MODEL', DEFAULT_MODEL))
    parser.add_argument('--endpoint', default=os.environ.get('ALIYUN_ENDPOINT', TOKYO_ENDPOINT))
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--jobs', type=int, choices=range(1, 10), default=3)
    parser.add_argument('--graph-root', type=Path, required=True)
    parser.add_argument('--queries-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    args = parser.parse_args()
    args.endpoint = normalize_endpoint(args.endpoint)
    args.graph_root = args.graph_root.resolve()
    args.queries_root = args.queries_root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    preflight(args)
    report = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {}
        for dataset in dict.fromkeys(args.datasets):
            local = job_args(args, dataset)
            futures[pool.submit(run_experiment, local, chat=TransportRetryChat(local))] = dataset
        for future in as_completed(futures):
            dataset = futures[future]
            summary = future.result()
            folder = args.output / dataset / 'adaptive_agent'
            rows = json.loads((folder / 'results.json').read_text(encoding='utf-8'))
            report[dataset] = {'completed': len(rows), 'expected': 30,
                'errors': summary['errors'],
                **{m: sum(r[m] for r in rows) / 30 for m in ('P@5', 'P@10', 'P@15', 'MRR')}}
            (args.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
