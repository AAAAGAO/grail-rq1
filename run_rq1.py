"""Portable entry point for the frozen V12 Full method."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

from scripts.run_agent_first_graph_pilot import DATASETS, TOKYO_ENDPOINT, run_experiment
from scripts.run_feedback_nine_suite import job_args, TransportRetryChat

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='deepseek-v4.1-flash')
    parser.add_argument('--endpoint', default=os.environ.get('ALIYUN_ENDPOINT', TOKYO_ENDPOINT))
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--jobs', type=int, choices=range(1, 10), default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    args = parser.parse_args()
    args.graph_root = ROOT / 'data/graphs'
    args.queries_root = ROOT / 'data/queries'
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {}
        for dataset in dict.fromkeys(args.datasets):
            local = job_args(args, dataset, 'adaptive_agent')
            local.terminal_ranking = 'independent-score'
            local.relation_mask = 'full'
            local.verification_policy = 'online'
            futures[pool.submit(run_experiment, local, chat=TransportRetryChat(local))] = dataset
        for future in as_completed(futures):
            dataset = futures[future]
            summary = future.result()
            folder = args.output / dataset / 'adaptive_agent'
            rows = json.loads((folder / 'results.json').read_text(encoding='utf-8'))
            report[dataset] = {'completed': len(rows), 'expected': 30,
                'errors': summary['errors'],
                **{m: sum(r[m] for r in rows) / 30 for m in ('P@5', 'P@10', 'P@15', 'MRR')}}
            (args.output / 'rq1_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
