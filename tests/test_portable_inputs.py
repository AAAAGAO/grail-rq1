import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_feedback_nine_suite import job_args, preflight
from scripts.run_agent_first_graph_pilot import DATASETS, TOKYO_ENDPOINT, run_experiment, _read_queries

ROOT = Path(__file__).resolve().parents[1]


class OfflineChat:
    def call(self, system, user):
        payload = json.loads(user)
        if 'legal_actions' in payload:
            answer = {'action_id': 'STOP'}
        else:
            answer = {'assessments': [{'id': c['id'], 'api_fit': 2, 'ku_support': 2}
                       for c in payload['candidate_pairs']]}
        return {'content': json.dumps(answer), 'usage': {}, 'seconds': 0}


class PortableInputsTest(unittest.TestCase):
    def test_all_nine_inputs_and_offline_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(graph_root=ROOT/'data/graphs', queries_root=ROOT/'data/queries',
                output=Path(tmp), model='deepseek-v4.1-flash', endpoint=TOKYO_ENDPOINT)
            self.assertEqual(len(preflight(args)), 45)
            for dataset in DATASETS:
                local = job_args(args, dataset, 'adaptive_agent')
                local.terminal_ranking = 'independent-score'
                local.continue_on_error = False
                queries = _read_queries(args.queries_root / f'{dataset}.csv')[:1]
                with patch('scripts.run_agent_first_graph_pilot._read_queries', return_value=queries):
                    result = run_experiment(local, chat=OfflineChat())
                self.assertEqual(result['completed_queries'], 1)
                self.assertEqual(result['status'], 'complete')
