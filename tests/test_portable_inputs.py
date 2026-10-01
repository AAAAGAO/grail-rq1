import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.rq1_llm import DEFAULT_MODEL, TOKYO_ENDPOINT
from scripts.rq1_retrieval import DATASETS, run_experiment
from scripts.rq1_runner import job_args, preflight
from scripts.rq1_support import read_queries


ROOT = Path(__file__).resolve().parents[1]


class OfflineChat:
    def call(self, system, user):
        payload = json.loads(user)
        if "legal_actions" in payload:
            answer = {"action_id": "STOP"}
        else:
            answer = {
                "assessments": [
                    {"id": card["id"], "api_fit": 2, "ku_support": 2}
                    for card in payload["candidate_pairs"]
                ]
            }
        return {"content": json.dumps(answer), "usage": {}, "seconds": 0}


class PortableInputsTest(unittest.TestCase):
    def test_all_nine_inputs_and_offline_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                graph_root=ROOT / "data/graphs",
                queries_root=ROOT / "data/queries",
                output=Path(tmp),
                model=DEFAULT_MODEL,
                endpoint=TOKYO_ENDPOINT,
            )
            self.assertEqual(len(preflight(args)), 45)
            for dataset in DATASETS:
                local = job_args(args, dataset)
                local.continue_on_error = False
                queries = read_queries(
                    args.queries_root / f"{dataset}.csv"
                )[:1]
                with patch(
                    "scripts.rq1_retrieval.read_queries",
                    return_value=queries,
                ):
                    result = run_experiment(local, chat=OfflineChat())
                self.assertEqual(result["completed_queries"], 1)
                self.assertEqual(result["status"], "complete")


if __name__ == "__main__":
    unittest.main()
