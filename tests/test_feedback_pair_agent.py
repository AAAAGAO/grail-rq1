from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts.rq1_feedback import acquire_feedback_candidates
from scripts.rq1_graph import DirectedGraph
from scripts.rq1_llm import DEFAULT_MODEL, TOKYO_ENDPOINT, normalize_endpoint
from scripts.rq1_retrieval import PilotBudget, final_rerank, run_resume_identity
from scripts.rq1_runner import job_args


class ScriptedChat:
    """Replace only the external model while exercising the real RQ1 controller."""

    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.payloads = []

    def call(self, system, user):
        payload = json.loads(user)
        self.payloads.append(payload)
        decision = next(self.decisions)
        if callable(decision):
            decision = decision(payload)
        if isinstance(decision, dict):
            decision = dict(decision)
            for action in payload.get("legal_actions", []):
                if decision.get("action_id") == action.get("semantic_id"):
                    decision["action_id"] = action["id"]
                    break
        return {"content": json.dumps(decision), "usage": {}, "seconds": 0.0}


class RQ1ControllerTests(unittest.TestCase):
    def test_default_model_and_endpoint(self):
        args = job_args(
            type(
                "Args",
                (),
                {
                    "graph_root": Path("graph"),
                    "queries_root": Path("queries"),
                    "output": Path("out"),
                    "model": DEFAULT_MODEL,
                    "endpoint": TOKYO_ENDPOINT,
                },
            )(),
            "resources",
        )
        self.assertEqual(args.model, "qwen3.8-flash")
        self.assertIn("ap-northeast-1.maas.aliyuncs.com", args.endpoint)
        self.assertEqual(
            normalize_endpoint(TOKYO_ENDPOINT + "/chat/completions"),
            TOKYO_ENDPOINT,
        )

    def run_loop(
        self,
        chat,
        *,
        budget=None,
        edges=None,
        verification_policy="online",
    ):
        lookup = {
            **{
                f"a{i}": {
                    "pair_id": f"a{i}",
                    "canonical_api": "A",
                    "raw_api": "A",
                    "reference": "API A",
                    "ku": f"A example {i}",
                }
                for i in range(1, 10)
            },
            "b1": {
                "pair_id": "b1",
                "canonical_api": "B",
                "raw_api": "B",
                "ku": "B example",
            },
            "c1": {
                "pair_id": "c1",
                "canonical_api": "C",
                "raw_api": "C",
                "ku": "C example",
            },
            "d1": {
                "pair_id": "d1",
                "canonical_api": "D",
                "raw_api": "D",
                "ku": "D example",
            },
        }
        return acquire_feedback_candidates(
            configuration="adaptive_agent",
            chat=chat,
            query="use A",
            initial_pairs=["a1", "b1"],
            pair_scores={p: 20 - i for i, p in enumerate(lookup)},
            pair_lookup=lookup,
            graph=DirectedGraph(edges or [], {"A", "B", "C", "D"}),
            api_documents={a: {"re_only": f"API {a}"} for a in "ABCD"},
            rows_by_api={
                a: [row for row in lookup.values() if row["canonical_api"] == a]
                for a in "ABCD"
            },
            node_lookup={a: {"kind": "class"} for a in "ABCD"},
            api_scores={},
            budget=budget or PilotBudget(),
            verification_policy=verification_policy,
        )

    def test_no_verification_preserves_adaptive_reads_and_text_order(self):
        chat = ScriptedChat(
            [{"action_id": "READ_PAIRS:A"}, {"action_id": "STOP"}]
        )
        pool, trace, _ = self.run_loop(chat, verification_policy="none")
        self.assertEqual(pool, [f"a{i}" for i in range(1, 8)] + ["b1"])
        self.assertEqual(trace[0]["action_policy"], "model")
        self.assertTrue(
            all(
                not event["explicit_drop_pair_ids"]
                and not event["restored_pair_ids"]
                for event in trace
            )
        )

    def test_no_verification_rejects_pair_updates(self):
        with self.assertRaisesRegex(ValueError, "verification is disabled"):
            self.run_loop(
                ScriptedChat(
                    [{"action_id": "STOP", "candidate_order": ["b1"]}] * 2
                ),
                verification_policy="none",
            )

    def test_stop_is_honored_without_forced_expansion(self):
        pool, trace, calls = self.run_loop(ScriptedChat([{"action_id": "STOP"}]))
        self.assertEqual(pool, ["a1", "b1"])
        self.assertEqual([event["action"] for event in trace], ["STOP"])
        self.assertEqual(len(calls), 1)

    def test_entry_api_can_supply_multiple_pages_of_distinct_pairs(self):
        chat = ScriptedChat(
            [
                {"action_id": "READ_PAIRS:A"},
                {"action_id": "READ_PAIRS:A"},
                {
                    "action_id": "STOP",
                    "candidate_order": [f"a{i}" for i in range(1, 10)],
                    "drop_pair_ids": ["b1"],
                },
            ]
        )
        pool, trace, _ = self.run_loop(chat)
        self.assertEqual(pool, [f"a{i}" for i in range(1, 10)])
        self.assertEqual(
            trace[0]["read_pair_ids"], ["a2", "a3", "a4", "a5", "a6", "a7"]
        )
        self.assertEqual(trace[1]["read_pair_ids"], ["a8", "a9"])
        serialized = json.dumps(chat.payloads)
        self.assertNotIn("relevant_groundtruth", serialized)
        self.assertNotIn("gold", serialized)

    def test_expansion_can_return_to_an_earlier_anchor(self):
        edges = [
            {
                "source": "A",
                "target": "C",
                "relation": "RETURNS",
                "symmetric": "0",
                "evidence": [],
            },
            {
                "source": "A",
                "target": "D",
                "relation": "ACCEPTS",
                "symmetric": "0",
                "evidence": [],
            },
        ]
        pool, trace, _ = self.run_loop(
            ScriptedChat(
                [
                    {"action_id": "EXPAND:A:RETURNS:forward"},
                    {"action_id": "EXPAND:A:ACCEPTS:forward"},
                    {"action_id": "READ_PAIRS:D"},
                    {"action_id": "STOP"},
                ]
            ),
            edges=edges,
        )
        self.assertEqual(trace[0]["returned_nodes"], ["C"])
        self.assertEqual(trace[1]["returned_nodes"], ["D"])
        self.assertIn("d1", pool)
        self.assertNotIn("c1", pool)

    def test_invalid_action_gets_one_recorded_correction(self):
        chat = ScriptedChat([{"action_id": "UNKNOWN"}, {"action_id": "STOP"}])
        _, trace, calls = self.run_loop(chat)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(trace[0]["response_repair_errors"]), 1)
        self.assertIn("response_correction", chat.payloads[1])

    def test_observation_parser_accepts_fenced_and_trailing_json(self):
        from scripts.rq1_feedback import parse_observation

        fenced, format_name = parse_observation(
            'Evidence is sufficient.\n```json\n{"action_id":"STOP"}\n```'
        )
        self.assertEqual(fenced["action_id"], "STOP")
        self.assertEqual(format_name, "fenced_json")
        trailing, format_name = parse_observation(
            'I can stop now.\n{"action_id":"STOP"}'
        )
        self.assertEqual(trailing["action_id"], "STOP")
        self.assertEqual(format_name, "trailing_json")

    def test_empty_pool_returns_no_fabricated_results(self):
        ranked, record = final_rerank(ScriptedChat([]), "query", [], {})
        self.assertEqual(ranked, [])
        self.assertEqual(record["usage"], {})

    def test_checkpoint_identity_changes_when_controller_source_changes(self):
        left = {"implementation": {"controller": "before"}}
        right = {"implementation": {"controller": "after"}}
        self.assertNotEqual(run_resume_identity(left), run_resume_identity(right))


if __name__ == "__main__":
    unittest.main()
