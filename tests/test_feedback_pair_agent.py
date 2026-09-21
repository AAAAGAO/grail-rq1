from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from scripts.run_agent_first_graph_pilot import (
    DirectedGraph, PilotBudget, acquire_candidates, final_rerank, pilot_resume_identity,
    parse_args,
)


class ScriptedChat:
    """Replace only the external model; run the real menu, state and pool code."""

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
            for action in payload["legal_actions"]:
                if decision.get("action_id") == action.get("semantic_id"):
                    decision["action_id"] = action["id"]
                    break
        return {"content": json.dumps(decision), "usage": {}, "seconds": 0.0}


class FeedbackPairAgentTests(unittest.TestCase):
    def test_default_model_and_endpoint_are_deepseek_tokyo(self):
        argv = ["pilot", "--dataset", "resources", "--configuration", "adaptive_agent",
                "--phase", "dev", "--graph-root", "graph", "--queries-root", "queries",
                "--output-dir", "out", "--repetition", "1"]
        with patch.dict("os.environ", {}, clear=True), patch("sys.argv", argv):
            args = parse_args()
        self.assertEqual(args.model, "deepseek-v4.1-flash")
        self.assertIn("ap-northeast-1.maas.aliyuncs.com", args.endpoint)
        self.assertFalse(args.deepseek_fallback)

    def run_loop(self, chat, *, configuration="adaptive_agent", budget=None, edges=None,
                 verification_policy="online"):
        lookup = {
            **{f"a{i}": {"pair_id": f"a{i}", "canonical_api": "A", "raw_api": "A",
                         "reference": "API A", "ku": f"A example {i}",
                         "relevant_groundtruth": "1", "identified_relevance": "1"}
               for i in range(1, 10)},
            "b1": {"pair_id": "b1", "canonical_api": "B", "raw_api": "B", "ku": "B example"},
            "c1": {"pair_id": "c1", "canonical_api": "C", "raw_api": "C", "ku": "C example"},
            "d1": {"pair_id": "d1", "canonical_api": "D", "raw_api": "D", "ku": "D example"},
        }
        return acquire_candidates(
            configuration=configuration, controller_version="v12", chat=chat, query="use A",
            initial_pairs=["a1", "b1"], pair_scores={p: 20 - i for i, p in enumerate(lookup)},
            pair_lookup=lookup, graph=DirectedGraph(edges or [], {"A", "B", "C", "D"}),
            api_documents={a: {"re_only": f"API {a}"} for a in "ABCD"},
            rows_by_api={a: [r for r in lookup.values() if r["canonical_api"] == a] for a in "ABCD"},
            node_lookup={a: {"kind": "class"} for a in "ABCD"}, api_scores={},
            budget=budget or PilotBudget(),
            verification_policy=verification_policy,
        )

    def test_no_verification_preserves_adaptive_reads_and_text_order(self):
        chat = ScriptedChat([{"action_id": "READ_PAIRS:A"}, {"action_id": "STOP"}])
        pool, trace, _ = self.run_loop(chat, verification_policy="none")
        self.assertEqual(pool, [f"a{i}" for i in range(1, 8)] + ["b1"])
        self.assertEqual(trace[0]["action_policy"], "model")
        self.assertTrue(all(not t["explicit_drop_pair_ids"] and not t["restored_pair_ids"]
                            for t in trace))

    def test_no_verification_rejects_pair_updates(self):
        with self.assertRaisesRegex(ValueError, "verification is disabled"):
            self.run_loop(ScriptedChat([{"action_id": "STOP", "candidate_order": ["b1"]}] * 2),
                          verification_policy="none")

    def test_stop_is_honored_without_forced_expansion_or_replacement(self):
        pool, trace, calls = self.run_loop(ScriptedChat([{"action_id": "STOP"}]))
        self.assertEqual(pool, ["a1", "b1"])
        self.assertEqual([event["action"] for event in trace], ["STOP"])
        self.assertEqual(len(calls), 1)

    def test_entry_api_can_supply_multiple_pages_of_distinct_pairs(self):
        chat = ScriptedChat([
            {"action_id": "READ_PAIRS:A"},
            {"action_id": "READ_PAIRS:A"},
            {"action_id": "STOP", "candidate_order": [f"a{i}" for i in range(1, 10)],
             "drop_pair_ids": ["b1"]},
        ])
        pool, trace, _ = self.run_loop(chat)
        self.assertEqual(pool, [f"a{i}" for i in range(1, 10)])
        self.assertEqual(trace[0]["read_pair_ids"], ["a2", "a3", "a4", "a5", "a6", "a7"])
        self.assertEqual(trace[1]["read_pair_ids"], ["a8", "a9"])
        serialized = json.dumps(chat.payloads)
        self.assertNotIn("relevant_groundtruth", serialized)
        self.assertNotIn("identified_relevance", serialized)
        self.assertNotIn("gold", serialized)

    def test_observation_can_remove_initial_pair_and_restore_it_later(self):
        chat = ScriptedChat([
            {"action_id": "READ_PAIRS:A", "drop_pair_ids": ["a1"]},
            {"action_id": "STOP", "restore_pair_ids": ["a1"], "candidate_order": ["a1", "a2"]},
        ])
        pool, trace, _ = self.run_loop(chat)
        self.assertIn("a1", pool)
        self.assertEqual(pool[:2], ["a1", "a2"])
        self.assertIn("a1", trace[0]["dropped_pair_ids"])
        self.assertIn("a1", trace[1]["restored_pair_ids"])
        self.assertIn("a1", [c["id"] for c in chat.payloads[1]["archived_pairs"]])

    def test_expansion_can_return_to_an_earlier_anchor(self):
        edges = [
            {"source": "A", "target": "C", "relation": "RETURNS", "symmetric": "0", "evidence": []},
            {"source": "A", "target": "D", "relation": "ACCEPTS", "symmetric": "0", "evidence": []},
        ]
        pool, trace, _ = self.run_loop(ScriptedChat([
            {"action_id": "EXPAND:A:RETURNS:forward"},
            {"action_id": "EXPAND:A:ACCEPTS:forward"},
            {"action_id": "READ_PAIRS:D"},
            {"action_id": "STOP"},
        ]), edges=edges)
        self.assertEqual(trace[0]["returned_nodes"], ["C"])
        self.assertEqual(trace[1]["returned_nodes"], ["D"])
        self.assertIn("d1", pool)
        self.assertNotIn("c1", pool)  # discovering an API is not reading its KU

    def test_final_observation_reviews_last_read_when_budget_exhausted(self):
        chat = ScriptedChat([
            {"action_id": "READ_PAIRS:A"},
            {"action_id": "STOP", "drop_pair_ids": ["a2"], "candidate_order": ["a7", "a1"]},
        ])
        pool, trace, _ = self.run_loop(chat, budget=PilotBudget(max_tool_actions=1))
        self.assertNotIn("a2", pool)
        self.assertEqual(pool[:2], ["a7", "a1"])
        self.assertEqual([a["id"] for a in chat.payloads[-1]["legal_actions"]], ["STOP"])

    def test_invalid_response_cannot_silently_turn_into_a_rule_action(self):
        with self.assertRaisesRegex(ValueError, "action"):
            self.run_loop(ScriptedChat([{"action_id": "UNKNOWN"}] * 2))

    def test_invalid_action_gets_one_recorded_correction(self):
        chat = ScriptedChat([{"action_id": "UNKNOWN"}, {"action_id": "STOP"}])
        _, trace, calls = self.run_loop(chat)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(trace[0]["response_repair_errors"]), 1)
        self.assertIn("response_correction", chat.payloads[1])

    def test_fixed_observer_does_not_need_an_action(self):
        pool, trace, calls = self.run_loop(
            ScriptedChat([{"drop_pair_ids": ["b1"]}, {}]),
            configuration="fixed_graph", budget=PilotBudget(max_tool_actions=1),
        )
        self.assertEqual(trace[0]["action"], "READ_PAIRS")
        self.assertIsNone(trace[0]["proposed_action_id"])
        self.assertEqual(len(calls), 2)
        self.assertNotIn("b1", pool)

    def test_action_menu_uses_short_ids_with_exact_mapping(self):
        chat = ScriptedChat([{"action_id": "A001"}, {"action_id": "STOP"}])
        _, trace, _ = self.run_loop(chat)
        self.assertEqual(trace[0]["semantic_action_id"], "READ_PAIRS:A")
        self.assertEqual(trace[0]["action_id"], "A001")

    def test_fenced_json_with_explanation_preserves_model_action(self):
        class FencedChat:
            def call(self, system, user):
                return {"content": 'Evidence is sufficient.\n```json\n{"action_id":"STOP"}\n```'}
        pool, trace, _ = self.run_loop(FencedChat())
        self.assertEqual(pool, ["a1", "b1"])
        self.assertEqual(trace[0]["action_id"], "STOP")
        self.assertEqual(trace[0]["response_format"], "fenced_json")

    def test_trailing_json_after_prose_preserves_model_action(self):
        class ProseChat:
            def call(self, system, user):
                return {"content": 'I can stop now.\n{"action_id":"STOP","reason":"enough evidence"}'}
        pool, trace, _ = self.run_loop(ProseChat())
        self.assertEqual(pool, ["a1", "b1"])
        self.assertEqual(trace[0]["action_id"], "STOP")
        self.assertEqual(trace[0]["response_format"], "trailing_json")

    def test_invented_pair_cannot_enter_pool(self):
        with self.assertRaisesRegex(ValueError, "pair"):
            self.run_loop(ScriptedChat([{"action_id": "STOP", "candidate_order": ["invented"]}] * 2))

    def test_conflicting_pool_update_is_corrected_before_mutation(self):
        chat = ScriptedChat([
            {"action_id": "STOP", "candidate_order": ["a1"], "drop_pair_ids": ["a1"]},
            {"action_id": "STOP", "candidate_order": ["a1"], "drop_pair_ids": []},
        ])
        pool, trace, calls = self.run_loop(chat)
        self.assertEqual(pool, ["a1", "b1"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(trace[0]["response_repair_errors"]), 1)

    def test_fixed_control_uses_same_observation_but_not_model_action(self):
        pool, trace, _ = self.run_loop(ScriptedChat([
            {"action_id": "STOP", "drop_pair_ids": ["b1"]},
            {"action_id": "STOP", "candidate_order": ["a7", "a1"]},
        ]), configuration="fixed_graph", budget=PilotBudget(max_tool_actions=1))
        self.assertEqual(trace[0]["action"], "READ_PAIRS")
        self.assertNotIn("b1", pool)
        self.assertIn("a7", pool)
        self.assertEqual(pool[0], "a7")

    def test_empty_pool_returns_no_fabricated_or_backfilled_results(self):
        pool, _, _ = self.run_loop(ScriptedChat([
            {"action_id": "STOP", "drop_pair_ids": ["a1", "b1"]},
        ]))
        ranked, record = final_rerank(ScriptedChat([]), "query", pool, {})
        self.assertEqual(ranked, [])
        self.assertEqual(record["usage"], {})

    def test_checkpoint_identity_changes_when_controller_source_changes(self):
        left = {"implementation": {"controller": "before"}}
        right = {"implementation": {"controller": "after"}}
        self.assertNotEqual(pilot_resume_identity(left), pilot_resume_identity(right))

    def test_full_pool_can_replace_any_initial_pair_with_same_api_evidence(self):
        initial = [f"old{i}" for i in range(30)]
        lookup = {p: {"pair_id": p, "canonical_api": "A", "ku": p} for p in initial + ["new"]}
        chat = ScriptedChat([
            {"action_id": "READ_PAIRS:A"},
            {"action_id": "STOP", "candidate_order": ["new"] + initial[1:], "drop_pair_ids": ["old0"]},
        ])
        pool, trace, _ = acquire_candidates(
            configuration="adaptive_agent", controller_version="v12", chat=chat, query="A",
            initial_pairs=initial, pair_scores={}, pair_lookup=lookup,
            graph=DirectedGraph([], {"A"}), api_documents={}, rows_by_api={"A": list(lookup.values())},
            node_lookup={}, api_scores={}, budget=PilotBudget(),
        )
        self.assertEqual(pool, ["new"] + initial[1:])
        self.assertEqual(len(pool), 30)
        self.assertEqual(len(chat.payloads[1]["candidate_pairs"]), 31)
        self.assertIn("old0", trace[1]["dropped_pair_ids"])


if __name__ == "__main__":
    unittest.main()
