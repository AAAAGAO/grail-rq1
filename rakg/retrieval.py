import json
from collections import defaultdict

from rakg.common import dataset_paths, pair_text, read_csv, write_json
from rakg.embeddings import Encoder
from rakg.graph import Graph
from rakg.llm import Client, parse_object


CONTROLLER_SYSTEM = (
    "You control adaptive API knowledge retrieval. Choose one listed action. "
    "READ_API deepens an API subgraph, EXPAND_API broadens through API relations, "
    "EXPAND_KU broadens through KU relations, and STOP terminates investigation. "
    "Use only the supplied evidence and action IDs. Return JSON only: "
    '{"action_id":"...","reason":"..."}'
)

VERIFICATION_SYSTEM = (
    "Assess each supplied API and knowledge unit pair for the query. "
    "api_fit is an integer from 0 to 3 and ku_support is an integer from 0 to 2. "
    "Return JSON only: "
    '{"assessments":[{"pair_id":"...","api_fit":0,"ku_support":0}]}'
)


class Retriever:
    def __init__(
        self,
        dataset_root,
        dataset,
        model_name=None,
        controller="llm",
        verifier="llm",
        client=None,
        max_actions=6,
        max_expansions=3,
        read_batch=6,
        candidate_cap=30,
    ):
        paths = dataset_paths(dataset_root, dataset)
        self.dataset = dataset
        self.pairs = read_csv(paths["pairs"])
        self.graph = Graph.load(paths["nodes"], paths["edges"])
        self.encoder = Encoder(model_name=model_name)
        self.controller = controller
        self.verifier = verifier
        self.client = client
        self.max_actions = max_actions
        self.max_expansions = max_expansions
        self.read_batch = read_batch
        self.candidate_cap = candidate_cap
        self.pair_by_id = {row["pair_id"]: row for row in self.pairs}
        self.node_by_pair = {
            row["pair_id"]: self.graph.pair_node(row["pair_id"])
            for row in self.pairs
        }
        self.pairs_by_api = defaultdict(list)
        for row in self.pairs:
            self.pairs_by_api[row["canonical_api"]].append(row["pair_id"])
        self.roles_by_pair = defaultdict(set)
        for edge in self.graph.edges:
            source = self.graph.node(edge["source"])
            target = self.graph.node(edge["target"])
            if source.get("kind") == "KU" and target.get("kind") == "API":
                self.roles_by_pair[source.get("pair_id", "")].add(edge["relation"])
            elif source.get("kind") == "API" and target.get("kind") == "KU":
                self.roles_by_pair[target.get("pair_id", "")].add(edge["relation"])

    def _scores(self, query):
        values = [pair_text(row) for row in self.pairs]
        scores = self.encoder.similarity(query, values)
        return {
            row["pair_id"]: float(score)
            for row, score in zip(self.pairs, scores)
        }

    def _initial(self, scores):
        return sorted(scores, key=lambda pair_id: (-scores[pair_id], pair_id))[:5]

    def _pair_preview(self, pair_id, scores):
        row = self.pair_by_id[pair_id]
        return {
            "pair_id": pair_id,
            "api": row["canonical_api"],
            "score": round(scores[pair_id], 6),
            "knowledge_unit": row.get("ku", ""),
        }

    def _actions(self, query, scores, observed, discovered, expanded):
        actions = []
        for api in sorted(discovered):
            unread_by_role = {}
            for pair_id in self.pairs_by_api.get(api, []):
                if pair_id in observed:
                    continue
                roles = self.roles_by_pair.get(pair_id, set())
                for role in sorted(roles or {"all"}):
                    unread_by_role.setdefault(role, []).append(pair_id)
            for role, unread in sorted(unread_by_role.items()):
                unread.sort(key=lambda item: (-scores[item], item))
                actions.append({
                    "id": f"READ_API:{api}:{role}",
                    "type": "READ_API",
                    "api": api,
                    "role": role,
                    "pair_ids": unread[:self.read_batch],
                })
        if expanded < self.max_expansions:
            api_targets = set()
            for api in sorted(discovered):
                source = self.graph.api_node(api).get("node_id", "")
                for target, edge in self.graph.neighbors(source):
                    if edge["relation"] not in {"subtype_of", "accepts", "returns"}:
                        continue
                    node = self.graph.node(target)
                    if node.get("kind") == "API" and node.get("api") not in discovered:
                        api_targets.add((api, edge["relation"], node.get("api", "")))
            for source, relation, target in sorted(api_targets):
                actions.append({
                    "id": f"EXPAND_API:{source}:{relation}:{target}",
                    "type": "EXPAND_API",
                    "source": source,
                    "relation": relation,
                    "target": target,
                })
            for pair_id in sorted(observed):
                node_id = self.node_by_pair[pair_id].get("node_id", "")
                for target, edge in self.graph.ku_neighbors(node_id):
                    target_node = self.graph.node(target)
                    if (
                        target_node.get("kind") == "KU"
                        and target_node.get("pair_id") not in observed
                    ):
                        actions.append({
                            "id": f"EXPAND_KU:{pair_id}:{edge['relation']}:{target_node['pair_id']}",
                            "type": "EXPAND_KU",
                            "source_pair": pair_id,
                            "relation": edge["relation"],
                            "target_pair": target_node["pair_id"],
                        })
        actions.append({"id": "STOP", "type": "STOP"})
        return actions

    def _choose(self, query, actions, evidence):
        if self.controller == "llm":
            if not self.client:
                raise RuntimeError("an LLM client is required for the llm controller")
            result = parse_object(self.client.call(
                CONTROLLER_SYSTEM,
                json.dumps({
                    "query": query,
                    "evidence": evidence,
                    "legal_actions": actions,
                }, ensure_ascii=False),
            ))
            action_id = result.get("action_id")
            if action_id not in {item["id"] for item in actions}:
                raise ValueError("the controller returned an illegal action")
            return action_id, result.get("reason", "")
        readable = [item for item in actions if item["type"] == "READ_API"]
        if readable:
            return readable[0]["id"], "deepen the highest-ranked discovered API"
        broadening = [item for item in actions if item["type"] != "STOP"]
        if broadening:
            return broadening[0]["id"], "broaden the current evidence graph"
        return "STOP", "no unread or legal graph action remains"

    def _verify(self, query, pair_ids, scores):
        if self.verifier == "llm":
            if not self.client:
                raise RuntimeError("an LLM client is required for llm verification")
            payload = {
                "query": query,
                "pairs": [
                    {
                        "pair_id": pair_id,
                        "api": self.pair_by_id[pair_id]["canonical_api"],
                        "knowledge_unit": self.pair_by_id[pair_id].get("ku", ""),
                    }
                    for pair_id in pair_ids
                ],
            }
            result = parse_object(self.client.call(
                VERIFICATION_SYSTEM,
                json.dumps(payload, ensure_ascii=False),
                max_tokens=3000,
            ))
            assessments = {
                item["pair_id"]: item
                for item in result.get("assessments", [])
                if item.get("pair_id") in pair_ids
            }
            if set(assessments) != set(pair_ids):
                raise ValueError("verification must score every retained pair")
            order = {pair_id: index for index, pair_id in enumerate(pair_ids)}
            return sorted(
                pair_ids,
                key=lambda pair_id: (
                    -int(int(assessments[pair_id]["ku_support"]) > 0),
                    -int(assessments[pair_id]["api_fit"]),
                    -int(assessments[pair_id]["ku_support"]),
                    order[pair_id],
                ),
            )
        return sorted(pair_ids, key=lambda pair_id: (-scores[pair_id], pair_id))

    def retrieve(self, query):
        scores = self._scores(query)
        initial = self._initial(scores)
        observed = set(initial)
        discovered = {
            self.pair_by_id[pair_id]["canonical_api"]
            for pair_id in initial
        }
        expanded = 0
        trace = []
        for step in range(self.max_actions):
            actions = self._actions(query, scores, observed, discovered, expanded)
            evidence = [
                self._pair_preview(pair_id, scores)
                for pair_id in sorted(observed, key=lambda item: (-scores[item], item))[:self.candidate_cap]
            ]
            action_id, reason = self._choose(query, actions, evidence)
            action = next(item for item in actions if item["id"] == action_id)
            event = {
                "step": step + 1,
                "action": action["type"],
                "action_id": action_id,
                "reason": reason,
                "read_pair_ids": [],
                "returned_pair_ids": [],
                "returned_api_ids": [],
            }
            if action["type"] == "STOP":
                trace.append(event)
                break
            if action["type"] == "READ_API":
                event["read_pair_ids"] = action["pair_ids"]
                event["role"] = action.get("role", "")
                observed.update(action["pair_ids"])
            elif action["type"] == "EXPAND_API":
                discovered.add(action["target"])
                event["returned_api_ids"] = [action["target"]]
                expanded += 1
            elif action["type"] == "EXPAND_KU":
                observed.add(action["target_pair"])
                discovered.add(self.pair_by_id[action["target_pair"]]["canonical_api"])
                event["returned_pair_ids"] = [action["target_pair"]]
                expanded += 1
            trace.append(event)
        retained = sorted(observed, key=lambda item: (-scores[item], item))[:self.candidate_cap]
        ranked = self._verify(query, retained, scores)[:15]
        return {
            "dataset": self.dataset,
            "query": query,
            "initial_pair_ids": initial,
            "observed_pair_ids": sorted(observed),
            "discovered_apis": sorted(discovered),
            "ranked_pairs": [
                {
                    "pair_id": pair_id,
                    "api": self.pair_by_id[pair_id]["canonical_api"],
                    "score": scores[pair_id],
                    "knowledge_unit": self.pair_by_id[pair_id].get("ku", ""),
                }
                for pair_id in ranked
            ],
            "trace": trace,
        }
