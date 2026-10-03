from collections import defaultdict

from rakg.common import API_KU_RELATIONS, API_RELATIONS, KU_RELATIONS, read_csv


class Graph:
    def __init__(self, nodes, edges):
        self.nodes = {row["node_id"]: row for row in nodes}
        self.edges = []
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)
        for row in edges:
            relation = row.get("relation", "").strip().lower()
            if relation not in API_RELATIONS | KU_RELATIONS | API_KU_RELATIONS:
                continue
            edge = {**row, "relation": relation}
            self.edges.append(edge)
            self.outgoing[edge["source"]].append(edge)
            self.incoming[edge["target"]].append(edge)

    @classmethod
    def load(cls, nodes_path, edges_path):
        return cls(read_csv(nodes_path), read_csv(edges_path))

    def node(self, node_id):
        return self.nodes.get(node_id, {})

    def api_node(self, api):
        return next(
            (
                row for row in self.nodes.values()
                if row.get("kind") == "API" and row.get("api") == api
            ),
            {},
        )

    def pair_node(self, pair_id):
        return next(
            (
                row for row in self.nodes.values()
                if row.get("kind") == "KU" and row.get("pair_id") == pair_id
            ),
            {},
        )

    def neighbors(self, node_id, relations=None):
        relations = set(relations or ())
        candidates = [
            (edge["target"], edge)
            for edge in self.outgoing.get(node_id, [])
        ] + [
            (edge["source"], edge)
            for edge in self.incoming.get(node_id, [])
        ]
        return [
            (target, edge) for target, edge in candidates
            if not relations or edge["relation"] in relations
        ]

    def api_neighbors(self, api):
        node = self.api_node(api)
        return self.neighbors(node.get("node_id", ""), API_RELATIONS)

    def ku_neighbors(self, node_id):
        return self.neighbors(node_id, KU_RELATIONS)
