"""Graph navigation and usage helpers for the RQ1 retrieval method."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.rq1_support import read_csv
from scripts.text_utils import compact_text, tokenize


STRUCTURAL = {"SUBTYPE_OF", "NESTED_IN", "ACCEPTS", "RETURNS"}
SEMANTIC = {
    "REQUIRES",
    "COLLABORATES_WITH",
    "CONVERTS_TO",
    "ALTERNATIVE_TO",
    "CREATES",
    "CONFIGURES",
}
RELATION_MEANINGS = {
    "SUBTYPE_OF": {
        "forward": "find the abstraction or contract",
        "reverse": "find implementations or concrete alternatives",
    },
    "NESTED_IN": {
        "forward": "find the enclosing API context",
        "reverse": "find nested capabilities",
    },
    "ACCEPTS": {
        "forward": "find required input or collaborator types",
        "reverse": "find APIs that consume this type",
    },
    "RETURNS": {
        "forward": "find values produced by this API",
        "reverse": "find APIs that produce this type",
    },
    "REQUIRES": {
        "forward": "find a required prerequisite",
        "reverse": "find APIs requiring this prerequisite",
    },
    "COLLABORATES_WITH": {
        "forward": "find an API used together in the cited task",
        "reverse": "find an API used together in the cited task",
    },
    "CONVERTS_TO": {
        "forward": "find the conversion result",
        "reverse": "find APIs convertible to this type",
    },
    "ALTERNATIVE_TO": {
        "forward": "find a task-specific alternative",
        "reverse": "find a task-specific alternative",
    },
    "CREATES": {
        "forward": "find objects created by this API",
        "reverse": "find APIs that create this object",
    },
    "CONFIGURES": {
        "forward": "find the API configured by this provider",
        "reverse": "find configuration providers for this API",
    },
}


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relation_mask(name: str) -> set[str]:
    if name == "full":
        return STRUCTURAL | SEMANTIC
    if name == "structural":
        return set(STRUCTURAL)
    if name == "semantic":
        return set(SEMANTIC)
    if name == "none":
        return set()
    raise ValueError(name)


def load_graph_edges(
    folder: Path,
    allowed: set[str],
) -> list[dict[str, object]]:
    evidence: dict[str, list[str]] = defaultdict(list)
    evidence_path = folder / "edge_evidence.csv"
    if evidence_path.exists():
        for row in read_csv(evidence_path):
            text = " | ".join(
                part for part in (row.get("scope", ""), row.get("quote", ""))
                if part
            )
            if text and text not in evidence[row["edge_id"]]:
                evidence[row["edge_id"]].append(text)
    result = []
    for row in read_csv(folder / "edges.csv"):
        if row["relation"] not in allowed:
            continue
        result.append({
            **row,
            "evidence": evidence.get(row["edge_id"], [])[:3],
        })
    return result


class DirectedGraph:
    def __init__(
        self,
        edges: list[dict[str, object]],
        target_ids: set[str],
    ):
        self.target_ids = target_ids
        self.adjacency: dict[
            str,
            dict[str, dict[str, list[dict[str, object]]]],
        ] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for edge in edges:
            relation = str(edge["relation"])
            source, target = str(edge["source"]), str(edge["target"])
            self.adjacency[relation]["forward"][source].append(
                {"neighbor": target, **edge}
            )
            self.adjacency[relation]["reverse"][target].append(
                {"neighbor": source, **edge}
            )
            if str(edge.get("symmetric", "0")) == "1":
                self.adjacency[relation]["forward"][target].append(
                    {"neighbor": source, **edge}
                )
                self.adjacency[relation]["reverse"][source].append(
                    {"neighbor": target, **edge}
                )

    def expand(
        self,
        anchors: list[str],
        action_id: str,
        visited: set[str],
    ) -> dict[str, list[dict[str, object]]]:
        relation, direction = action_id.split(":", 1)
        result: dict[str, list[dict[str, object]]] = defaultdict(list)
        for anchor in anchors:
            for edge in self.adjacency.get(relation, {}).get(
                direction, {}
            ).get(anchor, []):
                neighbor = str(edge["neighbor"])
                if neighbor not in visited:
                    result[neighbor].append(edge)
        return dict(result)

    def menu(
        self,
        anchors: list[str],
        visited: set[str],
        node_lookup: dict[str, dict[str, str]],
    ) -> list[dict[str, object]]:
        menu = []
        for relation in sorted(self.adjacency):
            for direction in ("forward", "reverse"):
                action_id = f"{relation}:{direction}"
                frontier = self.expand(anchors, action_id, visited)
                if not frontier:
                    continue
                previews = []
                for node_id, edge_rows in list(sorted(frontier.items()))[:4]:
                    previews.append({
                        "api": node_id,
                        "kind": node_lookup.get(node_id, {}).get("kind", ""),
                        "evidence": [
                            compact_text(str(item), 220)
                            for row in edge_rows
                            for item in row.get("evidence", [])
                        ][:2],
                    })
                menu.append({
                    "id": action_id,
                    "meaning": RELATION_MEANINGS.get(relation, {}).get(
                        direction, "related API"
                    ),
                    "neighbor_count": len(frontier),
                    "previews": previews,
                })
        return menu


def usage(calls: list[dict[str, object]]) -> dict[str, object]:
    prompt = completion = seconds = 0.0
    providers: Counter[str] = Counter()
    for call in calls:
        local = call.get("usage", {}) if call else {}
        prompt += float(local.get("prompt_tokens", 0) or 0)
        completion += float(local.get("completion_tokens", 0) or 0)
        seconds += float(call.get("seconds", 0) or 0)
        providers[str(call.get("provider_model", "unknown"))] += 1
    return {
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
        "model_seconds": seconds,
        "provider_models": dict(providers),
        "call_count": len(calls),
    }
