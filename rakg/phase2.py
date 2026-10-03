import hashlib
import re
from collections import defaultdict

from rakg.common import API_KU_RELATIONS, API_RELATIONS, KU_RELATIONS, normalize_text, read_csv, write_csv


def _identity(row):
    return hashlib.sha256((
        row.get("ku_source", "") + "\0" + normalize_text(row.get("ku", "")).casefold()
    ).encode()).hexdigest()


def _tokens(value):
    return set(re.findall(r"[a-z][a-z0-9_.$]*|\d+", str(value or "").lower()))


def _jaccard(left, right):
    union = left | right
    return len(left & right) / max(1, len(union))


def ku_candidates(pairs, same_api_k=3, cross_api_k=3):
    tokenized = {
        row["pair_id"]: _tokens(row.get("ku", ""))
        for row in pairs
    }
    result = {}
    for left in pairs:
        left_id = left["pair_id"]
        same_api = []
        cross_api = []
        for right in pairs:
            right_id = right["pair_id"]
            if left_id == right_id:
                continue
            score = _jaccard(tokenized[left_id], tokenized[right_id])
            if score <= 0:
                continue
            item = (score, right_id)
            if left["canonical_api"] == right["canonical_api"]:
                same_api.append(item)
            else:
                cross_api.append(item)
        same_api.sort(key=lambda item: (-item[0], item[1]))
        cross_api.sort(key=lambda item: (-item[0], item[1]))
        for score, right_id in [*same_api[:same_api_k], *cross_api[:cross_api_k]]:
            key = tuple(sorted((left_id, right_id)))
            result.setdefault(key, score)
        mentioned = [
            right for right in pairs
            if right["canonical_api"] != left["canonical_api"]
            and right["canonical_api"].lower() in left.get("ku", "").lower()
        ]
        for right in mentioned:
            score = _jaccard(tokenized[left_id], tokenized[right["pair_id"]])
            if score > 0:
                key = tuple(sorted((left_id, right["pair_id"])))
                result[key] = max(score, result.get(key, 0.0))
    return [
        {"left": left, "right": right, "score": score}
        for (left_id, right_id), score in sorted(
            result.items(),
            key=lambda item: (-item[1], item[0]),
        )
        for left in pairs
        if left["pair_id"] == left_id
        for right in pairs
        if right["pair_id"] == right_id
    ]


RELATION_SYSTEM = (
    "Judge the relations between two API knowledge units. Return JSON only with a "
    "relations array. Allowed relations are task_related, complements, constrains, "
    "and contrasts_with. Each item must contain relation, evidence_source, and "
    "evidence_target. Use empty relations when no relation is supported by the text."
)


def classify_ku_relations(candidates, client):
    from rakg.llm import parse_object

    edges = []
    for candidate in candidates:
        left = candidate["left"]
        right = candidate["right"]
        payload = {
            "left_api": left.get("canonical_api", ""),
            "left_ku": left.get("ku", ""),
            "right_api": right.get("canonical_api", ""),
            "right_ku": right.get("ku", ""),
        }
        result = parse_object(client.call(
            RELATION_SYSTEM,
            str(payload),
            max_tokens=1200,
        ))
        for item in result.get("relations", []):
            relation = str(item.get("relation", "")).strip().lower()
            if relation not in KU_RELATIONS - {"same_ku"}:
                continue
            edges.append({
                "source": f"{left.get('dataset', '')}:KU:{left['canonical_api']}:{left['pair_id']}",
                "target": f"{right.get('dataset', '')}:KU:{right['canonical_api']}:{right['pair_id']}",
                "relation": relation,
                "evidence": str(item.get("evidence_source", "")),
                "basis": "llm_quote_grounded",
                "evidence_source": str(item.get("evidence_source", "")),
                "evidence_target": str(item.get("evidence_target", "")),
            })
    return edges


def _api_node(dataset, api, row):
    return {
        "node_id": f"{dataset}:API:{api}",
        "dataset": dataset,
        "kind": "API",
        "api_node": "",
        "api": api,
        "canonical_owner": row.get("canonical_owner", ""),
        "pair_id": "",
        "ku_identity": "",
        "source": "",
        "task": "",
        "text": "",
        "reference": row.get("reference", ""),
        "reference_origin": row.get("reference_origin", ""),
    }


def build(
    pairs_path,
    output_dir,
    declarations_path=None,
    relations_path=None,
    client=None,
    same_api_k=3,
    cross_api_k=3,
):
    source_pairs = [
        row for row in read_csv(pairs_path)
        if row.get("identified_relevance", "1") == "1"
        and row.get("canonical_api", "").strip()
    ]
    dataset = source_pairs[0].get("dataset", "").strip() if source_pairs else ""
    dataset = dataset or "dataset"
    pairs = [
        {**row, "dataset": row.get("dataset", "").strip() or dataset}
        for row in source_pairs
    ]
    nodes = []
    api_rows = {}
    for row in pairs:
        api = row["canonical_api"].strip()
        api_rows.setdefault(api, row)
    nodes.extend(_api_node(dataset, api, api_rows[api]) for api in sorted(api_rows))
    for row in pairs:
        api = row["canonical_api"].strip()
        pair_id = row["pair_id"].strip()
        nodes.append({
            "node_id": f"{dataset}:KU:{api}:{pair_id}",
            "dataset": dataset,
            "kind": "KU",
            "api_node": f"{dataset}:API:{api}",
            "api": api,
            "pair_id": pair_id,
            "ku_identity": _identity(row),
            "source": row.get("ku_source", ""),
            "task": row.get("task", ""),
            "text": row.get("ku", ""),
            "reference": "",
            "reference_origin": "",
        })
    edges = []
    by_identity = defaultdict(list)
    for node in nodes:
        if node["kind"] == "KU":
            by_identity[node["ku_identity"]].append(node)
            pair = row_by_pair(pairs, node["pair_id"])
            roles = [
                item.strip()
                for item in pair.get("lamic_role", "").split("|")
                if item.strip()
            ]
            for role in roles:
                if role in API_KU_RELATIONS:
                    edges.append({
                        "source": node["node_id"],
                        "target": node["api_node"],
                        "relation": role,
                        "evidence": pair.get("lamic_evidence", ""),
                        "basis": "lamic",
                    })
    for group in by_identity.values():
        for left_index, left in enumerate(group):
            for right in group[left_index + 1:]:
                if left["api"] != right["api"]:
                    edges.append({
                        "source": left["node_id"],
                        "target": right["node_id"],
                        "relation": "same_ku",
                        "evidence": left["ku_identity"],
                        "basis": "deterministic_identity",
                    })
    if client:
        edges.extend(classify_ku_relations(
            ku_candidates(pairs, same_api_k, cross_api_k),
            client,
        ))
    allowed = API_RELATIONS | KU_RELATIONS | API_KU_RELATIONS
    for path, basis in ((declarations_path, "official_declaration"), (relations_path, "lamic")):
        if not path:
            continue
        for row in read_csv(path):
            relation = row.get("relation", "").strip().lower()
            if relation not in allowed:
                continue
            edges.append({
                "source": row.get("source", ""),
                "target": row.get("target", ""),
                "relation": relation,
                "evidence": row.get("evidence", ""),
                "basis": row.get("basis", basis),
            })
    node_ids = {node["node_id"] for node in nodes}
    edges = [
        edge for edge in edges
        if edge["source"] in node_ids
        and edge["target"] in node_ids
        and edge["source"] != edge["target"]
    ]
    unique = {}
    for edge in edges:
        unique[(edge["source"], edge["target"], edge["relation"])] = edge
    write_csv(output_dir / "nodes.csv", nodes)
    write_csv(output_dir / "edges.csv", unique.values())
    return {"nodes": len(nodes), "edges": len(unique)}


def row_by_pair(rows, pair_id):
    return next(row for row in rows if row.get("pair_id") == pair_id)
