#!/usr/bin/env python3
"""Measure whether an official API graph can recover text-missed target APIs."""

from __future__ import annotations

try:
    from scripts.graph_knowledge_retention import knowledge_entity_id, retain_knowledge_nodes
except ModuleNotFoundError:
    from graph_knowledge_retention import knowledge_entity_id, retain_knowledge_nodes

import argparse
from collections import Counter, defaultdict, deque
import csv
import json
from pathlib import Path
import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    split_camel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z0-9]+", split_camel.casefold())


def normalize_label(text: str) -> str:
    return "".join(tokenize(text))


def build_adjacency(
    edges: list[dict[str, str]],
    excluded_relations: set[str] | None = None,
) -> dict[str, list[tuple[str, str]]]:
    excluded = excluded_relations or set()
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in edges:
        relation = edge["relation"]
        if relation in excluded:
            continue
        source, target = edge["source"], edge["target"]
        adjacency[source].append((target, relation))
        adjacency[target].append((source, relation))
    for node in adjacency:
        adjacency[node].sort()
    return adjacency


def shortest_target_paths(
    seeds: list[str],
    adjacency: dict[str, list[tuple[str, str]]],
    target_ids: set[str],
    *,
    max_hops: int,
) -> dict[str, dict[str, object]]:
    paths: dict[str, dict[str, object]] = {}
    seen: dict[str, int] = {}
    queue: deque[tuple[str, list[str], list[str]]] = deque()
    for seed in seeds:
        if seed in seen:
            continue
        seen[seed] = 0
        queue.append((seed, [seed], []))
    while queue:
        node, node_path, relations = queue.popleft()
        distance = len(relations)
        if node in target_ids and node not in paths:
            paths[node] = {
                "distance": distance,
                "nodes": node_path,
                "relations": relations,
            }
        if distance >= max_hops:
            continue
        for neighbor, relation in adjacency.get(node, []):
            next_distance = distance + 1
            if seen.get(neighbor, max_hops + 1) <= next_distance:
                continue
            seen[neighbor] = next_distance
            queue.append((neighbor, node_path + [neighbor], relations + [relation]))
    return paths


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_queries(path: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.reader(handle), start=1):
            if not row or not row[0].strip():
                continue
            result.append({
                "query_id": index,
                "query": row[0].strip(),
                "gt_labels": [item.strip() for item in row[1:] if item.strip()],
            })
    return result


def build_target_documents(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, set[str]]]:
    try:
        from scripts.graph_reference_overrides import apply_reference_overrides
    except ModuleNotFoundError:
        from graph_reference_overrides import apply_reference_overrides
    rows = apply_reference_overrides(rows)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    aliases: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        canonical = knowledge_entity_id(row)
        if not canonical:
            continue
        grouped[canonical].append(row)
        aliases[normalize_label(row.get("raw_api", ""))].add(canonical)
    documents: dict[str, dict[str, str]] = {}
    for canonical, local in grouped.items():
        raw_names = " ".join(sorted({row.get("raw_api", "") for row in local}))
        references = " ".join(dict.fromkeys(row.get("reference", "") for row in local if row.get("reference", "")))
        kus = " ".join(row.get("ku", "") for row in local)
        base = f"{raw_names} {canonical} {references}".strip()
        documents[canonical] = {"re_only": base, "re_ku": f"{base} {kus}".strip()}
    return documents, aliases


def rank_targets(query: str, documents: dict[str, dict[str, str]], field: str) -> list[str]:
    node_ids = sorted(documents)
    corpus = [tokenize(documents[node_id][field]) for node_id in node_ids]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))
    return [node_id for _, node_id in sorted(zip(scores, node_ids), key=lambda item: (-item[0], item[1]))]


def _query_gt_nodes(query: dict[str, object], aliases: dict[str, set[str]]) -> tuple[set[str], list[str]]:
    nodes: set[str] = set()
    uncovered: list[str] = []
    for label in query["gt_labels"]:
        mapped = aliases.get(normalize_label(str(label)), set())
        if mapped:
            nodes.update(mapped)
        else:
            uncovered.append(str(label))
    return nodes, uncovered


def evaluate_configuration(
    queries: list[dict[str, object]],
    rankings: dict[int, list[str]],
    aliases: dict[str, set[str]],
    adjacency: dict[str, list[tuple[str, str]]],
    target_ids: set[str],
    *,
    seed_k: int,
    max_hops: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    direct_recalls: list[float] = []
    graph_recalls: list[float] = []
    improved = 0
    recovered_total = 0
    for query in queries:
        query_id = int(query["query_id"])
        gt_nodes, uncovered = _query_gt_nodes(query, aliases)
        if not gt_nodes:
            continue
        seeds = rankings[query_id][:seed_k]
        direct_hits = gt_nodes.intersection(seeds)
        paths = shortest_target_paths(seeds, adjacency, target_ids, max_hops=max_hops)
        graph_hits = gt_nodes.intersection(paths)
        recovered = graph_hits - direct_hits
        direct_recall = len(direct_hits) / len(gt_nodes)
        graph_recall = len(graph_hits) / len(gt_nodes)
        direct_recalls.append(direct_recall)
        graph_recalls.append(graph_recall)
        improved += graph_recall > direct_recall
        recovered_total += len(recovered)
        rows.append({
            "query_id": query_id,
            "query": query["query"],
            "covered_gt_count": len(gt_nodes),
            "uncovered_gt_labels": " | ".join(uncovered),
            "seed_k": seed_k,
            "max_hops": max_hops,
            "direct_hits": " | ".join(sorted(direct_hits)),
            "graph_hits": " | ".join(sorted(graph_hits)),
            "recovered_gt": " | ".join(sorted(recovered)),
            "direct_recall": direct_recall,
            "graph_oracle_recall": graph_recall,
            "reachable_target_count": len(paths),
            "recovered_paths": json.dumps(
                {node: paths[node] for node in sorted(recovered)}, ensure_ascii=False,
            ),
        })
    count = len(direct_recalls)
    summary = {
        "evaluated_queries": float(count),
        "direct_api_recall": sum(direct_recalls) / count if count else 0.0,
        "oracle_graph_api_recall": sum(graph_recalls) / count if count else 0.0,
        "delta": (sum(graph_recalls) - sum(direct_recalls)) / count if count else 0.0,
        "improved_queries": float(improved),
        "recovered_gt_nodes": float(recovered_total),
        "mean_reachable_target_count": sum(int(row["reachable_target_count"]) for row in rows) / count if count else 0.0,
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/jodatime_api_re_ku_disambiguated.csv"))
    parser.add_argument("--nodes", type=Path, default=Path("eval_outputs/authoritative_api_graph_jodatime_20260917/nodes.csv"))
    parser.add_argument("--edges", type=Path, default=Path("eval_outputs/authoritative_api_graph_jodatime_20260917/edges.csv"))
    parser.add_argument("--queries", type=Path, default=Path("eval_outputs/query_revised_9datasets_20260911/jodatime.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/authoritative_api_graph_opportunity_jodatime_20260917"))
    args = parser.parse_args()

    dataset_rows = _read_csv(args.dataset)
    nodes = _read_csv(args.nodes)
    edges = _read_csv(args.edges)
    queries = _read_queries(args.queries)
    documents, aliases = build_target_documents(dataset_rows)
    target_ids = {node["node_id"] for node in nodes if node.get("has_ku") == "1"}
    relations = sorted({edge["relation"] for edge in edges})
    adjacency = build_adjacency(edges)

    all_summaries: dict[str, dict[str, dict[str, float]]] = {}
    all_rows: list[dict[str, object]] = []
    rankings_by_field: dict[str, dict[int, list[str]]] = {}
    for field in ("re_only", "re_ku"):
        rankings = {
            int(query["query_id"]): rank_targets(str(query["query"]), documents, field)
            for query in queries
        }
        rankings_by_field[field] = rankings
        field_summary: dict[str, dict[str, float]] = {}
        for seed_k in (3, 5, 10):
            for max_hops in (1, 2):
                summary, rows = evaluate_configuration(
                    queries, rankings, aliases, adjacency, target_ids,
                    seed_k=seed_k, max_hops=max_hops,
                )
                key = f"k{seed_k}_h{max_hops}"
                field_summary[key] = summary
                for row in rows:
                    all_rows.append({"seed_field": field, **row})
        all_summaries[field] = field_summary

    primary_rankings = rankings_by_field["re_only"]
    # One relation action must be followed by reading evidence before another
    # navigation decision.  Therefore the structural opportunity gate is one
    # hop; the two-hop figures above are diagnostics for uncontrolled spread.
    base_summary, base_rows = evaluate_configuration(
        queries, primary_rankings, aliases, adjacency, target_ids,
        seed_k=5, max_hops=1,
    )
    ablations: dict[str, dict[str, float]] = {}
    for relation in relations:
        ablated, _ = evaluate_configuration(
            queries, primary_rankings, aliases,
            build_adjacency(edges, excluded_relations={relation}), target_ids,
            seed_k=5, max_hops=1,
        )
        ablations[relation] = {
            "oracle_graph_api_recall": ablated["oracle_graph_api_recall"],
            "delta_vs_full": ablated["oracle_graph_api_recall"] - base_summary["oracle_graph_api_recall"],
            "recovered_gt_nodes": ablated["recovered_gt_nodes"],
            "recovered_gt_delta_vs_full": ablated["recovered_gt_nodes"] - base_summary["recovered_gt_nodes"],
        }

    path_sequences: Counter[str] = Counter()
    relation_in_recovered_paths: Counter[str] = Counter()
    for row in base_rows:
        recovered_paths = json.loads(str(row["recovered_paths"]))
        for path in recovered_paths.values():
            relations_in_path = path["relations"]
            path_sequences[" -> ".join(relations_in_path)] += 1
            relation_in_recovered_paths.update(set(relations_in_path))

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_query.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    result = {
        "protocol": {
            "primary": "RE-only BM25 top-5 seeds; one relation action, then mandatory evidence reading",
            "ground_truth_use": "evaluation only; never used for graph construction or ranking",
            "target_definition": "resolved APIs with at least one KU",
        },
        "target_node_count": len(target_ids),
        "query_count": len(queries),
        "summaries": all_summaries,
        "primary": base_summary,
        "relation_ablation": ablations,
        "recovered_path_sequences": dict(path_sequences),
        "relations_in_recovered_paths": dict(relation_in_recovered_paths),
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
