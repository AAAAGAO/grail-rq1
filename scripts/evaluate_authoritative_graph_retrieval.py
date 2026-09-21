#!/usr/bin/env python3
"""Evaluate one-hop authoritative graph retrieval without using gold labels.

This is deliberately a small, frozen decision experiment.  Every method gets
the same 15-API budget and the same final pair scorer.  The graph method may
replace text candidates only through one authoritative relation action; it
cannot use labels, perform an uncontrolled second hop, or return connector-only
nodes that have no KU.
"""

from __future__ import annotations

try:
    from scripts.graph_knowledge_retention import knowledge_entity_id, retain_knowledge_nodes
except ModuleNotFoundError:
    from graph_knowledge_retention import knowledge_entity_id, retain_knowledge_nodes

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Iterable

from rank_bm25 import BM25Okapi

try:
    from scripts.evaluate_authoritative_graph_opportunity import (
        _query_gt_nodes,
        _read_csv,
        _read_queries,
        build_adjacency,
        build_target_documents,
        normalize_label,
        tokenize,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from evaluate_authoritative_graph_opportunity import (
        _query_gt_nodes,
        _read_csv,
        _read_queries,
        build_adjacency,
        build_target_documents,
        normalize_label,
        tokenize,
    )


def score_documents(
    query: str,
    documents: dict[str, dict[str, str]],
    field: str,
) -> dict[str, float]:
    identifiers = sorted(documents)
    corpus = [tokenize(documents[item][field]) for item in identifiers]
    model = BM25Okapi(corpus)
    scores = model.get_scores(tokenize(query))
    return {item: float(score) for item, score in zip(identifiers, scores)}


def ordered_by_score(scores: dict[str, float], allowed: Iterable[str] | None = None) -> list[str]:
    identifiers = set(scores) if allowed is None else set(allowed)
    return sorted(identifiers, key=lambda item: (-scores[item], item))


def expand_one_hop_targets(
    seeds: list[str],
    adjacency: dict[str, list[tuple[str, str]]],
    target_ids: set[str],
) -> tuple[set[str], dict[str, set[str]]]:
    seed_set = set(seeds)
    expanded: set[str] = set()
    provenance: dict[str, set[str]] = defaultdict(set)
    for seed in seeds:
        for neighbor, relation in adjacency.get(seed, []):
            if neighbor not in target_ids or neighbor in seed_set:
                continue
            expanded.add(neighbor)
            provenance[neighbor].add(relation)
    return expanded, dict(provenance)


def ranking_metrics(
    order: list[str],
    positive: set[str],
    *,
    cutoffs: tuple[int, ...] = (5, 10, 15),
) -> dict[str, float]:
    flags = [item in positive for item in order]
    result = {f"P@{cutoff}": sum(flags[:cutoff]) / cutoff for cutoff in cutoffs}
    result["MRR"] = next((1.0 / rank for rank, flag in enumerate(flags, start=1) if flag), 0.0)
    return result


def api_recall_metrics(
    order: list[str],
    positive: set[str],
    *,
    cutoffs: tuple[int, ...] = (5, 10, 15),
) -> dict[str, float]:
    denominator = len(positive)
    return {
        f"APIRecall@{cutoff}": (
            len(set(order[:cutoff]).intersection(positive)) / denominator if denominator else 0.0
        )
        for cutoff in cutoffs
    }


def build_pair_documents(
    rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, list[str]]]:
    try:
        from scripts.graph_reference_overrides import apply_reference_overrides
    except ModuleNotFoundError:
        from graph_reference_overrides import apply_reference_overrides
    rows = apply_reference_overrides(rows)
    documents: dict[str, dict[str, str]] = {}
    lookup: dict[str, dict[str, str]] = {}
    by_api: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        canonical = knowledge_entity_id(row)
        pair_id = row.get("pair_id", "")
        if not pair_id:
            continue
        text = " ".join(
            value for value in (
                row.get("raw_api", ""), canonical, row.get("reference", ""), row.get("ku", ""),
            ) if value
        )
        documents[pair_id] = {"evidence": text}
        lookup[pair_id] = row
        by_api[canonical].append(pair_id)
    return documents, lookup, dict(by_api)


def rank_pairs_for_apis(
    pair_scores: dict[str, float],
    pair_lookup: dict[str, dict[str, str]],
    api_order: list[str],
) -> list[str]:
    """Rank all eligible pairs from selected APIs with a deterministic tie break."""
    api_rank = {api: index for index, api in enumerate(api_order)}
    candidates = [
        pair_id for pair_id, row in pair_lookup.items()
        if knowledge_entity_id(row) in api_rank
    ]
    return sorted(
        candidates,
        key=lambda pair_id: (
            -pair_scores[pair_id],
            api_rank[knowledge_entity_id(pair_lookup[pair_id])],
            pair_id,
        ),
    )


def _pair_positives(
    pair_lookup: dict[str, dict[str, str]],
    gt_nodes: set[str],
) -> set[str]:
    return {
        pair_id for pair_id, row in pair_lookup.items()
        if row.get("relevant_groundtruth") == "1" and knowledge_entity_id(row) in gt_nodes
    }


def choose_candidates(
    method: str,
    re_only_order: list[str],
    re_ku_order: list[str],
    re_ku_scores: dict[str, float],
    adjacency: dict[str, list[tuple[str, str]]],
    target_ids: set[str],
    *,
    seed_k: int,
    budget: int,
) -> tuple[list[str], dict[str, set[str]], int]:
    if method == "text_re_only":
        selected = re_only_order[:budget]
        return ordered_by_score(re_ku_scores, selected), {}, len(selected)
    if method == "text_re_ku":
        selected = re_ku_order[:budget]
        return selected, {}, len(selected)
    if method != "graph_one_hop":
        raise ValueError(f"Unknown method: {method}")

    seeds = re_only_order[:seed_k]
    expanded, provenance = expand_one_hop_targets(seeds, adjacency, target_ids)
    pool = set(seeds).union(expanded)
    # If the relation frontier is small, fill from exactly the same text list.
    for node in re_only_order:
        if len(pool) >= budget:
            break
        pool.add(node)
    selected = ordered_by_score(re_ku_scores, pool)[:budget]
    selected_set = set(selected)
    selected_provenance = {
        node: relations for node, relations in provenance.items() if node in selected_set
    }
    return selected, selected_provenance, len(pool)


def macro_average(rows: list[dict[str, object]], metrics: list[str]) -> dict[str, float]:
    return {metric: mean(float(row[metric]) for row in rows) for metric in metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/jodatime_api_re_ku_disambiguated.csv"))
    parser.add_argument("--nodes", type=Path, default=Path("eval_outputs/authoritative_api_graph_jodatime_20260917/nodes.csv"))
    parser.add_argument("--edges", type=Path, default=Path("eval_outputs/authoritative_api_graph_jodatime_20260917/edges.csv"))
    parser.add_argument("--queries", type=Path, default=Path("eval_outputs/query_revised_9datasets_20260911/jodatime.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs/authoritative_api_graph_retrieval_jodatime_20260917"))
    parser.add_argument("--seed-k", type=int, default=5)
    parser.add_argument("--api-budget", type=int, default=15)
    args = parser.parse_args()

    dataset_rows = _read_csv(args.dataset)
    nodes = _read_csv(args.nodes)
    edges = _read_csv(args.edges)
    queries = _read_queries(args.queries)
    api_documents, aliases = build_target_documents(dataset_rows)
    pair_documents, pair_lookup, _ = build_pair_documents(dataset_rows)
    target_ids = {node["node_id"] for node in nodes if node.get("has_ku") == "1"}
    adjacency = build_adjacency(edges)
    methods = ("text_re_only", "text_re_ku", "graph_one_hop")
    metric_names = [
        "P@5", "P@10", "P@15", "MRR",
        "APIRecall@5", "APIRecall@10", "APIRecall@15",
    ]
    detail: list[dict[str, object]] = []
    relation_actions: Counter[str] = Counter()

    for query in queries:
        query_id = int(query["query_id"])
        query_text = str(query["query"])
        gt_nodes, uncovered = _query_gt_nodes(query, aliases)
        if not gt_nodes:
            continue
        pair_positive = _pair_positives(pair_lookup, gt_nodes)
        re_only_scores = score_documents(query_text, api_documents, "re_only")
        re_ku_scores = score_documents(query_text, api_documents, "re_ku")
        pair_scores = score_documents(query_text, pair_documents, "evidence")
        re_only_order = ordered_by_score(re_only_scores)
        re_ku_order = ordered_by_score(re_ku_scores)

        for method in methods:
            api_order, provenance, pool_size = choose_candidates(
                method, re_only_order, re_ku_order, re_ku_scores,
                adjacency, target_ids, seed_k=args.seed_k, budget=args.api_budget,
            )
            pair_order = rank_pairs_for_apis(pair_scores, pair_lookup, api_order)
            metrics = ranking_metrics(pair_order, pair_positive)
            metrics.update(api_recall_metrics(api_order, gt_nodes))
            if method == "graph_one_hop":
                for relations in provenance.values():
                    relation_actions.update(relations)
            detail.append({
                "query_id": query_id,
                "query": query_text,
                "method": method,
                **metrics,
                "gt_api_count": len(gt_nodes),
                "positive_pair_count": len(pair_positive),
                "uncovered_gt_labels": " | ".join(uncovered),
                "candidate_pool_size_before_budget": pool_size,
                "selected_apis": " | ".join(api_order),
                "selected_graph_apis": " | ".join(sorted(provenance)),
                "selected_graph_relations": json.dumps(
                    {node: sorted(relations) for node, relations in sorted(provenance.items())},
                    ensure_ascii=False,
                ),
                "ranked_pairs": " | ".join(pair_order[:15]),
            })

    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in detail:
        by_method[str(row["method"])].append(row)
    macro = {method: macro_average(by_method[method], metric_names) for method in methods}
    graph_delta = {
        baseline: {
            metric: macro["graph_one_hop"][metric] - macro[baseline][metric]
            for metric in metric_names
        }
        for baseline in ("text_re_only", "text_re_ku")
    }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_query.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    result = {
        "protocol": {
            "seed": f"top-{args.seed_k} APIs by RE-only BM25",
            "action": "one authoritative relation hop, then evidence reading and RE+KU reranking",
            "api_budget": args.api_budget,
            "final_output": "ranked <API, KU> pairs",
            "ground_truth_use": "scoring only",
            "fairness": "same API budget and same pair scorer for every method",
        },
        "query_count": len(by_method["graph_one_hop"]),
        "target_api_count": len(target_ids),
        "metrics": macro,
        "graph_delta": graph_delta,
        "selected_relation_action_counts": dict(relation_actions),
    }
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
