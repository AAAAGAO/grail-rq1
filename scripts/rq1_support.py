"""Shared data, retrieval, and evaluation helpers for the RQ1 artifact."""

from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
import re
from statistics import mean
from typing import Iterable

from rank_bm25 import BM25Okapi

from scripts.rq1_knowledge import knowledge_entity_id
from scripts.rq1_overrides import apply_reference_overrides
from scripts.text_utils import tokenize


def normalize_label(text: str) -> str:
    return "".join(tokenize(text))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_queries(path: Path) -> list[dict[str, object]]:
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


def build_target_documents(
    rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, set[str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    aliases: dict[str, set[str]] = defaultdict(set)
    for row in apply_reference_overrides(rows):
        canonical = knowledge_entity_id(row)
        if not canonical:
            continue
        grouped[canonical].append(row)
        aliases[normalize_label(row.get("raw_api", ""))].add(canonical)
    documents: dict[str, dict[str, str]] = {}
    for canonical, local in grouped.items():
        raw_names = " ".join(sorted({row.get("raw_api", "") for row in local}))
        references = " ".join(
            dict.fromkeys(row.get("reference", "") for row in local if row.get("reference"))
        )
        kus = " ".join(row.get("ku", "") for row in local)
        base = f"{raw_names} {canonical} {references}".strip()
        documents[canonical] = {
            "re_only": base,
            "re_ku": f"{base} {kus}".strip(),
        }
    return documents, aliases


def query_ground_truth(
    query: dict[str, object],
    aliases: dict[str, set[str]],
) -> tuple[set[str], list[str]]:
    nodes: set[str] = set()
    uncovered: list[str] = []
    for label in query["gt_labels"]:
        mapped = aliases.get(normalize_label(str(label)), set())
        if mapped:
            nodes.update(mapped)
        else:
            uncovered.append(str(label))
    return nodes, uncovered


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


def ordered_by_score(
    scores: dict[str, float],
    allowed: Iterable[str] | None = None,
) -> list[str]:
    identifiers = set(scores) if allowed is None else set(allowed)
    return sorted(identifiers, key=lambda item: (-scores[item], item))


def ranking_metrics(
    order: list[str],
    positive: set[str],
    *,
    cutoffs: tuple[int, ...] = (5, 10, 15),
) -> dict[str, float]:
    flags = [item in positive for item in order]
    result = {f"P@{cutoff}": sum(flags[:cutoff]) / cutoff for cutoff in cutoffs}
    result["MRR"] = next(
        (1.0 / rank for rank, flag in enumerate(flags, start=1) if flag),
        0.0,
    )
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
            len(set(order[:cutoff]).intersection(positive)) / denominator
            if denominator
            else 0.0
        )
        for cutoff in cutoffs
    }


def build_pair_documents(
    rows: list[dict[str, str]],
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, list[str]],
]:
    documents: dict[str, dict[str, str]] = {}
    lookup: dict[str, dict[str, str]] = {}
    by_api: dict[str, list[str]] = defaultdict(list)
    for row in apply_reference_overrides(rows):
        canonical = knowledge_entity_id(row)
        pair_id = row.get("pair_id", "")
        if not pair_id:
            continue
        text = " ".join(
            value
            for value in (
                row.get("raw_api", ""),
                canonical,
                row.get("reference", ""),
                row.get("ku", ""),
            )
            if value
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
    api_rank = {api: index for index, api in enumerate(api_order)}
    candidates = [
        pair_id
        for pair_id, row in pair_lookup.items()
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


def pair_positives(
    pair_lookup: dict[str, dict[str, str]],
    ground_truth_nodes: set[str],
) -> set[str]:
    return {
        pair_id
        for pair_id, row in pair_lookup.items()
        if row.get("relevant_groundtruth") == "1"
        and knowledge_entity_id(row) in ground_truth_nodes
    }


def macro_average(
    rows: list[dict[str, object]],
    metrics: list[str],
) -> dict[str, float]:
    return {
        metric: mean(float(row[metric]) for row in rows)
        for metric in metrics
    }
