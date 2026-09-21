#!/usr/bin/env python3
"""No-graph API ranking baseline: BM25 pair recall -> API evidence -> LLM rank."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("jodatime", "math", "official", "jenkov", "smack", "graphics", "resources", "text", "data")
TOKYO_ENDPOINT = "https://ws-l8mmsqf3krhsrnad.ap-northeast-1.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"


@dataclass(frozen=True)
class Pair:
    pair_id: str
    api: str
    source: str
    ki: str
    truth: bool = True


def load_pairs(path: Path) -> list[Pair]:
    pairs = []
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("relevance", "").strip() != "1":
                continue
            api, ki = row.get("API", "").strip(), row.get("KI", "").strip()
            if not api or not ki:
                continue
            pairs.append(Pair(f"P{len(pairs) + 1}", api, row.get("source", "").strip(), ki,
                              row.get("relevant-groundtruth", "").strip() == "1"))
    return pairs


def load_references(path: Path, pairs: list[Pair]) -> tuple[dict[str, list[str]], list[str]]:
    """RE is extra prompt evidence, never an evaluated output pair."""
    apis = sorted({pair.api for pair in pairs})
    references: dict[str, list[str]] = {api: [] for api in apis}
    unmatched = []
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2 or (len(row) > 2 and row[2].strip() != "1"):
                continue
            full_api, description = row[0].strip(), row[1].strip()
            matches = [api for api in apis if full_api == api or full_api.endswith("." + api)]
            if len(matches) == 1 and description:
                references[matches[0]].append(description)
            elif description:
                unmatched.append(full_api)
    return references, unmatched


def load_merged_corpus(path: Path) -> dict[str, dict]:
    """Read the prepared nine-dataset API/RE/KU table once, without using truth for retrieval."""
    datasets: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset"].strip()
            api = row["api"].strip()
            if not dataset or not api:
                continue
            data = datasets.setdefault(dataset, {"pairs": [], "references": {}, "reference_origins": {}})
            reference = row["reference"].strip()
            if not reference:
                raise ValueError(f"Missing RE for {dataset}:{api}")
            if api in data["references"] and data["references"][api] != reference:
                raise ValueError(f"Conflicting RE for {dataset}:{api}")
            data["references"][api] = reference
            data["reference_origins"][api] = row["reference_origin"].strip()
            if row["identified_relevance"].strip() != "1":
                continue
            source, ki = row["ku_source"].strip(), row["ku"].strip()
            if source not in {"TU", "SO"} or not ki:
                raise ValueError(f"Invalid identified KU for {dataset}:{api}")
            pairs = data["pairs"]
            pairs.append(Pair(f"P{len(pairs) + 1}", api, source, ki,
                              row["relevant_groundtruth"].strip() == "1"))
    return datasets


def read_queries(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for i, row in enumerate(csv.reader(handle), 1):
            if row and row[0].strip():
                rows.append({"query_id": i, "query": row[0].strip(),
                             "gt_apis": [api.strip() for api in row[1:] if api.strip()]})
    return rows


def tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9_.$]*|\d+", value.lower())


def bm25_indices(pairs: list[Pair], query: str, count: int) -> tuple[list[int], list[float]]:
    corpus = [tokenize(f"{pair.api} {pair.ki}") or ["empty"] for pair in pairs]
    scores = BM25Okapi(corpus).get_scores(tokenize(query))
    indices = sorted(range(len(pairs)), key=lambda i: (-scores[i], i))[:count]
    return indices, [float(scores[i]) for i in indices]


def retrieve_pair_candidates(pairs: list[Pair], query: str, count: int) -> list[dict]:
    """Return atomic pair candidates in BM25 order without API-level pruning."""
    indices, scores = bm25_indices(pairs, query, count)
    return [{"pair": pairs[index], "score": score} for index, score in zip(indices, scores)]


class EvidenceIndex:
    """BM25 over atomic TU/SO pairs and one RE document per API."""

    def __init__(self, pairs: list[Pair], references: dict[str, str]):
        self.pairs = pairs
        self.api_to_pairs: dict[str, list[int]] = {}
        documents = []
        self.doc_apis = []
        for i, pair in enumerate(pairs):
            self.api_to_pairs.setdefault(pair.api, []).append(i)
            documents.append(tokenize(f"{pair.api} {pair.ki}") or ["empty"])
            self.doc_apis.append(pair.api)
        for api in self.api_to_pairs:
            if api not in references:
                continue
            if not references[api].strip():
                raise ValueError(f"Empty RE evidence for {api}")
            documents.append(tokenize(f"{api} {references[api]}") or ["empty"])
            self.doc_apis.append(api)
        self.bm25 = BM25Okapi(documents)

    def retrieve(self, query: str, *, candidate_docs: int, max_apis: int,
                 evidence_per_api: int) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        top_docs = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:candidate_docs]
        api_scores: dict[str, float] = {}
        for i in top_docs:
            api = self.doc_apis[i]
            api_scores[api] = max(api_scores.get(api, float("-inf")), float(scores[i]))
        ordered_apis = sorted(api_scores, key=lambda api: (-api_scores[api], api))[:max_apis]
        groups = []
        for rank, api in enumerate(ordered_apis, 1):
            pair_indices = sorted(self.api_to_pairs[api], key=lambda i: (-scores[i], i))[:evidence_per_api]
            groups.append({"api_id": f"A{rank}", "api": api, "score": api_scores[api],
                           "pairs": [self.pairs[i] for i in pair_indices]})
        return groups


def group_candidates(pairs: list[Pair], indices: list[int], scores: list[float],
                     max_apis: int, evidence_per_api: int) -> list[dict]:
    grouped: dict[str, dict] = {}
    for index, score in zip(indices, scores):
        pair = pairs[index]
        if pair.api not in grouped and len(grouped) >= max_apis:
            continue
        group = grouped.setdefault(pair.api, {"api": pair.api, "score": score, "pairs": []})
        if len(group["pairs"]) < evidence_per_api:
            group["pairs"].append(pair)
    return [{"api_id": f"A{i}", **group} for i, group in enumerate(grouped.values(), 1)]


def compact_text(value: str, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", value)).strip()
    return value[:max_chars] + ("…" if len(value) > max_chars else "")


def query_relevant_text(value: str, query: str, max_chars: int) -> tuple[str, bool]:
    """Select a contiguous query-relevant window instead of blindly keeping the prefix."""
    clean = re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", value)).strip()
    if len(clean) <= max_chars:
        return clean, False
    query_terms = set(tokenize(query))
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+|\n+", clean) if part.strip()]
    if not sentences:
        return compact_text(clean, max_chars), True
    best = max(range(len(sentences)), key=lambda i: (len(query_terms & set(tokenize(sentences[i]))), -i))
    selected = sentences[best]
    left, right = best - 1, best + 1
    while True:
        options = []
        if left >= 0:
            options.append((left, sentences[left] + " " + selected))
        if right < len(sentences):
            options.append((right, selected + " " + sentences[right]))
        fitting = [(i, text) for i, text in options if len(text) <= max_chars]
        if not fitting:
            break
        i, selected = max(fitting, key=lambda item: len(query_terms & set(tokenize(sentences[item[0]]))))
        if i == left:
            left -= 1
        else:
            right += 1
    return compact_text(selected, max_chars), True


def make_pair_prompt(query: str, candidates: list[dict], references: dict[str, str | list[str]],
                     re_chars: int, ku_chars: int) -> tuple[str, list[dict]]:
    lines = ["Rank the candidate <API, knowledge-unit> pairs for the developer query.",
             "Use RE only as supporting API evidence; judge whether each KU helps answer the query.",
             "Return every supplied pair ID exactly once as JSON:",
             '{"ranking": ["P1", "P2", ...]}', f"Query: {query}", "Candidates:"]
    truncation = []
    for candidate in candidates:
        pair = candidate["pair"]
        source = references.get(pair.api, "")
        ref = ([source] if isinstance(source, str) else source)[0] if source else ""
        ref_text, ref_cut = query_relevant_text(ref, query, re_chars) if ref.strip() else ("", False)
        ku_text, ku_cut = query_relevant_text(pair.ki, query, ku_chars)
        lines.append(f"{pair.pair_id} [{pair.source}] API: {pair.api}")
        if ref_text:
            lines.append(f"  RE: {ref_text}")
        lines.append(f"  KU: {ku_text}")
        truncation.append({"pair_id": pair.pair_id, "re_truncated": ref_cut, "ku_truncated": ku_cut})
    return "\n".join(lines), truncation


def parse_pair_ranking(content: str, valid_ids: list[str]) -> list[str]:
    valid = set(valid_ids)
    try:
        value = json.loads(content)
        raw = value.get("ranking", []) if isinstance(value, dict) else value
        matches = [str(item).upper() for item in raw]
    except (json.JSONDecodeError, TypeError):
        matches = [item.upper() for item in re.findall(r"\bP\d+\b", content, flags=re.IGNORECASE)]
    recognized = [item for item in matches if item in valid]
    if not recognized:
        raise ValueError(f"Model returned no valid pair IDs: {content[:200]!r}")
    ranking = []
    for item in recognized + valid_ids:
        if item in valid and item not in ranking:
            ranking.append(item)
    return ranking


def make_prompt(query: str, groups: list[dict], references: dict[str, str | list[str]], max_chars: int) -> str:
    lines = ["Rank the candidate APIs for the developer query. Use only the supplied API IDs.",
             "Judge whether the API can solve the query using the cited knowledge. Return JSON:",
             '{"ranking": ["A1", "A2", ...]}. Include every supplied API ID once; no explanation.',
             f"Query: {query}", "Candidates:"]
    for group in groups:
        lines.append(f"{group['api_id']} API: {group['api']}")
        source = references.get(group["api"], [])
        for ref in ([source] if isinstance(source, str) else source)[:1]:
            if ref.strip():
                lines.append(f"  RE: {compact_text(ref, max_chars)}")
        for pair in group["pairs"]:
            lines.append(f"  {pair.pair_id} [{pair.source}]: {compact_text(pair.ki, max_chars)}")
    return "\n".join(lines)


def parse_api_ranking(content: str, valid_ids: list[str]) -> list[str]:
    valid = set(valid_ids)
    try:
        value = json.loads(content)
        raw = value.get("ranking", []) if isinstance(value, dict) else value
        matches = [str(item).upper() for item in raw]
    except (json.JSONDecodeError, TypeError):
        matches = re.findall(r"\bA\d+\b", content, flags=re.IGNORECASE)
        matches = [item.upper() for item in matches]
    recognized = [item for item in matches if item in valid]
    if not recognized:
        raise ValueError(f"Model returned no valid API IDs: {content[:200]!r}")
    ranking = []
    for item in recognized + valid_ids:
        if item in valid and item not in ranking:
            ranking.append(item)
    return ranking


def rank_pairs(groups: list[dict], api_ranking: list[str], top_k: int) -> list[str]:
    by_id = {group["api_id"]: group for group in groups}
    return [pair.pair_id for api_id in api_ranking if api_id in by_id
            for pair in by_id[api_id]["pairs"]][:top_k]


def evaluate(rows: list[dict], pair_lookup: dict[str, Pair]) -> dict[str, float]:
    if not rows:
        raise ValueError("No query results to evaluate")
    values = {"P@5": [], "P@10": [], "P@15": [], "MRR": [], "APIRecall@5": []}
    for row in rows:
        gt = {api.casefold() for api in row["gt_apis"]}
        ranked = [pair_lookup[pid] for pid in row["ranked_pairs"]]
        flags = [int(pair.api.casefold() in gt and pair.truth) for pair in ranked]
        for k in (5, 10, 15):
            values[f"P@{k}"].append(sum(flags[:k]) / k)
        values["MRR"].append(next((1 / i for i, flag in enumerate(flags, 1) if flag), 0.0))
        values["APIRecall@5"].append(len({p.api.casefold() for p in ranked[:5]} & gt) / len(gt) if gt else 0.0)
    return {key: sum(items) / len(items) for key, items in values.items()}


def make_chat_payload(prompt: str, model: str) -> dict:
    payload = {"model": model, "temperature": 0, "max_tokens": 1024,
               "messages": [{"role": "system", "content": "You are a grounded API ranking baseline. Return valid JSON only."},
                            {"role": "user", "content": prompt}]}
    if model.startswith("qwen3.8-"):
        payload["enable_thinking"] = False
    return payload


def call_llm(prompt: str, *, endpoint: str, model: str, api_key: str, retries: int) -> tuple[str, dict]:
    payload = make_chat_payload(prompt, model)
    request = Request(endpoint, data=json.dumps(payload).encode("utf-8"),
                      headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=120) as response:
                body = json.load(response)
            return body["choices"][0]["message"]["content"], body.get("usage", {})
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def call_with_fallback(prompt: str, args: argparse.Namespace, valid_ids: list[str] | None = None,
                       ranking_unit: str = "api") -> dict:
    parser = parse_pair_ranking if ranking_unit == "pair" else parse_api_ranking
    try:
        content, usage = call_llm(prompt, endpoint=args.endpoint, model=args.model,
                                  api_key=os.environ[args.api_key_env], retries=args.retries)
        ranking = parser(content, valid_ids) if valid_ids is not None else None
        return {"content": content, "usage": usage, "provider": "aliyun_tokyo", "model": args.model,
                "ranking": ranking}
    except Exception:
        if not args.deepseek_fallback or not os.getenv(args.deepseek_key_env):
            raise
        content, usage = call_llm(prompt, endpoint=args.deepseek_endpoint, model=args.deepseek_model,
                                  api_key=os.environ[args.deepseek_key_env], retries=args.retries)
        ranking = parser(content, valid_ids) if valid_ids is not None else None
        return {"content": content, "usage": usage, "provider": "deepseek", "model": args.deepseek_model,
                "ranking": ranking}


def save_checkpoint(path: Path, signature: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"signature": signature, "queries": rows}, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def load_checkpoint(path: Path, signature: str) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("signature") != signature:
        raise ValueError(f"Existing checkpoint has different experiment settings: {path}; use --restart")
    return payload["queries"]


def experiment_signature(dataset: str, args: argparse.Namespace, query_path: Path) -> str:
    sources = [args.merged_csv, query_path]
    record = {"dataset": dataset, "sources": [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
                                                for path in sources],
              "settings": {key: getattr(args, key, None) for key in ("ranking_unit", "candidate_pairs", "max_apis", "evidence_per_api",
                                                                "max_chars", "re_chars", "ku_chars", "top_k", "max_queries", "no_re",
                                                                "model", "endpoint", "deepseek_fallback",
                                                                "deepseek_model", "deepseek_endpoint")}}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()


def run_dataset(dataset: str, args: argparse.Namespace, corpus: dict[str, dict]) -> dict:
    query_path = args.query_dir / f"{dataset}.csv"
    data = corpus[dataset]
    pairs, queries = data["pairs"], read_queries(query_path)
    if not pairs or not queries:
        raise ValueError(f"Empty pairs or queries: {dataset}")
    with_re = not getattr(args, "no_re", False)
    references = data["references"] if with_re else {api: "" for api in data["references"]}
    index = EvidenceIndex(pairs, data["references"] if with_re else {})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{dataset}_{args.mode}.json"
    checkpoint = args.output_dir / f"{dataset}_{args.mode}.checkpoint.json"
    signature = experiment_signature(dataset, args, query_path) if args.mode == "run" else ""
    rows = ([] if getattr(args, "restart", False) or args.mode == "prepare"
            else load_checkpoint(checkpoint, signature))
    completed = {row["query_id"] for row in rows}
    limit = len(queries) if args.max_queries is None else min(args.max_queries, len(queries))
    for item in queries[:limit]:
        if item["query_id"] in completed:
            continue
        ranking_unit = getattr(args, "ranking_unit", "api")
        if ranking_unit == "pair":
            candidates = retrieve_pair_candidates(pairs, item["query"], args.candidate_pairs)
            prompt, truncation = make_pair_prompt(item["query"], candidates, references,
                                                   args.re_chars, args.ku_chars)
            valid_ids = [candidate["pair"].pair_id for candidate in candidates]
            candidate_api_ids = {}
            groups = []
        else:
            groups = index.retrieve(item["query"], candidate_docs=args.candidate_pairs,
                                    max_apis=args.max_apis, evidence_per_api=args.evidence_per_api)
            prompt = make_prompt(item["query"], groups, references, args.max_chars)
            valid_ids = [group["api_id"] for group in groups]
            candidate_api_ids = {g["api_id"]: g["api"] for g in groups}
            truncation = []
        if args.mode == "prepare":
            ranking, usage, response = valid_ids, {}, None
        else:
            response = call_with_fallback(prompt, args, valid_ids, ranking_unit=ranking_unit)
            ranking, usage = response["ranking"], response["usage"]
        ranked_pairs = ranking[:args.top_k] if ranking_unit == "pair" else rank_pairs(groups, ranking, args.top_k)
        candidate_pair_ids = valid_ids if ranking_unit == "pair" else [pair.pair_id for group in groups for pair in group["pairs"]]
        rows.append({"query_id": item["query_id"], "query": item["query"],
                     "gt_apis": item["gt_apis"], "candidate_api_ids": candidate_api_ids,
                     "candidate_pair_ids": candidate_pair_ids,
                     "ranked_api_ids": ranking if ranking_unit == "api" else [], "ranked_pairs": ranked_pairs,
                     "truncation": truncation,
                     "prompt": prompt if args.mode == "prepare" else None, "usage": usage,
                     "provider": response["provider"] if response else None,
                     "model": response["model"] if response else None,
                     "raw_response": response["content"] if response else None})
        if args.mode == "run":
            save_checkpoint(checkpoint, signature, rows)
            print(f"[{dataset}] {len(rows)}/{limit}", flush=True)
    rows.sort(key=lambda row: row["query_id"])
    summary = {"dataset": dataset, "mode": args.mode, "pair_count": len(pairs),
               "pairs_with_decode_replacement": sum("\ufffd" in pair.ki for pair in pairs),
               "api_count": len({pair.api for pair in pairs}), "query_count": limit,
               "with_re": with_re, "re_api_coverage": sum(bool(v) for v in references.values()),
               "reference_origins": {origin: sum(value == origin for value in data["reference_origins"].values())
                                     for origin in sorted(set(data["reference_origins"].values()))},
               "candidate_docs": args.candidate_pairs,
               "ranking_unit": getattr(args, "ranking_unit", "api"),
               "max_apis": args.max_apis, "evidence_per_api": args.evidence_per_api,
               "model": args.model if args.mode == "run" else None,
               "provider_counts": {name: sum(row.get("provider") == name for row in rows)
                                   for name in sorted({row.get("provider") for row in rows if row.get("provider")})},
               "usage_totals": {key: sum(int(row.get("usage", {}).get(key, 0) or 0) for row in rows)
                                for key in ("prompt_tokens", "completion_tokens", "total_tokens")}}
    if args.mode == "run":
        summary.update(evaluate(rows, {pair.pair_id: pair for pair in pairs}))
    out.write_text(json.dumps({"signature": signature if args.mode == "run" else None,
                               "summary": summary, "queries": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(out), **summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run"), default="prepare")
    parser.add_argument("--datasets", default="jodatime", help="Comma-separated names, or 'all'")
    parser.add_argument("--merged-csv", type=Path, default=ROOT / "data/API_RE_KU_9datasets.csv")
    parser.add_argument("--query-dir", type=Path, default=ROOT / "eval_outputs/query_revised_9datasets_20260911")
    parser.add_argument("--no-re", action="store_true", help="Ablation: remove RE from retrieval and prompt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval_outputs/api_evidence_llm")
    parser.add_argument("--candidate-pairs", type=int, default=60)
    parser.add_argument("--ranking-unit", choices=("api", "pair"), default="api")
    parser.add_argument("--max-apis", type=int, default=20)
    parser.add_argument("--evidence-per-api", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--re-chars", type=int, default=300)
    parser.add_argument("--ku-chars", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--model", default=os.getenv("ALIYUN_MODEL", "qwen3.8-flash"))
    parser.add_argument("--endpoint", default=os.getenv("ALIYUN_ENDPOINT", TOKYO_ENDPOINT))
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--deepseek-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deepseek-endpoint", default=DEEPSEEK_ENDPOINT)
    parser.add_argument("--deepseek-model", default="deepseek-v4-flash")
    parser.add_argument("--deepseek-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--restart", action="store_true", help="Ignore a previous matching checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.candidate_pairs, args.max_apis, args.evidence_per_api, args.max_chars, args.top_k) < 1:
        raise SystemExit("Candidate and output limits must be positive")
    if args.mode == "run" and not os.getenv(args.api_key_env):
        raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")
    datasets = DATASETS if args.datasets == "all" else tuple(name.strip() for name in args.datasets.split(","))
    corpus = load_merged_corpus(args.merged_csv)
    for dataset in datasets:
        if dataset not in DATASETS:
            raise SystemExit(f"Unknown dataset: {dataset}")
        if dataset not in corpus:
            raise SystemExit(f"Dataset missing from merged corpus: {dataset}")
        print(json.dumps(run_dataset(dataset, args, corpus), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
