import csv
import hashlib
import os
import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from rakg.common import normalize_text, pair_text, read_csv, write_csv
from rakg.embeddings import Encoder


def _value(row, names):
    for name in names:
        if row.get(name):
            return row[name]
    return ""


def collect(
    tutorial_path,
    stackoverflow_path,
    specification_path,
    output_path,
    dataset="",
):
    sources = (
        ("SG", tutorial_path, ("api", "API", "raw_api"), ("ku", "KI", "segment", "SG")),
        ("QA", stackoverflow_path, ("api", "API", "raw_api"), ("ku", "KI", "answer", "QA")),
        ("DP", specification_path, ("api", "API", "raw_api"), ("ku", "KI", "description", "DP")),
    )
    rows = []
    index = 1
    for source, path, api_names, ku_names in sources:
        for row in read_csv(path):
            api = normalize_text(_value(row, api_names))
            ku = normalize_text(_value(row, ku_names))
            if not api or not ku:
                continue
            rows.append({
                "pair_id": f"PAIR-{index:06d}",
                "dataset": dataset,
                "ku_source": source,
                "raw_api": api,
                "canonical_api": normalize_text(_value(row, ("canonical_api", "resolved_api"))),
                "reference": normalize_text(_value(row, ("reference", "DP", "description"))),
                "ku": ku,
                "identified_relevance": "",
                "relevant_groundtruth": "",
            })
            index += 1
    write_csv(output_path, rows)


def _tokens(value):
    return re.findall(r"[a-z][a-z0-9_.$]*|\d+", value.lower())


_NLP = None


def _signature(value):
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load(
            os.environ.get("RAKG_SPACY_MODEL", "en_core_web_sm"),
            disable=["parser", "ner", "lemmatizer"],
        )
    return [token.pos_ for token in _NLP(str(value or ""))]


def _jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / max(1, len(left | right))


def select_demonstrations(rows, target, encoder, count=6):
    texts = [pair_text(row) for row in rows]
    tokens = [_tokens(text) for text in texts]
    bm25 = BM25Okapi(tokens)
    lexical = bm25.get_scores(_tokens(pair_text(target)))
    semantic = encoder.similarity(pair_text(target), texts)
    structural = [
        _jaccard(_signature(target.get("ku", "")), _signature(row.get("ku", "")))
        for row in rows
    ]
    rankings = []
    for values in (lexical, semantic, structural):
        order = sorted(range(len(rows)), key=lambda i: (-float(values[i]), i))
        rankings.append({index: rank + 1 for rank, index in enumerate(order)})
    scores = {
        index: sum(1 / (60 + ranking[index]) for ranking in rankings)
        for index in range(len(rows))
    }
    order = sorted(scores, key=lambda i: (-scores[i], i))[:count]
    return [rows[index] for index in order]


LAMIC_SYSTEM = (
    "Classify whether the supplied API knowledge unit explains how to use the API. "
    "Use the demonstrations and the supplied API and knowledge text. Return JSON only: "
    '{"identified_relevance":"1 or 0","roles":["has_explanation","has_guidance","has_example","has_constraint"],'
    '"clues":["exact supporting excerpts"],"reasoning":"brief explanation"}'
)


def identify(input_path, output_path, demonstrations_path=None, model_name=None, client=None):
    rows = read_csv(input_path)
    labeled = [
        row for row in rows
        if row.get("identified_relevance") in {"0", "1"}
    ]
    if not client and len(labeled) != len(rows):
        raise RuntimeError("LAMIC identification requires an LLM client for unlabeled pairs")
    demo_rows = read_csv(demonstrations_path) if demonstrations_path else labeled
    if not demo_rows:
        raise ValueError("LAMIC requires a labeled demonstration pool")
    encoder = Encoder(model_name=model_name)
    output = []
    for row in rows:
        if row.get("identified_relevance") in {"0", "1"} and not client:
            output.append(row)
            continue
        demos = select_demonstrations(demo_rows, row, encoder)
        if not client:
            row = {**row, "identified_relevance": "1"}
            output.append(row)
            continue
        prompt = {
            "target": row,
            "demonstrations": demos,
        }
        result = client.call(LAMIC_SYSTEM, str(prompt))
        from rakg.llm import parse_object
        parsed = parse_object(result)
        roles = parsed.get("roles", [])
        if isinstance(roles, str):
            roles = [item.strip() for item in roles.split("|") if item.strip()]
        clues = parsed.get("clues", [])
        if isinstance(clues, str):
            clues = [clues]
        output.append({
            **row,
            "identified_relevance": str(parsed.get("identified_relevance", "0")),
            "lamic_role": "|".join(str(item) for item in roles),
            "lamic_evidence": " | ".join(str(item) for item in clues),
            "lamic_reasoning": str(parsed.get("reasoning", "")),
        })
    write_csv(output_path, output)
