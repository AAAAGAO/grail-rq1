import csv
import hashlib
import json
from pathlib import Path

from rakg import DATASETS

API_RELATIONS = {
    "subtype_of",
    "accepts",
    "returns",
}

KU_RELATIONS = {
    "task_related",
    "complements",
    "constrains",
    "contrasts_with",
    "same_ku",
}

API_KU_RELATIONS = {
    "has_explanation",
    "has_guidance",
    "has_example",
    "has_constraint",
}

RELATIONS = API_RELATIONS | KU_RELATIONS | API_KU_RELATIONS


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value):
    return " ".join(str(value or "").split())


def pair_text(row):
    return normalize_text(" ".join(
        value for value in (
            row.get("raw_api", ""),
            row.get("canonical_api", ""),
            row.get("canonical_owner", ""),
            row.get("reference", ""),
            row.get("ku", ""),
        )
        if value
    ))


def load_queries(path, dataset=None):
    rows = read_csv(path)
    result = []
    for row in rows:
        if dataset and row.get("dataset") != dataset:
            continue
        labels = [
            item.strip()
            for item in row.get("ground_truth_apis", "").split("|")
            if item.strip()
        ]
        result.append({
            "dataset": row["dataset"],
            "query_id": int(row["query_id"]),
            "query": row["query"],
            "ground_truth_apis": labels,
        })
    return sorted(result, key=lambda item: (item["dataset"], item["query_id"]))


def dataset_paths(root, dataset):
    folder = Path(root) / dataset
    return {
        "folder": folder,
        "pairs": folder / "knowledge_pairs.csv",
        "nodes": folder / "nodes.csv",
        "edges": folder / "edges.csv",
    }


def validate_release(root, queries_path):
    root = Path(root)
    queries = load_queries(queries_path)
    if {row["dataset"] for row in queries} != set(DATASETS):
        raise ValueError("the query file must contain all nine datasets")
    for dataset in DATASETS:
        local = [row for row in queries if row["dataset"] == dataset]
        if [row["query_id"] for row in local] != list(range(1, 31)):
            raise ValueError(f"{dataset} must contain query IDs 1 through 30")
        paths = dataset_paths(root, dataset)
        for path in paths.values():
            if path.name and not path.exists():
                raise FileNotFoundError(path)
    return queries
