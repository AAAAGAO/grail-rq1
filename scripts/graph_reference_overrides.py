"""Apply occurrence-specific reference corrections without changing knowledge."""
import hashlib
import json
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[1] / "data/RE_reference/occurrence_reference_overrides.json"


def apply_reference_overrides(rows, registry=None):
    if registry is None:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else {}
    output = []
    for row in rows:
        result = dict(row)
        override = registry.get(row.get("pair_id"))
        if override:
            digest = hashlib.sha256(row.get("ku", "").encode()).hexdigest()
            if digest != override["ku_sha256"]:
                raise ValueError(f"Reference correction source changed: {row['pair_id']}")
            result.update(override["fields"])
        output.append(result)
    return output
