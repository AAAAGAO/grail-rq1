"""Apply frozen occurrence reference corrections without changing knowledge."""
import hashlib
import json
import os
from pathlib import Path

REGISTRY_ENV = "GRAIL_REFERENCE_OVERRIDE_REGISTRY"


def apply_reference_overrides(rows, registry=None):
    if registry is None:
        registry_path = os.environ.get(REGISTRY_ENV)
        registry = (
            json.loads(Path(registry_path).read_text(encoding="utf-8"))
            if registry_path and Path(registry_path).exists()
            else {}
        )
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
