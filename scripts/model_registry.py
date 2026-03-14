#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = ROOT / "models" / "registry.json"


def load_registry(path: Path = DEFAULT_REGISTRY) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing registry file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data.get("models", [])
    if not isinstance(models, list) or not models:
        raise SystemExit("Registry must contain a non-empty 'models' list.")
    seen_names = set()
    for entry in models:
        for key in ("name", "extractor", "feature_params", "svm_params"):
            if key not in entry:
                raise SystemExit(f"Model entry missing '{key}': {entry}")
        model_name = entry["name"]
        if model_name in seen_names:
            raise SystemExit(f"Duplicate model name in registry: {model_name}")
        seen_names.add(model_name)
    return models


def find_model(models: list[dict], name: str) -> dict:
    for entry in models:
        if entry.get("name") == name:
            return entry
    raise SystemExit(f"Model '{name}' not found in registry.")
