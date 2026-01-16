#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from datetime import datetime

from cnn_train import LABELS
from cnn_validate import main as validate_main


def load_meta(path: Path):
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise SystemExit(f"Missing meta file: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Compare CNN checkpoints using their meta files.")
    parser.add_argument("--models", required=True, help="Comma-separated list of .pt paths.")
    args = parser.parse_args()

    model_paths = [Path(p.strip()) for p in args.models.split(",") if p.strip()]
    results = {
        "metric": "val_accuracy",
        "models": {},
    }

    for model_path in model_paths:
        meta = load_meta(model_path)
        results["models"][model_path.name] = {
            "arch": meta.get("arch"),
            "best_val_acc": meta.get("best_val_acc"),
            "image_size": meta.get("image_size"),
            "use_rgb": meta.get("use_rgb"),
            "epochs": meta.get("epochs"),
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path("reports") / f"cnn_comparison_{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
