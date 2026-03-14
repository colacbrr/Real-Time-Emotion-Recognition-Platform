#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from datetime import datetime


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
    if not model_paths:
        raise SystemExit("Provide at least one model path.")
    results = {
        "metric": "val_accuracy",
        "models": {},
    }

    for model_path in model_paths:
        if not model_path.exists():
            raise SystemExit(f"Missing model checkpoint: {model_path}")
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
    markdown_output = Path("reports") / f"cnn_comparison_{timestamp}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [
        "# CNN Comparison",
        "",
        f"Metric: {results['metric']}",
        "",
    ]
    for model_name, stats in results["models"].items():
        lines.extend(
            [
                f"## {model_name}",
                f"- architecture: {stats['arch'] or '-'}",
                f"- best validation accuracy: {stats['best_val_acc'] if stats['best_val_acc'] is not None else '-'}",
                f"- image size: {stats['image_size'] or '-'}",
                f"- RGB input: {stats['use_rgb']}",
                f"- epochs: {stats['epochs'] or '-'}",
                "",
            ]
        )
    markdown_output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {output}")
    print(f"Saved {markdown_output}")


if __name__ == "__main__":
    main()
