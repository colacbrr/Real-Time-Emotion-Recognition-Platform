#!/usr/bin/env python3
import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from skimage import io, transform
from skimage.feature import hog, local_binary_pattern

from model_registry import load_registry, find_model

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "dataset" / "train"
VAL_DIR = ROOT / "dataset" / "val"
TRAIN_CSV = ROOT / "dataset" / "train.csv"
VAL_CSV = ROOT / "dataset" / "val.csv"
REPORTS_DIR = ROOT / "reports"

IMAGE_SIZE = (48, 48)


def load_image(image_id: int, base_dir: Path) -> np.ndarray:
    path = base_dir / f"{image_id}.jpg"
    image = io.imread(path, as_gray=True)
    if image.shape != IMAGE_SIZE:
        image = transform.resize(image, IMAGE_SIZE, anti_aliasing=True)
    return image.astype("float32")


def extract_hog(image: np.ndarray, params: dict) -> np.ndarray:
    return hog(image, **params)


def extract_lbp(image: np.ndarray, params: dict) -> np.ndarray:
    lbp = local_binary_pattern(image, params["P"], params["R"], method=params["method"])
    n_bins = params["P"] + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype("float32")


def build_features(image_ids, base_dir, extractor_name, params):
    features = []
    for image_id in image_ids:
        image = load_image(image_id, base_dir)
        if extractor_name == "hog":
            feat = extract_hog(image, params)
        elif extractor_name == "lbp":
            feat = extract_lbp(image, params)
        else:
            raise ValueError(f"Unknown extractor: {extractor_name}")
        features.append(feat)
    return np.stack(features)


def load_labels(csv_path: Path):
    image_ids = []
    labels = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_ids.append(int(row["id"]))
            labels.append(row["label"])
    return np.array(image_ids), np.array(labels)


def evaluate_model(spec: dict, train_ids, train_labels, val_ids, val_labels):
    train_features = build_features(
        train_ids,
        TRAIN_DIR,
        spec["extractor"],
        spec["feature_params"],
    )
    val_features = build_features(
        val_ids,
        VAL_DIR,
        spec["extractor"],
        spec["feature_params"],
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(**spec["svm_params"]))
    ])
    pipeline.fit(train_features, train_labels)
    preds = pipeline.predict(val_features)
    accuracy = accuracy_score(val_labels, preds)

    return {"accuracy": float(accuracy)}


def main():
    parser = argparse.ArgumentParser(description="Compare registry models on the val split.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "models" / "registry.json",
        help="Path to models registry JSON.",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated model names to compare (default: all).",
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train_ids, train_labels = load_labels(TRAIN_CSV)
    val_ids, val_labels = load_labels(VAL_CSV)

    registry = load_registry(args.registry)
    if args.models:
        selected = []
        for name in args.models.split(","):
            selected.append(find_model(registry, name.strip()))
        specs = selected
    else:
        specs = registry

    results = {
        "metric": "accuracy",
        "models": {},
    }

    for spec in specs:
        results["models"][spec["name"]] = {
            "accuracy": evaluate_model(
            spec,
            train_ids,
            train_labels,
            val_ids,
            val_labels,
            )["accuracy"],
            "config": {
                "extractor": spec["extractor"],
                "feature_params": spec["feature_params"],
                "svm_params": spec["svm_params"],
                "description": spec.get("description", ""),
            },
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / f"comparison_{timestamp}.json"
    md_path = REPORTS_DIR / f"comparison_{timestamp}.md"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# Model comparison",
        "",
        f"Metric: {results['metric']}",
        "",
    ]
    for name, stats in results["models"].items():
        lines.extend([
            f"## {name}",
            f"Accuracy (val): {stats['accuracy']:.4f}",
            "",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
