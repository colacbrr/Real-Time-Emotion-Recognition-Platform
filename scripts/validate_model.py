#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from skimage import io, transform
from skimage.feature import hog, local_binary_pattern

from model_registry import load_registry, find_model

ROOT = Path(__file__).resolve().parent.parent
VAL_DIR = ROOT / "dataset" / "val"
VAL_CSV = ROOT / "dataset" / "val.csv"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

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


def load_meta(model_path: Path) -> dict:
    meta_path = model_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise SystemExit(f"Missing meta file: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Validate a trained model on val split.")
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to .joblib model file.",
    )
    parser.add_argument(
        "--model-name",
        help="Registry model name (e.g., hog_svm).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "models" / "registry.json",
        help="Path to models registry JSON.",
    )
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        registry = load_registry(args.registry)
        if args.model_name:
            spec = find_model(registry, args.model_name)
        else:
            spec = registry[0]
        model_path = MODELS_DIR / f"{spec['name']}.joblib"
    if not model_path.exists():
        raise SystemExit(f"Missing model file: {model_path}")

    meta = load_meta(model_path)
    extractor = meta["extractor"]
    feature_params = meta["feature_params"]
    label_order = meta.get("labels", [])

    val_ids, val_labels = load_labels(VAL_CSV)
    if not len(val_ids):
        raise SystemExit(f"No validation labels found in {VAL_CSV}")

    features = build_features(val_ids, VAL_DIR, extractor, feature_params)
    pipeline = joblib.load(model_path)
    preds = pipeline.predict(features)

    accuracy = float(accuracy_score(val_labels, preds))
    if not label_order:
        label_order = sorted(set(val_labels.tolist()))
    cm = confusion_matrix(val_labels, preds, labels=label_order)
    per_class = np.divide(
        np.diag(cm),
        cm.sum(axis=1),
        out=np.zeros(len(label_order), dtype=float),
        where=cm.sum(axis=1) != 0,
    )
    per_class_accuracy = {label: float(acc) for label, acc in zip(label_order, per_class)}

    report = {
        "metric": "accuracy",
        "accuracy": accuracy,
        "labels": label_order,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": cm.tolist(),
        "model": {
            "path": str(model_path),
            "extractor": extractor,
            "feature_params": feature_params,
            "svm_params": meta.get("svm_params", {}),
            "image_size": meta.get("image_size", IMAGE_SIZE),
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / f"validation_{timestamp}.json"
    md_path = REPORTS_DIR / f"validation_{timestamp}.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Validation report",
        "",
        f"Metric: {report['metric']}",
        f"Accuracy (val): {accuracy:.4f}",
        "",
        "## Per-class accuracy",
        "",
    ]
    for label, acc in per_class_accuracy.items():
        lines.append(f"- {label}: {acc:.4f}")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
