#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from skimage import io, transform
from skimage.feature import hog, local_binary_pattern

from model_registry import load_registry, find_model

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "dataset" / "train"
VAL_DIR = ROOT / "dataset" / "val"
TRAIN_CSV = ROOT / "dataset" / "train.csv"
VAL_CSV = ROOT / "dataset" / "val.csv"
PREDICT_DIR = ROOT / "dataset" / "predict"
MODELS_DIR = ROOT / "models"
OUTPUT_CSV = ROOT / "submission.csv"

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


def list_predict_ids():
    if not PREDICT_DIR.exists():
        raise SystemExit(f"Missing prediction directory: {PREDICT_DIR}")
    image_ids = []
    for path in PREDICT_DIR.glob("*.jpg"):
        try:
            image_ids.append(int(path.stem))
        except ValueError:
            continue
    return sorted(image_ids)


def load_labels(csv_path: Path):
    if not csv_path.exists():
        raise SystemExit(f"Missing labels CSV: {csv_path}")
    image_ids = []
    labels = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_ids.append(int(row["id"]))
            labels.append(row["label"])
    return np.array(image_ids), np.array(labels)


def train_pipeline(spec, train_ids, train_labels):
    features = build_features(
        train_ids,
        TRAIN_DIR,
        spec["extractor"],
        spec["feature_params"],
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(**spec["svm_params"]))
    ])
    pipeline.fit(features, train_labels)
    return pipeline


def evaluate_model(spec, train_ids, train_labels, val_ids, val_labels) -> float:
    pipeline = train_pipeline(spec, train_ids, train_labels)
    val_features = build_features(
        val_ids,
        VAL_DIR,
        spec["extractor"],
        spec["feature_params"],
    )
    preds = pipeline.predict(val_features)
    return float(accuracy_score(val_labels, preds))


def ensure_trained_model(spec, train_ids, train_labels, force: bool = False) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{spec['name']}.joblib"
    meta_path = model_path.with_suffix(".meta.json")

    if model_path.exists() and meta_path.exists() and not force:
        return model_path

    pipeline = train_pipeline(spec, train_ids, train_labels)
    joblib.dump(pipeline, model_path)

    meta = {
        "model_name": spec["name"],
        "extractor": spec["extractor"],
        "feature_params": spec["feature_params"],
        "svm_params": spec["svm_params"],
        "image_size": IMAGE_SIZE,
        "labels": sorted(set(train_labels.tolist())),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model_path


def select_best_model(registry, train_ids, train_labels, val_ids, val_labels):
    best_spec = None
    best_acc = -1.0
    for spec in registry:
        acc = evaluate_model(spec, train_ids, train_labels, val_ids, val_labels)
        if acc > best_acc:
            best_acc = acc
            best_spec = spec
    return best_spec, best_acc


def resolve_feature_params(model_path: Path, fallback_spec: Optional[dict]):
    meta_path = model_path.with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta["extractor"], meta["feature_params"]
    if fallback_spec:
        return fallback_spec["extractor"], fallback_spec["feature_params"]
    raise SystemExit(f"Missing meta file: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate submission using a trained model.")
    parser.add_argument(
        "--model-name",
        help="Registry model name to use (e.g., hog_svm).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to a .joblib model file.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "models" / "registry.json",
        help="Path to models registry JSON.",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Select the highest-accuracy registry model on val, then generate submission.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain the selected model before generating the submission.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="Path to the output CSV file.",
    )
    args = parser.parse_args()

    if args.best and (args.model_path or args.model_name):
        raise SystemExit("--best cannot be combined with --model-name or --model-path.")

    selected_spec = None

    if args.best:
        registry = load_registry(args.registry)
        train_ids, train_labels = load_labels(TRAIN_CSV)
        val_ids, val_labels = load_labels(VAL_CSV)
        selected_spec, best_acc = select_best_model(
            registry,
            train_ids,
            train_labels,
            val_ids,
            val_labels,
        )
        if selected_spec is None:
            raise SystemExit("No models found in registry.")
        model_path = ensure_trained_model(
            selected_spec,
            train_ids,
            train_labels,
            force=args.retrain,
        )
        print(f"Selected best model: {selected_spec['name']} (val acc={best_acc:.4f})")
    elif args.model_path:
        model_path = args.model_path
    else:
        registry = load_registry(args.registry)
        if args.model_name:
            spec = find_model(registry, args.model_name)
        else:
            spec = registry[0]
        selected_spec = spec
        model_path = MODELS_DIR / f"{spec['name']}.joblib"

    if not model_path.exists():
        raise SystemExit("Missing model file. Run scripts/train_model.py first.")

    extractor, feature_params = resolve_feature_params(model_path, selected_spec)

    pipeline = joblib.load(model_path)

    predict_ids = list_predict_ids()
    if not predict_ids:
        raise SystemExit(f"No prediction images found in {PREDICT_DIR}")

    features = build_features(predict_ids, PREDICT_DIR, extractor, feature_params)
    preds = pipeline.predict(features)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for image_id, label in zip(predict_ids, preds):
            writer.writerow([image_id, label])

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
