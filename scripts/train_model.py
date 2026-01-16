#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from skimage import io, transform
from skimage.feature import hog, local_binary_pattern

from model_registry import load_registry, find_model

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "dataset" / "train"
TRAIN_CSV = ROOT / "dataset" / "train.csv"
MODELS_DIR = ROOT / "models"

IMAGE_SIZE = (48, 48)


def load_image(image_id: int) -> np.ndarray:
    path = TRAIN_DIR / f"{image_id}.jpg"
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


def build_features(image_ids, extractor_name, params):
    features = []
    for image_id in image_ids:
        image = load_image(image_id)
        if extractor_name == "hog":
            feat = extract_hog(image, params)
        elif extractor_name == "lbp":
            feat = extract_lbp(image, params)
        else:
            raise ValueError(f"Unknown extractor: {extractor_name}")
        features.append(feat)
    return np.stack(features)


def load_labels():
    image_ids = []
    labels = []
    with TRAIN_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_ids.append(int(row["id"]))
            labels.append(row["label"])
    return np.array(image_ids), np.array(labels)


def train_single(model_spec, image_ids, labels):
    extractor = model_spec["extractor"]
    feature_params = model_spec["feature_params"]
    svm_params = model_spec["svm_params"]

    features = build_features(image_ids, extractor, feature_params)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(**svm_params))
    ])
    pipeline.fit(features, labels)

    model_name = model_spec["name"]
    model_path = MODELS_DIR / f"{model_name}.joblib"
    meta_path = MODELS_DIR / f"{model_name}.meta.json"

    joblib.dump(pipeline, model_path)
    meta = {
        "model_name": model_name,
        "extractor": extractor,
        "feature_params": feature_params,
        "svm_params": svm_params,
        "image_size": IMAGE_SIZE,
        "labels": sorted(set(labels.tolist())),
        "description": model_spec.get("description", ""),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved {model_path}")
    print(f"Saved {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Train feature+SVM models from the registry.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "models" / "registry.json",
        help="Path to models registry JSON.",
    )
    parser.add_argument(
        "--model",
        help="Registry model name to train (e.g., hog_svm).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all models from the registry.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available registry models and exit.",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models = load_registry(args.registry)
    if args.list:
        for entry in models:
            desc = entry.get("description", "")
            print(f"{entry['name']}\t{entry['extractor']}\t{desc}")
        return

    image_ids, labels = load_labels()

    if args.all:
        for entry in models:
            train_single(entry, image_ids, labels)
        return

    if args.model:
        spec = find_model(models, args.model)
    else:
        spec = models[0]

    train_single(spec, image_ids, labels)


if __name__ == "__main__":
    main()
