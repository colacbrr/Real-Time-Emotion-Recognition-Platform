#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from skimage import io, transform
from skimage.feature import hog, local_binary_pattern

from model_registry import load_registry, find_model

ROOT = Path(__file__).resolve().parent.parent
PREDICT_DIR = ROOT / "dataset" / "predict"
MODELS_DIR = ROOT / "models"
OUTPUT_CSV = ROOT / "submission.csv"

IMAGE_SIZE = (48, 48)


def load_image(image_id: int) -> np.ndarray:
    path = PREDICT_DIR / f"{image_id}.jpg"
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


def list_predict_ids():
    image_ids = []
    for path in PREDICT_DIR.glob("*.jpg"):
        try:
            image_ids.append(int(path.stem))
        except ValueError:
            continue
    return sorted(image_ids)


def load_meta(model_path: Path) -> dict:
    meta_path = model_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise SystemExit(f"Missing meta file: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Generate submission using a trained model.")
    parser.add_argument(
        "--model",
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
    args = parser.parse_args()

    if args.model_path:
        model_path = args.model_path
    else:
        registry = load_registry(args.registry)
        if args.model:
            spec = find_model(registry, args.model)
        else:
            spec = registry[0]
        model_path = MODELS_DIR / f"{spec['name']}.joblib"

    if not model_path.exists():
        raise SystemExit("Missing model file. Run scripts/train_model.py first.")

    meta = load_meta(model_path)
    extractor = meta["extractor"]
    feature_params = meta["feature_params"]

    pipeline = joblib.load(model_path)

    predict_ids = list_predict_ids()
    if not predict_ids:
        raise SystemExit(f"No prediction images found in {PREDICT_DIR}")

    features = build_features(predict_ids, extractor, feature_params)
    preds = pipeline.predict(features)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for image_id, label in zip(predict_ids, preds):
            writer.writerow([image_id, label])

    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
