#!/usr/bin/env python3
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from skimage import io, transform
from skimage.feature import hog

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "dataset" / "train"
VAL_DIR = ROOT / "dataset" / "val"
TRAIN_CSV = ROOT / "dataset" / "train.csv"
VAL_CSV = ROOT / "dataset" / "val.csv"
REPORTS_DIR = ROOT / "reports"

IMAGE_SIZE = (48, 48)


@dataclass(frozen=True)
class HogConfig:
    name: str
    params: dict


@dataclass(frozen=True)
class SvmConfig:
    name: str
    params: dict


def load_image(image_id: int, base_dir: Path) -> np.ndarray:
    path = base_dir / f"{image_id}.jpg"
    image = io.imread(path, as_gray=True)
    if image.shape != IMAGE_SIZE:
        image = transform.resize(image, IMAGE_SIZE, anti_aliasing=True)
    return image.astype("float32")


def extract_hog(image: np.ndarray, params: dict) -> np.ndarray:
    return hog(image, **params)


def build_features(image_ids, base_dir, params):
    features = []
    for image_id in image_ids:
        image = load_image(image_id, base_dir)
        features.append(extract_hog(image, params))
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


def evaluate_config(train_features, train_labels, val_features, val_labels, svm_params, label_order):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(**svm_params)),
    ])
    pipeline.fit(train_features, train_labels)
    preds = pipeline.predict(val_features)

    cm = confusion_matrix(val_labels, preds, labels=label_order)
    per_class = np.divide(
        np.diag(cm),
        cm.sum(axis=1),
        out=np.zeros(len(label_order), dtype=float),
        where=cm.sum(axis=1) != 0,
    )

    return {
        "accuracy": float(accuracy_score(val_labels, preds)),
        "per_class_accuracy": {label: float(acc) for label, acc in zip(label_order, per_class)},
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train_ids, train_labels = load_labels(TRAIN_CSV)
    val_ids, val_labels = load_labels(VAL_CSV)
    label_order = sorted(set(train_labels.tolist()))

    hog_configs = [
        HogConfig(
            name="hog_9_8x8_2x2",
            params={
                "orientations": 9,
                "pixels_per_cell": (8, 8),
                "cells_per_block": (2, 2),
                "block_norm": "L2-Hys",
            },
        ),
        HogConfig(
            name="hog_9_6x6_2x2",
            params={
                "orientations": 9,
                "pixels_per_cell": (6, 6),
                "cells_per_block": (2, 2),
                "block_norm": "L2-Hys",
            },
        ),
        HogConfig(
            name="hog_12_8x8_2x2",
            params={
                "orientations": 12,
                "pixels_per_cell": (8, 8),
                "cells_per_block": (2, 2),
                "block_norm": "L2-Hys",
            },
        ),
    ]

    svm_configs = [
        SvmConfig(
            name="svm_rbf_c10_scale",
            params={"kernel": "rbf", "C": 10.0, "gamma": "scale"},
        ),
        SvmConfig(
            name="svm_rbf_c30_scale",
            params={"kernel": "rbf", "C": 30.0, "gamma": "scale"},
        ),
        SvmConfig(
            name="svm_rbf_c10_scale_balanced",
            params={"kernel": "rbf", "C": 10.0, "gamma": "scale", "class_weight": "balanced"},
        ),
        SvmConfig(
            name="svm_linear_c1_balanced",
            params={"kernel": "linear", "C": 1.0, "class_weight": "balanced"},
        ),
        SvmConfig(
            name="svm_linear_c3_balanced",
            params={"kernel": "linear", "C": 3.0, "class_weight": "balanced"},
        ),
    ]

    results = {
        "metric": "accuracy",
        "labels": label_order,
        "hog_configs": {cfg.name: cfg.params for cfg in hog_configs},
        "svm_configs": {cfg.name: cfg.params for cfg in svm_configs},
        "results": {},
    }

    best_key = None
    best_score = -1.0

    for hog_cfg in hog_configs:
        train_features = build_features(train_ids, TRAIN_DIR, hog_cfg.params)
        val_features = build_features(val_ids, VAL_DIR, hog_cfg.params)
        results["results"][hog_cfg.name] = {}
        for svm_cfg in svm_configs:
            stats = evaluate_config(
                train_features,
                train_labels,
                val_features,
                val_labels,
                svm_cfg.params,
                label_order,
            )
            results["results"][hog_cfg.name][svm_cfg.name] = stats
            if stats["accuracy"] > best_score:
                best_score = stats["accuracy"]
                best_key = (hog_cfg.name, svm_cfg.name, stats)

    if best_key:
        hog_name, svm_name, stats = best_key
        results["best"] = {
            "hog_config": hog_name,
            "svm_config": svm_name,
            "accuracy": stats["accuracy"],
            "per_class_accuracy": stats["per_class_accuracy"],
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / f"hog_search_{timestamp}.json"
    md_path = REPORTS_DIR / f"hog_search_{timestamp}.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# HOG + SVM search",
        "",
        f"Metric: {results['metric']}",
        "",
        "## Results",
        "",
    ]
    for hog_name, svm_map in results["results"].items():
        lines.append(f"### {hog_name}")
        for svm_name, stats in svm_map.items():
            lines.append(f"- {svm_name}: {stats['accuracy']:.4f}")
        lines.append("")

    best = results.get("best")
    if best:
        lines.extend([
            "## Best config",
            "",
            f"HOG: {best['hog_config']}",
            f"SVM: {best['svm_config']}",
            f"Accuracy (val): {best['accuracy']:.4f}",
            "",
            "Per-class accuracy:",
        ])
        for label, acc in best["per_class_accuracy"].items():
            lines.append(f"- {label}: {acc:.4f}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
