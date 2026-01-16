#!/usr/bin/env python3
import csv
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "dataset" / "dataset"
TRAIN_DIR = ROOT / "dataset" / "train"
VAL_DIR = ROOT / "dataset" / "val"
PREDICT_DIR = ROOT / "dataset" / "predict"

SOURCE_LABELS = ROOT / "train.csv"
TRAIN_CSV = ROOT / "dataset" / "train.csv"
VAL_CSV = ROOT / "dataset" / "val.csv"

TRAIN_COUNT = 2000
VAL_COUNT = 700
RANDOM_SEED = 42


def load_labeled_ids():
    image_ids = []
    labels = {}
    with SOURCE_LABELS.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = int(row["id"])
            image_ids.append(image_id)
            labels[image_id] = row["label"]
    return image_ids, labels


def list_image_ids(directory: Path):
    ids = []
    for path in directory.glob("*.jpg"):
        try:
            ids.append(int(path.stem))
        except ValueError:
            continue
    return ids


def move_image(image_id: int, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{image_id}.jpg"
    if target_path.exists():
        return
    source_path = RAW_DIR / f"{image_id}.jpg"
    if source_path.exists():
        shutil.move(str(source_path), str(target_path))
        return
    raise FileNotFoundError(f"Missing image {image_id}.jpg in {RAW_DIR}")


def write_split_csv(path: Path, image_ids, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for image_id in image_ids:
            writer.writerow([image_id, labels[image_id]])


def main():
    labeled_ids, labels = load_labeled_ids()
    labeled_ids = sorted(labeled_ids)
    if len(labeled_ids) != TRAIN_COUNT + VAL_COUNT:
        raise SystemExit(
            f"Expected {TRAIN_COUNT + VAL_COUNT} labeled images, got {len(labeled_ids)}."
        )

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(labeled_ids)
    train_ids = sorted(labeled_ids[:TRAIN_COUNT])
    val_ids = sorted(labeled_ids[TRAIN_COUNT:])

    raw_ids = set(list_image_ids(RAW_DIR))
    labeled_set = set(labeled_ids)
    predict_ids = sorted(raw_ids - labeled_set)

    for image_id in train_ids:
        move_image(image_id, TRAIN_DIR)
    for image_id in val_ids:
        move_image(image_id, VAL_DIR)
    for image_id in predict_ids:
        move_image(image_id, PREDICT_DIR)

    write_split_csv(TRAIN_CSV, train_ids, labels)
    write_split_csv(VAL_CSV, val_ids, labels)

    print(f"Train images: {len(train_ids)} -> {TRAIN_DIR}")
    print(f"Val images: {len(val_ids)} -> {VAL_DIR}")
    print(f"Predict images: {len(predict_ids)} -> {PREDICT_DIR}")
    print(f"Wrote {TRAIN_CSV} and {VAL_CSV}")


if __name__ == "__main__":
    main()
