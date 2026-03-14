#!/usr/bin/env python3
import argparse
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

DEFAULT_TRAIN_COUNT = 2000
DEFAULT_VAL_COUNT = 700
DEFAULT_RANDOM_SEED = 42


def load_labeled_ids():
    if not SOURCE_LABELS.exists():
        raise SystemExit(f"Missing source labels CSV: {SOURCE_LABELS}")
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
    parser = argparse.ArgumentParser(description="Split raw labeled images into train/val/predict folders.")
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--val-count", type=int, default=DEFAULT_VAL_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of moving them out of the raw dataset directory.",
    )
    args = parser.parse_args()

    if not RAW_DIR.exists():
        raise SystemExit(f"Missing raw dataset directory: {RAW_DIR}")

    labeled_ids, labels = load_labeled_ids()
    labeled_ids = sorted(labeled_ids)
    expected_count = args.train_count + args.val_count
    if len(labeled_ids) != expected_count:
        raise SystemExit(
            f"Expected {expected_count} labeled images, got {len(labeled_ids)}."
        )

    rng = random.Random(args.seed)
    rng.shuffle(labeled_ids)
    train_ids = sorted(labeled_ids[: args.train_count])
    val_ids = sorted(labeled_ids[args.train_count :])

    raw_ids = set(list_image_ids(RAW_DIR))
    labeled_set = set(labeled_ids)
    predict_ids = sorted(raw_ids - labeled_set)

    transfer = shutil.copy2 if args.copy else shutil.move
    for image_id, target_dir in (
        *((image_id, TRAIN_DIR) for image_id in train_ids),
        *((image_id, VAL_DIR) for image_id in val_ids),
        *((image_id, PREDICT_DIR) for image_id in predict_ids),
    ):
        target_dir.mkdir(parents=True, exist_ok=True)
        source_path = RAW_DIR / f"{image_id}.jpg"
        target_path = target_dir / f"{image_id}.jpg"
        if target_path.exists():
            continue
        if not source_path.exists():
            raise FileNotFoundError(f"Missing image {image_id}.jpg in {RAW_DIR}")
        transfer(str(source_path), str(target_path))

    write_split_csv(TRAIN_CSV, train_ids, labels)
    write_split_csv(VAL_CSV, val_ids, labels)

    print(f"Train images: {len(train_ids)} -> {TRAIN_DIR}")
    print(f"Val images: {len(val_ids)} -> {VAL_DIR}")
    print(f"Predict images: {len(predict_ids)} -> {PREDICT_DIR}")
    print(f"Wrote {TRAIN_CSV} and {VAL_CSV}")


if __name__ == "__main__":
    main()
