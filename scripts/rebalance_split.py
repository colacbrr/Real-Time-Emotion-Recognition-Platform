#!/usr/bin/env python3
import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "dataset" / "train"
VAL_DIR = ROOT / "dataset" / "val"
TRAIN_CSV = ROOT / "dataset" / "train.csv"
VAL_CSV = ROOT / "dataset" / "val.csv"


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerows(rows)


def move_image(image_id: str, src_dir: Path, dst_dir: Path):
    src = src_dir / f"{image_id}.jpg"
    dst = dst_dir / f"{image_id}.jpg"
    if not src.exists():
        raise FileNotFoundError(f"Missing image: {src}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def main():
    parser = argparse.ArgumentParser(description="Move images from val to train (stratified).")
    parser.add_argument("--keep-val", type=int, default=100, help="How many val samples to keep.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backup", action="store_true", help="Write train_orig.csv/val_orig.csv backups.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_rows = load_rows(TRAIN_CSV)
    val_rows = load_rows(VAL_CSV)

    if args.keep_val < 0 or args.keep_val > len(val_rows):
        raise SystemExit(f"keep-val must be between 0 and {len(val_rows)}")

    by_label = defaultdict(list)
    for row in val_rows:
        by_label[row["label"]].append(row)

    keep_rows = []
    move_rows = []
    total_val = len(val_rows)
    for label, rows in by_label.items():
        rng.shuffle(rows)
        keep_count = round(len(rows) * args.keep_val / total_val)
        keep_rows.extend(rows[:keep_count])
        move_rows.extend(rows[keep_count:])

    # Fix rounding drift
    if len(keep_rows) > args.keep_val:
        overflow = len(keep_rows) - args.keep_val
        move_rows.extend(keep_rows[-overflow:])
        keep_rows = keep_rows[:-overflow]
    elif len(keep_rows) < args.keep_val:
        deficit = args.keep_val - len(keep_rows)
        move_rows.sort(key=lambda r: rng.random())
        keep_rows.extend(move_rows[:deficit])
        move_rows = move_rows[deficit:]

    if args.backup:
        TRAIN_CSV.with_name("train_orig.csv").write_text(
            TRAIN_CSV.read_text(encoding="utf-8"), encoding="utf-8"
        )
        VAL_CSV.with_name("val_orig.csv").write_text(
            VAL_CSV.read_text(encoding="utf-8"), encoding="utf-8"
        )

    for row in move_rows:
        move_image(row["id"], VAL_DIR, TRAIN_DIR)

    new_train_rows = train_rows + move_rows
    write_rows(TRAIN_CSV, new_train_rows)
    write_rows(VAL_CSV, keep_rows)

    print(f"Moved {len(move_rows)} samples to train.")
    print(f"Kept {len(keep_rows)} samples in val.")


if __name__ == "__main__":
    main()
