#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
VAL_DIR = ROOT / "dataset" / "val"
VAL_CSV = ROOT / "dataset" / "val.csv"
REPORTS_DIR = ROOT / "reports"


class EmotionDataset(Dataset):
    def __init__(self, csv_path, images_dir, label_to_idx, transform, use_rgb):
        self.images_dir = images_dir
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.use_rgb = use_rgb
        self.items = []
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = int(row["id"])
                label = row["label"]
                self.items.append((image_id, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_id, label = self.items[idx]
        path = self.images_dir / f"{image_id}.jpg"
        img = Image.open(path)
        if self.use_rgb:
            img = img.convert("RGB")
        else:
            img = img.convert("L")
        img = self.transform(img)
        label_idx = self.label_to_idx[label]
        return img, label_idx


def build_transforms(image_size, use_rgb, mean, std):
    if use_rgb:
        return transforms.Compose([
            transforms.Resize(256 if image_size == 224 else image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def main():
    parser = argparse.ArgumentParser(description="Validate CNN model on val split.")
    parser.add_argument("--model", type=Path, required=True, help="Path to .pt model file.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device)
    labels = checkpoint["labels"]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    use_rgb = checkpoint.get("use_rgb", False)
    mean = checkpoint.get("mean", (0.5,))
    std = checkpoint.get("std", (0.5,))
    image_size = checkpoint.get("image_size", 48)

    if checkpoint["arch"] == "resnet18":
        from torchvision import models
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(labels))
    elif checkpoint["arch"] == "resnet50":
        from torchvision import models
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(labels))
    elif checkpoint["arch"] == "efficientnet_b0":
        from torchvision import models
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(labels))
    else:
        from cnn_train import SmallCNN
        model = SmallCNN(len(labels))

    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    tfm = build_transforms(image_size, use_rgb, mean, std)
    dataset = EmotionDataset(VAL_CSV, VAL_DIR, label_to_idx, tfm, use_rgb)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels_batch in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.tolist())

    acc = float(accuracy_score(all_labels, all_preds))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(labels))))
    totals = cm.sum(axis=1)
    per_class_accuracy = {}
    for idx, label in enumerate(labels):
        denom = totals[idx]
        per_class_accuracy[label] = float(cm[idx, idx] / denom) if denom else 0.0

    report = {
        "metric": "accuracy",
        "accuracy": acc,
        "labels": labels,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": cm.tolist(),
        "model": {
            "path": str(args.model),
            "arch": checkpoint["arch"],
            "image_size": image_size,
            "use_rgb": use_rgb,
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / f"cnn_validation_{timestamp}.json"
    md_path = REPORTS_DIR / f"cnn_validation_{timestamp}.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# CNN Validation report",
        "",
        f"Metric: {report['metric']}",
        f"Accuracy (val): {acc:.4f}",
        "",
        "## Per-class accuracy",
        "",
    ]
    for label, acc_value in per_class_accuracy.items():
        lines.append(f"- {label}: {acc_value:.4f}")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
