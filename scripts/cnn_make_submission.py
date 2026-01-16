#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PREDICT_DIR = ROOT / "dataset" / "predict"
OUTPUT_CSV = ROOT / "submission.csv"


class PredictDataset(Dataset):
    def __init__(self, image_ids, images_dir, transform, use_rgb):
        self.image_ids = image_ids
        self.images_dir = images_dir
        self.transform = transform
        self.use_rgb = use_rgb

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = self.images_dir / f"{image_id}.jpg"
        img = Image.open(path)
        if self.use_rgb:
            img = img.convert("RGB")
        else:
            img = img.convert("L")
        img = self.transform(img)
        return image_id, img


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


def list_predict_ids():
    image_ids = []
    for path in PREDICT_DIR.glob("*.jpg"):
        try:
            image_ids.append(int(path.stem))
        except ValueError:
            continue
    return sorted(image_ids)


def _load_checkpoint(model_path: Path, device):
    checkpoint = torch.load(model_path, map_location=device)
    labels = checkpoint["labels"]
    use_rgb = checkpoint.get("use_rgb", False)
    mean = checkpoint.get("mean", (0.5,))
    std = checkpoint.get("std", (0.5,))
    image_size = checkpoint.get("image_size", 48)

    if checkpoint["arch"] == "resnet18":
        from torchvision import models
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
    elif checkpoint["arch"] == "resnet50":
        from torchvision import models
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
    elif checkpoint["arch"] == "efficientnet_b0":
        from torchvision import models
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(labels))
    else:
        from cnn_train import SmallCNN
        model = SmallCNN(len(labels))

    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    return checkpoint, model, labels, use_rgb, mean, std, image_size


def main():
    parser = argparse.ArgumentParser(description="Generate submission using a trained CNN model.")
    parser.add_argument("--model", type=Path, help="Path to a single .pt model file.")
    parser.add_argument("--models", help="Comma-separated list of .pt model files to ensemble.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = []
    if args.models:
        model_paths = [Path(p.strip()) for p in args.models.split(",") if p.strip()]
    elif args.model:
        model_paths = [args.model]
    else:
        raise SystemExit("Provide --model or --models.")

    checkpoints = []
    models = []
    labels = None
    use_rgb = None
    mean = None
    std = None
    image_size = None

    for path in model_paths:
        checkpoint, model, ck_labels, ck_use_rgb, ck_mean, ck_std, ck_size = _load_checkpoint(path, device)
        if labels is None:
            labels = ck_labels
            use_rgb = ck_use_rgb
            mean = ck_mean
            std = ck_std
            image_size = ck_size
        else:
            if ck_labels != labels or ck_use_rgb != use_rgb or ck_size != image_size or ck_mean != mean or ck_std != std:
                raise SystemExit("All ensemble models must share labels and preprocessing.")
        checkpoints.append(checkpoint)
        models.append(model)

    predict_ids = list_predict_ids()
    if not predict_ids:
        raise SystemExit(f"No prediction images found in {PREDICT_DIR}")

    tfm = build_transforms(image_size, use_rgb, mean, std)
    dataset = PredictDataset(predict_ids, PREDICT_DIR, tfm, use_rgb)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    preds = []
    with torch.no_grad():
        for batch in loader:
            ids, imgs = batch
            imgs = imgs.to(device)
            prob_sum = None
            for model in models:
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                prob_sum = probs if prob_sum is None else prob_sum + probs
            probs = prob_sum / len(models)
            batch_preds = probs.argmax(dim=1).cpu().tolist()
            clean_ids = [int(x) for x in ids]
            preds.extend(zip(clean_ids, batch_preds))

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for image_id, pred_idx in preds:
            writer.writerow([image_id, labels[pred_idx]])

    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
