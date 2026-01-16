#!/usr/bin/env python3
import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_DIR = ROOT / "dataset" / "train"
DEFAULT_VAL_DIR = ROOT / "dataset" / "val"
DEFAULT_TRAIN_CSV = ROOT / "dataset" / "train.csv"
DEFAULT_VAL_CSV = ROOT / "dataset" / "val.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


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


class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_transforms(arch, train):
    if arch in {"resnet18", "resnet50", "efficientnet_b0"}:
        if arch == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
        elif arch == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT
        else:
            weights = models.ResNet18_Weights.DEFAULT
        mean = weights.meta.get("mean", (0.485, 0.456, 0.406))
        std = weights.meta.get("std", (0.229, 0.224, 0.225))
        if train:
            tfm = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            tfm = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        return tfm, True, mean, std, 224

    mean = (0.5,)
    std = (0.5,)
    if train:
        tfm = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        tfm = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return tfm, False, mean, std, 48


def build_model(arch, num_classes):
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if arch == "small_cnn":
        return SmallCNN(num_classes)
    raise ValueError(f"Unknown arch: {arch}")


def accuracy(preds, labels):
    return (preds == labels).float().mean().item()


def _mixup_data(inputs, targets, alpha):
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    lam = random.betavariate(alpha, alpha)
    index = torch.randperm(inputs.size(0), device=inputs.device)
    mixed = lam * inputs + (1 - lam) * inputs[index]
    return mixed, targets, targets[index], lam


def _rand_bbox(width, height, lam):
    cut_rat = (1.0 - lam) ** 0.5
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)
    cx = random.randint(0, width - 1)
    cy = random.randint(0, height - 1)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)
    return x1, y1, x2, y2


def _cutmix_data(inputs, targets, alpha):
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    lam = random.betavariate(alpha, alpha)
    index = torch.randperm(inputs.size(0), device=inputs.device)
    x1, y1, x2, y2 = _rand_bbox(inputs.size(3), inputs.size(2), lam)
    mixed = inputs.clone()
    mixed[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (inputs.size(2) * inputs.size(3)))
    return mixed, targets, targets[index], lam


def train_epoch(model, loader, criterion, optimizer, device, mixup_alpha, cutmix_alpha):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        use_mix = mixup_alpha > 0 or cutmix_alpha > 0
        if use_mix and random.random() < 0.5 and mixup_alpha > 0:
            mixed, target_a, target_b, lam = _mixup_data(imgs, labels, mixup_alpha)
            outputs = model(mixed)
            loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
        elif use_mix and cutmix_alpha > 0:
            mixed, target_a, target_b, lam = _cutmix_data(imgs, labels, cutmix_alpha)
            outputs = model(mixed)
            loss = lam * criterion(outputs, target_a) + (1 - lam) * criterion(outputs, target_b)
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total_acc += accuracy(outputs.argmax(dim=1), labels) * imgs.size(0)
    count = len(loader.dataset)
    return total_loss / count, total_acc / count


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            total_acc += accuracy(outputs.argmax(dim=1), labels) * imgs.size(0)
    count = len(loader.dataset)
    return total_loss / count, total_acc / count


def load_label_counts(train_csv: Path):
    counts = {label: 0 for label in LABELS}
    with train_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[row["label"]] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Train CNN model for emotion recognition.")
    parser.add_argument("--arch", choices=["small_cnn", "resnet18", "resnet50", "efficientnet_b0"], default="resnet50")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--cutmix-alpha", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (epochs).")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tfm, use_rgb, mean, std, image_size = build_transforms(args.arch, train=True)
    val_tfm, _, _, _, _ = build_transforms(args.arch, train=False)

    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}

    train_ds = EmotionDataset(args.train_csv, args.train_dir, label_to_idx, train_tfm, use_rgb)
    val_ds = EmotionDataset(args.val_csv, args.val_dir, label_to_idx, val_tfm, use_rgb)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.arch, len(LABELS)).to(device)

    counts = load_label_counts(args.train_csv)
    total = sum(counts.values())
    weights = [total / max(1, counts[label]) for label in LABELS]
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    lr = args.lr
    if lr is None:
        if args.arch == "small_cnn":
            lr = 1e-3
        else:
            lr = 1e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.output:
        model_path = args.output
    else:
        model_path = MODELS_DIR / f"cnn_{args.arch}.pt"

    best_acc = 0.0
    epochs_no_improve = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.mixup_alpha,
            args.cutmix_alpha,
        )
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "arch": args.arch,
                "model_state": model.state_dict(),
                "labels": LABELS,
                "use_rgb": use_rgb,
                "mean": mean,
                "std": std,
                "image_size": image_size,
            }, model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val_acc={best_acc:.4f})")
                break
        print(f"Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    meta = {
        "arch": args.arch,
        "model_path": str(model_path),
        "labels": LABELS,
        "image_size": image_size,
        "use_rgb": use_rgb,
        "mean": mean,
        "std": std,
        "best_val_acc": best_acc,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "mixup_alpha": args.mixup_alpha,
        "cutmix_alpha": args.cutmix_alpha,
        "label_smoothing": args.label_smoothing,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "history": history,
    }
    meta_path = model_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {model_path}")
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
