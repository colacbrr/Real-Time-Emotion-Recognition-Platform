# Accuracy workbench

The primary metric is validation accuracy. The unlabeled 300-image set can only
be scored with labels, so we track model confidence and class balance there.

## Validation (labeled data)

Compare all registered models:

```bash
python3 scripts/compare_models.py
```

Validate a specific registered model:

```bash
python3 scripts/validate_model.py --model-name hog_svm
```

Both commands write JSON + Markdown reports to `reports/`.

## CNN validation (GPU recommended)

Train a CNN model (ResNet50 with transfer learning by default):

```bash
python3 scripts/cnn_train.py --arch resnet50 --epochs 25
```

Train EfficientNet-B0:

```bash
python3 scripts/cnn_train.py --arch efficientnet_b0 --epochs 25
```

Validate a trained CNN checkpoint:

```bash
python3 scripts/cnn_validate.py --model models/cnn_resnet50.pt
```

## Unlabeled 300-image set

Generate predictions for submission/testing (auto-picks the best validation model):

```bash
python3 scripts/make_submission.py --best
```

The script writes a CSV in the repo root. Use the output to compute mean
confidence or inspect class balance before public testing.

## CNN submission

Generate submission from the CNN checkpoint:

```bash
python3 scripts/cnn_make_submission.py --model models/cnn_resnet50.pt
```

Compare CNN checkpoints (uses stored meta metrics):

```bash
python3 scripts/cnn_compare_models.py --models models/cnn_resnet18.pt,models/cnn_resnet50.pt
```

Ensemble two CNNs for submission:

```bash
python3 scripts/cnn_make_submission.py --models models/cnn_resnet50.pt,models/cnn_efficientnet_b0.pt
```
