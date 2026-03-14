# Scripts Guide

The `scripts/` directory contains the model-training and evaluation workflows shared with the core emotion-prediction repository.

## Classical Models

- `train_model.py`: train one registry-defined feature model such as `hog_svm` or `lbp_svm`
- `validate_model.py`: validate a saved classical model on the labeled validation split
- `compare_models.py`: train and compare all registry models on the validation split
- `hog_model_search.py`: search HOG parameter combinations

## CNN Models

- `cnn_train.py`: train a CNN checkpoint such as `resnet18`, `resnet50`, `efficientnet_b0`, or `small_cnn`
- `cnn_validate.py`: evaluate a saved CNN checkpoint and write a validation report
- `cnn_compare_models.py`: compare stored CNN checkpoint metadata
- `cnn_make_submission.py`: generate `submission.csv` from one or more CNN checkpoints

## Data Preparation

- `split_dataset.py`: create dataset splits
- `rebalance_split.py`: rebalance split distributions when needed

## Submission and Prediction

- `make_submission.py`: generate a submission CSV from the best classical model
- `predict_submission.py`: run prediction-oriented workflows against the unlabeled split

## Model Metadata

- `model_registry.py`: load and query the `models/registry.json` configuration
