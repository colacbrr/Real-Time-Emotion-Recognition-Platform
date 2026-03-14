# Emotion Face Prediction MIR

Facial emotion recognition project built around two model families:
- classical pipelines using HOG/LBP features with SVM classifiers
- CNN-based pipelines using architectures such as ResNet and EfficientNet

The repository also includes a FastAPI backend, a React frontend, model registry metadata, and utility scripts for training, validation, comparison, and submission generation.

## Project Goals

- compare classical computer vision baselines against CNN models
- expose prediction and comparison workflows through a local web interface
- support dataset split management, model validation, and reproducible experiments
- add MIR-style search and indexing support around the image collection

## Repository Layout

```text
Emotion-Face-Prediction-MIR/
├── backend/                # FastAPI application
├── frontend/               # React + Vite interface
├── scripts/                # training and evaluation scripts
├── models/                 # model metadata and checkpoints
├── docs/                   # accuracy and architecture notes
├── start.sh                # convenience startup script
└── stop.sh                 # stop helper
```

Expected data directories during local work:
- `dataset/train`
- `dataset/val`
- `dataset/predict`

These are intentionally not committed.

## Implemented Areas

- FastAPI backend for prediction and status endpoints
- React frontend for interactive exploration
- classical feature pipelines with HOG/LBP + SVM
- CNN training and validation scripts
- model comparison utilities
- submission generation workflow
- architecture and accuracy notes under `docs/`

## Local Setup

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn search_server:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Convenience Startup

From the repository root:

```bash
chmod +x start.sh
./start.sh
```

Stop services later with:

```bash
./stop.sh
```

## Training Examples

Recommended CNN commands:

```bash
python3 scripts/cnn_train.py --arch=resnet50 --epochs=35 --patience=8
python3 scripts/cnn_train.py --arch=efficientnet_b0 --epochs=35 --patience=8
python3 scripts/cnn_make_submission.py --models models/cnn_resnet50.pt,models/cnn_efficientnet_b0.pt
```

Other useful scripts in `scripts/`:
- `train_model.py`
- `validate_model.py`
- `cnn_validate.py`
- `cnn_compare_models.py`
- `compare_models.py`
- `make_submission.py`
- `predict_submission.py`

## Documentation

- `Documentation_LATEX.md`
- `docs/ARCHITECTURE.md`
- `docs/ACCURACY.md`
- `Schema.md`

## Notes For Reviewers

- local machine paths were removed from this repo to keep setup portable
- datasets and large model checkpoints are expected to live outside version control
- the project is organized to show both experimentation code and a usable demo interface
