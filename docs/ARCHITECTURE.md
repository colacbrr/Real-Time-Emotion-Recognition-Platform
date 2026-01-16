# Architecture

This project merges two pipelines into a single, standalone stack:

- ASPICV facial emotion inference (HOG/LBP feature extraction + SVM classifiers).
- TEM MIR search (CLIP embeddings + FAISS index + optional RAG explanations).

## Top-level layout

- `backend/`: FastAPI API that serves predictions, comparisons, reports, and MIR.
- `frontend/`: Vite + React control room UI.
- `dataset/`: `train/`, `val/`, and `predict/` image splits plus CSV metadata.
- `models/`: trained model artifacts and `registry.json` metadata.
- `reports/`: validation and model comparison reports.
- `outputs/`: generated CLIP embeddings + FAISS metadata for MIR search.
- `scripts/`: training/validation utilities for ASPICV models.

## Data flow

1. Training utilities (`scripts/`) build or validate models using `dataset/`.
2. Model artifacts land in `models/` with JSON metadata for reproducibility.
3. Backend loads `models/` and `dataset/` to serve `/predict` and `/compare`.
4. MIR builds CLIP embeddings from labeled images and stores them in `outputs/`.
5. Frontend visualizes predictions, comparisons, MIR results, and reports.

## Accuracy focus

Accuracy on the labeled validation set is the primary metric. Use the scripts
in `scripts/` to compare model configs, validate chosen models, and generate
prediction outputs for the unlabeled set.
