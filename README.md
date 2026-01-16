# Colacel_Cristian-Proiect-ASPICV-Python

Combined architecture: TEM-style MIR (CLIP + FAISS + RAG) plus the ASPICV facial
emotion pipeline (HOG/LBP + SVM) in one FastAPI + Vite stack.

## Layout
- `backend/`: FastAPI server (prediction + comparison + reports + CLIP search + RAG).
- `frontend/`: Vite + React UI.
- `dataset/`, `models/`, `reports/`: expected ASPICV assets.
- `outputs/`: generated CLIP embeddings + FAISS manifests for MIR.
- `scripts/`: training + validation utilities (HOG/LBP + SVM).

## Quick start

Backend:
```bash
cd /home/colac/School/Colacel_Cristian-Proiect-ASPICV-Python/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python search_server.py
```

Frontend:
```bash
cd /home/colac/School/Colacel_Cristian-Proiect-ASPICV-Python/frontend
npm install
npm run dev
```

## Environment overrides
```bash
export ASPICV_DATASET_DIR=/path/to/dataset
export ASPICV_MODELS_DIR=/path/to/models
export ASPICV_REPORTS_DIR=/path/to/reports

# MIR tuning (optional)
export MIR_INDEX_TYPE=flat # or hnsw
export MIR_HNSW_M=32
export MIR_HNSW_EF=64
export MIR_RERANK=true
export MIR_RERANK_ALPHA=0.25
export MIR_RAG_MODEL=llama3.1:8b
export MIR_MAX_IMAGES=0 # 0 = all labeled train/val images
```

## Notes
- Default dataset/model/report paths are `dataset/`, `models/`, `reports/` in this folder.
- Ensure `dataset/`, `models/`, and `reports/` exist or set the env overrides.
- MIR uses labeled `train/` and `val/` images to build the CLIP + FAISS index.
- Accuracy workbench notes: see `docs/ACCURACY.md`.
- For the final CSV, use `python3 scripts/make_submission.py --best` to pick the top val model.
- For higher accuracy, train the CNN pipeline (`scripts/cnn_train.py`) and submit via `scripts/cnn_make_submission.py`.
