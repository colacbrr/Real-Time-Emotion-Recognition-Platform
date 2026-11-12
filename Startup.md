# Startup

This project has a backend (FastAPI) and a frontend (Vite).

## One-command startup

From the repository root:

```bash
chmod +x start.sh
./start.sh
```

To stop the backend later:

```bash
./stop.sh
```

Stopping is scoped to this repository's `backend/` and `frontend/` working directories on Linux. It does not terminate other projects' development servers. Inspect the selection without sending signals:

```bash
python3 scripts/stop_local_app.py --dry-run
```

## Backend

From `backend`:

```bash
cd backend
. .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```

Notes:
- Startup loads local classification models and counts images; it does not train or download models.
- Without a trained model, prediction returns an explicit error.
- Configure local inputs with `ASPICV_DATASET_DIR`, `ASPICV_MODELS_DIR`, and `ASPICV_REPORTS_DIR`.
- The demonstration serves compatible historical checkpoints from `models/`; the scientific study has a separate [evaluation workflow](docs/publicare/studiu_v2/README.md).

## Frontend

From `frontend`:

```bash
npm install
npm run dev
```
