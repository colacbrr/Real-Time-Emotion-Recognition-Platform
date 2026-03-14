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

## Backend

From `backend`:

```bash
cd backend
. .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn search_server:app --reload --port 8000
```

Notes:
- The first run builds CLIP embeddings for the MIR index (may take a while).
- Optional: install and run Ollama to enable RAG explanations.

## Frontend

From `frontend`:

```bash
npm install
npm run dev
```
