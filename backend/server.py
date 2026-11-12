import base64
import json
import os
import threading
import time
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

try:
    import torch
except ImportError:
    torch = None

from PIL import Image

try:
    from torchvision import models as tv_models
    from torchvision import transforms as tv_transforms
except Exception:
    tv_models = None
    tv_transforms = None

try:
    import joblib
    import numpy as np
    from skimage import io, transform
    from skimage.feature import hog, local_binary_pattern
except Exception:
    joblib = None
    np = None
    io = None
    transform = None
    hog = None
    local_binary_pattern = None

ROOT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_DATASET_DIR = ROOT_DIR / "dataset"
DEFAULT_MODELS_DIR = ROOT_DIR / "models"
DEFAULT_REPORTS_DIR = ROOT_DIR / "reports"

dataset_env = os.getenv("ASPICV_DATASET_DIR", "").strip()
models_env = os.getenv("ASPICV_MODELS_DIR", "").strip()
reports_env = os.getenv("ASPICV_REPORTS_DIR", "").strip()

DATASET_DIR = Path(dataset_env) if dataset_env and dataset_env != "." else None
MODELS_DIR = Path(models_env) if models_env and models_env != "." else None
REPORTS_DIR = Path(reports_env) if reports_env and reports_env != "." else None

if DATASET_DIR is None or not DATASET_DIR.exists():
    DATASET_DIR = DEFAULT_DATASET_DIR
if MODELS_DIR is None or not MODELS_DIR.exists():
    MODELS_DIR = DEFAULT_MODELS_DIR
if REPORTS_DIR is None or not REPORTS_DIR.exists():
    REPORTS_DIR = DEFAULT_REPORTS_DIR

FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"

IMAGE_SIZE = (48, 48)

app = FastAPI(title="Facial Expression Classification API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if DATASET_DIR.exists():
    app.mount("/dataset", StaticFiles(directory=str(DATASET_DIR)), name="dataset")
if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

_init_lock = threading.Lock()
_ready = False
_init_error = None
_progress = {
    "stage": "idle",
    "message": "Idle",
    "current": 0,
    "total": 0,
    "percent": 0,
}

_log_lines = []
_log_lock = threading.Lock()

_metrics_lock = threading.Lock()
_metrics = {
    "model_load_time_sec": None,
    "cnn_model_load_time_sec": None,
    "query_count": 0,
    "query_total_ms": 0.0,
    "last_query_ms": None,
}

_models = {}
_cnn_models = {}
_dataset_summary = {
    "train_count": 0,
    "val_count": 0,
    "predict_count": 0,
}


def _log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] {message}"
    with _log_lock:
        _log_lines.append(line)
        if len(_log_lines) > 200:
            _log_lines.pop(0)


def _set_progress(stage: str, message: str, current: int = 0, total: int = 0) -> None:
    percent = int((current / total) * 100) if total else 0
    _progress.update(
        {
            "stage": stage,
            "message": message,
            "current": current,
            "total": total,
            "percent": percent,
        }
    )
    _log(f"{stage}: {message} ({current}/{total})")


def _append_metrics_event(event: dict) -> None:
    event["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    line = json.dumps(event)
    with _metrics_lock:
        log_file = ROOT_DIR / "metrics.jsonl"
        try:
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            pass


def _load_models() -> dict:
    if joblib is None:
        return {}

    models = {}
    if not MODELS_DIR.exists():
        return models

    for meta_path in MODELS_DIR.glob("*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            model_name = meta_path.name.replace(".meta.json", "")
            model_path = MODELS_DIR / f"{model_name}.joblib"
            if not model_path.exists():
                continue
            pipeline = joblib.load(model_path)
            models[model_name] = {"meta": meta, "pipeline": pipeline}
        except Exception:
            continue
    return models


def _build_cnn_model(arch: str, num_classes: int):
    if torch is None or tv_models is None:
        raise RuntimeError("Torch/Torchvision not available.")
    if arch == "resnet18":
        model = tv_models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "resnet50":
        model = tv_models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "efficientnet_b0":
        model = tv_models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if arch == "small_cnn":
        class SmallCNN(torch.nn.Module):
            def __init__(self, classes: int):
                super().__init__()
                self.features = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(32),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(2),
                )
                self.classifier = torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Dropout(0.4),
                    torch.nn.Linear(128 * 6 * 6, 256),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.4),
                    torch.nn.Linear(256, classes),
                )

            def forward(self, x):
                x = self.features(x)
                return self.classifier(x)

        return SmallCNN(num_classes)
    raise ValueError(f"Unknown CNN arch: {arch}")


def _load_cnn_models() -> dict:
    if torch is None or tv_transforms is None:
        return {}
    if not MODELS_DIR.exists():
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for path in MODELS_DIR.glob("*.pt"):
        try:
            checkpoint = torch.load(path, map_location=device)
            labels = checkpoint.get("labels") or []
            arch = checkpoint.get("arch")
            if not arch or not labels:
                continue
            model = _build_cnn_model(arch, len(labels))
            model.load_state_dict(checkpoint["model_state"])
            model.to(device)
            model.eval()
            models[path.stem] = {
                "checkpoint": checkpoint,
                "model": model,
                "device": device,
            }
        except Exception:
            continue
    return models


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.glob("*.jpg"))


def _load_dataset_summary() -> dict:
    return {
        "train_count": _count_images(DATASET_DIR / "train"),
        "val_count": _count_images(DATASET_DIR / "val"),
        "predict_count": _count_images(DATASET_DIR / "predict"),
    }


def _load_state() -> None:
    global _ready, _init_error, _models, _cnn_models, _dataset_summary
    with _init_lock:
        if _ready:
            return
        try:
            _set_progress("init", "Loading models", 0, 1)
            start = time.perf_counter()
            _models = _load_models()
            _metrics["model_load_time_sec"] = round(time.perf_counter() - start, 3)
            start = time.perf_counter()
            _cnn_models = _load_cnn_models()
            _metrics["cnn_model_load_time_sec"] = round(time.perf_counter() - start, 3)
            _set_progress("init", "Loading dataset summary", 0, 1)
            _dataset_summary = _load_dataset_summary()
            _ready = True
            _set_progress("ready", "Ready", 1, 1)
        except Exception as exc:
            _init_error = str(exc)
            _ready = False
            _set_progress("error", _init_error or "Initialization failed")


def _decode_image(payload: dict) -> bytes:
    image_path = payload.get("image_path")
    image_base64 = payload.get("image_base64")

    if image_path:
        candidate = Path(image_path).expanduser()
        if not candidate.exists():
            raise FileNotFoundError("image_path not found")
        return candidate.read_bytes()

    if image_base64:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        try:
            return base64.b64decode(image_base64, validate=True)
        except ValueError as exc:
            raise ValueError("invalid base64") from exc

    raise ValueError("provide image_path or image_base64")


def _preprocess_image(image_bytes: bytes):
    if io is None or transform is None or np is None:
        raise RuntimeError("Missing image dependencies.")
    image = io.imread(BytesIO(image_bytes), as_gray=True)
    if image.shape != IMAGE_SIZE:
        image = transform.resize(image, IMAGE_SIZE, anti_aliasing=True)
    return image.astype("float32")


def _extract_features(image, extractor: str, params: dict):
    if extractor == "hog":
        return hog(image, **params)
    if extractor == "lbp":
        lbp = local_binary_pattern(image, params["P"], params["R"], method=params["method"])
        n_bins = params["P"] + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return hist.astype("float32")
    raise ValueError(f"Unknown extractor: {extractor}")


def _predict_with_model(model_name: str, image_bytes: bytes):
    entry = _models.get(model_name)
    if not entry:
        return None, 0.0, None, None, "Model not available. Train a model first."

    meta = entry["meta"]
    pipeline = entry["pipeline"]
    image = _preprocess_image(image_bytes)
    features = _extract_features(image, meta["extractor"], meta["feature_params"])
    pred = pipeline.predict([features])[0]
    confidence = 0.0
    scores = None
    normalized_scores = None
    labels = list(getattr(pipeline, "classes_", [])) or meta.get("labels", [])

    if hasattr(pipeline, "predict_proba"):
        try:
            prob_scores = pipeline.predict_proba([features])[0]
            if len(prob_scores) == len(labels):
                normalized_scores = {label: float(value) for label, value in zip(labels, prob_scores)}
                confidence = max(normalized_scores.values()) if normalized_scores else 0.0
                scores = normalized_scores
        except Exception:
            normalized_scores = None
            scores = None

    if normalized_scores is None and hasattr(pipeline, "decision_function"):
        try:
            raw_scores = pipeline.decision_function([features])
            if raw_scores.ndim == 1 and len(labels) == 2:
                score = float(raw_scores[0])
                scores = {labels[0]: -score, labels[1]: score}
                prob = 1.0 / (1.0 + np.exp(-score))
                normalized_scores = {labels[0]: 1.0 - prob, labels[1]: prob}
                confidence = max(normalized_scores.values())
            elif raw_scores.ndim == 2 and raw_scores.shape[1] == len(labels):
                values = raw_scores[0].astype(float)
                scores = {label: float(value) for label, value in zip(labels, values)}
                stabilized = values - np.max(values)
                exp_scores = np.exp(stabilized)
                probs = exp_scores / np.sum(exp_scores)
                normalized_scores = {label: float(value) for label, value in zip(labels, probs)}
                confidence = max(normalized_scores.values())
        except Exception:
            scores = None
            confidence = 0.0

    warning = None if normalized_scores else "Model did not return score distribution."
    return pred, confidence, scores, normalized_scores, warning


def _build_cnn_transform(checkpoint: dict):
    image_size = int(checkpoint.get("image_size", 48))
    use_rgb = bool(checkpoint.get("use_rgb", False))
    mean = checkpoint.get("mean") or ((0.485, 0.456, 0.406) if use_rgb else (0.5,))
    std = checkpoint.get("std") or ((0.229, 0.224, 0.225) if use_rgb else (0.5,))

    if use_rgb:
        return tv_transforms.Compose([
            tv_transforms.Resize(256 if image_size == 224 else image_size),
            tv_transforms.CenterCrop(image_size),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=mean, std=std),
        ])
    return tv_transforms.Compose([
        tv_transforms.Resize((image_size, image_size)),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=mean, std=std),
    ])


def _predict_with_cnn(model_name: str, image_bytes: bytes):
    entry = _cnn_models.get(model_name)
    if not entry:
        return None, 0.0, None, None, "CNN model not available."
    if Image is None or torch is None or tv_transforms is None:
        return None, 0.0, None, None, "Missing CNN dependencies."

    checkpoint = entry["checkpoint"]
    labels = checkpoint.get("labels", [])
    if not labels:
        return None, 0.0, None, None, "Missing CNN labels."

    img = Image.open(BytesIO(image_bytes))
    img = img.convert("RGB" if checkpoint.get("use_rgb", False) else "L")
    transform = _build_cnn_transform(checkpoint)
    tensor = transform(img).unsqueeze(0).to(entry["device"])

    with torch.no_grad():
        logits = entry["model"](tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().tolist()

    scores = {label: float(value) for label, value in zip(labels, probs)}
    pred_idx = int(torch.tensor(probs).argmax().item())
    confidence = float(max(probs))
    return labels[pred_idx], confidence, scores, scores, None


def _load_reports_summary():
    summary = {
        "comparison": None,
        "validations": {},
    }
    if not REPORTS_DIR.exists():
        return summary

    comparison_files = sorted(REPORTS_DIR.glob("comparison_*.json"))
    if comparison_files:
        try:
            summary["comparison"] = json.loads(comparison_files[-1].read_text(encoding="utf-8"))
        except Exception:
            summary["comparison"] = None

    validation_files = sorted(REPORTS_DIR.glob("validation_*.json"))
    for path in validation_files:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            model_path = report.get("model", {}).get("path", "")
            model_name = Path(model_path).stem if model_path else "unknown"
            summary["validations"][model_name] = report
        except Exception:
            continue

    return summary


def _update_query_metrics(latency_ms: float) -> None:
    _metrics["query_count"] += 1
    _metrics["query_total_ms"] += latency_ms
    _metrics["last_query_ms"] = round(latency_ms, 2)


@app.on_event("startup")
def startup_event():
    threading.Thread(target=_load_state, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def root():
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    return HTMLResponse(
        """
        <html>
          <head><title>ASPICV Emotion API</title></head>
          <body style="font-family: Arial, sans-serif; margin: 40px;">
            <h1>ASPICV Emotion API</h1>
            <p>Backend is running. Useful endpoints:</p>
            <ul>
              <li><a href="/status">/status</a></li>
              <li><a href="/docs">/docs</a></li>
            </ul>
            <pre>cd frontend
npm install
npm run dev</pre>
          </body>
        </html>
        """
    )


@app.get("/status")
def status():
    avg_query_ms = None
    if _metrics["query_count"]:
        avg_query_ms = round(_metrics["query_total_ms"] / _metrics["query_count"], 2)

    return {
        "ready": _ready,
        "error": _init_error,
        "progress": _progress,
        "models": sorted(list(_models.keys()) + list(_cnn_models.keys())),
        "classic_models": sorted(_models.keys()),
        "cnn_models": sorted(_cnn_models.keys()),
        "dataset": _dataset_summary,
        "paths": {
            "dataset": str(DATASET_DIR),
            "models": str(MODELS_DIR),
            "reports": str(REPORTS_DIR),
        },
        "metrics": {
            **_metrics,
            "avg_query_ms": avg_query_ms,
        },
    }


@app.get("/events")
def events():
    def event_stream():
        last_payload = None
        while True:
            payload = dict(_progress)
            payload["ready"] = _ready
            payload["error"] = _init_error
            if payload != last_payload:
                yield f"data: {json.dumps(payload)}\n\n"
                last_payload = payload
            time.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/logs")
def logs():
    def log_stream():
        last_len = 0
        while True:
            with _log_lock:
                current_len = len(_log_lines)
                if current_len > last_len:
                    chunk = _log_lines[last_len:current_len]
                    last_len = current_len
                else:
                    chunk = []
            for line in chunk:
                yield f"data: {json.dumps({'line': line})}\n\n"
            time.sleep(1)

    return StreamingResponse(log_stream(), media_type="text/event-stream")


@app.get("/models")
def list_models():
    return {
        "models": sorted(list(_models.keys()) + list(_cnn_models.keys())),
        "classic_models": sorted(_models.keys()),
        "cnn_models": sorted(_cnn_models.keys()),
    }


@app.get("/reports")
def reports():
    return _load_reports_summary()


@app.post("/predict")
def predict(payload: dict):
    if not _ready:
        return {"error": _init_error or "Initialization in progress"}

    if not payload:
        return {"error": "missing payload"}

    try:
        image_bytes = _decode_image(payload)
    except Exception as exc:
        return {"error": str(exc)}

    model_name = payload.get("model_name")
    chosen_model = model_name or (next(iter(_models.keys()), None))
    if chosen_model is None and _cnn_models:
        chosen_model = next(iter(_cnn_models.keys()), None)

    start = time.perf_counter()
    warning = None
    label = None
    confidence = 0.0
    scores = None
    normalized_scores = None

    if chosen_model and (_models or _cnn_models):
        try:
            if chosen_model in _cnn_models:
                label, confidence, scores, normalized_scores, warning = _predict_with_cnn(
                    chosen_model,
                    image_bytes,
                )
            else:
                label, confidence, scores, normalized_scores, warning = _predict_with_model(
                    chosen_model,
                    image_bytes,
                )
        except Exception as exc:
            warning = f"Prediction failed: {exc}"
            label = None

    if not label:
        return {"error": warning or "No model available. Load a trained model first.",
                "model": chosen_model}

    latency_ms = (time.perf_counter() - start) * 1000.0
    _update_query_metrics(latency_ms)
    _append_metrics_event(
        {
            "type": "predict",
            "model": chosen_model,
            "latency_ms": round(latency_ms, 2),
        }
    )

    return {
        "label": label,
        "confidence": confidence,
        "bytes": len(image_bytes),
        "model": chosen_model,
        "scores": scores,
        "normalized_scores": normalized_scores,
        "warning": warning,
        "latency_ms": round(latency_ms, 2),
    }


@app.post("/compare")
def compare(payload: dict):
    if not _ready:
        return {"error": _init_error or "Initialization in progress"}

    if not payload:
        return {"error": "missing payload"}

    try:
        image_bytes = _decode_image(payload)
    except Exception as exc:
        return {"error": str(exc)}

    if not _models and not _cnn_models:
        return {"error": "no models available"}

    results = []
    start = time.perf_counter()
    for name in sorted(list(_models.keys()) + list(_cnn_models.keys())):
        try:
            if name in _cnn_models:
                pred, confidence, scores, normalized_scores, warning = _predict_with_cnn(
                    name,
                    image_bytes,
                )
            else:
                pred, confidence, scores, normalized_scores, warning = _predict_with_model(
                    name,
                    image_bytes,
                )
            results.append(
                {
                    "model": name,
                    "label": pred,
                    "confidence": confidence,
                    "scores": scores,
                    "normalized_scores": normalized_scores,
                    "warning": warning,
                }
            )
        except Exception as exc:
            results.append({"model": name, "error": str(exc)})

    latency_ms = (time.perf_counter() - start) * 1000.0
    _update_query_metrics(latency_ms)
    _append_metrics_event(
        {
            "type": "compare",
            "models": len(results),
            "latency_ms": round(latency_ms, 2),
        }
    )

    return {"results": results, "bytes": len(image_bytes), "latency_ms": round(latency_ms, 2)}


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("ASPICV_HOST", "0.0.0.0")
    port = int(os.environ.get("ASPICV_PORT", "8000"))
    uvicorn.run("server:app", host=host, port=port, reload=False)
