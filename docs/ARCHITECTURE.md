# Architecture

The repository contains an offline facial-expression study and a local classification demonstration.

## Scientific study

The manifest identifies images, labels, hashes and known duplicate groups. Evaluation keeps groups together, fits preprocessing on training data only, and compares HOG with frozen ResNet18 features under common SVM heads. Predictions, class supports, accuracy, macro-F1, recall, confusion matrices and conditional uncertainty are exported.

The [controlled study](publicare/studiu_v2/README.md) records configurations and folds. Historical reports remain separate. Demonstration scores do not replace this evaluation.

## Local demonstration

- `backend/server.py`: FastAPI service for prediction, model comparison, availability, reports and initialization events.
- `frontend/`: React/Vite interface for selecting an image and inspecting predictions.
- `dataset/`: local images and academic labels, not included in Git.
- `models/`: compatible classical `.joblib` artifacts with `.meta.json`, and historical CNN `.pt` checkpoints.
- `reports/`: local historical validation/comparison reports.
- `outputs/`: offline study inventories, fixed features, model artifacts and predictions; ignored by Git.
- `scripts/`: training, provenance, audit, evaluation and document tools.

Startup loads local model artifacts and counts images, without training or network downloads. Classical prediction extracts HOG or LBP and applies the saved pipeline. CNN prediction reconstructs the checkpoint architecture and preprocessing. Without a usable model, prediction returns an explicit error.

The backend accepts historical checkpoint formats. The scientific v2 SVMs on cached HOG/ResNet18 features are evaluated through dedicated scripts and are not automatically imported into the demonstration.

See [Startup.md](../Startup.md). The interface does not establish deployment readiness or real-time performance.
