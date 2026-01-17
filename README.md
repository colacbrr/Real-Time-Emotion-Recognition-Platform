# Colacel_Cristian-Proiect-ASPICV-Python

Recunoaștere emoții faciale din imagini 48×48, cu două familii de modele:
HOG/LBP + SVM (baseline clasic) și CNN (ResNet/EfficientNet) pentru acuratețe ridicată.
Proiectul include backend FastAPI, UI React și scripturi pentru antrenare, validare și generarea `submission.csv`.

## Structură minimă
- `backend/`, `frontend/`, `scripts/`
- `dataset/` (train/val/predict + CSV-uri)
- `models/` (checkpoint-uri .pt/.joblib)

## Rulare UI
```bash
cd /home/colac/School/Colacel_Cristian-Proiect-ASPICV-Python/backend
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python search_server.py
```
```bash
cd /home/colac/School/Colacel_Cristian-Proiect-ASPICV-Python/frontend
npm install
npm run dev
```

## CNN (recomandat)
```bash
python3 scripts/cnn_train.py --arch=resnet50 --epochs=35 --patience=8
python3 scripts/cnn_train.py --arch=efficientnet_b0 --epochs=35 --patience=8
python3 scripts/cnn_make_submission.py --models models/cnn_resnet50.pt,models/cnn_efficientnet_b0.pt
```

## Documentație completă
- `Documentation_LATEX.md`
- `docs/ACCURACY.md`
