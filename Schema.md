# Fluxurile proiectului

## Studiul reproductibil

```text
Imagini și etichete
  -> manifest cu ID, hash și grup de duplicate
  -> inventar eligibil și separare după sursă
  -> subseturi comune de antrenare și evaluare
  -> HOG sau trăsături ResNet18 fixe
  -> StandardScaler ajustat exclusiv pe antrenare
  -> SVM liniar sau RBF, cu selecție internă a parametrilor
  -> predicții individuale
  -> metrici, incertitudine, tabele și figuri
```

Reperul majoritar folosește numai etichetele de antrenare. [Protocolul complet](docs/publicare/studiu_v2/README.md) precizează gruparea, împărțirile și interpretarea rezultatelor.

## Demonstrația locală

```text
Interfață React: imagine sau cale locală
  -> FastAPI, backend/server.py
  -> model local compatibil
     -> HOG/LBP + pipeline SVM salvat
     -> sau CNN cu arhitectura și transformările checkpoint-ului
  -> etichetă, scoruri și timpul cererii
  -> interfață React
```

Fără model funcțional, răspunsul conține o eroare, fără etichetă implicită. Scorurile afișate nu certifică probabilități calibrate.

## Rutele API

- `POST /predict`: clasifică o imagine cu modelul selectat.
- `POST /compare`: compară predicțiile modelelor pentru aceeași imagine; nu evaluează un set de validare.
- `GET /models`: listează modelele încărcate.
- `GET /reports`: afișează rapoartele locale.
- `GET /status`: starea modelelor și rezumatul imaginilor disponibile.
- `GET /events`: evenimente despre inițializare.
- `GET /logs`: jurnalul serverului.

La pornire se încarcă artefactele locale, fără antrenare sau descărcare. Demonstrația și studiul controlat au fluxuri de evaluare distincte.
