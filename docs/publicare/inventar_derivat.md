# Inventarul derivat și protocolul intern v1

Stare: 2026-11-09. Regula de excludere documentată a fost aplicată numai unor **CSV-uri derivate**, fără mutarea, copierea, ștergerea sau reetichetarea imaginilor originale.

## Evidența tuturor imaginilor

`outputs/internal-inventory-v1/` conține:

| Fișier | Rânduri | Rol |
|---|---:|---|
| `eligible_inventory.csv` | 2677 | Inventar etichetat pentru experimentul intern retrospectiv |
| `excluded_inventory.csv` | 23 | Exemple reținute în afara dezvoltării interne, cu motivul original al deciziei |
| `official_unlabeled_inventory.csv` | 300 | Testul oficial fără etichete, păstrat separat |
| `selection_ledger.csv` | 3000 | Evidența completă a ID-urilor și rolului fiecărui exemplu |
| `summary.json` | — | Hash-uri, distribuții, verificarea sursei și limite |

Cele 23 de excluderi înseamnă 13 exemple din grupuri cu dezacord de etichetă, 3 din grupuri de potrivire incertă și încă 7 cu suprapunere față de predict. Toate ID-urile sunt păstrate în evidență, fără suprapuneri între cele trei categorii. Nu există hash-uri sau grupuri revizuite comune între inventarul eligibil și celelalte două categorii.

Pentru exemplele eligibile, coloana originală `split` este păstrată sub numele **`historical_split`**, pentru trasabilitate; nu este utilizată tacit ca noul split. `subject_id` rămâne gol, iar `duplicate_group` este păstrat separat. Cele 300 de exemple oficiale rămân fără etichete; nu se completează din duplicate sau predicții.

| Clasă | Eligibile | Excluse |
|---|---:|---:|
| angry | 354 | 1 |
| disgust | 38 | 1 |
| fear | 407 | 6 |
| happy | 639 | 1 |
| neutral | 488 | 1 |
| sad | 448 | 3 |
| surprise | 303 | 10 |

Cele 2677 de imagini eligibile formează 2645 grupuri de imagini. Copiile rămase cu etichete concordante sunt păstrate împreună la împărțire; nu sunt eliminate automat. Prin urmare, metricile vor fi **per imagine**, nu per fotografie unică. Nu afirmăm că acest inventar este complet curat sau independent pe persoane: screeningul de similaritate este incomplet, revizuirea este singulară, iar metadatele de subiect lipsesc.

## Protocol comun fixat înaintea noilor antrenări

Manifestul nou este `outputs/internal-protocol-v1/manifest.csv`. Hash SHA-256:

`29aa1ebdad60bc55a672a3abc5bc2d3685bfa05654a2da53c7d5a74b9376dfc8`.

Regulile și configurațiile sunt în [internal_protocol_v1.json](internal_protocol_v1.json), scris înaintea antrenărilor și calculării scorurilor noi. Împărțirea folosește seed 42 și prima partiție a două `StratifiedGroupKFold` imbricate (5, apoi 4 fold-uri), cu grupare pe hash-uri, grupuri de imagini revizuite și eventuale ID-uri de subiect furnizate. Nu se aleg seed-uri după performanță.

| Clasă | Train | Validare | Test intern |
|---|---:|---:|---:|
| angry | 212 | 71 | 71 |
| disgust | 23 | 7 | 8 |
| fear | 244 | 81 | 82 |
| happy | 383 | 128 | 128 |
| neutral | 293 | 98 | 97 |
| sad | 269 | 90 | 89 |
| surprise | 181 | 61 | 61 |
| **Total** | **1605** | **536** | **536** |

Toate clasele apar în fiecare split. Auditul verifică imaginile și absența grupurilor cunoscute comune între split-uri. Cele 300 de imagini oficiale nu participă la această împărțire. Testul intern folosește imagini deja explorate în trecut și **nu este numit neatins**; nu înlocuiește testul oficial.

Configurații fixate: predictor majoritar din train, HOG-SVM C=10/RBF/gamma=scale cu scaler ajustat numai pe train, respectiv ResNet18 cu backbone înghețat și cap liniar antrenat 10 epoci. Alegerea epocii CNN folosește exclusiv macro-F1 pe validare; la egalitate se păstrează prima epocă. Evaluarea de test folosește artefactele salvate, fără fit.

## Proveniența weights CNN și bugetul de calcul

Weights **ResNet18 IMAGENET1K_V1** au fost recuperați explicit din URL-ul publicat în [sursa Torchvision v0.28.0](https://github.com/pytorch/vision/blob/v0.28.0/torchvision/models/resnet.py), cu transformările descrise în [documentația modelului](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). URL-ul este `https://download.pytorch.org/models/resnet18-f37072fd.pth`.

Fișier local: `outputs/pretrained/resnet18-f37072fd.pth`, 46,830,571 bytes. SHA-256 complet calculat:

`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.

Prefixul corespunde celui publicat în numele oficial al fișierului; acesta nu este o semnătură sau un checksum complet publicat independent. Încărcarea strictă în arhitectura ResNet18 originală cu 1000 clase a trecut. Metadatele, versiunea bibliotecilor și transformările sunt salvate în fișierul `.provenance.json` alăturat. Aceasta documentează pretraining-ul ImageNet declarat de Torchvision și nu stabilește proveniența sau licența imaginilor ASPICV.

Pilotul **anterior rulării CNN**, numai pe 16 imagini train, a măsurat 1.404 s pentru extracție cu încărcare și a extrapolat 234.84 s pentru toate cele 2677 de imagini, plus circa 3.89 s pentru capul liniar la 10 epoci. Este o estimare conservatoare, cu overhead de încărcare multiplicat și unele costuri excluse; nu este măsurătoarea finală. Buget orientativ comunicat: aproximativ 6 minute, fără grid search, un singur seed, CPU i5-13600KF, maximum 4 thread-uri Torch, fără CUDA. Raportul pilotului este `outputs/internal-protocol-v1/cnn-cost-estimate.json`.

## Reproducere

Folosește directoare noi pentru fiecare versiune. Prima comandă reverifică sursa și reconstruiește propunerea înainte de orice scriere derivată:

```bash
backend/.venv/bin/python scripts/build_internal_inventory.py --recovery outputs/recovery-v1 --evidence outputs/duplicate-review-v2 --review docs/publicare/duplicate_review.csv --record docs/publicare/duplicate_review.provenance.json --policy outputs/duplicate-policy-v1 --source /home/colac/Projects/Academic/03_WSL_SCHOOL_COPY/School_WSL_HOME/aspicv-project --output outputs/internal-inventory-v1
backend/.venv/bin/python scripts/split_protocol.py --source outputs/internal-inventory-v1/eligible_inventory.csv --root /home/colac/Projects/Academic/03_WSL_SCHOOL_COPY/School_WSL_HOME/aspicv-project --output outputs/internal-protocol-v1/splits.csv --seed 42
backend/.venv/bin/python scripts/evaluation.py prepare --source outputs/internal-protocol-v1/splits.csv --root /home/colac/Projects/Academic/03_WSL_SCHOOL_COPY/School_WSL_HOME/aspicv-project --output outputs/internal-protocol-v1/manifest.csv
```

`build_internal_inventory.py` refuză acțiuni necunoscute, ID-uri duplicate, exemple eligibile marcate cu risc și grupuri/hash-uri comune cu datele excluse sau oficiale. Cazurile incerte sunt reținute în afară; această operație nu rezolvă incertitudinea sau conflictele.

## Evaluarea executată

Comparația a fost finalizată fără schimbarea protocolului fixat. [Interpretarea](rezultate_interne_v1/interpretare.md) și [raportul numeric](rezultate_interne_v1/report.md) prezintă rezultatele pe același test intern: HOG 38.62% accuracy / 0.3579 macro-F1, CNN 40.49% / 0.3171, baseline 23.88% / 0.0551. Fit-ul CNN efectiv a durat 25.08 s CPU, sub estimarea conservatoare. Nu s-au repetat antrenările după test.
