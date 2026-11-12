# Fișa datelor — actualizată la 2026-11-09

## Date recuperate și integritate

La auditul inițial, checkout-ul repository-ului nu conținea date sau checkpoint-uri. Ulterior, utilizatorul a indicat copia academică:

`/home/colac/Projects/Academic/03_WSL_SCHOOL_COPY/School_WSL_HOME/aspicv-project`

Aceasta conține **3000 imagini JPEG**, ID-uri 1–3000, toate grayscale (`L`), 48×48. Există 2700 etichete în `train.csv`, distribuite istoric în 2000 train și 700 val; ultimele 300 sunt fără etichete în `dataset/predict`. Reuniunea celor două CSV-uri istorice coincide exact cu etichetele originale. Toate cele 3000 de imagini corespund byte cu byte membrilor din `dataset.zip`.

SHA-256 al arhivei: `0fc6e5271a0064fd54e30e94f1ffc288ab0ff3ef6b32de24cc2a5a2553697456`.

Imaginile efective sunt în `dataset/train`, `dataset/val`, `dataset/predict`. Directorul `dataset/dataset` conține numai fișiere auxiliare Windows `Zone.Identifier`, fără JPEG-uri; numărul său de fișiere nu este un număr de imagini.

Auditul reproductibil este în `scripts/audit_recovery.py`. Rezultatele locale sunt în `outputs/recovery-v1/`: `historical_manifest.csv` (3000 rânduri, ID, cale, etichetă, SHA-256 fișier/pixeli, split istoric, subject_id gol), `audit.json`, `similarity_candidates.csv`. Hash-ul manifestului: `b17ee848b9b3d145061554785d9f3e94fcc640e8a024418cbfe32535c097489a`. Datele originale nu au fost mutate, copiate în Git sau modificate.

| Clasă | Train istoric | Val istoric | Total etichetat |
|---|---:|---:|---:|
| angry | 268 | 87 | 355 |
| disgust | 27 | 12 | 39 |
| fear | 293 | 120 | 413 |
| happy | 467 | 173 | 640 |
| neutral | 379 | 110 | 489 |
| sad | 332 | 119 | 451 |
| surprise | 234 | 79 | 313 |

## Proveniență demonstrată și necunoscute

Cerința recuperată, `Cerinta proiect ASPICV IISC.docx.pdf`, precizează că participanții primesc `dataset.zip`, imaginile 1–3000 și etichetele primelor 2700; ultimele 300 sunt testul competiției. Descrierea corespunde fișierelor recuperate. Aceasta documentează **furnizarea în cadrul proiectului ASPICV**, nu datasetul public originar sau drepturile lui.

Investigația online ulterioară a identificat familia **FER2013** și o distribuție publică de referință: [Kaggle msambare/fer2013, versiunea 1](https://www.kaggle.com/datasets/msambare/fer2013). Toate cele **3000 de imagini ASPICV au potriviri byte cu byte** în acea arhivă, verificate prin SHA-256. Aceasta confirmă corespondența conținutului; nu dovedește exact de la ce uploader a descărcat autorul pachetului academic.

Metadatele distribuției declară codul Kaggle **DbCL-1.0**, afișat ca ODbL pentru baza de date și Database Contents License pentru conținut. [Raportul sursei și licenței](sursa_fer2013/raport.md) păstrează sursele primare, versiunea, hash-urile și limitele atribuirii. Declarația uploaderului nu verifică independent toate drepturile fotografiilor originale. Fișierele locale `Zone.Identifier` nu documentează lanțul de descărcare: conțin numai `ZoneId=3`.

Cerința spune că `test.csv` urma să fie publicat după deadline. Acest fișier **nu există în directorul recuperat**; etichetele testului oficial rămân de recuperat. `submission.csv` conține predicții, nu adevăr de referință. Nu se completează etichetele lipsă prin deducție din duplicate.

## Duplicate și persoane

Există **20 grupuri de duplicate exacte** (42 fișiere, 22 copii suplimentare); grupurile sunt identice după hash-ul de fișier și cel al pixelilor.

- **7 grupuri train–val:** `464/2414`, `700/1638`, `912/2681`, `1058/1532`, `1175/1280/2668`, `1798/2575`, `2235/2352`.
- **3 grupuri etichetat–predict:** `351/1260/2840`, `447/2972`, `2675/2927`.
- **Conflict de etichete:** `351=surprise`, `1260=fear`; imaginea este identică, iar `2840` este o copie fără etichetă. Nu alegem automat una dintre etichete.

Celelalte grupuri sunt în interiorul unui split. Lista completă este în audit. Split-ul istoric este contaminat și nu poate susține o evaluare independentă prin simpla redenumire a validării în test.

Screeningul 64-bit dHash, prag Hamming ≤4, identifică suplimentar **126 perechi candidate**, dintre care 75 traversează split-urile istorice. Perechile exacte au fost excluse din acest număr. Toate cele 126 au fost revizuite ulterior: 37 perechi susțin reutilizarea aceleiași fotografii, 87 reprezintă imagini diferite, iar 2 rămân incerte. Este o singură revizuire vizuală făcută de asistent, fără deducerea identității persoanelor. Vezi [deciziile și propunerea de tratare](duplicate_review.md). Metoda nu detectează toate crop-urile, recomprimările sau cadrele similare.

CSV-urile recuperate au numai ID și etichetă; nu există metadate de persoană/sesiune. Separarea pe persoane nu este demonstrată. Nu deducem identitatea din recunoaștere facială.

## Checkpoint-uri și rapoarte recuperate

- `models/hog_svm.joblib` și `.meta.json`, SHA-256 checkpoint `cec35012252d83b3e932615a8876719b510bbca0e9ca0215e433e9c6b8a84dbe`.
- `models/lbp_svm.joblib` și `.meta.json`, SHA-256 checkpoint `3f0ebbd093f50cb333dede60f619b4a4032d86c759d8ba86532f209378590da2`.
- Șase rapoarte de comparație și două rapoarte de validare JSON, cu hash-uri inventariate.
- Niciun checkpoint CNN istoric `.pt`/`.pth` în copia academică indicată. Ulterior au fost recuperați explicit weights oficiali ResNet18 IMAGENET1K_V1 și s-a antrenat un cap nou în protocolul intern; [proveniență, hash și protocol](inventar_derivat.md). Nu se pretinde reproducerea scorurilor CNN istorice.

Checkpoint-ul HOG a fost reevaluat pe validarea istorică, fără antrenare: accuracy 0.432857 și macro-F1 0.390162. Accuracy și matricea completă coincid cu raportul `validation_20260116_032850.json`. Predicțiile și recall/clasă sunt acum exportate în `outputs/recovered-validation-v2/hog/`. Baseline-ul majoritar ajustat din train este în directorul alăturat `baseline/`. Acestea sunt rezultate exploratorii pe un split contaminat, nu un nou test independent.

Încărcarea checkpoint-ului produce avertisment scikit-learn 1.8.0→1.9.0. Coincidența raportului este o verificare utilă, dar nu certifică compatibilitatea generală. Checkpoint-ul nu conține un manifest al ID-urilor originale de antrenare. Avertismentele și versiunile sunt salvate în export.

## Ce mai trebuie recuperat sau decis

1. Familia FER2013 și licența declarată a distribuției Kaggle sunt acum documentate prin surse și potriviri de fișiere. Rămâne neconfirmat lanțul exact de furnizare academică și dreptul uploaderului asupra fiecărei fotografii originale.
2. `test.csv` oficial și istoricul consultării testului; până atunci testul oficial nu este evaluabil.
3. Metadatele de persoană/sesiune, dacă există, și a doua verificare pentru perechile incerte 847/1286 și 1291/2957; screeningul nu este exhaustiv.
4. Clarificarea celor patru grupuri cu etichete neuniforme (unul exact, trei aproape identice) și evaluarea propunerii de carantinare, fără modificarea originalelor.
5. Pentru noul experiment, weights ResNet18 oficiali, pilotul de cost și antrenarea au fost finalizate; scriptul de recuperare este explicit și evaluatorul nu descarcă nimic implicit. [Rezultatele interne](rezultate_interne_v1/interpretare.md) sunt retrospective, distincte de testul oficial.
6. Protocolul intern comun este acum fixat și executat pe 2677 de imagini eligibile, cu 23 excluse prin evidențe CSV. Rămâne retrospectiv/exploratoriu. Split-ul oficial are prioritate pentru raportarea benchmarkului, dar etichetele sale lipsesc.

Checkpoint-urile CNN istorice sunt necesare numai dacă dorim reproducerea scorurilor lor; un experiment nou poate porni de la weights ImageNet documentate.
