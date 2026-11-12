# Revizuirea duplicatelor și propunerea de tratare

Verificare finală: 2026-11-09.

## Rezultatul acestei intervenții

Au fost inspectate toate cele **126 de perechi** candidate din auditul recuperării, pe șapte planșe locale. Revizuirea este realizată de asistent, cu sprijin numeric, fără un al doilea evaluator independent.

| Decizie asupra imaginii | Perechi | Interpretare |
|---|---:|---|
| `near_duplicate` | 37 | Compoziție fotografică potrivită cu reutilizarea aceleiași imagini, cu mici diferențe de poziționare, scară, contrast, compresie sau watermark; comparația pixelilor sprijină observația |
| `different_image` | 87 | Conținut fotografic, poziție sau compoziție diferită; asemănarea dHash nu confirmă duplicarea |
| `uncertain` | 2 | Dovezi insuficiente pentru confirmare; necesită a doua revizuire |

Deciziile individuale și motivele sunt în [duplicate_review.csv](duplicate_review.csv). Identitățile persoanelor **nu au fost comparate sau deduse**. `different_image` nu înseamnă persoane distincte, iar grupurile de mai jos reprezintă imagini, nu subiecți independenți.

Cele două perechi incerte sunt **847/1286** și **1291/2957**. Prima traversează train–val; a doua leagă train de predict. Ambele sunt tratate conservator în propunere, fără a le declara duplicate confirmate.

## Dovezi și limite ale metodei

Planșele folosesc pixelii originali grayscale, măriți 2× prin nearest-neighbor. Etichetele claselor nu sunt afișate pe planșe. Aceasta nu reprezintă un experiment orb validat sau o adjudecare umană independentă.

Pentru fiecare pereche se păstrează dHash, SSIM și RMSE fără aliniere, plus corelația normalizată (NCC) după o aliniere limitată. Se compară centrul de 40×40 pixeli, explorând 13 scări între 0.85 și 1.15 și translații întregi de −3…3 pixeli, cu interpolare liniară: 637 transformări. Sunt tolerate diferențe de luminozitate prin centrarea intensităților. Metricile susțin revizuirea, **nu atribuie automat decizii** și nu măsoară identitatea. Alegerea celei mai bune transformări poate produce asemănări și pentru imagini diferite.

Deciziile sunt specifice eșantionului examinat. Screeningul inițial verifică numai dHash cu distanță ≤4; poate rata alte copii, crop-uri mari, cadre vecine sau imagini întoarse în oglindă. Nu se pretinde o rată de detecție validată. Nu s-au inferat etichetele celor 300 de imagini fără adevăr de referință.

Fișierele locale din `outputs/duplicate-review-v2/` conțin planșele, dovezile numerice și proveniența. Planșele sunt identice byte cu byte cu cele din prima randare, care au fost inspectate. [Proveniența revizuirii](duplicate_review.provenance.json) păstrează hash-urile intrărilor, adnotărilor și planșelor. Imaginile nu sunt incluse în documentația versionabilă și nu au fost publicate.

## Grupuri și dezacorduri de etichete

Componentele conexe formate din duplicatele exacte, cele 37 de perechi susținute vizual și cele 2 perechi incerte produc **39 de grupuri cu cel puțin două imagini, însumând 91 de ID-uri**. Numărul include **2 grupuri incerte**; nu înseamnă 39 de grupuri confirmate.

În această propunere conservatoare, 14 grupuri traversează train–val și 10 leagă date etichetate de predict. Dintre acestea, câte unul conține o pereche incertă. Grupurile exacte rămân cele 20 documentate anterior, dintre care 7 train–val și 3 etichetat–predict.

Patru componente au etichete neuniforme:

| ID-uri | Etichete cunoscute | Natura dovezii |
|---|---|---|
| 351, 1260, 2840 | 351=surprise; 1260=fear; 2840 fără etichetă | Duplicate exacte; conflict de etichete demonstrat |
| 607, 643, 708, 2542, 2978 | 607/2542=surprise; 643/708=fear; 2978 fără etichetă | Imagini aproape identice susținute vizual și numeric; dezacord de adnotare de clarificat |
| 1034, 1202, 1743, 2499 | 1034/1202/2499=surprise; 1743=fear | Imagini aproape identice susținute vizual și numeric; dezacord de adnotare de clarificat |
| 1592, 2036, 2048 | 1592/2048=sad; 2036=fear | Imagini aproape identice susținute vizual și numeric; dezacord de adnotare de clarificat |

Nu se alege eticheta majoritară și nu se corectează etichetele după predicția modelului sau reinterpretarea expresiei. Pentru imaginile aproape identice, dezacordul nu dovedește singur care adnotare este greșită.

## Propunerea inițială și aplicarea ulterioară în CSV-uri derivate

`outputs/duplicate-policy-v1/group_proposal.csv` păstrează toate cele 3000 de rânduri, etichetele și split-urile originale, adăugând `duplicate_group`, indicatori de risc și `proposed_action`. `summary.json` descrie componentele și proveniența.

Prioritatea propusă pentru un eventual experiment intern este:

1. Păstrarea celor **300 predict fără etichete**, în split-ul oficial original. Nicio completare a etichetelor din duplicate.
2. Carantinarea celor **13 exemple etichetate** din componentele cu dezacord, până la clarificarea sursei.
3. Carantinarea a **3 exemple etichetate** din componentele incerte, până la a doua revizuire.
4. Excluderea din dezvoltarea internă a încă **7 exemple etichetate** legate prin duplicare de predict, după regulile anterioare.
5. Păstrarea împreună a celor **58 de exemple etichetate** rămase în grupuri de imagini, dacă se aprobă metodologic o nouă împărțire internă.
6. Celelalte **2619 exemple etichetate** nu au restricții din acest screening, dar rămân supuse verificării sursei, identităților și protocolului.

Regulile sunt exclusive în această ordine. Astfel, 23 de exemple etichetate sunt propuse pentru reținere în afara dezvoltării interne; cele 2677 rămase **nu sunt certificate drept dataset curat sau independent pe persoane**. În etapa de propunere nu s-a efectuat filtrare, reetichetare sau antrenare. Ulterior regulile au fost aplicate în [CSV-uri derivate](inventar_derivat.md), apoi s-a fixat un split intern retrospectiv și s-a executat comparația. Fișierele și etichetele originale au rămas nemodificate. Raportarea pe testul oficial va păstra split-ul său și va declara suprapunerile, fără a-l înlocui tacit cu un test selectat ulterior.

Manifestul evaluatorului și generatorul de split acceptă acum opțional `duplicate_group`, separat de `subject_id`. Evaluatorul blochează traversarea split-urilor de către un astfel de grup și etichetele contradictorii în interiorul lui. Generatorul păstrează împreună componentele conexe ale grupurilor revizuite, hash-urilor și identităților furnizate. Propunerea completă, cu predict și conflicte, **nu este direct un manifest evaluabil**.

## Reproducerea infrastructurii

Comenzile cer directoare de ieșire noi; exemplul de mai jos arată numele folosite în această intervenție:

```bash
backend/.venv/bin/python scripts/review_candidates.py --recovery outputs/recovery-v1 --source /home/colac/Projects/Academic/03_WSL_SCHOOL_COPY/School_WSL_HOME/aspicv-project --output outputs/duplicate-review-v2
backend/.venv/bin/python scripts/duplicate_policy.py --recovery outputs/recovery-v1 --evidence outputs/duplicate-review-v2 --review docs/publicare/duplicate_review.csv --output outputs/duplicate-policy-v1
backend/.venv/bin/python -m unittest discover -s tests -v
```

Randarea și metricile sunt reproductibile; deciziile vizuale sunt adnotări păstrate în CSV, nu rezultate regenerate automat. Propunerea respinge revizuiri incomplete, perechi sau dovezi numerice schimbate și lipsa motivelor/recenzentului. **22 teste trec în verificarea curentă**, inclusiv alinierea unei translații sintetice, imaginile constante, politica de carantinare, păstrarea etichetelor necunoscute și blocarea scurgerii grupurilor revizuite. Nu există scoruri noi de clasificare din această intervenție.

## Verificare integrală a artefactelor

`verify_duplicate_review.py` verifică hash-urile înregistrate pentru adnotări, intrări, codul de revizuire/grupare și cele șapte planșe, apoi recalculează hash-urile celor 3000 de imagini și 21 de artefacte recuperate. Reconstruiește propunerea într-un director temporar și cere un CSV și JSON identice byte cu byte. Verifică și păstrarea tuturor câmpurilor originale în fiecare rând al propunerii, inclusiv etichetele goale din predict.

```bash
backend/.venv/bin/python scripts/verify_duplicate_review.py --recovery outputs/recovery-v1 --evidence outputs/duplicate-review-v2 --review docs/publicare/duplicate_review.csv --record docs/publicare/duplicate_review.provenance.json --policy outputs/duplicate-policy-v1 --source /home/colac/Projects/Academic/03_WSL_SCHOOL_COPY/School_WSL_HOME/aspicv-project --output outputs/duplicate-verification-v1.json
```

Verificarea pe datele recuperate a trecut; raportul este în `outputs/duplicate-verification-v1.json`. Integritatea și reproductibilitatea artefactelor nu constituie o a doua revizuire vizuală și nu certifică independența persoanelor. Testul sintetic verifică și respingerea unei revizuiri modificate după înregistrarea hash-ului.
