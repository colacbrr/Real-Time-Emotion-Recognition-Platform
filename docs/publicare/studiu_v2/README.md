# Extensie științifică v2 — 12 noiembrie 2026

Încadrare: **studiu educațional de audit și reproducibilitate**, cu experimente retrospective. Contribuția proprie este reconstrucția provenienței și evaluarea controlată, nu inventarea HOG, SVM, ResNet sau a unei metode statistice. [Manuscrisul](../articol/manuscript.tex) este separat de raportul academic istoric.

## Ce aduce față de v1

Potrivirea exactă în FER2013 arată amestecul partițiilor sursei: dintre toate cele 3000 imagini, 2324 apar numai în `train`, 562 numai în `test`, iar 114 în ambele. Acestea sunt apartenențe ale conținutului, nu presupuneri despre împărțirea academică. [Tabelul complet](results/source_membership.csv) păstrează și split-ul academic.

V2 pornește din cele 2677 eligibile după politica anterioară și respectă apartenența sursei:

- 2069 exclusiv source-train pentru dezvoltare;
- 96 cu exemplare exacte în ambele partiții excluse;
- 512 exclusiv source-test, dintre care trei excluse deoarece grupul lor de imagini apare în dezvoltare;
- 509 pentru verificarea separată, retrospectivă. Nu reprezintă testul FER2013 integral, un domeniu nou sau un test istoric neatins.

Cele 300 imagini ASPICV predict rămân fără etichete academice. Nu sunt folosite în metricile noi.

Comparația 2×2 folosește HOG/ResNet18 înghețat × SVM liniar/RBF, plus baseline majoritar. Aceleași cinci folduri externe și trei folduri interne pentru toate configurațiile; grupurile cunoscute rămân împreună. Se selectează C din 0.1/1/10 exclusiv prin macro-F1 intern. StandardScaler se ajustează numai pe train-ul fiecărui fold. Backbone-ul este singurul CNN și nu este fine-tuned. Acest control separă mai bine reprezentarea de familia clasificatorului, fără a egaliza dimensionalitatea sau avantajul pretraining-ului.

`protocol.json`, `cohort.csv` și `folds.json` au fost fixate înainte de rezultatele v2; este un plan local datat, nu preregistrare publică. Designul cunoștea rezultatele v1. Fișierele și codul v1 au fost păstrate identice.

## Rezultate noi

| Pipeline | OOF accuracy / macro-F1, n=2069 | Source-test accuracy / macro-F1, n=509 |
|---|---:|---:|
| Majoritar | 24.55% / 0.0563 | 24.75% / 0.0567 |
| HOG + SVM liniar | 28.66% / 0.2520 | 30.84% / 0.2758 |
| HOG + SVM RBF | 39.29% / 0.3096 | 39.49% / 0.3157 |
| ResNet18 + SVM liniar | 37.60% / 0.3263 | 33.99% / 0.3011 |
| ResNet18 + SVM RBF | 44.03% / 0.3692 | 39.49% / 0.3528 |

Pe source-test, HOG-RBF și ResNet-RBF au fiecare 201 clasificări corecte, dar alte confuzii. Diferența macro-F1 ResNet−HOG este +0.0371, cu interval bootstrap pereche 95% [−0.0235, +0.0972]. Nu susține o concluzie fermă de superioritate. Pentru OOF, diferența este +0.0596 [0.0317, 0.0876], însă intervalele nu includ dependența creată de train-urile suprapuse.

Incertitudinea: 2000 replici bootstrap, grupuri de imagini resamplate în fiecare clasă, aceleași grupuri pentru toate modelele; intervale percentile 95% și 98.75% pentru patru contraste, per metrică și per evaluare. Sunt **condiționate de predicțiile salvate**, fără refit, nu intervale complete pentru algoritm. Nu includ variația antrenării, alegerii foldurilor, selecției parametrilor, review-ului sau persoanelor necunoscute. Nu se aplică un t-test tratând foldurile ca independente. Baseline-ul poate avea interval degenerat deoarece distribuția claselor este condiționată.

Rezultatele integrale se află în [results.json](results/results.json), [scoruri și intervale](results/scores.csv), [diferențe perechi](results/paired_differences.csv), [recall și suport](results/recall_per_class.csv), [predicții OOF](results/oof_predictions.csv), [predicții source-test](results/holdout_predictions.csv) și [traseul selecției C](results/tuning.json). Matricile CSV au rânduri adevărate și coloane prezise, în ordinea claselor din JSON.

## Cost estimat înainte și cost măsurat

[Pilotul](cost_estimate.json) a folosit 140 imagini de dezvoltare, numai pentru cost: circa 65 s extrapolare CNN; buget conservator 5–15 minute CPU pentru extragere și 240 fit-uri SVM. Rularea efectivă a durat 114.80 s, incluzând bootstrap. Extragerea pe 2578 imagini active: HOG 0.95 s, ResNet18 27.18 s. Hardware i5-13600KF, Torch maximum patru fire, fără CUDA. Sunt măsurători unice pe loturi, cu cache pentru trăsături fixe; nu certifică real-time sau latența unei aplicații.

Intervențiile au țintit blocuri de aproximativ 5–10 minute active, fără a pretinde cronometrare: (1) audit al apartenenței la sursă; (2) protocol și folduri comune; (3) teste și pilot de cost; (4) rulare și exporturi; (5) verificări fără fit și analiză; (6) redactare, figuri și verificare vizuală PDF. Costul calculului este raportat separat de redactare.

## Verificare fără antrenare și fără rețea

### Pachet numeric, fără dataset sau checkpoint-uri

```bash
python -m pip install -r requirements/verify.txt
python scripts/verify_published_results.py
```

Această cale verifică numai pachetul de dovezi prezent în repository: integritate, codul științific fixat, etichete/grupuri din cohortă, separarea hash-urilor cunoscute, acoperirea tuturor subseturilor interne/externe și alinierea predicțiilor. Recalculează metricile și bootstrap-ul din exporturi. Căile istorice absolute ale datelor nu sunt necesare. Nu verifică fișierele originale, checkpoint-urile, etichetele ca adevăr independent sau corectitudinea modelului prin reexecutare. Toleranța comparațiilor numerice este 1e-12 pentru diferențe de rotunjire între medii.

### Verificarea completă a pachetului local recuperat

Din rădăcina repository-ului, cu datele recuperate în locația cunoscută:

```bash
backend/.venv/bin/python -m unittest discover -s tests
backend/.venv/bin/python scripts/scientific_study.py verify \
  --plan docs/publicare/studiu_v2 \
  --root /home/colac/Projects/Academic/03_WSL_SCHOOL_COPY/School_WSL_HOME/aspicv-project \
  --output outputs/scientific-study-v2
backend/.venv/bin/python scripts/verify_scientific_artifacts.py \
  --plan docs/publicare/studiu_v2 --run outputs/scientific-study-v2 \
  --root /home/colac/Projects/Academic/03_WSL_SCHOOL_COPY/School_WSL_HOME/aspicv-project \
  --public docs/publicare/studiu_v2/results \
  --output docs/publicare/studiu_v2/verification.json
```

Primul verificator recalculează metricile și bootstrap-ul din predicții, după hash-uri. Al doilea verifică separarea grupurilor, acoperirea OOF, mărimea train-ului scalerelor finale, reproduce predicțiile holdout din modelele/trăsăturile salvate și verifică exporturile publicabile. Niciunul nu antrenează. Integritatea nu dovedește validitatea etichetelor sau independența persoanelor.

Artefactele mari din `outputs/` sunt ignorate de Git și trebuie păstrate în backup local: inventarul eligibil, corespondențele FER, weights, trăsături `.npy`, modele `.joblib`, rezultate și integritate. `protocol.json` înregistrează căi și hash-uri ale intrărilor de la rularea originală; verificarea acelui pachet cere acces la ele. O clonă fără date poate verifica logic prin testele sintetice și examina toate predicțiile numerice, dar nu poate verifica hash-urile imaginilor sau reantrena.

## Reproducere completă într-o versiune nouă

Păstrează protocolul original. Pe altă cale/clonă se generează un nou protocol cu aceleași reguli, cod și intrări identice, dar cu identificatori de cale locali. Nu înlocui `studiu_v2` sau rezultatele sale.

```bash
backend/.venv/bin/python scripts/scientific_study.py plan \
  --inventory outputs/internal-inventory-v1/eligible_inventory.csv \
  --matches outputs/fer2013-content-match-v1/content_matches.csv \
  --weights outputs/pretrained/resnet18-f37072fd.pth --output outputs/reproduction-v2-plan
backend/.venv/bin/python scripts/scientific_study.py pilot \
  --plan outputs/reproduction-v2-plan --root /cale/aspicv-project \
  --weights outputs/pretrained/resnet18-f37072fd.pth --output outputs/reproduction-v2-cost.json
backend/.venv/bin/python scripts/scientific_study.py run \
  --plan outputs/reproduction-v2-plan --root /cale/aspicv-project \
  --weights outputs/pretrained/resnet18-f37072fd.pth \
  --pilot outputs/reproduction-v2-cost.json --output outputs/reproduction-v2-run
```

Dependențele de antrenare sunt aceleași ca v1. Versiunile numerice sunt în `results.json` și protocolul v1; mediul instalat a fost reutilizat. Pentru exportul figurilor: Python cu NumPy/Matplotlib; aici acestea sunt disponibile în Python-ul sistemului. Nu s-a instalat nimic. Pentru PDF: două treceri pdflatex, vezi [instrucțiunile manuscrisului](../articol/README.md).

## Ce mai trebuie înainte de trimiterea unui articol

Verificare umană a manuscrisului, confirmarea autorului/afilierii, al doilea reviewer pentru duplicate și clarificarea materialelor redistribuibile. Declarația uploader-ului nu certifică independent drepturile fotografiilor. Lipsesc metadatele persoanelor, procedura de eșantionare academică și etichetele testului academic oficial. Un test pe alt domeniu și evaluarea pedagogică nu au fost făcute. Aceste lipsuri delimitează articolul; nu sunt completate prin afirmații mai puternice. Nu s-a publicat nimic.
