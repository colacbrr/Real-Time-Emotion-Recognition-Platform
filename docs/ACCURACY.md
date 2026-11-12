# Protocol comun de evaluare

**Actualizare 12 noiembrie 2026:** extensia [științifică v2](publicare/studiu_v2/README.md) adaugă respectarea apartenenței FER train/test, comparația reprezentare × clasificator, selecția grupată imbricată și incertitudinea pereche. Protocolul v1 de mai jos și artefactele sale sunt păstrate ca etapă retrospectivă distinctă.

Datele și checkpoint-urile HOG/LBP au fost găsite în copia academică indicată de utilizator; vezi [fișa datelor](publicare/dataset_card.md). Auditul constată duplicate între split-uri. Reevaluarea HOG pe cele 700 de imagini istorice reproduce accuracy 0.432857 și matricea de confuzie originală; exportă acum macro-F1 0.390162 și predicții individuale. Acest rezultat rămâne exploratoriu. Comparația nouă pe test intern comun este executată: [rezultate și limite](publicare/rezultate_interne_v1/interpretare.md). Testul este retrospectiv și nu înlocuiește cele 300 de imagini oficiale. Valorile istorice 0.32/0.38 pentru LBP/HOG pe 2000/700 și 0.57–0.59 pentru CNN/ensemble pe 2600/100 sunt exploratorii, nereproduse și nu demonstrează superioritatea unui model.

## Pași în intervenții de 5–10 minute active

1. Recuperează sursa, licența, imaginile și etichetele; completează fișa datelor. Recuperarea poate dura separat.
2. Construiește inventarul CSV și verifică hash-uri/duplicate/identități. Analiza dublurilor similare poate depăși o intervenție.
3. Fixează train/val/test: split oficial prioritar; altfel împărțire stratificată justificată, cu grupare pe persoane și duplicate înaintea stratificării. Păstrează seed-ul, algoritmul și listele ID-urilor. Nu există un split automat implicit în evaluator. Generatorul separat `split_protocol.py` este destinat numai unui protocol intern neoficial, justificat înainte de folosire; nu înlocuiește testul oficial. Fiecare clasă trebuie să apară în fiecare split; în caz contrar scriptul oprește protocolul, nu inventează exemple.
4. Rulează baseline și HOG pe validare; configurația inițială HOG este fixă. Ajustările folosesc exclusiv val.
5. Estimează costul CNN, apoi rulează ResNet18 cu backbone înghețat și strat final liniar. Checkpoint-ul stratului final este ales prin macro-F1 pe val, primul la egalitate. Nu există MixUp/CutMix, ensemble sau alte backbone-uri în comparația nouă.
6. După înghețarea alegerilor evaluează artefactele salvate cu `evaluate-saved`, exportă și verifică tabelul comun. Această comandă nu face fit și verifică hash-urile manifestului, codului și artefactului; pentru CNN cere exact aceiași weights de backbone. Comanda veche `run --split test` încă reantrenează și nu este fluxul recomandat pentru test. Nu se antrenează pe train+val. O rulare de test nu împiedică tehnic reutilizarea testului: istoricul experimentelor trebuie păstrat.

## Comenzi

Folosește mediul existent `backend/.venv/bin/python` în loc de `python`, dacă este disponibil. Dependințe minime: numpy, Pillow, scikit-learn, scikit-image, joblib; pentru CNN și torch/torchvision. Mediul inspectat: Python 3.14.9, numpy 2.5.1, sklearn 1.9.0, skimage 0.26.0, torch 2.13.0+cu130, torchvision 0.28.0+cu130; CUDA indisponibil. Nu a fost instalat nimic.

Pregătește `inventory.csv` cu `id,path,label,split,subject_id` (ultima coloană poate fi goală). Căile imaginilor sunt relative la `dataset`; split-urile sunt `train`, `val`, `test`. Setul fără etichete rămâne separat și nu este numit test evaluabil.

```bash
python scripts/evaluation.py prepare --source inventory.csv --root dataset --output outputs/protocol-v1.csv
python scripts/evaluation.py run --manifest outputs/protocol-v1.csv --root dataset --model baseline --output outputs/baseline-val
python scripts/evaluation.py run --manifest outputs/protocol-v1.csv --root dataset --model hog --output outputs/hog-val
python scripts/evaluation.py estimate-cnn --manifest outputs/protocol-v1.csv --root dataset --weights /cale/resnet18-imagenet-v1.pth --sample 16 --epochs 10
python scripts/evaluation.py run --manifest outputs/protocol-v1.csv --root dataset --model cnn --weights /cale/resnet18-imagenet-v1.pth --epochs 10 --seed 42 --output outputs/cnn-val
```

După selecție, evaluează artefactele existente, apoi compară:

```bash
python scripts/evaluation.py evaluate-saved --manifest outputs/protocol-v1.csv --root dataset --run outputs/baseline-val --output outputs/baseline-test
python scripts/evaluation.py evaluate-saved --manifest outputs/protocol-v1.csv --root dataset --run outputs/hog-val --output outputs/hog-test
python scripts/evaluation.py evaluate-saved --manifest outputs/protocol-v1.csv --root dataset --run outputs/cnn-val --weights /cale/resnet18-imagenet-v1.pth --output outputs/cnn-test
python scripts/evaluation.py compare --manifest outputs/protocol-v1.csv --root dataset --runs outputs/baseline-test outputs/hog-test outputs/cnn-test --output outputs/comparison.json
python -m unittest discover -s tests -v
```

Manifestul nu suprascrie versiuni existente; conținutul imaginilor este reverificat la fiecare rulare. Comparația cere aceleași ID-uri și adevăruri, același hash de manifest și split test, recalculează metricile din predicții și cere toate cele trei modele. Nu combină scoruri istorice sau doar metadate de validare.

Fiecare rulare produce `predictions.csv` (ID, adevăr, predicție, split), `metrics.json`, modelul/stratul final. Metricile: accuracy, macro-F1 cu lista completă de clase, recall și suport pe clasă, matrice de confuzie (rând=adevăr). Diviziunile fără suport folosesc zero, iar manifestul cere suport în toate split-urile. Baseline-ul alege clasa majoritară exclusiv din train; egalitățile sunt rezolvate lexical. StandardScaler și SVM sunt ajustate exclusiv pe train. HOG folosește grayscale 48×48, valori /255; CNN folosește transformarea oficială `IMAGENET1K_V1` (RGB, resize/crop și normalizare ImageNet). Preprocesările diferă justificat între metode, exemplele și etichetele sunt comune.

Hash-urile codului, manifestului și weights, configurația, seed-ul, versiunile, duratele CPU și dimensiunile artefactelor sunt salvate. Pentru CNN `head.pt` necesită și weights externi identificați prin hash pentru reconstrucție; nu este checkpoint complet de backbone. Costul de inferență CNN include reconstruirea/încărcarea backbone-ului și preprocesarea, deci este timp de rulare batch, nu latență de serviciu sau dovadă real-time.

## Costul antrenării — separat de implementare

Weights oficiali ResNet18 au fost recuperați cu proveniență și checksum; pilotul anterior experimentului a estimat aproximativ 4 minute pentru extracție și câteva secunde pentru cap. Rularea efectivă a măsurat 25.08 s fit CNN pe CPU; detaliile și limitele comparației dintre estimare și măsurare sunt în [raportul rezultatelor](publicare/rezultate_interne_v1/interpretare.md). Nu s-au lansat experimente lungi. `estimate-cnn` verifică manifestul, folosește numai imagini train, cronometrează extracția pentru maximum 16 exemple și 10 pași ai stratului final, apoi extrapolează extracția tuturor split-urilor și antrenarea capului. Aceasta este estimare orientativă cu overhead de încărcare inclus; validarea repetată, auditul și serializarea sunt excluse. Bugetul se decide după acel pilot, cu marjă. Backbone-ul se extrage o dată per split, nu la fiecare epocă. HOG/SVM are configurație fixă, fără grid search; nu extrapola liniar costul unui SVM dintr-un eșantion mic.

Pentru fine-tuning complet, seed-uri multiple sau GPU este necesar un pilot și un buget separat înainte de lansare. Timpii testelor sintetice nu estimează performanța sau costul experimentului real.

## Audit și reevaluare a copiei recuperate

Fișierele originale rămân nemodificate. Exemplu de reproducere (alege directoare de ieșire noi dacă acestea există):

```bash
python scripts/audit_recovery.py --source /home/colac/Projects/Academic/03_WSL_SCHOOL_COPY/School_WSL_HOME/aspicv-project --output outputs/recovery-v1
python scripts/evaluate_recovered.py --source /home/colac/Projects/Academic/03_WSL_SCHOOL_COPY/School_WSL_HOME/aspicv-project --recovery outputs/recovery-v1 --output outputs/recovered-validation-v2
```

`evaluate_recovered.py` folosește preprocesarea originală din `validate_model.py`, nu noua variantă PIL; astfel nu schimbă preprocesarea checkpoint-ului recuperat. Nu reantrenează. Exportul include hash-uri, versiunile, avertismentele de compatibilitate și legătura verificată cu raportul istoric. Se încarcă doar checkpoint-ul propriu din sursa de recuperare indicată de utilizator.

Manifestul istoric conține `train/val/predict`, cu etichete goale la predict. Acesta este un inventar forensic, **nu** un manifest train/val/test acceptabil pentru noua comparație. Evaluatorul strict îl respinge; nu am mutat duplicatele sau inventat etichetele testului pentru a obține un scor.

Pentru un protocol intern nou, numai după verificarea split-ului oficial, a conflictelor și a perechilor similare, `split_protocol.py` poate primi un inventar etichetat fără coloana split:

```bash
python scripts/split_protocol.py --source inventory-unsplit.csv --root /cale/date --output outputs/internal-split.csv --seed 42
```

Gruparea folosește componente conexe ale hash-urilor de fișier/pixeli și ID-urilor de persoană furnizate; stratificarea folosește două `StratifiedGroupKFold`, cu prima partiție aleasă determinist, implicit aproximativ 60/20/20. Clasele absente opresc procesul. CSV-ul și raportul păstrează seed, algoritm, distribuții și hash-uri. Nu mută imagini, nu rezolvă conflictele de etichete și nu garantează detectarea persoanelor ori dublurilor similare. O împărțire nouă a celor 2700 de imagini deja explorate rămâne retrospectivă. Împărțirea retrospectivă a fost executată pe cele 2677 de exemple rămase după excluderi documentate; [inventarul derivat](publicare/inventar_derivat.md) păstrează motivele și distribuțiile.

## Grupurile de imagini revizuite

Cele 126 de perechi au fost inspectate și documentate în [raportul duplicatelor](publicare/duplicate_review.md). Schema manifestului acceptă opțional `duplicate_group`, separat de `subject_id`. `prepare` păstrează adnotarea, `load_verified`/audit blochează scurgerea sa între split-uri, iar `split_protocol.py` folosește grupurile în componentele conexe înainte de stratificare. Conflictele de etichetă rămân blocate, nu sunt corectate automat. Regulile au fost aplicate numai CSV-urilor derivate: 2677 eligibile, 23 excluse, 300 oficiale fără etichete. Imaginile și protocolul oficial nu au fost modificate.

## Rularea internă v1 efectuată

Comenzile generice de mai sus descriu interfața. Rularea efectivă folosește `outputs/internal-protocol-v1/manifest.csv`, rădăcina copiei academice din fișa datelor și `outputs/internal-runs-v1/{baseline,hog,cnn}-val` pentru ajustare/selecție. Testele sunt în directoarele omonime `*-test`; `comparison.json` este recalculat din predicții. [Protocolul fixat](publicare/internal_protocol_v1.json) specifică configurațiile înainte de antrenare, iar [interpretarea](publicare/rezultate_interne_v1/interpretare.md) separă aceste rezultate de scorurile istorice. Nicio căutare suplimentară sau schimbare de hiperparametri după test.

Pentru verificarea integrală a artefactelor înregistrate fără antrenare sau rețea, folosește [comanda unică de verificare](publicare/reproducere_studiu_v1.md). Aceasta verifică sursa, inventarele, manifestul fixat, checkpoint-urile și raportul recalculat din predicții.

## Proveniență clarificată ulterior experimentului

[Sursa FER2013 și licența declarată](publicare/sursa_fer2013/raport.md) au fost investigate separat: 3000/3000 potriviri byte cu byte cu distribuția Kaggle msambare/fer2013 v1, cu declarație DbCL-1.0 / ODbL. Această verificare nu schimbă etichetele, split-ul, modelul sau caracterul retrospectiv al rezultatelor. Referința publică nu a fost folosită pentru a crea un test.csv ASPICV sau pentru a calcula scorul testului oficial.
