# Clasificarea expresiilor faciale — studiu educațional

[Documentația finală în română (PDF)](output/pdf/Real-Time%20Emotion%20Recognition%20Platform.pdf) · [Sursa LaTeX și regenerarea figurilor](docs/publicare/prezentare_proiect/README.md)

Proiectul compară descriptori HOG/LBP cu SVM și transfer learning pentru clasificarea **expresiilor etichetate în imagini**. Nu demonstrează deducerea stării emoționale interioare. Prima comparație verificabilă vizează un baseline majoritar, HOG-SVM și un singur CNN ResNet18 cu backbone înghețat și strat final antrenat.

Contribuția proprie urmărită pentru articol este auditul provenienței și al separărilor, evidența deciziilor și evaluarea controlată reproductibilă. HOG, SVM și ResNet sunt metode standard; nu se propune un algoritm nou. Demonstrația FastAPI/React rămâne componentă secundară.

**Extensia științifică v2 (12 noiembrie 2026):** [protocol, rezultate și reproducere](docs/publicare/studiu_v2/README.md), [manuscris în engleză](docs/publicare/articol/README.md). Apartenența la partițiile FER este acum verificată: dezvoltare pe 2069 imagini exclusiv source-train, evaluare retrospectivă separată pe 509 exclusiv source-test după excluderi. HOG și ResNet18 înghețat sunt comparate cu aceleași capete SVM liniar/RBF și selecție grupată imbricată, plus baseline. Pe source-test, ambele RBF au 39.49% accuracy; macro-F1 este 0.3157 HOG și 0.3528 ResNet, dar intervalul pereche al diferenței include zero. OOF și source-test sunt raportate separat. Manuscrisul este un draft local pentru verificare umană, nu un articol trimis sau publicat.

**Stare:** datele au fost recuperate și auditate. După reținerea separată a 23 de exemple problematice, s-a executat o comparație internă retrospectivă pe același manifest 1605/536/536: baseline 23.88% accuracy / 0.0551 macro-F1, HOG-SVM 38.62% / 0.3579, ResNet18 înghețat + cap liniar 40.49% / 0.3171. CNN are accuracy mai mare, dar HOG macro-F1 mai bun; nu există un câștig uniform. Testul intern nu este istoric neatins, independența pe persoane nu este demonstrată, iar etichetele testului oficial rămân necunoscute. Investigația ulterioară confirmă potriviri byte cu byte pentru toate cele 3000 de imagini în distribuția FER2013 `msambare/fer2013` v1, care declară DbCL-1.0 / ODbL; drepturile originale ale fotografiilor nu sunt certificate independent. Numele repository-ului este istoric; funcționarea „real-time” nu este demonstrată.


- [Sursa FER2013 și licența declarată](docs/publicare/sursa_fer2013/raport.md)
- [Verificare integrală fără antrenare](docs/publicare/reproducere_studiu_v1.md)
- [Rezultate comparative noi și limite](docs/publicare/rezultate_interne_v1/interpretare.md)
- [Inventarul derivat și protocolul fixat](docs/publicare/inventar_derivat.md)
- [Protocol, comenzi și costul antrenării](docs/ACCURACY.md)
- [Disponibilitate, proveniență și lista de recuperare](docs/publicare/dataset_card.md)
- [Raportul intervențiilor și verificărilor](docs/publicare/stare_evaluare.md)
- [Revizuirea duplicatelor și propunerea de tratare](docs/publicare/duplicate_review.md)
- [Planul de îmbunătățiri](docs/publicare/plan_imbunatatiri.md)

Rezultatele numerice pot fi verificate **fără imagini sau checkpoint-uri**, dintr-o clonă care conține manifestele și exporturile:

```bash
python -m pip install -r requirements/verify.txt
python scripts/verify_published_results.py
```

Comanda verifică integritatea exporturilor, separarea grupurilor și acoperirea subseturilor, apoi recalculează metricile și bootstrap-ul pereche. Nu descarcă date, nu ajustează modele și nu accesează căile locale istorice ale datasetului. Verifică consistența dovezilor salvate, nu adevărul etichetelor sau predicțiile prin reexecutarea modelelor.

Pentru toate testele, inclusiv API și CNN sintetic:

```bash
backend/.venv/bin/python -m pip install -r tests/requirements.txt
backend/.venv/bin/python -m unittest discover -s tests -v
backend/.venv/bin/python scripts/evaluation.py --help
```

Testele folosesc exclusiv imagini și weights sintetice în directoare temporare. Verifică infrastructura, nu clasificarea expresiilor. Scripturile vechi rămân pentru trasabilitate; comenzile lor de split/rebalansare și comparațiile pe metadate nu constituie protocolul comun.

Demonstrația locală folosește `backend/server.py`, fără servicii de căutare sau modele lingvistice. Testele API verifică predicțiile clasice/CNN și erorile explicite când lipsește sau eșuează un model. Nu se returnează o etichetă fictivă în aceste situații. `tests/requirements.txt` include clientul HTTP necesar testării; acesta nu este o dependență a serverului în producție.

Structură: `scripts/` pentru experimente, `tests/` pentru controalele protocolului, `models/registry.json` pentru configurația HOG/LBP, `docs/` pentru documentație, `backend/` și `frontend/` pentru demonstrația locală. `Documentation_LATEX.md` păstrează raportul istoric, cu avertizare privind statutul rezultatelor.

Demonstrația locală se pornește conform [Startup.md](Startup.md) și [ghidului de arhitectură](docs/ARCHITECTURE.md). Ea depinde de date, modele și dependențele backend/frontend; nu a fost validată ca parte a evaluării noi.

Configurația [GitHub Actions](.github/workflows/checks.yml) rulează verificarea numerică fără dataset, testele sintetice și build-ul frontend la push și pull request. Starea rulărilor se consultă în [Actions](https://github.com/colacbrr/Real-Time-Emotion-Recognition-Platform/actions). `stop.sh` selectează procesele după comanda și directorul acestui repository; nu oprește global toate procesele Vite sau Uvicorn.

Limite: lanțul exact de furnizare academică și drepturile fotografiilor neconfirmate independent, independență pe persoane neconfirmată, screening incomplet al dublurilor similare și două perechi incerte, lipsa unui test istoric neatins și lipsa evaluării pe alt domeniu. LBP rămâne extensie ulterioară; mai multe modele nu rezolvă aceste lipsuri.
