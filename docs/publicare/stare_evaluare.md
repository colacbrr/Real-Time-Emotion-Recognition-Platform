# Starea studiului — actualizare 2026-11-12

## Extensie pentru articol educațional de reproducibilitate

S-a realizat [studiul v2](studiu_v2/README.md) și un [manuscris nou în engleză](articol/README.md). Nu s-a lucrat la interfață. V2 constată amestecul partițiilor FER în subsetul academic, separă 2069 imagini de dezvoltare de 509 imagini source-test după excluderi și compară HOG/ResNet18 înghețat cu aceleași capete SVM liniar/RBF. Cinci folduri externe, trei interne, baseline comun, predicții OOF și source-test, recall/suport, matrici și bootstrap pereche condiționat, fără pretenție de test istoric neatins sau independență pe persoane.

Pilot de cost separat înaintea celor 240 fit-uri; buget 5–15 min CPU, efectiv 114.80 s inclusiv bootstrap. ResNet-RBF are OOF macro-F1 0.3692 față de HOG-RBF 0.3096. Pe source-test ambele au accuracy 0.3949, iar macro-F1 0.3528 versus 0.3157; intervalul pereche al diferenței include zero. 29 teste sintetice au trecut în 17.245 s; verificările v1 și v2 au trecut fără antrenare sau rețea. Rezultatele și codul fixat v1 rămân identice.

Rămân: verificare umană și autor/afiliere înainte de trimitere, review independent al duplicatelor, clarificarea redistribuirii datelor, metadatele persoanelor și eșantionării, test independent/alt domeniu și etichetele testului academic oficial. Lipsurile sunt declarate în articol. Nu s-a publicat nimic și nu s-au modificat PDF-urile originale.

Documentarea de mai jos păstrează istoricul etapei v1; numărul de teste și starea sursei din acea etapă trebuie citite cu actualizarea de mai sus.

## Contribuție și rezultat

Studiu educațional de clasificare a expresiilor etichetate, prin HOG/LBP-SVM și transfer learning. Contribuția proprie este integrarea software și evaluarea verificabilă, nu inventarea algoritmilor standard. Nu se demonstrează deducerea stării interioare sau funcționarea real-time.

După recuperarea și auditul datelor s-a executat **o comparație internă comună** între baseline majoritar, HOG-SVM și ResNet18 înghețat cu cap liniar. Este retrospectivă și exploratorie, nu evaluarea testului oficial. [Protocol, rezultate și limite](rezultate_interne_v1/interpretare.md).

## Intervențiile efectuate

Blocuri cu țintă de 5–10 minute active, nu un jurnal de cronometrare; costul antrenării este măsurat separat:

1. Citirea instrucțiunilor locale și planului; inventarul inițial, fără date în checkout.
2. Manifestul și controalele hash/split/identități cunoscute; metrici și exporturi comune.
3. Baseline, HOG, un singur CNN; evaluarea artefactelor salvate fără fit și pilot de cost separat.
4. Recuperarea copiei academice indicate de utilizator; verificarea arhivei, etichetelor și checkpoint-urilor.
5. Reevaluarea HOG istoric și baseline pe cele 700 de imagini originale de validare.
6. Screeningul și revizuirea celor 126 de perechi similare; adnotări, dovezi și propunere de carantinare.
7. Verificarea integrală a hash-urilor și regenerarea identică a propunerii.
8. Aplicarea regulilor în CSV-uri derivate: 2677 eligibile, 23 excluse, 300 oficiale fără etichete; fără modificarea imaginilor.
9. Fixarea protocolului intern 1605/536/536 înainte de antrenare; recuperarea weights ResNet18 oficiali și pilotul CPU.
10. O singură rulare per configurație, cu test din artefactele selectate pe validare, comparație recalculată și raport comun.

## Verificări și artefacte

- 3000 imagini 48×48 grayscale, identice byte cu byte cu arhiva recuperată; 2700 etichete concordante între CSV-ul original și cele istorice.
- 20 grupuri de duplicate exacte, dintre care 7 train–val și 3 etichetat–predict; un grup are conflict exact de etichete.
- 126 perechi suplimentare revizuite: 37 aproape duplicate, 87 imagini diferite, 2 incerte; o singură revizuire de asistent, fără deduceri de identitate.
- Patru componente cu etichete neuniforme, reținute în afara noului experiment. Cele două perechi incerte rămân nerezolvate și excluse conservator.
- CSV-urile derivate păstrează toate ID-urile în evidență; inventarul eligibil nu are grupuri/hash-uri comune cu datele excluse sau oficiale.
- Split-ul nou păstrează grupurile cunoscute împreună, are toate clasele în fiecare subset și păstrează istoricul separat.
- Weights ResNet18 IMAGENET1K_V1: URL oficial Torchvision, prefix SHA-256 publicat verificat, hash complet și încărcare strictă a arhitecturii păstrate în proveniență.
- **24 teste sintetice trecute**, ultima rulare 19.041 s. Sunt acoperite metricile, scurgerile, identitățile furnizate, artefactele modificate, datele sintetice CNN, split-urile grupate, auditul arhivei, alinierea, politica de excludere și inventarul derivat.
- Verificarea integrală reutilizabilă a trecut: `verify_internal_study.py` reproduce inventarele și raportul din artefactele salvate, fără fit sau rețea; inventarul alterat este respins. [Comandă și artefacte necesare](reproducere_studiu_v1.md).
- Exportatorul rezultatelor a reverificat codul/protocolul fixat, manifestul, legătura artefactelor train–val–test și a recalculat comparația din predicții.

[Inventar și protocol](inventar_derivat.md), [regulile fixate](internal_protocol_v1.json), [raportul numeric](rezultate_interne_v1/report.md), [datele complete ale raportului](rezultate_interne_v1/results.json).

## Rezultate noi pe testul intern comun

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majoritar, happy din train | 0.238806 | 0.055077 |
| HOG-SVM | 0.386194 | 0.357925 |
| ResNet18 înghețat + cap liniar | 0.404851 | 0.317053 |

Același test de 536 imagini, același train de 1605 și aceeași validare de 536. CNN are accuracy mai mare, iar HOG macro-F1 mai bun; nu rezultă un câștig uniform. CNN are recall 0/8 la disgust și 11/89 la sad, față de HOG 2/8 și 27/89. Un singur seed și suportul mic al claselor limitează concluzia.

Au fost exportate câte 536 predicții individuale per model, accuracy, macro-F1, recall/suport pe clasă și matrici de confuzie JSON/CSV. Predicțiile sunt în `outputs/internal-runs-v1/*-test/predictions.csv`. Matricile, recall-ul și metadatele se află și în `docs/publicare/rezultate_interne_v1/`.

Pilotul anterior CNN a estimat conservator circa 4 minute pentru extracție și câteva secunde pentru cap. Fit-ul efectiv a durat circa 25.08 s CPU; HOG circa 1.60 s. Inferența pe lotul test a durat circa 8.04 s CNN și 0.48 s HOG, cu încărcare/preprocesare incluse. Nu sunt latențe de serviciu și nu demonstrează real-time.

## Rezultate istorice păstrate separat

Checkpoint-ul HOG recuperat reproduce accuracy 0.432857 și matricea originală pe cele 700 de imagini istorice, cu macro-F1 exportat acum 0.390162; baseline 0.247143 accuracy / 0.056619 macro-F1. Acest split este contaminat și a fost folosit la selecție. Există avertismente de compatibilitate scikit-learn 1.8→1.9 și lipsește un manifest al antrenării originale. Rezultatele sunt în `outputs/recovered-validation-v2/`.

Aceste scoruri nu sunt comparate direct cu noile valori interne; nici scorurile CNN istorice 2600/100 nu sunt combinate cu noul tabel.

## Ce depinde încă de surse externe sau clarificări

- Ulterior experimentului, familia FER2013 a fost identificată prin potrivirea byte cu byte a tuturor celor 3000 de imagini cu distribuția Kaggle msambare/fer2013 v1; declarația de licență DbCL-1.0 / ODbL este documentată. [Surse și limite](sursa_fer2013/raport.md). Lanțul exact de furnizare academică și drepturile fotografiilor rămân neconfirmate independent.
- `test.csv` oficial pentru cele 300 imagini și istoricul consultării lui. Testul intern nu îl înlocuiește și nu este istoric neatins.
- Metadatele de persoană/sesiune, dacă există; screeningul imaginilor nu stabilește independența persoanelor.
- A doua revizuire a celor două perechi incerte și clarificarea celor patru componente cu etichete neuniforme, pentru o eventuală versiune ulterioară a datelor.

Repetarea seed-urilor, un experiment controlat de fine-tuning sau evaluarea pe alt domeniu sunt extensii viitoare cu buget separat; nu au fost executate în această versiune. Nu se schimbă modelul pentru a îmbunătăți retrospectiv testul consultat.

Protocolul și raportul numeric fixate înaintea investigației păstrează mențiunea inițială despre licență; noul addendum completează proveniența fără rescrierea experimentului. Potrivirile de referință nu au modificat etichetele sau modelele.

Nu s-au modificat copia academică sau PDF-urile din Tema S1, nu s-a publicat nimic și nu au fost lansate căutări sau antrenări lungi. Backendul și interfața nu au fost pornite sau validate în această etapă.
