# Sursa FER2013 și licența declarată — verificare 2026-11-09

## Concluzie verificată prin conținut

**Toate cele 3000 de imagini din arhiva ASPICV au corespondent identic byte cu byte în distribuția publică Kaggle `msambare/fer2013`, versiunea 1.** Nicio imagine nu a rămas fără potrivire. Identificarea folosește SHA-256 al întregului fișier JPEG, nu asemănarea fețelor, rezoluția sau denumirea claselor.

În documentație putem atribui acum datele drept **subset FER2013, renumerotat și repartizat pentru proiectul ASPICV, cu conținut verificat față de distribuția Kaggle msambare/fer2013 v1**. Potrivirea fișierelor nu dovedește că profesorul/autorul arhivei a descărcat exact de la acest uploader; lanțul efectiv de furnizare academic rămâne nedocumentat.

## Originea datasetului și distribuția găsită

[Raportul original al organizatorilor](https://arxiv.org/html/1307.0414v1), secțiunea 3, atribuie crearea FER2013 lui Pierre Luc Carrier și Aaron Courville. Acesta descrie colectarea prin căutare de imagini Google, decuparea/prelucrarea și subsetul de 35,887 de imagini grayscale 48×48, cu șapte clase. Aceasta este sursa bibliografică pentru familia datasetului, nu un text de licență pentru arhiva ASPICV.

Distribuția comparată este [FER-2013, uploader Manas Sambare](https://www.kaggle.com/datasets/msambare/fer2013), dataset ID 786787, versiunea 1, creată la 2020-07-19. Arhiva publică descărcată conține 35,887 imagini. [Metadatele API Kaggle](https://www.kaggle.com/api/v1/datasets/view/msambare/fer2013) și hash-urile lor sunt salvate local, împreună cu data verificării.

## Licența: ce este confirmat și ce nu

Metadatele acestei distribuții declară `Database: Open Database, Contents: Database Contents`. [Documentația oficială Kaggle](https://raw.githubusercontent.com/Kaggle/kaggle-api/main/docs/datasets_metadata.md), secțiunea Licenses, asociază această denumire cu codul **`DbCL-1.0`**, afișat ca pereche:

- [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) pentru baza de date;
- [Database Contents License 1.0](https://opendatacommons.org/licenses/dbcl/1-0/) pentru conținut.

Textul DbCL, secțiunea 2.2, cere respectarea ODbL. ODbL tratează distinct drepturile bazei de date și ale elementelor individuale. Prin urmare, consemnăm **licența declarată de distribuitor**, cu link-uri și snapshot de metadate, fără a transforma declarația într-o verificare independentă a tuturor drepturilor fotografiilor originale sau într-o nouă licență acordată de acest proiect. Licența codului repository-ului rămâne separată.

[Regulile competiției originale](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/rules) includ cerințe de licențiere a soluțiilor câștigătoare; acea prevedere nu este o acordare explicită de licență asupra tuturor imaginilor. [Microsoft FER+](https://github.com/microsoft/FERPlus) publică adnotări suplimentare și indică faptul că nu găzduiește imaginile originale. Licența adnotărilor/codului FER+ nu este folosită aici ca licență a fotografiilor FER2013. Nu s-au importat adnotări FER+.

## Dovada tehnică

| Element | Rezultat |
|---|---|
| Imagini locale verificate | 3000 |
| Potriviri SHA-256 ale întregului fișier | 3000 |
| Imagini locale fără potrivire | 0 |
| Imagini în arhiva publică de referință | 35,887 |
| Etichete locale pentru care toate copiile exacte găsite concordă | 2691 |
| Etichete locale cu copii exacte care au mai multe adnotări în referință, dintre care cel puțin una concordă | 9 |
| Etichete locale fără niciun candidat concordant | 0 |

Cele nouă cazuri nu sunt corectate automat: distribuția de referință conține ea însăși copii cu adnotări diferite. Potrivirea de conținut nu desemnează automat o singură etichetă drept adevăr.

Cele 300 de imagini predict sunt incluse numai în verificarea conținutului. În exportul de corespondențe, clasa de referință și calea care ar divulga acea clasă sunt ascunse pentru ele. Nu au fost completate etichete ASPICV, creat un `test.csv` sau calculate scoruri pe testul oficial din etichetele copiei publice.

Arhiva publică, 63,252,113 bytes, are SHA-256:

`9f3422c7a45b676c2965ef4fc5f3099932aa98fd13eaaa2c254f620a9894a6e9`.

Metadatele API salvate au SHA-256:

`94b305f2b83df60331632be838ed220559828172b9944198051f79a5b0508484`.

Manifestul ASPICV comparat este cel istoric, cu SHA-256:

`b17ee848b9b3d145061554785d9f3e94fcc640e8a024418cbfe32535c097489a`.

[Evidența structurată](evidence.json) include versiunea, licența declarată, numărătorile și hash-urile. Datele de referință rămân local în `outputs/fer2013-reference-v2/`; corespondențele și inventarul pe ID în `outputs/fer2013-content-match-v1/`. Nu au fost publicate imagini.

## Reproducere și efect asupra studiului

Descărcarea explicită cere acces de rețea și un director nou; comparația ulterioară este offline:

```bash
backend/.venv/bin/python scripts/fetch_fer2013_reference.py --output outputs/fer2013-reference-v2
backend/.venv/bin/python scripts/match_fer2013_reference.py --manifest outputs/recovery-v1/historical_manifest.csv --source /home/colac/Projects/Academic/03_WSL_SCHOOL_COPY/School_WSL_HOME/aspicv-project --reference outputs/fer2013-reference-v2 --output outputs/fer2013-content-match-v1
```

Endpoint-urile publice au răspuns HTTP 200; nu au fost folosite acreditări, acceptate condiții în numele utilizatorului sau ocolite restricții de autentificare. Scriptul refuză un răspuns care nu este arhivă ZIP și verifică integritatea referinței înainte de potrivire.

Această investigație completează proveniența **după** experiment. Protocolul, checkpoint-urile, predicțiile, timpii și raportul numeric înghețate nu au fost rescrise. Mențiunea anterioară „sursă/licență necunoscută” din snapshot-urile JSON descrie starea de cunoaștere la momentul experimentului; prezentul addendum consemnează distribuția identificată și licența ei declarată. Rezultatele rămân interne și retrospective, fără independență demonstrată pe persoane și fără echivalare cu benchmarkul oficial FER2013.
