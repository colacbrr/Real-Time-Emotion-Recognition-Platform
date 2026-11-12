# Verificarea studiului intern v1

## Verificare integrală fără antrenare

Din rădăcina repository-ului, cu copia academică accesibilă, rulează:

```bash
backend/.venv/bin/python scripts/verify_internal_study.py --source /home/colac/Projects/Academic/03_WSL_SCHOOL_COPY/School_WSL_HOME/aspicv-project
```

Pentru salvarea unui raport JSON, adaugă `--output outputs/verificare-noua.json`, cu un nume inexistent. Nu se descarcă date și nu se face fit în această comandă. Sunt folosite implicit artefactele locale v1; opțiunile `--help` permit indicarea unor căi diferite.

Verificatorul:

1. Controlează înregistrarea revizuirii, hash-urile celor șapte planșe, cele 3000 de imagini și 21 de artefacte ale sursei recuperate.
2. Regenerează propunerea și inventarele derivate într-un director temporar și cere CSV-uri/JSON identice cu cele înregistrate.
3. Verifică păstrarea tuturor ID-urilor, etichetelor, hash-urilor și split-urilor originale în evidența completă; inventarul eligibil, cel exclus și cel oficial sunt disjuncte și acoperă toate cele 3000 de ID-uri.
4. Cere ca manifestul experimentului să conțină exact cele 2677 de exemple eligibile și să păstreze distribuțiile de clasă fixate. Auditul verifică absența grupurilor/hash-urilor cunoscute comune între split-uri.
5. Verifică hash-urile codului și protocolului anterior antrenării, artefactele selectate pe validare și proveniența evaluării fără fit.
6. Recalculează metricile din cele trei exporturi de predicții și regenerează raportul numeric, recall-ul și matricile CSV. Cere rezultate identice cu artefactele salvate.

Rularea efectuată a trecut; raportul este `outputs/internal-final-verification-v2.json`. Un control negativ pe o copie temporară a inventarului a confirmat că adăugarea unui rând neînregistrat este respinsă. Originalele nu au fost modificate în acest control.

Această verificare demonstrează integritatea și coerența artefactelor înregistrate. Nu reprezintă o a doua revizuire vizuală, o certificare a licenței, a independenței persoanelor sau a unui test istoric neatins.

## Artefactele locale necesare

| Director/fișier | Conținut |
|---|---|
| Copia academică indicată prin `--source` | Imagini originale, etichete, arhivă, checkpoint-uri și rapoarte recuperate |
| `outputs/recovery-v1/` | Manifestul istoric, auditul și cele 126 de perechi candidate |
| `outputs/duplicate-review-v2/` | Planșe și dovezi numerice de revizuire |
| `docs/publicare/duplicate_review.csv` și `.provenance.json` | Decizii, motive și hash-uri |
| `outputs/duplicate-policy-v1/` | Componente de imagini și politica de selecție |
| `outputs/internal-inventory-v1/` | 2677 eligibile, 23 excluse, 300 oficiale, evidență completă |
| `outputs/internal-protocol-v1/` | Split, manifest, protocol anterior antrenării și pilot CPU |
| `outputs/internal-runs-v1/` | Artefacte, predicții și metrici de validare/test pentru cele trei modele |
| `docs/publicare/rezultate_interne_v1/` | Rezultate JSON, raport numeric, interpretare și matrici CSV |

Imaginile și modelele sunt locale și ignorate de Git, conform organizării repository-ului. Un checkout nou fără aceste artefacte nu poate verifica automat rezultatele salvate. Fișa datelor documentează sursa de recuperare cunoscută și lipsurile; nu presupunem existența unei adrese publice de descărcare pentru dataset.

## Comenzile experimentului executat

Configurația completă este în [internal_protocol_v1.json](internal_protocol_v1.json). Comenzile de mai jos arată fluxul folosit; directoarele existente nu se suprascriu. Orice reproducere prin reantrenare necesită directoare noi și păstrarea configurației; nu este o ocazie de ajustare retrospectivă pe testul consultat.

```bash
# Se folosește aceeași rădăcină a datelor pentru toate comenzile.
ASPICV_STUDY_SOURCE=/home/colac/Projects/Academic/03_WSL_SCHOOL_COPY/School_WSL_HOME/aspicv-project

backend/.venv/bin/python scripts/evaluation.py run --manifest outputs/internal-protocol-v1/manifest.csv --root "$ASPICV_STUDY_SOURCE" --model baseline --output outputs/internal-runs-v1/baseline-val
backend/.venv/bin/python scripts/evaluation.py run --manifest outputs/internal-protocol-v1/manifest.csv --root "$ASPICV_STUDY_SOURCE" --model hog --output outputs/internal-runs-v1/hog-val
backend/.venv/bin/python scripts/evaluation.py run --manifest outputs/internal-protocol-v1/manifest.csv --root "$ASPICV_STUDY_SOURCE" --model cnn --weights outputs/pretrained/resnet18-f37072fd.pth --epochs 10 --seed 42 --batch-size 16 --output outputs/internal-runs-v1/cnn-val

backend/.venv/bin/python scripts/evaluation.py evaluate-saved --manifest outputs/internal-protocol-v1/manifest.csv --root "$ASPICV_STUDY_SOURCE" --run outputs/internal-runs-v1/baseline-val --output outputs/internal-runs-v1/baseline-test
backend/.venv/bin/python scripts/evaluation.py evaluate-saved --manifest outputs/internal-protocol-v1/manifest.csv --root "$ASPICV_STUDY_SOURCE" --run outputs/internal-runs-v1/hog-val --output outputs/internal-runs-v1/hog-test
backend/.venv/bin/python scripts/evaluation.py evaluate-saved --manifest outputs/internal-protocol-v1/manifest.csv --root "$ASPICV_STUDY_SOURCE" --run outputs/internal-runs-v1/cnn-val --weights outputs/pretrained/resnet18-f37072fd.pth --batch-size 16 --output outputs/internal-runs-v1/cnn-test

backend/.venv/bin/python scripts/evaluation.py compare --manifest outputs/internal-protocol-v1/manifest.csv --root "$ASPICV_STUDY_SOURCE" --runs outputs/internal-runs-v1/baseline-test outputs/internal-runs-v1/hog-test outputs/internal-runs-v1/cnn-test --output outputs/internal-runs-v1/comparison.json
```

Weights ImageNet au fost recuperați explicit, separat de evaluator. Pilotul și costul măsurat sunt în [inventarul derivat](inventar_derivat.md) și [interpretarea rezultatelor](rezultate_interne_v1/interpretare.md). Timpii diferă de la o rulare la alta; identitatea byte cu byte verificată aici privește regenerarea din artefactele salvate, nu promisiunea că orice mediu va reproduce exact toate fișierele unei reantrenări.
