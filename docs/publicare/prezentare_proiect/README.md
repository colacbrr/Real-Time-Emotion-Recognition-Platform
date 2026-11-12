# Documentație de prezentare a proiectului

`documentatie.tex` este varianta în română, cu două coloane, adaptată colecției de documentații din profil. Explică scopul, datele, protocolul, metodele, rezultatele și limitele studiului. Nu include data curentă; data creării este omisă și din metadatele PDF-ului.

Documentul folosește exclusiv rezultatele deja salvate ale studiului v2. Graficele și tabelele în română din `figuri/` sunt generate de `scripts/build_publication_figures_ro.py`, care verifică integritatea sursei numerice înainte de export. Traducerea etichetelor claselor este păstrată în `figuri/provenienta.json`. Nu sunt adăugate experimente sau rezultate noi prin redactare. Numele istoric al fișierului PDF este păstrat pentru compatibilitatea cu linkurile din profil; titlul din document descrie clasificarea expresiilor faciale.

Compilare din rădăcina repository-ului, cu pdfLaTeX și pachetele standard folosite în sursă:

```bash
MPLCONFIGDIR=/tmp/matplotlib-public-ro python scripts/build_publication_figures_ro.py
mkdir -p tmp/pdfs/portfolio output/pdf
cd docs/publicare/prezentare_proiect
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../../../tmp/pdfs/portfolio documentatie.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../../../tmp/pdfs/portfolio documentatie.tex
cp ../../../tmp/pdfs/portfolio/documentatie.pdf '../../../output/pdf/Real-Time Emotion Recognition Platform.pdf'
```

PDF-ul are patru pagini, cu pagini tematice și coloane aliniate. Termenii metricilor sunt explicați înaintea rezultatelor. Precizările despre populația evaluată și incertitudine apar în protocol; nu există secțiuni separate „Limitele care rămân” sau „Asistență și responsabilitate”. Pentru control vizual se randează toate paginile cu `pdftoppm`; `pdfinfo` verifică metadatele. Sursa editabilă rămâne în repository. Originalul din colecția profilului se păstrează în arhivă înaintea înlocuirii.
