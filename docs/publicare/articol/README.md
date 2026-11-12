# Manuscris educațional de reproducibilitate

Documentul nou în engleză prezintă studiul ca audit empiric și exemplu educațional. Nu propune o arhitectură nouă și nu pretinde eficacitate pedagogică măsurată, superioritate generală, inferență a stării interioare sau real-time. Datele și experimentele au fost executate; autorul/afilierea, revizia umană și cerințele unei publicații nu sunt confirmate de acest draft.

- [PDF pentru review](../../../output/pdf/fer2013_reproducibility_study.pdf)
- [Sursă LaTeX editabilă](manuscript.tex)
- [Interpretare cantitativă](results_discussion.tex)
- [Protocol și reproducerea experimentelor](../studiu_v2/README.md)

PDF-ul include tabele generate din JSON, figuri cu intervale condiționate, matrici de confuzie și 11 referințe primare/documentație. Nu conține fotografii faciale. Originalele din Tema S1 rămân intacte; `Documentation_LATEX.md` continuă să fie raportul istoric.

Din rădăcina repository-ului, reconstruirea PDF-ului folosește TeX Live/pdflatex și imaginile/tabelele deja incluse în `studiu_v2/results`:

```bash
mkdir -p tmp/pdfs output/pdf
cd docs/publicare/articol
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../../../tmp/pdfs manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../../../tmp/pdfs manuscript.tex
cp ../../../tmp/pdfs/manuscript.pdf ../../../output/pdf/fer2013_reproducibility_study.pdf
```

Nu este nevoie de date sau checkpoint-uri pentru compilarea manuscrisului. Reantrenarea și verificarea imaginilor sunt operații distincte, documentate în studiul v2. Tabelele pot fi urmărite până la predicțiile individuale; figurile au fost generate cu Matplotlib. Versiunea PDF a fost verificată prin randare, nu numai prin compilare.

Înainte de trimitere trebuie confirmate autorul și afilierea, verificate de un om afirmațiile și bibliografia, repetat independent review-ul duplicatelor și stabilit ce artefacte se pot distribui. Textul include declararea asistenței AI și nu inventează aprobări etice, finanțare sau un DOI al unui depozit public. Nu s-a trimis sau publicat nimic.
