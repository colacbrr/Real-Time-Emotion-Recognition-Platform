# Expresii faciale - plan pentru o comparație corectă a clasificatorilor

## Ce este proiectul

Titlu propus: „Clasificarea expresiilor faciale din imagini de rezoluție redusă: un studiu educațional HOG/LBP-SVM și CNN”. Titlul trebuie să descrie etichetele imaginilor și experimentul efectuat; capacitatea de a deduce starea interioară a unei persoane nu este demonstrată.

Scop: să construiesc și să compar, într-un protocol comun, o abordare cu descriptori calculați explicit și una cu reprezentări învățate. Utilitatea este un exemplu de experimentare ML care arată cum se evaluează metodele pe puține date și clase dezechilibrate.

Întrebarea principală: „Cum diferă performanța și costul metodelor când au acces la aceleași date de antrenare și același test?” Scorul mare nu este singurul rezultat util; o comparație credibilă poate arăta și limite sau absența unui câștig clar.

## Situația verificată și dependența principală

Există scripturi pentru HOG/LBP-SVM, CNN, comparație și submission. Raportul combină 2000/700 pentru metodele clasice cu 2600/100 pentru CNN. Copia locală examinată nu conține datasetul, checkpoint-urile sau rapoartele complete de antrenare. Proveniența exactă a celor 3000 de imagini trebuie confirmată, nu dedusă doar din rezoluție și etichete.

Fără date sau predicții salvate, nu putem reconstitui scorurile și nu putem transforma acest raport într-un studiu comparativ verificat numai prin rescriere. Recuperarea datelor este primul pas, nu adăugarea unui model nou.

## P0 - implementări necesare pentru rezultate comparative

| Pas | Implementare propusă | Dovada de acceptare | Efort activ estimat |
|---|---|---|---|
| 1 | Inventar dataset: sursa exactă, versiunea, condițiile de utilizare, număr de exemple și clase | `dataset_card.md` și verificare că fișierele corespund descrierii | 5-10 min după găsirea sursei; recuperarea poate dura mai mult |
| 2 | Manifest pe imagini: ID, etichetă, hash și split; verificarea duplicatelor exacte | Niciun hash comun între train și test; excepțiile sunt investigate | 5-10 min pentru primul script |
| 3 | Separare fixă train/validare/test, stratificată când datele permit | Toate modelele citesc același manifest; testul nu este folosit la tuning | 5-10 min dacă manifestul există |
| 4 | Control de persoane/dubluri similare dacă există metadate relevante | Identitățile nu traversează seturile sau limitarea este declarată explicit | Timp variabil; fără promisiune de rezolvare în 10 min |
| 5 | Export comun de predicții și metrici | `predictions.csv`: ID, adevăr, predicție; `metrics.json`: accuracy, macro-F1, recall/clasă și matrice de confuzie | 1-2 intervenții de 5-10 min |
| 6 | Evaluarea unui predictor simplu și a modelelor păstrate pe același test | Baseline majoritar, HOG-SVM și un CNN au rezultate comparabile ca protocol | Configurare 5-10 min per pas; antrenarea CNN poate dura mult mai mult |

Dacă imaginile din test au fost deja folosite pentru alegerea repetată a modelelor, nu numim setul retrospectiv „neatins”. Alegem o evaluare nouă justificată sau păstrăm caracterul exploratoriu. Dacă datele provin dintr-un benchmark cu split oficial, acesta are prioritate față de o împărțire arbitrară, după verificarea condițiilor lui.

Pentru a compara metodele pe puține date este suficient inițial HOG-SVM versus un singur CNN. LBP poate rămâne o referință suplimentară. Nu sunt necesare ResNet18, ResNet50, EfficientNet și ensemble în prima versiune verificată.

## P1 - implementări care fac rezultatul mai util

- [ ] Repetarea antrenării cu câteva seed-uri, cu testul păstrat fix, pentru a vedea sensibilitatea la inițializare. Costul de calcul trebuie estimat înainte.
- [ ] Comparație controlată a unei singure alegeri: de exemplu, CNN fără/cu augmentare. Aceeași împărțire, același buget de antrenare și același criteriu de selecție.
- [ ] Exportul timpului de antrenare, dimensiunii modelului și latenței de inferență. CPU și GPU sunt raportate separat, cu batch size și hardware.
- [ ] Exemple de erori pe clase apropiate și distribuția claselor. Folosirea imaginilor în PDF depinde de drepturile datasetului; pot fi suficiente ID-uri și statistici dacă redistribuirea nu este permisă.
- [ ] Interval de incertitudine pentru scorul de test, explicând metoda și faptul că nu corectează toate efectele selecției modelelor.

Într-un set mic, diferența de câteva imagini nu este un argument pentru un câștig stabil. Performanța slabă a unui model este un rezultat publicabil ca raport educațional dacă evaluarea este corectă și discuția este utilă.

## Modificări în documentație, în structura existentă

- Introducere: de ce am ales comparația clasic versus transfer learning și de ce rezoluția mică face experimentul interesant.
- Date: proveniență, dimensiunea reală, clase, split-uri și ce știm despre duplicate/identități.
- Implementare: ce am construit eu în jurul algoritmilor standard; preprocesarea, scalerul ajustat numai pe train, maparea etichetelor, pretraining-ul și configurarea modelelor.
- Protocol: cum aleg hiperparametrii/checkpoint-ul pe validare și când folosesc testul. Antrenarea finală pe toate datele se descrie separat de evaluare.
- Rezultate: tabel comun pe același test; valorile istorice pe split-uri diferite se separă sau se înlocuiesc numai după existența experimentelor noi.
- Discuție: clasele rare, confuziile observate, compromis accuracy/cost și incertitudinea. Nu atribuim efecte MixUp/CutMix dacă nu au fost izolate.
- Limite: eșantion mic, diferența dintre expresie etichetată și emoție interioară, testarea pe alt domeniu neefectuată.
- Titlu/rezumat: „real-time” se păstrează numai dacă este demonstrat prin latență și cerința de funcționare definită; altfel folosim „pipeline de clasificare”.

Formulare propusă pentru contribuție: „Am construit un flux de antrenare și evaluare pentru compararea unor descriptori clasici cu transfer learning pe imagini faciale de rezoluție redusă. Am urmărit nu numai accuracy, ci și comportamentul pe clase rare și costul de inferență.” Ultima propoziție devine afirmație de rezultat numai după implementarea și rularea metricilor.

## Ce nu merită adăugat acum

Mai multe backbone-uri, recunoaștere live din webcam sau un dashboard nou nu rezolvă lipsa unei evaluări comune. Etichetarea drept produs medical ori instrument de monitorizare psihologică ar depăși dovezile existente.

## Prag pentru distribuire

Pentru un raport de implementare exploratorie: prezentăm contribuția software și delimităm explicit scorurile istorice. Pentru un raport comparativ: toate punctele P0 privind datele și evaluarea trebuie finalizate, cu predicții și metrici verificabile. Dacă datele nu sunt recuperate, păstrăm prima încadrare; nu completăm scoruri estimate.
