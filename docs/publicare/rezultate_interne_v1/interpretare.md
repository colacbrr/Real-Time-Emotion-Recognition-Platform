# Interpretarea experimentului intern v1

## Întrebarea și protocolul

Cum diferă HOG-SVM și transfer learning atunci când folosesc aceleași exemple de antrenare și același test? Experimentul compară un predictor majoritar, HOG-SVM și ResNet18 cu backbone înghețat, cap liniar și 10 epoci. Este un studiu educațional retrospectiv, nu un benchmark independent.

Din cele 2700 de exemple etichetate recuperate, 23 au fost reținute în afara dezvoltării interne prin reguli documentate înaintea antrenării: dezacorduri de etichetă, incertitudine de duplicare sau suprapunere cu predict. Cele 2677 rămase au fost împărțite, cu seed 42 și gruparea duplicatelor cunoscute, în 1605 train, 536 validare și 536 test intern. Cele 300 de imagini oficiale fără etichete nu au fost folosite.

[Protocolul fixat anterior antrenării](../internal_protocol_v1.json) păstrează configurațiile, hash-urile, distribuțiile și hardware-ul. Nu s-au căutat hiperparametri pe test. Pentru CNN s-a păstrat epoca 9, selectată exclusiv prin macro-F1 pe validare; evaluarea finală a încărcat același artefact, fără reantrenare. Baseline-ul și scalerul/SVM au folosit numai train pentru ajustare. HOG folosește grayscale 48×48; CNN folosește RGB și transformările oficiale ImageNet de resize/crop/normalizare. Exemplele sunt comune, preprocesarea este adaptată metodei și documentată.

## Rezultate noi pe același test intern

| Model | Corecte / 536 | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Majoritar (happy) | 128 | 23.88% | 0.0551 |
| HOG-SVM | 207 | 38.62% | 0.3579 |
| ResNet18 înghețat + cap liniar | 217 | 40.49% | 0.3171 |

CNN clasifică corect cu 10 imagini mai mult decât HOG, adică +1.87 puncte procentuale accuracy. HOG are macro-F1 cu aproximativ 0.0409 mai mare. Nu există un câștig uniform al CNN: concluzia depinde de metrica urmărită. Rezultatul descrie aceste configurații, pe acest split și un singur seed; nu stabilește superioritatea generală a uneia dintre familii.

[Recall-ul pe clase](recall_per_class.csv) arată de ce accuracy singur este insuficient. Pentru `disgust`, HOG recuperează 2/8 exemple, iar CNN 0/8. Pentru `sad`, HOG recuperează 27/89, iar CNN 11/89. CNN are recall mai mare pentru `angry`, `fear`, `happy` și `surprise`, iar HOG pentru `neutral`, `sad` și `disgust`. În clasa cu numai opt exemple, un singur caz schimbă recall cu 12.5 puncte procentuale; aceste valori nu sunt estimări stabile pentru populația generală.

Nu atribuim diferențele unei cauze izolate precum pretraining, augmentare sau dezechilibru: nu s-a efectuat un experiment controlat care să separe aceste efecte. Nu au fost folosite MixUp/CutMix, ensemble sau fine-tuning integral. Nu s-au modificat configurațiile după observarea testului.

## Cost observat față de estimarea anterioară

Hardware: Intel Core i5-13600KF, CPU, Torch limitat la maximum patru thread-uri, fără CUDA. Pilotul pe 16 imagini train a extrapolat conservator circa 235 s pentru extracția tuturor imaginilor și circa 4 s pentru cap. Bugetul comunicat înainte de rulare a fost circa 6 minute.

Fit-ul măsurat a durat 1.60 s pentru HOG și 25.08 s pentru CNN. Aceste durate includ extracția train/validare, ajustarea, alegerea checkpoint-ului acolo unde se aplică și salvarea artefactului; nu sunt exclusiv timpi de optimizare. Evaluarea lotului de 536 imagini test, inclusiv încărcarea artefactului și preprocesarea, a durat 0.48 s pentru HOG și 8.04 s pentru CNN. Încărcarea și overhead-ul pe eșantionul mic fac extrapolarea pilotului conservatoare; pilotul nu a fost înlocuit retrospectiv cu durata efectivă.

HOG salvat ocupă 11,536,557 bytes. Capul CNN salvat ocupă 16,589 bytes, dar necesită și backbone-ul extern de 46,830,571 bytes, identificat prin hash. Dimensiunea capului singur nu este dimensiunea totală a modelului utilizabil.

Acestea sunt măsurători ale scripturilor pe loturi, nu latențe ale API-ului sau măsurători ale unui sistem live. Nu justifică termenul real-time.

## Dovezi și limite

[results.json](results.json) conține scorurile complete, suportul, recall-ul, matricile de confuzie, timpii, versiunile și hash-urile artefactelor/predicțiilor. Matricile sunt și în CSV: [baseline](baseline_confusion_matrix.csv), [HOG](hog_confusion_matrix.csv), [CNN](cnn_confusion_matrix.csv), cu rânduri=adevăr și coloane=predicție. Predicțiile individuale rămân local în `outputs/internal-runs-v1/{baseline,hog,cnn}-test/predictions.csv`, câte 536 rânduri fiecare. Exportatorul recalculează metricile din aceste predicții și verifică legătura cu artefactele de validare și protocolul fixat.

Rămân limite materiale:

- Datele etichetate au fost deja explorate istoric; noul test intern nu este istoric neatins.
- Nu există ID-uri de persoană/sesiune. Gruparea pe imagini nu demonstrează independența pe persoane.
- Screeningul dublurilor similare nu este exhaustiv și există o singură revizuire vizuală de asistent. Cazurile incerte au fost excluse conservator, nu rezolvate.
- Copiile concordante dintr-un split au fost păstrate; metricile sunt per imagine, nu per fotografie unică.
- La executarea experimentului, sursa/licența era neclarificată. [Investigația ulterioară](../sursa_fer2013/raport.md) confirmă corespondența tuturor imaginilor cu distribuția publică FER2013 msambare/fer2013 v1 și consemnează licența declarată DbCL-1.0 / ODbL. Lanțul drepturilor fotografiilor nu este certificat independent, iar test.csv oficial ASPICV rămâne nerecuperat. Snapshot-urile protocolului/raportului numeric nu au fost rescrise.
- Este o singură configurație per metodă și un singur seed, fără intervale de incertitudine sau evaluare pe alt domeniu.

Reevaluarea istorică HOG de 43.29% pe alte 700 de imagini rămâne separată. Nu comparăm direct acest scor cu 38.62% din noul protocol și nu numim diferența îmbunătățire sau degradare a modelului.
