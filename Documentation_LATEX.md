\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts

\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{algorithm}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{microtype}
\usepackage{fvextra}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\usetikzlibrary{shapes.geometric, arrows, positioning, fit, backgrounds}

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\linespread{1.08}
\setlength{\parskip}{0.2em}
\setlength{\parindent}{1em}
\setlength{\columnsep}{0.28in}
\setlength{\textfloatsep}{10pt}
\setlength{\intextsep}{10pt}
\setlength{\abovecaptionskip}{6pt}
\setlength{\belowcaptionskip}{6pt}
\sloppy
\emergencystretch=1em
\raggedbottom
\DefineVerbatimEnvironment{Code}{Verbatim}{breaklines,breakanywhere,fontsize=\small}

\begin{document}

\title{Recunoașterea Emoțiilor Faciale din Imagini 48\,$\times$\,48\\
cu HOG/LBP, SVM și CNN-uri moderne\\
}

\author{\IEEEauthorblockN{Colăcel Cristian-Adrian}
\IEEEauthorblockA{\textit{Facultatea de Electronică, Telecomunicații și Tehnologia Informației} \\
\textit{Universitatea Politehnica din București}\\
\textit{Ingineria Informației}\\
București, România \\
}
}

\maketitle

%==============================================================================
% ABSTRACT
%==============================================================================
\begin{abstract}
Lucrarea prezintă un sistem complet pentru recunoașterea emoțiilor faciale din imagini alb-negru de 48\,$\times$\,48 pixeli, combinând un pipeline clasic (HOG/LBP + SVM) cu modele CNN moderne (ResNet18/ResNet50 și EfficientNet-B0). Proiectul folosește un set etichetat împărțit în antrenare și validare, plus un set separat de predicție pentru generarea fișierului \texttt{submission.csv}. Arhitectura include un backend FastAPI pentru inferență și raportare, o interfață React pentru inspecție și scripturi reproductibile pentru antrenare, validare, comparație de modele, augmentări (MixUp/CutMix) și ensembling. Se discută în detaliu motivația alegerilor metodologice, fluxul de date, metricile utilizate și limitările inerente dataset-ului, cu accent pe maximizarea acurateții în faza de verificare.
\end{abstract}

\begin{IEEEkeywords}
recunoaștere emoții faciale, HOG, LBP, SVM, CNN, ResNet, EfficientNet, MixUp, CutMix, ensembling, FastAPI, validare
\end{IEEEkeywords}

%==============================================================================
% I. INTRODUCERE
%==============================================================================
\section{Introducere}

\subsection{Motivația și Contextul Aplicațiilor}

Recunoașterea emoțiilor faciale reprezintă un subdomeniu esențial al \textit{affective computing}, cu aplicații în interfețe om-calculator, educație adaptivă, asistență medicală, analiză comportamentală și sisteme de recomandare. Într-un context academic și practic, aceste aplicații necesită modele robuste, explicabile și eficiente computațional, capabile să funcționeze pe imagini de rezoluție redusă și în condiții variabile de iluminare sau expresivitate.

În proiectul de față, ne concentrăm pe clasificarea a șapte emoții de bază (\textit{angry}, \textit{disgust}, \textit{fear}, \textit{happy}, \textit{neutral}, \textit{sad}, \textit{surprise}) din imagini 48\,$\times$\,48 alb-negru. Dimensiunea redusă a imaginii și variabilitatea expresiilor introduc provocări specifice, care justifică utilizarea unor metode bine înțelese și reproductibile, precum HOG/LBP + SVM, în detrimentul unor rețele profunde dificil de calibrat fără date și resurse suficiente.

\subsection{Problema de Cercetare}

Problema principală este estimarea corectă a emoției faciale pe baza unor imagini mici, cu detalii limitate și potențial zgomot de etichetare. Aceasta implică:

\begin{itemize}
    \item \textbf{Robustețe la variații locale}: expresiile se bazează pe diferențe fine la nivel de regiuni faciale (sprâncene, gură, ochi).
    \item \textbf{Separarea claselor apropiate}: emoții precum \textit{sad} și \textit{neutral} sau \textit{fear} și \textit{surprise} pot fi vizual similare.
    \item \textbf{Dezechilibru de clase}: distribuția etichetelor este neuniformă, ceea ce poate favoriza clasele majoritare.
\end{itemize}

\subsection{Contribuțiile Proiectului}

Contribuțiile principale sunt:

\begin{enumerate}
    \item Un pipeline complet, reproductibil, pentru clasificarea emoțiilor faciale, bazat pe HOG/LBP + SVM și CNN-uri moderne.
    \item O arhitectură software integrată (backend FastAPI + frontend React) care permite atât inferență, cât și vizualizarea progresului și a rapoartelor.
    \item Un flux de antrenare CNN cu transfer learning (ResNet/EfficientNet), augmentări MixUp/CutMix și ensembling la inferență.
    \item Mecanism automat de selecție a celui mai bun model și generare a fișierului final \texttt{submission.csv}.
    \item Documentarea clară a alegerilor tehnice, metricilor și limitărilor, în scopul maximizării acurateții la verificare.
\end{enumerate}

\subsection{Structura Documentului}

Secțiunea II prezintă fundamentele teoretice ale recunoașterii emoțiilor faciale și justifică alegerea HOG/LBP + SVM. Secțiunea III descrie arhitectura sistemului și fluxul de date. Secțiunea IV detaliază metodologia experimentală și configurațiile exacte folosite. Secțiunea V discută interpretarea rezultatelor și modul de analiză. Secțiunile VI și VII abordează implicațiile, limitările și direcțiile viitoare. Secțiunea VIII sintetizează concluziile.

%==============================================================================
% II. FUNDAMENTE TEORETICE ȘI STUDIUL LITERATURII
%==============================================================================
\section{Fundamente Teoretice și Studiul Literaturii}

\subsection{Evoluția Recunoașterii Emoțiilor Faciale}

\subsubsection{Psihologie și Modele de Emoție}

Modelele psihologice clasice, precum teoria emoțiilor de bază a lui Ekman, propun existența unui set universal de expresii faciale asociate unor emoții fundamentale \cite{ekman1992}. Sistemul FACS (Facial Action Coding System) descrie expresiile în termeni de \textit{action units}, permițând decompozarea unei expresii în mișcări musculare elementare \cite{facs}. Aceste teorii fundamentează etichetele folosite în dataset-uri moderne de recunoaștere a emoțiilor, inclusiv cele de tip FER.

\subsubsection{Abordări Clasice}

Înaintea valului deep learning, metodele tradiționale se bazau pe:
\begin{itemize}
    \item \textbf{Caracteristici geometrice}: distanțe și unghiuri între repere faciale (ochi, gură, sprâncene).
    \item \textbf{Caracteristici de aspect}: descrieri locale ale texturii sau gradientului (HOG, LBP, Gabor).
\end{itemize}

Aceste metode sunt atractive pentru dataset-uri moderate deoarece sunt interpretabile și au cerințe compute reduse.

\subsubsection{Abordări Deep Learning}

Rețelele convoluționale au devenit dominante prin capacitatea de a învăța reprezentări ierarhice direct din pixeli. Totuși, pentru imagini foarte mici (48\,$\times$\,48) și seturi de antrenare limitate, riscul de supraînvățare este ridicat, iar beneficiile față de metodele clasice nu sunt întotdeauna decisive fără augmentări și infrastructură adecvată \cite{fer2013}.

În acest proiect, deep learning-ul este integrat prin transfer learning cu ResNet și EfficientNet. Aceste arhitecturi oferă un compromis optim între capacitate și cost computațional, iar generalizarea este îmbunătățită prin augmentări MixUp/CutMix și regularizare prin label smoothing.

\subsection{HOG: Histogram of Oriented Gradients}

HOG descrie distribuția orientărilor gradientului de intensitate pe regiuni locale. Procedura standard include:

\begin{enumerate}
    \item Calculul gradientului pe axele \textit{x} și \textit{y}.
    \item Împărțirea imaginii în celule (\textit{cells}) și acumularea orientărilor în histograme.
    \item Gruparea celulelor în blocuri și normalizarea pentru stabilitate la iluminare.
\end{enumerate}

Pentru o imagine \(I\), gradientul este:
\begin{equation}
G_x = I \ast \begin{bmatrix}-1 & 0 & 1\end{bmatrix}, \quad
G_y = I \ast \begin{bmatrix}-1 \\ 0 \\ 1\end{bmatrix}
\end{equation}

Magnitudinea și orientarea se calculează prin:
\begin{equation}
M = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\left(\frac{G_y}{G_x}\right)
\end{equation}

În proiect, HOG este configurat cu \texttt{orientations=9}, \texttt{pixels\_per\_cell=(8,8)}, \texttt{cells\_per\_block=(2,2)} și normalizare \texttt{L2-Hys}, setare clasică recomandată pentru fețe \cite{dalal2005}.

\subsection{LBP: Local Binary Patterns}

LBP codifică textura locală comparând intensitatea fiecărui pixel cu vecinii săi. Pentru un pixel central \(p_c\) și vecini \(p_i\), codul LBP este:
\begin{equation}
\text{LBP}(p_c) = \sum_{i=0}^{P-1} s(p_i - p_c) \cdot 2^i, \quad
s(x) = \begin{cases}1, & x \ge 0 \\ 0, & x < 0\end{cases}
\end{equation}

Folosim varianta \textit{uniform}, cu \(P=8\) și \(R=1\), care reduce dimensionalitatea și este robustă la zgomot \cite{ojala2002}. Histogramarea LBP pe întreaga imagine oferă un descriptor compact, potrivit pentru clasificare cu SVM.

\subsection{SVM: Support Vector Machines}

SVM maximizează marginea dintre clase și este eficient în spații de caracteristici de dimensiune moderată. Pentru clasificare multi-clasă, scikit-learn utilizează strategia \textit{one-vs-one}. Funcția de decizie pentru un exemplu \(x\) este:
\begin{equation}
 f(x) = \text{sign}\left(\sum_{i=1}^{n} \alpha_i y_i K(x_i, x) + b\right)
\end{equation}

În proiect se folosește kernel RBF, cu \(C=10\) și \(\gamma=\text{scale}\), combinație care oferă un compromis între flexibilitate și generalizare \cite{svm1995}.

\subsection{Metrici de Evaluare}

Pentru clasificare, metrica principală este acuratețea:
\begin{equation}
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\end{equation}

În plus, folosim matricea de confuzie și acuratețea pe clasă pentru a evalua performanța pe clase rare (ex. \textit{disgust}), evitând concluzii eronate date de clasele majoritare.

%==============================================================================
% III. ARHITECTURA SISTEMULUI PROPUS
%==============================================================================
\section{Arhitectura Sistemului Propus}

\subsection{Viziune de Ansamblu}

Sistemul este organizat în patru componente principale:
\begin{itemize}
    \item \textbf{Date}: imagini și fișiere CSV pentru antrenare, validare și predicție.
    \item \textbf{Pipeline de modelare}: scripturi pentru antrenare, validare, comparație și selecție.
    \item \textbf{Backend FastAPI}: endpoint-uri pentru inferență, raportare și status.
    \item \textbf{Frontend React}: interfață de control și vizualizare rezultate.
\end{itemize}

\subsection{Implementare Software: Backend și Frontend}

Backend-ul expune rute pentru:
\begin{itemize}
    \item încărcarea modelelor și starea pipeline-ului,
    \item predicții individuale sau batch,
    \item compararea rezultatelor între modele,
    \item acces la rapoarte și metadate.
\end{itemize}

Frontend-ul oferă un panou unic pentru interacțiune, permițând încărcarea imaginilor, vizualizarea predicțiilor și inspecția rapoartelor.

\begin{figure}[htbp]
\centerline{\includegraphics[width=\columnwidth]{Webpage1.png}}
\caption{Interfața React: zonă de predicție și feedback vizual.}
\label{fig:frontend_main}
\end{figure}

\begin{figure}[htbp]
\centerline{\includegraphics[width=\columnwidth]{Webpage2.png}}
\caption{Interfața React: vizualizarea rapoartelor și comparațiilor.}
\label{fig:frontend_reports}
\end{figure}

\subsection{Pipeline de Antrenare și Validare}

Pipeline-ul offline utilizează scripturile din \texttt{scripts/}:
\begin{itemize}
    \item \texttt{train\_model.py}: antrenează un model pe datele de train.
    \item \texttt{validate\_model.py}: validează un model și generează rapoarte.
    \item \texttt{compare\_models.py}: compară mai multe configurații.
    \item \texttt{make\_submission.py --best}: selectează automat modelul cu cea mai bună acuratețe pe \texttt{val} și generează \texttt{submission.csv}.
    \item \texttt{cnn\_train.py}: antrenează un CNN (ResNet18/ResNet50 sau SmallCNN).
    \item \texttt{cnn\_validate.py}: validează un checkpoint CNN pe \texttt{val}.
    \item \texttt{cnn\_make\_submission.py}: generează \texttt{submission.csv} din CNN.
\end{itemize}

\begin{figure}[htbp]
\centerline{\includegraphics[width=\columnwidth]{Schema1.png}}
\caption{Schema pipeline-ului de antrenare, validare și generare a \texttt{submission.csv}.}
\label{fig:pipeline}
\end{figure}

\subsection{Registry și Metadate}

Configurațiile de modele sunt declarate în \texttt{models/registry.json}. Fiecare antrenare produce un fișier \texttt{.joblib} și un fișier \texttt{.meta.json} cu parametri, ordine etichete și data antrenării. Acest mecanism asigură reproducibilitatea și trasabilitatea rezultatelor.

\subsection{Algoritm de Antrenare (Pseudo-cod)}

\begin{algorithm}[htbp]
\caption{Pipeline de antrenare și validare}
\begin{algorithmic}[1]
\STATE Încarcă \texttt{train.csv} și imaginile asociate
\STATE Preprocesează imaginile (grayscale, resize 48\,$\times$\,48)
\STATE Extrage caracteristici HOG/LBP
\STATE Standardizează vectorii de caracteristici
\STATE Antrenează SVM cu kernel RBF
\STATE Evaluează pe \texttt{val.csv} și calculează acuratețea
\STATE Salvează modelul și metadatele
\end{algorithmic}
\end{algorithm}

\subsection{Pipeline CNN și Transfer Learning}

Pentru creșterea acurateții, proiectul include un pipeline CNN bazat pe transfer learning. Modelele ResNet18/ResNet50 și EfficientNet-B0 sunt pre-antrenate pe ImageNet și adaptate pentru cele 7 clase de emoții. Imaginile grayscale sunt convertite la 3 canale pentru compatibilitate, iar preprocesarea urmează normalizarea specifică ImageNet. Antrenarea utilizează augmentări moderate (rotație, translatare, zoom, flip) plus MixUp/CutMix și label smoothing, iar ponderile de clasă contrabalansează dezechilibrul.

CNN-ul este integrat în același backend FastAPI: modelul este încărcat din fișierul \texttt{.pt}, iar predicțiile sunt returnate prin endpoint-urile existente, alături de scoruri softmax pentru fiecare emoție. Pentru scor final, inferența permite ensembling între mai multe checkpoint-uri (ex.: ResNet50 + EfficientNet-B0).

%==============================================================================
% IV. METODOLOGIE EXPERIMENTALĂ
%==============================================================================
\section{Metodologie Experimentală}

\subsection{Configurația Hardware și Software}

Pipeline-ul clasic (HOG/LBP + SVM) rulează eficient pe CPU. Pentru CNN, se utilizează GPU (dacă este disponibil) pentru reducerea timpului de antrenare. Stack-ul software include Python, scikit-learn, scikit-image, PyTorch, FastAPI și un frontend Vite + React.

\subsection{Dataset și Distribuții}

Setul de date este împărțit astfel:
\begin{itemize}
    \item \textbf{Train}: 2000 imagini
    \item \textbf{Val}: 700 imagini
    \item \textbf{Predict}: 300 imagini (fără etichete)
\end{itemize}

Pentru maximizarea scorului final, este posibilă realocarea unei părți din \texttt{val} către \texttt{train} (ex.: păstrarea a 100 de imagini pentru validare rapidă sau utilizarea tuturor celor 2700 de imagini etichetate la antrenarea finală).

Imaginile sunt alb-negru, 48\,$\times$\,48 pixeli, iar etichetele sunt cele șapte emoții de bază. Tabelul \ref{tab:labeldist} evidențiază dezechilibrul de clase.

\begin{table}[htbp]
\caption{Distribuția etichetelor în train și val}
\label{tab:labeldist}
\centering
\begin{tabular}{lrr}
\toprule
\textbf{Emoție} & \textbf{Train} & \textbf{Val} \\
\midrule
angry & 268 & 87 \\
disgust & 27 & 12 \\
fear & 293 & 120 \\
happy & 467 & 173 \\
neutral & 379 & 110 \\
sad & 332 & 119 \\
surprise & 234 & 79 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Preprocesare}

Fiecare imagine este încărcată în grayscale, redimensionată la 48\,$\times$\,48 (dacă este necesar) și convertită la \texttt{float32}. Această uniformizare asigură consistența descriptorilor și stabilitate numerică.

\subsection{Configurația Modelelor}

\textbf{Model HOG + SVM}:
\begin{itemize}
    \item HOG: \texttt{orientations=9}, \texttt{pixels\_per\_cell=(8,8)}, \texttt{cells\_per\_block=(2,2)}, \texttt{block\_norm=L2-Hys}
    \item SVM: \texttt{kernel=rbf}, \texttt{C=10}, \texttt{gamma=scale}
\end{itemize}

\textbf{Model LBP + SVM}:
\begin{itemize}
    \item LBP: \texttt{P=8}, \texttt{R=1}, \texttt{method=uniform}
    \item SVM: \texttt{kernel=rbf}, \texttt{C=10}, \texttt{gamma=scale}
\end{itemize}

\subsection{Augmentări și Regularizare pentru CNN}

Pentru modelele CNN s-au aplicat augmentări standard (rotație, flip, translatare, scalare) și tehnici de regularizare avansate:
\begin{itemize}
    \item \textbf{MixUp} și \textbf{CutMix} pentru creșterea robusteții la variații locale.
    \item \textbf{Label smoothing} pentru stabilizarea predicțiilor și reducerea supraîncrederii.
    \item \textbf{Ponderi pe clase} pentru a trata dezechilibrul (în special clasa \textit{disgust}).
    \item \textbf{Reducere adaptivă a ratei de învățare} și \textbf{early stopping} pentru convergență stabilă.
\end{itemize}

\subsection{Protocol de Evaluare}

Fiecare model este antrenat pe \texttt{train} și evaluat pe \texttt{val}. Scriptul de comparație produce rapoarte în format JSON și Markdown, iar scriptul de generare a predicțiilor selectează automat modelul cu acuratețea cea mai mare pe validare.

%==============================================================================
% V. REZULTATE ȘI ANALIZĂ
%==============================================================================
\section{Rezultate și Analiză}

\subsection{Interpretarea Rapoartelor}

Rapoartele generate includ acuratețea globală, matricea de confuzie și acuratețea pe clasă. Analiza se concentrează pe:
\begin{itemize}
    \item performanța pe clase rare (\textit{disgust}),
    \item confuziile tipice (\textit{sad} vs. \textit{neutral}, \textit{fear} vs. \textit{surprise}),
    \item stabilitatea performanței între modele.
\end{itemize}

\subsection{Discuție asupra Dificultății Claselor}

Clasele cu puține exemple tind să aibă acuratețe mai scăzută și rate mai mari de confuzie. Acest lucru este un efect direct al distribuției dezechilibrate și al rezoluției scăzute. Pentru o interpretare corectă a performanței, este necesară analiza per-clasă, nu doar acuratețea globală.

\subsection{Selectarea Modelului Final}

Modelul final utilizat pentru \texttt{submission.csv} este ales automat pe baza acurateței pe validare. Această strategie reduce riscul de selecție manuală subiectivă și maximizează șansele de generalizare în faza de verificare.

\subsection{Comparație LBP vs. HOG vs. CNN}

Tabelul \ref{tab:compare} sumarizează diferențele observate între modelele clasice și CNN, în funcție de mărimea setului de antrenare disponibil. Valorile pot varia ușor în funcție de seed și split, dar tendința este consistentă: CNN-ul depășește net descriptorii clasici atunci când există resurse GPU și un set mai mare de date.

\begin{table}[htbp]
\caption{Comparație între modele și configurații de date}
\label{tab:compare}
\centering
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Train} & \textbf{Val} & \textbf{Acuratețe} \\
\midrule
LBP + SVM & 2000 & 700 & \textasciitilde 0.32 \\
HOG + SVM & 2000 & 700 & \textasciitilde 0.38 \\
CNN ResNet50 & 2600 & 100 & 0.57 \\
CNN EfficientNet-B0 & 2600 & 100 & \textasciitilde 0.58 \\
Ensemble (R50 + B0) & 2600 & 100 & \textasciitilde 0.59 \\
CNN final (toate datele) & 2700 & 0 & n/a \\
\bottomrule
\end{tabular}
\end{table}

Pentru generarea fișierului final, s-a folosit varianta CNN antrenată pe toate cele 2700 de imagini etichetate, maximizând capacitatea de generalizare către setul de verificare.

\subsection{Eficiență Computațională}

Pipeline-ul clasic (HOG/LBP + SVM) rulează rapid pe CPU și este potrivit pentru inferență locală fără GPU. CNN-urile necesită timp de antrenare mai mare, însă pe GPU costul devine acceptabil, iar acuratețea obținută justifică investiția. Ensembling-ul crește timpul de inferență proporțional cu numărul de modele, dar oferă un câștig constant de performanță pentru scenariul de evaluare finală.

%==============================================================================
% VI. DISCUȚII: IMPLICAȚII ȘI LIMITĂRI
%==============================================================================
\section{Discuții: Implicații și Limitări}

\subsection{Aplicabilitate Practică}

Sistemul poate fi integrat în aplicații de monitorizare a stării emoționale, analiză comportamentală în educație sau suport pentru UX. Datorită cerințelor compute reduse, poate rula local, fără infrastructură specializată.

\subsection{Limitări}

Limitările principale sunt:
\begin{itemize}
    \item \textbf{Rezoluție scăzută}: detalii fine ale expresiei pot fi pierdute.
    \item \textbf{Dezechilibru de clase}: emoțiile rare sunt dificil de învățat.
    \item \textbf{Ambiguitatea expresiilor}: emoțiile reale sunt complexe și nu întotdeauna bine încadrate în clase discrete.
\end{itemize}

\subsection{Considerații Etice}

Recunoașterea emoțiilor implică aspecte de confidențialitate și consimțământ. Un astfel de sistem trebuie folosit transparent, cu informarea utilizatorilor și fără a lua decizii critice exclusiv pe baza predicțiilor automate.

%==============================================================================
% VII. IMPACT TEHNIC ȘI ȘTIINȚIFIC
%==============================================================================
\section{Impact Tehnic și Științific în Inginerie și Data Science}

Proiectul demonstrează că metodele clasice, corect calibrate și evaluate, pot obține performanțe solide pe dataset-uri specifice, oferind un baseline robust și interpretabil. Din perspectiva ingineriei, arhitectura modulară (scripturi + backend + frontend) facilitează reproducibilitatea și testarea sistematică.

%==============================================================================
% VIII. CONCLUZII ȘI DIRECȚII VIITOARE
%==============================================================================
\section{Concluzii și Direcții Viitoare}

\subsection{Contribuții Principale}

Lucrarea a prezentat un pipeline complet și reproductibil pentru recunoașterea emoțiilor faciale, integrând atât metode clasice (HOG/LBP + SVM), cât și CNN-uri moderne (ResNet, EfficientNet) și ensembling într-un sistem software coerent. Selecția automată a modelului pe validare și generarea \texttt{submission.csv} asigură o strategie obiectivă orientată către acuratețe.

\subsection{Direcții Viitoare}

Direcțiile viitoare includ:
\begin{itemize}
    \item augmentare de date pentru îmbunătățirea claselor rare,
    \item extinderea ensemble-urilor cu backbone-uri mai puternice și calibrare probabilistică,
    \item validare k-fold pentru estimări mai stabile.
\end{itemize}

\subsection{Reflecții Finale}

Proiectul evidențiază faptul că, într-un context academic orientat pe acuratețe și reproducibilitate, soluțiile clasice rămân relevante și oferă un punct de referință solid pentru viitoare îmbunătățiri.

%==============================================================================
% REFERENCES
%==============================================================================
\begin{thebibliography}{00}
\bibitem{ekman1992} P. Ekman, \textit{An Argument for Basic Emotions}, Cognition and Emotion, 1992.
\bibitem{facs} P. Ekman and W. V. Friesen, \textit{Facial Action Coding System (FACS)}, Consulting Psychologists Press, 1978.
\bibitem{dalal2005} N. Dalal and B. Triggs, \textit{Histograms of Oriented Gradients for Human Detection}, CVPR, 2005.
\bibitem{ojala2002} T. Ojala, M. Pietikäinen, and T. Mäenpää, \textit{Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns}, IEEE TPAMI, 2002.
\bibitem{svm1995} C. Cortes and V. Vapnik, \textit{Support-Vector Networks}, Machine Learning, 1995.
\bibitem{fer2013} I. Goodfellow et al., \textit{Challenges in Representation Learning: A Report on Three Machine Learning Contests}, NeurIPS Workshop (FER2013), 2013.
\end{thebibliography}

\end{document}
