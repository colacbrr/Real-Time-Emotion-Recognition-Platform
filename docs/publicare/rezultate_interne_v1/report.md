# Rezultate interne v1 — exploratorii

Aceleași 536 imagini de test intern; 1605 train și 536 validare. Nu sunt rezultatele testului oficial și nu se afirmă că testul intern este istoric neatins.

| Model | Accuracy | Macro-F1 | Fit CPU (s) | Test batch CPU (s) |
|---|---:|---:|---:|---:|
| baseline | 0.238806 | 0.055077 | 0.006 | 0.000 |
| hog | 0.386194 | 0.357925 | 1.600 | 0.482 |
| cnn | 0.404851 | 0.317053 | 25.077 | 8.041 |

## Recall pe clase

| Clasă | Suport test | baseline | hog | cnn |
|---|---:|---:|---:|---:|
| angry | 71 | 0.000000 | 0.267606 | 0.450704 |
| disgust | 8 | 0.000000 | 0.250000 | 0.000000 |
| fear | 82 | 0.000000 | 0.146341 | 0.195122 |
| happy | 128 | 1.000000 | 0.703125 | 0.742188 |
| neutral | 97 | 0.000000 | 0.371134 | 0.340206 |
| sad | 89 | 0.000000 | 0.303371 | 0.123596 |
| surprise | 61 | 0.000000 | 0.344262 | 0.491803 |

## Limite

- Original labeled pool previously explored; test is not historically untouched
- No subject metadata; person independence not established
- Near-duplicate screening incomplete and single assistant review
- Source dataset license and official test labels unresolved
- Remaining within-split copies retained with groups; metrics are per image and not per unique photograph
- Single seed and frozen-backbone configuration; no general claim that one model family is superior

Timpii includ costurile precizate în results.json; nu sunt măsurători de latență a serviciului sau dovadă real-time.

Predicțiile sunt păstrate local în outputs/internal-runs-v1; hash-urile și traseul lor sunt în results.json. Matricile CSV au rânduri=adevăr și coloane=predicție. Nu s-au modificat hiperparametrii după consultarea testului.
