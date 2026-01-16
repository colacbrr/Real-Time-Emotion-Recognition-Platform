# Schema arhitectura locala

## Flux general

```
[USER / Frontend React]
        |
        v
[FastAPI Backend]
        |
        +--> [Pipeline ASPICV]
        |           |
        |           v
        |   [Preprocesare imagini 48x48]
        |           |
        |           v
        |    [HOG / LBP Features]
        |           |
        |           v
        |      [SVM Model]
        |           |
        |           v
        |      [Eticheta + Scor]
        |
        +--> [Pipeline MIR (optional)]
        |           |
        |           v
        |     [CLIP Encoder]
        |           |
        |           v
        |      [FAISS Index]
        |           |
        |           v
        |   [Rezultate + Captions]
        |           |
        |           v
        |   [RAG Prompt Builder]
        |           |
        |           v
        |      [Ollama (LLM)]
        |           |
        |           v
        |   [Output Explicativ]
        |
        v
[Frontend React]
```

## Detaliu: predictie (imagine -> eticheta)

```
Imagine (48x48, grayscale)
  -> normalizare / resize
  -> extractie HOG sau LBP
  -> vector caracteristici
  -> SVM (predictie)
  -> eticheta emotie + scor
  -> trimite catre UI
```

## Detaliu: validare (val.csv)

```
val.csv + imagini
  -> build features (HOG/LBP)
  -> predictii SVM
  -> acuratete globala
  -> matrice de confuzie
  -> acuratete pe clasa
  -> raport JSON/MD
```

## Detaliu: MIR (text -> imagini + explicatii)

```
Query text
  -> CLIP text encoder (embedding)
  -> FAISS search (top-k)
  -> rezultate: fisiere + scoruri + caption
  -> RAG prompt + Ollama (opțional)
  -> raspuns explicativ
  -> trimite catre UI
```

## Endpoint-uri relevante

- `/predict` : imagine -> eticheta emotie
- `/compare` : comparatie modele pe setul de validare
- `/reports` : rapoarte generate (validare/comparatie)
- `/status` : stare backend
- `/events` : progres incarcare modele
- `/search` : cautare text -> imagini (MIR)
- `/rag` : cautare + raspuns LLM (MIR)
- `/explain` : raspuns LLM pe rezultate deja obtinute (MIR)
- `/ollama/models` : modele disponibile (MIR)
