# Car-ing is Sharing — LLM Chatbot Prototype

A multi-task NLP prototype built for **Car-ing is Sharing**, an auto dealership company, using pre-trained Hugging Face LLMs to handle diverse customer inquiries.

---

## Project Overview

The CTO commissioned this prototype to explore how Large Language Models can automate and enhance customer-facing and agent-support tasks. The solution processes car reviews and demonstrates four core NLP capabilities.

---

## Tasks Implemented

### Task 1 — Sentiment Classification
- **Model:** `distilbert-base-uncased-finetuned-sst-2-english`
- **What it does:** Classifies each of the 5 car reviews as POSITIVE or NEGATIVE
- **Output variables:**
  - `predicted_labels` — raw model output
  - `predictions` — binary list `{0, 1}` mapped from labels
  - `accuracy_result` — classification accuracy score
  - `f1_result` — F1 score of predictions

### Task 2 — English-to-Spanish Translation + BLEU Score
- **Model:** `Helsinki-NLP/opus-mt-en-es`
- **What it does:** Translates the first two sentences of the first review into Spanish, then evaluates quality against reference translations
- **Output variables:**
  - `translated_review` — Spanish translation text
  - `bleu_score` — dictionary of BLEU score results via `evaluate.load("bleu").compute()`

### Task 3 — Extractive Question Answering
- **Model:** `deepset/minilm-uncased-squad2`
- **What it does:** Answers the question *"What did he like about the brand?"* using the 2nd review as context
- **Output variables:**
  - `question` — the input question
  - `context` — the 2nd review text
  - `answer` — extracted answer text

### Task 4 — Text Summarization
- **Model:** `sshleifer/distilbart-cnn-12-6`
- **What it does:** Summarizes the last review into approximately 50–55 tokens
- **Output variable:**
  - `summarized_text` — the generated summary

---

## Project Structure

```
/data/
  ├── car_reviews.csv              # Dataset of 5 car reviews with sentiment labels
  └── reference_translations.txt  # Reference Spanish translations for BLEU scoring

notebook.ipynb                     # Main Jupyter notebook with all tasks
```

---

## ⚙️ Setup & Dependencies

```python
# Cell 1 — Imports
import subprocess
subprocess.run(["pip", "install", "sacrebleu"], capture_output=True)

import pandas as pd
import torch
from transformers import logging, pipeline
import evaluate
from sklearn.metrics import accuracy_score, f1_score

logging.set_verbosity(logging.WARNING)
```

---

## 📊 Dataset

The `car_reviews.csv` file is semicolon-separated with two columns:

| Column   | Description                        |
|----------|------------------------------------|
| `Review` | Full text of the car review        |
| `Class`  | Ground truth label: POSITIVE / NEGATIVE |

---

## Key Implementation Notes

- CSV uses **semicolon** (`;`) as delimiter — load with `pd.read_csv(..., sep=";")`
- BLEU score uses the `evaluate` library's `.compute()` method which returns a **dictionary**
- All models are loaded via Hugging Face `pipeline()` for clean, unified inference
- Translation extracts exactly the **first two sentences** using `.split(".")[:2]`

---

## Tools & Libraries

| Library | Purpose |
|---------|---------|
| `transformers` | Hugging Face pipeline and model loading |
| `pandas` | CSV data loading and manipulation |
| `torch` | Backend tensor computation |
| `evaluate` | BLEU score computation |
| `sklearn` | Accuracy and F1 score metrics |

---

*Built as a prototype for Car-ing is Sharing — powered by Hugging Face Transformers.*
