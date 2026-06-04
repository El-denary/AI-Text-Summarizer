# 📰 AI Text Summarizer

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-BART-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/facebook/bart-large-cnn)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![NLTK](https://img.shields.io/badge/NLTK-3.8+-4A90D9?style=flat-square)](https://nltk.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

> Dual-mode NLP summarizer that produces both an **extractive** (TF-IDF) and an **abstractive** (BART transformer) summary from any text — with a clean dark Streamlit UI, compression stats, and an adjustable sentence slider.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Models & Techniques](#-models--techniques)
- [Getting Started](#-getting-started)
- [Dependencies](#-dependencies)
- [Design Decisions](#-design-decisions)

---

## 🔍 Overview

Summarization is one of the most practical NLP tasks — yet most tools give you only one approach. This project implements **both paradigms** side by side:

- **Extractive** summarization selects the most important *existing* sentences using TF-IDF scoring — fast, transparent, zero hallucinations.
- **Abstractive** summarization uses `facebook/bart-large-cnn` to *generate new sentences* that capture the meaning — more fluent, more concise, reads like a human wrote it.

The app also shows compression stats (original vs. summary word counts and reduction %) alongside each result.

---

## ⚙️ How It Works

```
Raw text input
    ↓  clean (URLs, whitespace, non-ASCII)
    ↓
    ├── EXTRACTIVE PATH ──────────────────────────────────────
    │       ↓  sentence tokenize (NLTK)
    │       ↓  word tokenize + stopword removal per sentence
    │       ↓  compute TF-IDF weights for every word
    │       ↓  score each sentence (mean TF-IDF of its words)
    │       ↓  rank → pick top-N → restore original order
    │       → Extractive summary
    │
    └── ABSTRACTIVE PATH ─────────────────────────────────────
            ↓  truncate to 900 words (BART's 1024-token limit)
            ↓  facebook/bart-large-cnn pipeline (HuggingFace)
            → Abstractive summary
```

---

## 📁 Project Structure

```
├── app.py              # Streamlit web UI
├── summarizer.py       # Extractive + abstractive logic
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🔬 Methodology

### 1. Text Cleaning

Strips URLs, non-ASCII characters, and extra whitespace before any processing.

### 2. Extractive — TF-IDF Sentence Scoring

- Tokenizes the text into sentences using NLTK `sent_tokenize`.
- For each sentence, tokenizes words, removes stopwords, and computes per-sentence TF (term frequency).
- Computes IDF (inverse document frequency) across all sentences, treating each sentence as a "document".
- Scores every sentence by averaging the TF-IDF weights of its content words.
- Selects the top-N highest-scoring sentences and returns them in their **original reading order** to preserve narrative flow.
- N is controlled by the slider in the UI (1–6 sentences).

### 3. Abstractive — BART Transformer

- Uses `facebook/bart-large-cnn`, a seq2seq transformer pre-trained on CNN/DailyMail news data — the standard benchmark for news summarization.
- Input is truncated to 900 words to stay safely within BART's 1024-token context window.
- Output length is controlled via `max_length=130` and `min_length=40`.
- Model is loaded once and cached with `@st.cache_resource` so subsequent runs are instant.

### 4. Streamlit UI

- Dark modern interface.
- Slider to control extractive summary length (1–6 sentences).
- Compression stats displayed as stat cards (original words, extractive words, abstractive words, % reduction).
- Both summaries shown in labeled cards with color-coded badges indicating which technique was used.

---

## 🏆 Models & Techniques

| Technique | Method | Speed | Faithfulness | Fluency |
|---|---|---|---|---|
| **Extractive** | TF-IDF + sentence ranking | ⚡ Instant | ✅ 100% (exact quotes) | ⚠️ Can feel choppy |
| **Abstractive** | facebook/bart-large-cnn | 🕐 ~5–15s (CPU) | ✅ Generally accurate | ✅ Natural, fluent |

> **Why BART?** `facebook/bart-large-cnn` was fine-tuned specifically on the CNN/DailyMail news summarization dataset. It consistently produces high-quality factual summaries compared to general-purpose language models.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/El-denary/news-text-summarizer.git
cd news-text-summarizer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Paste any text, adjust the extractive length slider, and click **Summarize**.

> **Note:** On the first run, HuggingFace will download `facebook/bart-large-cnn` (~1.6 GB). This happens once and is cached locally.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.32 | Web UI |
| `transformers` | ≥ 4.40 | BART model |
| `torch` | ≥ 2.2 | PyTorch backend for BART |
| `nltk` | ≥ 3.8 | Sentence & word tokenization |
| `numpy` | ≥ 1.26 | Numerical operations |

---

## 💡 Design Decisions

**Why TF-IDF over pure frequency counting for extractive?**
Raw word frequency favors common words that survive stopword removal (e.g., "said", "year"). TF-IDF down-weights words that appear in many sentences and up-weights words unique to a few, which are usually the most topically informative.

**Why restore original sentence order in extractive output?**
Ranking selects the best sentences, but returning them in ranked order breaks logical flow. Sorting selected indices by their original position produces a summary that reads like a coherent paragraph.

**Why cache the BART model with `@st.cache_resource`?**
BART is ~1.6 GB. Loading it on every Streamlit rerun would take 10–20 seconds per click. `@st.cache_resource` loads it once per app session and reuses it across all requests.

**Why truncate to 900 words instead of 1024 tokens?**
BART's hard limit is 1024 *tokens*, not words. Tokenization expands words via subword splitting, so 1024 words can easily exceed 1024 tokens. 900 words is a safe proxy that avoids truncation errors without losing meaningful content.


