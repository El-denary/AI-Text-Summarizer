"""
summarizer.py
-------------
Two summarization strategies:
  - Extractive : TF-IDF weighted sentence scoring
  - Abstractive: facebook/bart-large-cnn via HuggingFace pipeline
"""

import re
import math
import streamlit as st
from collections import defaultdict

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from transformers import pipeline

# ── NLTK downloads (run once) ──────────────────────────────────────────────────
_NLTK_RESOURCES = {
    "punkt":     "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab/english/",
    "stopwords": "corpora/stopwords",
}

for pkg, path in _NLTK_RESOURCES.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))


# ── Helpers ────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove URLs, extra whitespace, and non-ASCII characters."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_words(text: str) -> int:
    return len(text.split())


def _tokenize_words(sentence: str) -> list[str]:
    """Lowercase, tokenize, remove stopwords and punctuation."""
    words = word_tokenize(sentence.lower())
    return [w for w in words if w.isalpha() and w not in STOP_WORDS]


# ── Extractive: TF-IDF sentence scoring ───────────────────────────────────────

def _compute_tfidf(sentences: list[str]) -> dict[str, float]:
    """
    Compute a TF-IDF score for every unique word across all sentences.
    Returns word → tfidf_score mapping.
    """
    N = len(sentences)
    tf: dict[int, dict[str, float]] = {}
    df: dict[str, int] = defaultdict(int)

    for i, sent in enumerate(sentences):
        words = _tokenize_words(sent)
        total = len(words) or 1
        freq: dict[str, int] = defaultdict(int)
        for w in words:
            freq[w] += 1
        tf[i] = {w: c / total for w, c in freq.items()}
        for w in set(words):
            df[w] += 1

    tfidf_global: dict[str, float] = {}
    for i, sent_tf in tf.items():
        for w, score in sent_tf.items():
            idf = math.log((N + 1) / (df[w] + 1)) + 1
            tfidf_global[w] = tfidf_global.get(w, 0) + score * idf

    return tfidf_global


def _sentence_scores(sentences: list[str], tfidf: dict[str, float]) -> list[float]:
    """Score each sentence by averaging TF-IDF weights of its words."""
    scores = []
    for sent in sentences:
        words = _tokenize_words(sent)
        score = sum(tfidf.get(w, 0) for w in words) / (len(words) or 1)
        scores.append(score)
    return scores


def extractive_summary(text: str, num_sentences: int = 3) -> str:
    """
    Extract the top-N most informative sentences using TF-IDF scoring,
    returned in their original reading order to preserve narrative flow.
    """
    text = clean_text(text)
    sents = sent_tokenize(text)

    # Never request more sentences than exist
    num_sentences = min(num_sentences, len(sents))

    tfidf  = _compute_tfidf(sents)
    scores = _sentence_scores(sents, tfidf)

    ranked  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_idx = sorted(ranked[:num_sentences])

    return " ".join(sents[i] for i in top_idx)


# ── Abstractive: BART ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_bart():
    """Load BART once and cache it across Streamlit reruns."""
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        framework="pt",
        device=-1       # CPU; set to 0 for GPU
    )


def abstractive_summary(text: str, max_len: int = 130, min_len: int = 40) -> str:
    """
    Generate a fluent abstractive summary using facebook/bart-large-cnn.
    Input is truncated to 900 words to stay safely within BART's 1024-token limit.
    """
    text = clean_text(text)

    words = text.split()
    if len(words) > 900:
        text = " ".join(words[:900])

    summarizer = _load_bart()
    result = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        truncation=True
    )
    return result[0]["summary_text"]
