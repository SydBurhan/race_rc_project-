"""
src/inference.py
================
Unified inference API for the Streamlit UI and corpus evaluation.

Public functions:
    predict_answer(article, question, options) -> (best_label, confidence)
    generate_question(article, correct_answer) -> str
    generate_distractors(article, question, correct_answer) -> list[str] (len 3)
    generate_hints(article, question, correct_answer) -> list[str] (len 3)

Models are loaded lazily once per process via @lru_cache.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import joblib
import numpy as np

from src.template_generator import (
    generate_questions, rank_questions, load_ranker, OHE_PATH as TG_OHE,
)
from src.model_b_train import (
    generate_distractors as _b_generate_distractors,
    generate_hints as _b_generate_hints,
    load_word2vec, load_vectorizer,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_A_TRAD = MODELS_DIR / "model_a" / "traditional"
OHE_PATH = MODEL_A_TRAD / "ohe_vectorizer.pkl"


# ══════════════════════════════════════════════════════════════════════════════
#  Cached loaders
# ══════════════════════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=1)
def _load_ohe():
    if not OHE_PATH.exists():
        raise FileNotFoundError(f"OHE vectorizer missing: {OHE_PATH}. "
                                f"Run src/preprocessing.py first.")
    return joblib.load(OHE_PATH)


@functools.lru_cache(maxsize=1)
def _load_ensemble():
    """Return (lr, svm, nb) — soft-vote ensemble. Returns None if missing."""
    paths = {n: MODEL_A_TRAD / f"{n}.pkl" for n in ("lr", "svm", "nb")}
    if not all(p.exists() for p in paths.values()):
        return None
    return tuple(joblib.load(p) for p in (paths["lr"], paths["svm"], paths["nb"]))


@functools.lru_cache(maxsize=1)
def _load_question_ranker():
    return load_ranker()


@functools.lru_cache(maxsize=1)
def _load_w2v_safe():
    try:
        return load_word2vec()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _featurise_option(article: str, question: str, option: str, vectorizer):
    combined = _clean(article) + " " + _clean(question) + " " + _clean(option)
    X_ohe = vectorizer.transform([combined])
    # Lexical features (must match preprocessing.build_lexical_features ordering)
    a_words = _clean(article).split()
    q_words = _clean(question).split()
    o_words = _clean(option).split()
    a_set = set(a_words)
    lex = np.array([[
        len(a_words),
        len(q_words),
        len(o_words),
        sum(1 for w in o_words if w in a_set),
        1.0 if option and option in article else 0.0,
        0.0,  # option_position unknown at single-option time → 0
    ]], dtype=np.float32)
    import scipy.sparse as sp
    return sp.hstack([X_ohe, sp.csr_matrix(lex)], format="csr")


# ══════════════════════════════════════════════════════════════════════════════
#  predict_answer — soft-vote ensemble over 4 options
# ══════════════════════════════════════════════════════════════════════════════

def predict_answer(article: str, question: str, options: dict) -> tuple[str, float]:
    if not article or not article.strip():
        raise ValueError("article must be a non-empty string")
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(options, dict) or not options:
        raise ValueError("options must be a non-empty dict {label: text}")

    ensemble = _load_ensemble()
    if ensemble is None:
        # Fallback: pick option with highest cosine similarity to article
        vec = _load_ohe()
        from sklearn.metrics.pairwise import cosine_similarity
        article_vec = vec.transform([_clean(article)])
        scores = {}
        for label, text in options.items():
            ov = vec.transform([_clean(text)])
            scores[label] = float(cosine_similarity(article_vec, ov).ravel()[0])
        best = max(scores, key=scores.get)
        return best, scores[best]

    lr, svm, nb = ensemble
    vec = _load_ohe()
    scores = {}
    for label, text in options.items():
        X = _featurise_option(article, question, str(text), vec)
        p = (lr.predict_proba(X)[0, 1]
             + svm.predict_proba(X)[0, 1]
             + nb.predict_proba(X)[0, 1]) / 3.0
        scores[label] = float(p)
    best = max(scores, key=scores.get)
    return best, scores[best]


# ══════════════════════════════════════════════════════════════════════════════
#  generate_question — template + SVM ranker
# ══════════════════════════════════════════════════════════════════════════════

def generate_question(article: str, correct_answer: str) -> str:
    if not article or not article.strip():
        raise ValueError("article must be a non-empty string")
    if not correct_answer or not str(correct_answer).strip():
        raise ValueError("correct_answer must be a non-empty string")

    vec = _load_ohe()
    cands = generate_questions(article, str(correct_answer), vec, n=3)
    if not cands:
        return ""
    ranker = _load_question_ranker()
    ranked = rank_questions(cands, article, str(correct_answer), vec, ranker)
    return ranked[0]["question"] if ranked else cands[0]["question"]


# ══════════════════════════════════════════════════════════════════════════════
#  generate_distractors — TF-IDF MMR + frequency + W2V
# ══════════════════════════════════════════════════════════════════════════════

def generate_distractors(article: str, question: str,
                          correct_answer: str) -> list[str]:
    if not article or not article.strip():
        raise ValueError("article must be a non-empty string")
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    vec = _load_ohe()
    w2v = _load_w2v_safe()
    out = _b_generate_distractors(article, question, str(correct_answer),
                                   vectorizer=vec, w2v_model=w2v, n=3)
    # Pad to exactly 3 with placeholders if generator returned fewer
    while len(out) < 3:
        out.append(f"option_{len(out) + 1}")
    return out[:3]


# ══════════════════════════════════════════════════════════════════════════════
#  generate_hints — graduated 3-tier
# ══════════════════════════════════════════════════════════════════════════════

def generate_hints(article: str, question: str,
                    correct_answer: str = "") -> list[str]:
    if not article or not article.strip():
        raise ValueError("article must be a non-empty string")
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    vec = _load_ohe()
    out = _b_generate_hints(article, question, str(correct_answer),
                             vectorizer=vec)
    while len(out) < 3:
        out.append("(no further hint available)")
    return out[:3]
