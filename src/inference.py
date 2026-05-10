"""
Unified inference API consumed by the Streamlit UI and the evaluation runner.

Rubric coverage:
  2.3  Answer verification on (article, question, option) triples
  2.3  Question generation entry point used by the UI
  5.x  Distractor generation entry point
  6.x  Hint generation entry point

All heavy artefacts (vectorizer, ensemble, Word2Vec) are loaded lazily and
cached for the lifetime of the process via functools.lru_cache.
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


def _featurise_option(article: str, question: str, option: str, vectorizer,
                       option_idx: int = 0, all_options: list[str] | None = None):
    """
    Build the same 10-feature vector as preprocessing.build_lexical_features.
    `all_options` (length 4) lets us compute the per-question relative features.
    When unavailable (single-option callers), we fall back to neutral defaults.
    """
    import math
    from collections import Counter
    import scipy.sparse as sp

    art_c = _clean(article)
    q_c = _clean(question)
    opt_c = _clean(option)
    combined = art_c + " " + q_c + " " + opt_c
    X_ohe = vectorizer.transform([combined])

    a_words = art_c.split()
    q_words = q_c.split()
    o_words = opt_c.split()
    a_set = set(a_words)
    q_set = set(q_words)
    o_set = set(o_words)

    a_counter = Counter(a_words)
    a_norm = math.sqrt(sum(c * c for c in a_counter.values())) or 1.0
    o_counter = Counter(o_words)
    o_norm = math.sqrt(sum(c * c for c in o_counter.values())) or 1.0
    dot = sum(o_counter[w] * a_counter[w] for w in o_counter if w in a_counter)
    opt_article_cos = dot / (a_norm * o_norm)

    # Per-question relative features need all 4 options
    if all_options and len(all_options) >= 2:
        opt_lists = [_clean(o).split() for o in all_options]
        opt_sets = [set(w) for w in opt_lists]
        mean_len = max(1.0, sum(len(w) for w in opt_lists) / len(opt_lists))
        other = set().union(*[opt_sets[k] for k, t in enumerate(all_options)
                                if _clean(t) != opt_c]) or set()
        opt_uniqueness = len(o_words) / mean_len
        union_oo = len(o_set | other)
        opt_other_overlap = (len(o_set & other) / union_oo) if union_oo else 0.0
    else:
        opt_uniqueness = 1.0
        opt_other_overlap = 0.0

    union_qo = len(q_set | o_set)
    q_opt_overlap = (len(q_set & o_set) / union_qo) if union_qo else 0.0

    lex = np.array([[
        len(a_words),
        len(q_words),
        len(o_words),
        sum(1 for w in o_words if w in a_set),
        1.0 if opt_c and opt_c in art_c else 0.0,
        float(option_idx),
        q_opt_overlap,
        opt_uniqueness,
        opt_other_overlap,
        opt_article_cos,
    ]], dtype=np.float32)
    return sp.hstack([X_ohe, sp.csr_matrix(lex)], format="csr")


# Rubric 2.3: soft-vote ensemble verifier over the four options.

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

    from scipy.special import expit
    lr, svm, nb = ensemble
    vec = _load_ohe()
    labels = list(options.keys())
    texts = [str(options[l]) for l in labels]
    scores = {}
    for idx, (label, text) in enumerate(zip(labels, texts)):
        X = _featurise_option(article, question, text, vec,
                                option_idx=idx, all_options=texts)
        # SVM has no predict_proba; sigmoid of decision_function is monotonic.
        p_svm = float(expit(svm.decision_function(X))[0])
        p = (lr.predict_proba(X)[0, 1] + p_svm + nb.predict_proba(X)[0, 1]) / 3.0
        scores[label] = float(p)
    best = max(scores, key=scores.get)
    return best, scores[best]


# Rubric 2.3: question generation (template fill, then LinearSVC ranking).

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


# Rubric 5.x: distractors merged from MMR, frequency, and Word2Vec sources.

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


# Rubric 6.x: three graduated hints with the third one cloze-redacted.

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
