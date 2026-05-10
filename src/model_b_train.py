"""
src/model_b_train.py
====================
Model B — Distractor & Hint Generation (traditional ML only).

Distractor strategies (combined into final list of 3):
  1. TF-IDF / OHE cosine + MMR diversity (extractive from passage)
  2. Frequency-based substitution (top content words)
  3. Word2Vec nearest-neighbours (gensim, pre-trained, optional & cached)

Hint generation:
  - Splits article into sentences
  - Scores each sentence with a trained Logistic Regression hint scorer
    (features: cos sim to question, cos sim to answer, keyword overlap,
     sentence position, sentence length); falls back to cosine ranking
     when the model is missing.
  - Returns 3 graduated hints: general -> specific -> near-explicit
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_A_TRAD = ROOT / "models" / "model_a" / "traditional"
MODEL_B_TRAD = ROOT / "models" / "model_b" / "traditional"
OHE_PATH = MODEL_A_TRAD / "ohe_vectorizer.pkl"
HINT_SCORER_PATH = MODEL_B_TRAD / "hint_scorer.pkl"
W2V_CACHE_PATH = MODEL_B_TRAD / "word2vec_kv.bin"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
N_DISTRACTORS = 3
N_HINTS = 3
MMR_DIVERSITY = 0.6
MIN_SENTENCE_WORDS = 5
NGRAM_SIZES = [1, 2, 3]
POOL_SIZE = 200

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "have", "has", "had", "do", "does", "did", "will", "would",
    "should", "could", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "they", "them", "their", "we", "our",
    "you", "your", "he", "she", "his", "her", "i", "my", "not", "no",
    "so", "if", "then", "than", "there", "here", "when", "where", "which",
    "who", "what", "how", "why", "all", "more", "very", "just", "also",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [s.strip() for s in raw if len(s.split()) >= MIN_SENTENCE_WORDS]


def _cosine(texts_a: list[str], texts_b: list[str], vectorizer) -> np.ndarray:
    return cosine_similarity(vectorizer.transform(texts_a),
                              vectorizer.transform(texts_b))


def load_vectorizer(path: Path = OHE_PATH):
    if not path.exists():
        log.error("Vectorizer not found: %s. Run preprocessing.py first.", path)
        sys.exit(1)
    return joblib.load(path)


# ══════════════════════════════════════════════════════════════════════════════
#  Distractor — N-gram + TF-IDF/OHE cosine + MMR
# ══════════════════════════════════════════════════════════════════════════════

def _extract_candidate_phrases(article: str, correct_answer: str) -> list[str]:
    tokens = _tokenise(article)
    correct_tok = _tokenise(correct_answer)
    correct_str = " ".join(correct_tok)
    counter: Counter = Counter()
    for n in NGRAM_SIZES:
        for i in range(len(tokens) - n + 1):
            gram = tokens[i: i + n]
            phrase = " ".join(gram)
            if n == 1 and gram[0] in STOPWORDS:
                continue
            if phrase in correct_str or correct_str in phrase:
                continue
            counter[phrase] += 1
    return [p for p, _ in counter.most_common(POOL_SIZE)]


def _mmr_select(scores: np.ndarray, inter_sim: np.ndarray, n: int,
                 diversity: float = MMR_DIVERSITY) -> list[int]:
    remaining = list(range(len(scores)))
    selected: list[int] = []
    for _ in range(min(n, len(remaining))):
        if not selected:
            best = max(remaining, key=lambda i: scores[i])
        else:
            def _score(i):
                redundancy = max(inter_sim[i, j] for j in selected)
                return scores[i] - diversity * redundancy
            best = max(remaining, key=_score)
        selected.append(best)
        remaining.remove(best)
    return selected


def _tfidf_mmr_distractors(article: str, correct_answer: str,
                            vectorizer, n: int = N_DISTRACTORS) -> list[str]:
    candidates = _extract_candidate_phrases(article, correct_answer)
    if not candidates:
        return []
    sim_to_ans = _cosine(candidates, [correct_answer], vectorizer).ravel()
    inter = _cosine(candidates, candidates, vectorizer)
    idx = _mmr_select(sim_to_ans, inter, n)
    return [candidates[i] for i in idx]


# ══════════════════════════════════════════════════════════════════════════════
#  Distractor — frequency-based substitution
# ══════════════════════════════════════════════════════════════════════════════

def frequency_substitution_distractors(article: str, correct_answer: str,
                                        top_n: int = 3) -> list[str]:
    """
    Returns top_n high-frequency content words from the article that are NOT
    substrings of the correct answer. Words are kept with original case so
    proper-noun-like tokens (capitalised) are preferred candidates.
    """
    raw_tokens = re.findall(r"[A-Za-z]{3,}", str(article))
    correct_lc = str(correct_answer).lower()

    freq: Counter = Counter()
    for tok in raw_tokens:
        low = tok.lower()
        if low in STOPWORDS:
            continue
        if low in correct_lc or correct_lc in low:
            continue
        freq[tok] += 1

    # Prefer proper-noun-like (capitalised) and frequent words
    ranked = sorted(freq.items(),
                    key=lambda kv: (-kv[1], 0 if kv[0][0].isupper() else 1))
    out: list[str] = []
    seen_lc: set[str] = set()
    for tok, _ in ranked[:50]:
        if tok.lower() in seen_lc:
            continue
        seen_lc.add(tok.lower())
        out.append(tok)
        if len(out) >= top_n:
            break
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Distractor — Word2Vec nearest neighbours
# ══════════════════════════════════════════════════════════════════════════════

_W2V_MODEL = None


def load_word2vec():
    """Lazily load gensim's pre-trained Word2Vec (Google News 300d)."""
    global _W2V_MODEL
    if _W2V_MODEL is not None:
        return _W2V_MODEL

    MODEL_B_TRAD.mkdir(parents=True, exist_ok=True)
    if W2V_CACHE_PATH.exists():
        try:
            from gensim.models import KeyedVectors
            log.info("Loading cached Word2Vec from %s ...", W2V_CACHE_PATH)
            _W2V_MODEL = KeyedVectors.load(str(W2V_CACHE_PATH))
            return _W2V_MODEL
        except Exception as e:
            log.warning("Cache load failed: %s; will redownload.", e)

    try:
        import gensim.downloader as api
        log.info("Downloading Word2Vec (google-news-300, ~1.6 GB) ...")
        kv = api.load("word2vec-google-news-300")
        kv.save(str(W2V_CACHE_PATH))
        _W2V_MODEL = kv
        return kv
    except Exception as e:
        log.warning("Word2Vec unavailable (%s). Distractor pipeline will skip W2V.", e)
        return None


def get_word2vec_distractors(correct_answer: str, article: str,
                              w2v_model, top_n: int = 10) -> list[str]:
    if w2v_model is None:
        return []
    article_words = set(_tokenise(article))
    candidates: list[tuple[str, float]] = []
    for tok in _tokenise(correct_answer):
        if tok not in w2v_model:
            continue
        try:
            for word, sim in w2v_model.most_similar(tok, topn=top_n):
                w_lc = word.lower()
                if w_lc in article_words or w_lc in STOPWORDS:
                    continue
                if w_lc in correct_answer.lower():
                    continue
                candidates.append((word, float(sim)))
        except KeyError:
            continue
    candidates.sort(key=lambda kv: -kv[1])
    seen: set[str] = set()
    out: list[str] = []
    for word, _ in candidates:
        if word.lower() in seen:
            continue
        seen.add(word.lower())
        out.append(word)
        if len(out) >= 3:
            break
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Combined distractor generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_distractors(article: str, question: str, correct_answer: str,
                          vectorizer=None, w2v_model=None,
                          n: int = N_DISTRACTORS) -> list[str]:
    """
    Merge three pipelines (TF-IDF/OHE+MMR, frequency, optional W2V),
    deduplicate against correct answer, return exactly `n` distractors.
    """
    if vectorizer is None:
        vectorizer = load_vectorizer()

    pool: list[str] = []
    pool.extend(_tfidf_mmr_distractors(article, correct_answer, vectorizer, n=n + 2))
    pool.extend(frequency_substitution_distractors(article, correct_answer, top_n=n + 2))
    pool.extend(get_word2vec_distractors(correct_answer, article, w2v_model, top_n=10))

    correct_lc = str(correct_answer).lower().strip()
    seen: set[str] = set()
    cleaned: list[str] = []
    for cand in pool:
        if not cand:
            continue
        c_lc = cand.lower().strip()
        if not c_lc or c_lc == correct_lc:
            continue
        if c_lc in correct_lc or correct_lc in c_lc:
            continue
        if c_lc in seen:
            continue
        seen.add(c_lc)
        cleaned.append(cand)

    if not cleaned:
        return []

    # Diversity-filtered top-n via MMR over cleaned pool
    sim_to_ans = _cosine(cleaned, [correct_answer], vectorizer).ravel()
    inter = _cosine(cleaned, cleaned, vectorizer)
    idx = _mmr_select(sim_to_ans, inter, n)
    selected = [cleaned[i] for i in idx]

    # Pad up to n if fewer
    while len(selected) < n and len(cleaned) > len(selected):
        for c in cleaned:
            if c not in selected:
                selected.append(c)
                break
    return selected[:n]


# ══════════════════════════════════════════════════════════════════════════════
#  Hint scorer (Logistic Regression)
# ══════════════════════════════════════════════════════════════════════════════

HINT_FEATURE_NAMES = ["cos_q", "cos_ans", "keyword_overlap", "position", "length"]


def _sentence_features(sentence: str, question: str, correct_answer: str,
                        position_norm: float, vectorizer) -> np.ndarray:
    s_vec = vectorizer.transform([sentence])
    q_vec = vectorizer.transform([question])
    a_vec = vectorizer.transform([correct_answer])
    cos_q = float(cosine_similarity(s_vec, q_vec).ravel()[0])
    cos_a = float(cosine_similarity(s_vec, a_vec).ravel()[0])
    s_words = set(_tokenise(sentence))
    q_words = set(_tokenise(question))
    keyword_overlap = float(len(s_words & q_words))
    length = float(len(sentence.split()))
    return np.array([cos_q, cos_a, keyword_overlap, position_norm, length],
                    dtype=np.float32)


def train_hint_scorer(train_df: pd.DataFrame, vectorizer,
                      save_path: Path = HINT_SCORER_PATH,
                      max_rows: int = 3000) -> LogisticRegression:
    log.info("Training hint scorer on up to %d articles ...", max_rows)
    train_df = train_df.head(max_rows).reset_index(drop=True)

    X, y = [], []
    for _, row in train_df.iterrows():
        article = str(row.get("article", "")).strip()
        question = str(row.get("question", "")).strip()
        correct_label = str(row.get("answer", "A")).strip().upper()
        answer_text = str(row.get(correct_label, "")).strip()
        if not article or not question or not answer_text:
            continue

        sentences = _split_sentences(article)
        if not sentences:
            continue
        n_sent = len(sentences)
        ans_lc = answer_text.lower()

        for i, sent in enumerate(sentences):
            label = 1 if ans_lc in sent.lower() else 0
            pos_norm = i / max(1, n_sent - 1)
            feat = _sentence_features(sent, question, answer_text,
                                       pos_norm, vectorizer)
            X.append(feat)
            y.append(label)

    if not X:
        log.error("No hint scorer training data produced.")
        sys.exit(1)
    X_arr = np.vstack(X)
    y_arr = np.array(y, dtype=np.int8)
    log.info("Hint scorer training: %d samples (%d pos / %d neg)",
             len(y_arr), int(y_arr.sum()), int((y_arr == 0).sum()))

    clf = LogisticRegression(class_weight="balanced", max_iter=500,
                              random_state=42)
    clf.fit(X_arr, y_arr)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "feature_names": HINT_FEATURE_NAMES}, save_path)
    log.info("Hint scorer saved -> %s", save_path)
    return clf


def _load_hint_scorer():
    if not HINT_SCORER_PATH.exists():
        return None
    return joblib.load(HINT_SCORER_PATH)


# ══════════════════════════════════════════════════════════════════════════════
#  Hint generation (graduated: general -> specific -> near-explicit)
# ══════════════════════════════════════════════════════════════════════════════

def generate_hints(article: str, question: str, correct_answer: str = "",
                   vectorizer=None) -> list[str]:
    if vectorizer is None:
        vectorizer = load_vectorizer()
    sentences = _split_sentences(article)
    if not sentences:
        return []

    bundle = _load_hint_scorer()

    n_sent = len(sentences)
    if bundle is not None:
        clf = bundle["clf"]
        feats = []
        for i, s in enumerate(sentences):
            pos = i / max(1, n_sent - 1)
            feats.append(_sentence_features(s, question, correct_answer,
                                             pos, vectorizer))
        X = np.vstack(feats)
        scores = clf.predict_proba(X)[:, 1]
    else:
        scores = _cosine(sentences, [question], vectorizer).ravel()

    # Sort indices by score descending
    order = np.argsort(-scores)
    if len(order) == 0:
        return []

    # Near-explicit (highest), specific (rank 2), general (median of top half)
    near = sentences[order[0]]
    specific = sentences[order[1]] if len(order) > 1 else near
    top_half = order[: max(1, len(order) // 2)]
    median_pos = top_half[len(top_half) // 2]
    general = sentences[median_pos]

    # Ensure 3 unique hints when possible
    out = [general, specific, near]
    seen, deduped = set(), []
    for h in out:
        key = h.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)
    # Pad with remaining sorted sentences if needed
    for i in order:
        if len(deduped) >= 3:
            break
        s = sentences[i]
        if s.strip().lower() in seen:
            continue
        seen.add(s.strip().lower())
        deduped.append(s)
    return deduped[:3]


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point — train hint scorer
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=3000,
                        help="Max training rows for hint scorer.")
    parser.add_argument("--skip-w2v", action="store_true",
                        help="Skip Word2Vec download/cache.")
    args = parser.parse_args()

    vectorizer = load_vectorizer()
    train_csv = PROCESSED_DIR / "train.csv"
    if not train_csv.exists():
        log.error("Train CSV not found: %s. Run preprocessing.py first.", train_csv)
        sys.exit(1)
    train_df = pd.read_csv(train_csv)

    train_hint_scorer(train_df, vectorizer, save_path=HINT_SCORER_PATH,
                       max_rows=args.max_rows)

    if not args.skip_w2v:
        load_word2vec()  # warm cache


if __name__ == "__main__":
    main()
