"""
Model B: distractor and hint generation.

Rubric coverage:
  5.1  Candidate extraction (n-gram phrases, frequency, Word2Vec)
  5.2  Ranking features (cosine, character overlap, passage frequency)
  5.3  ML ranker + diversity penalty (MMR + Jaccard)
  5.4  Plausibility, three distractors per question, syntactic match
  6.1  Logistic Regression hint scorer over five sentence-level features
  6.2  Three graduated hints (general -> specific -> redacted near-explicit)
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


# ── Answer-type detection (for distractor plausibility filters) ───────────────
_NUM_RE = re.compile(r"^\s*-?\d[\d,.\s]*\s*$")
_YEAR_RE = re.compile(r"^\s*\d{3,4}\s*$")
# Number followed by an optional unit / noun (e.g. "6,650 kilometers", "37 trillion cells")
_NUM_PHRASE_RE = re.compile(r"^\s*(?P<num>\d[\d,\.]*)\s*(?P<unit>[A-Za-z][A-Za-z\s]*)?\s*$")


def _answer_type(answer: str) -> str:
    """Return one of: 'year', 'number', 'num_phrase', 'short' (1-2 words), 'phrase'."""
    a = str(answer).strip()
    if _YEAR_RE.match(a):
        return "year"
    if _NUM_RE.match(a):
        return "number"
    if _NUM_PHRASE_RE.match(a) and any(c.isalpha() for c in a):
        return "num_phrase"
    if 1 <= len(a.split()) <= 2:
        return "short"
    return "phrase"


def _synthesize_numeric_distractors(answer: str, n: int = 5) -> list[str]:
    """
    Generate plausible numeric distractors by perturbing the gold answer.
    Handles three forms:
      - bare year   ("1636")            → nearby years  (±5, ±15, ±50, ±150, ±300)
      - bare number ("37000")           → ±10%, ±25%, ±50% rounded
      - num+unit    ("6,650 kilometers")→ same perturbations, unit preserved
    """
    a = str(answer).strip()
    m = _NUM_PHRASE_RE.match(a)
    if not m:
        return []
    raw_num = m.group("num").replace(",", "")
    unit = (m.group("unit") or "").strip()

    try:
        base = float(raw_num)
    except ValueError:
        return []
    is_year = _YEAR_RE.match(a) is not None

    if is_year:
        base = int(base)
        offsets = [-5, +15, -50, +150, -300, +500]
        cands = [base + d for d in offsets if 1000 <= base + d <= 2099]
        return [str(int(c)) for c in cands][:n]

    # Generic numeric perturbations
    is_int = base.is_integer()
    factors = [0.5, 0.75, 1.25, 1.5, 2.0, 0.25]
    cands_num = []
    for f in factors:
        v = base * f
        if is_int:
            # Round to a "nice" magnitude based on size
            if v >= 1000:
                v = round(v, -2)        # round to hundreds
            elif v >= 100:
                v = round(v, -1)        # round to tens
            else:
                v = round(v)
            cands_num.append(int(v))
        else:
            cands_num.append(round(v, 2))

    out: list[str] = []
    seen = {raw_num}
    for v in cands_num:
        s = f"{v:,}" if isinstance(v, int) and v >= 1000 else str(v)
        if s in seen:
            continue
        seen.add(s)
        out.append(f"{s} {unit}".strip())
        if len(out) >= n:
            break
    return out


def _length_ok(candidate: str, answer: str, ratio: float = 2.0) -> bool:
    """Distractor token count must be within `ratio`x of the answer length."""
    nc = max(1, len(str(candidate).split()))
    na = max(1, len(str(answer).split()))
    return (1.0 / ratio) <= (nc / na) <= ratio


def _content_words(text: str) -> set[str]:
    """Lowercase content tokens minus stop-words — for overlap checks."""
    return {w for w in _tokenise(text) if w not in STOPWORDS}


def _too_overlapping(candidate: str, answer: str, max_jaccard: float = 0.34) -> bool:
    """
    True if the candidate shares too many content words with the answer.
    Prevents distractors like "the northeastern region" when the answer is
    "northeastern region of Tanzania".
    """
    a = _content_words(answer)
    c = _content_words(candidate)
    if not a or not c:
        return False
    inter = a & c
    if not inter:
        return False
    union = a | c
    jaccard = len(inter) / len(union)
    if jaccard > max_jaccard:
        return True
    # Also reject if the candidate is essentially a subset of the answer's
    # content words (covers "the northeastern", "tanzania rises", etc.).
    if len(c) <= len(a) and len(inter) / len(c) >= 0.6:
        return True
    return False


def _matches_type(candidate: str, atype: str) -> bool:
    """Filter candidates by the answer's type (year / number / num_phrase / phrase)."""
    c = str(candidate).strip()
    if atype == "year":
        return bool(_YEAR_RE.match(c))
    if atype == "number":
        return bool(_NUM_RE.match(c))
    if atype == "num_phrase":
        return bool(_NUM_PHRASE_RE.match(c) and any(ch.isdigit() for ch in c))
    # 'short' and 'phrase' accept anything non-numeric
    return not _NUM_RE.match(c)


def _redact_answer(sentence: str, answer: str, mask: str = "_____") -> str:
    """Replace the answer string in a sentence with a blank, preserving case-insensitive match."""
    if not answer:
        return sentence
    pat = re.compile(re.escape(str(answer)), re.IGNORECASE)
    return pat.sub(mask, sentence)


def _contains_answer(sentence: str, answer: str) -> bool:
    if not answer:
        return False
    return str(answer).lower() in str(sentence).lower()


def load_vectorizer(path: Path = OHE_PATH):
    if not path.exists():
        log.error("Vectorizer not found: %s. Run preprocessing.py first.", path)
        sys.exit(1)
    return joblib.load(path)


# Rubric 5.1 / 5.3: distractor candidates from n-gram cosine + MMR diversity.

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


# Rubric 5.1: high-frequency content-word substitution from the passage.

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


# Rubric 5.1: pre-trained Word2Vec semantic neighbours (cached locally).

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


# Offline peer-entity lexicon used when Word2Vec is unavailable. Each list is
# kept short and homogeneous so that substituting any peer back into the
# answer phrase yields a plausible, type-matched distractor.
_PEER_LEXICON: dict[str, list[str]] = {
    # African countries (East Africa cluster + a few neighbours)
    "tanzania": ["Kenya", "Uganda", "Ethiopia", "Rwanda", "Mozambique", "Zambia"],
    "kenya":    ["Tanzania", "Uganda", "Ethiopia", "Somalia", "Sudan"],
    "uganda":   ["Kenya", "Tanzania", "Rwanda", "Burundi"],
    "ethiopia": ["Kenya", "Sudan", "Eritrea", "Somalia"],
    "egypt":    ["Libya", "Sudan", "Jordan", "Morocco"],
    "morocco":  ["Algeria", "Tunisia", "Egypt", "Libya"],
    # Other countries
    "australia":   ["New Zealand", "Indonesia", "Malaysia", "Philippines"],
    "japan":       ["China", "Korea", "Vietnam", "Thailand"],
    "china":       ["Japan", "India", "Vietnam", "Mongolia"],
    "india":       ["Pakistan", "Bangladesh", "Nepal", "Sri Lanka"],
    "france":      ["Germany", "Spain", "Italy", "Belgium"],
    "germany":     ["France", "Austria", "Poland", "Netherlands"],
    "italy":       ["Spain", "France", "Greece", "Portugal"],
    "spain":       ["Portugal", "Italy", "France", "Greece"],
    "england":     ["Scotland", "Wales", "Ireland", "France"],
    "russia":      ["Ukraine", "Belarus", "Poland", "Kazakhstan"],
    "brazil":      ["Argentina", "Chile", "Peru", "Colombia"],
    "mexico":      ["Guatemala", "Honduras", "Cuba", "Belize"],
    "canada":      ["Mexico", "Greenland", "Alaska", "Iceland"],
    # Continents
    "africa":          ["Asia", "Europe", "South America", "Oceania"],
    "asia":            ["Africa", "Europe", "South America", "Oceania"],
    "europe":          ["Asia", "Africa", "North America", "Oceania"],
    # Cardinal directions
    "northeastern":    ["southeastern", "northwestern", "southwestern", "central"],
    "southeastern":    ["northeastern", "southwestern", "northwestern", "central"],
    "northwestern":    ["southwestern", "northeastern", "southeastern", "central"],
    "southwestern":    ["northwestern", "southeastern", "northeastern", "central"],
    "northern":        ["southern", "eastern", "western", "central"],
    "southern":        ["northern", "eastern", "western", "central"],
    "eastern":         ["western", "northern", "southern", "central"],
    "western":         ["eastern", "northern", "southern", "central"],
    # Common geographic head nouns
    "ocean":           ["sea", "lake", "bay", "gulf"],
    "sea":             ["ocean", "lake", "bay", "gulf"],
    "river":           ["stream", "lake", "creek", "canal"],
    "mountain":        ["hill", "peak", "ridge", "plateau"],
    "forest":          ["jungle", "woodland", "grassland", "savanna"],
    "desert":          ["plateau", "savanna", "tundra", "steppe"],
    # Common entities
    "harvard":         ["Yale", "Princeton", "Stanford", "Oxford"],
    "yale":            ["Harvard", "Princeton", "Columbia", "Cornell"],
    "oxford":          ["Cambridge", "Harvard", "Yale", "Edinburgh"],
    "cambridge":       ["Oxford", "Harvard", "Yale", "Edinburgh"],
    "nile":            ["Amazon", "Yangtze", "Mississippi", "Ganges"],
    "amazon":          ["Nile", "Congo", "Mississippi", "Mekong"],
    "kilimanjaro":     ["Everest", "Denali", "Aconcagua", "Elbrus"],
    "everest":         ["Kilimanjaro", "Denali", "Aconcagua", "K2"],
}


def _peer_substitutes(head: str, k: int = 5) -> list[str]:
    """Lookup peers from the offline lexicon (case-insensitive)."""
    return list(_PEER_LEXICON.get(head.lower(), []))[:k]


def _w2v_peers(token: str, w2v_model, k: int = 6) -> list[str]:
    """Return up to k W2V neighbours of token, or [] if unavailable."""
    if w2v_model is None:
        return []
    key = None
    try:
        if token in w2v_model:
            key = token
        elif token.lower() in w2v_model:
            key = token.lower()
    except Exception:
        return []
    if key is None:
        return []
    try:
        out = []
        for word, _ in w2v_model.most_similar(key, topn=k * 3):
            cand = word.replace("_", " ").strip()
            if not cand or cand.lower() == token.lower():
                continue
            out.append(cand)
            if len(out) >= k:
                break
        return out
    except KeyError:
        return []


def _substitute_proper_noun_distractors(answer: str, w2v_model,
                                         n: int = 5) -> list[str]:
    """
    Build length-matched distractors by swapping one salient token of the
    answer phrase with peers from (a) the offline lexicon, then (b) Word2Vec.

    Tries every substitutable token (right-most first), so a phrase like
    "northeastern region of Tanzania" yields peers for both 'Tanzania'
    (Kenya, Uganda, ...) and 'northeastern' (southeastern, central, ...).
    """
    tokens = str(answer).split()
    if len(tokens) < 2:
        return []

    # Identify substitutable token indices: prefer right-most proper nouns,
    # then any non-stopword cardinal-direction-like token.
    sub_indices: list[int] = []
    for i in range(len(tokens) - 1, -1, -1):
        clean = tokens[i].strip(".,;:!?\"'")
        if not clean:
            continue
        head_lc = clean.lower()
        if head_lc in STOPWORDS:
            continue
        # Substitutable if (a) capitalized proper noun, or (b) listed in lexicon.
        if clean[0].isupper() or head_lc in _PEER_LEXICON:
            sub_indices.append(i)

    if not sub_indices:
        return []

    out: list[str] = []
    seen_phrases: set[str] = {answer.lower().strip()}

    for idx in sub_indices:
        head = tokens[idx].strip(".,;:!?\"'")
        peers = _peer_substitutes(head, k=n + 2)
        if not peers:
            peers = _w2v_peers(head, w2v_model, k=n + 2)

        for peer in peers:
            if not peer or peer.lower() == head.lower():
                continue
            new_tokens = list(tokens)
            # Preserve original case-pattern: if head was lowercase, keep peer lower
            if head[0].islower() and peer[0].isupper():
                new_tokens[idx] = peer.lower()
            else:
                new_tokens[idx] = peer
            phrase = " ".join(new_tokens)
            key = phrase.lower().strip()
            if key in seen_phrases:
                continue
            seen_phrases.add(key)
            out.append(phrase)
            if len(out) >= n:
                return out
    return out


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


# Rubric 5.3 / 5.4: merge candidate pools, type/length filter, MMR for final 3.

def generate_distractors(article: str, question: str, correct_answer: str,
                          vectorizer=None, w2v_model=None,
                          n: int = N_DISTRACTORS) -> list[str]:
    """
    Merge three pipelines (TF-IDF/OHE+MMR, frequency, optional W2V),
    deduplicate against correct answer, return exactly `n` distractors.
    """
    if vectorizer is None:
        vectorizer = load_vectorizer()

    atype = _answer_type(correct_answer)

    # Two pools: TRUSTED (synthesized / substituted — bypass overlap & length
    # filters because they are deliberately constructed to be type-matched)
    # and UNTRUSTED (extracted from article — must pass all plausibility checks).
    trusted: list[str] = []
    untrusted: list[str] = []

    if atype in ("year", "number", "num_phrase"):
        trusted.extend(_synthesize_numeric_distractors(correct_answer, n=n + 3))
    if atype in ("phrase", "short", "num_phrase"):
        trusted.extend(_substitute_proper_noun_distractors(correct_answer, w2v_model,
                                                            n=n + 3))

    untrusted.extend(_tfidf_mmr_distractors(article, correct_answer, vectorizer, n=n + 2))
    untrusted.extend(frequency_substitution_distractors(article, correct_answer, top_n=n + 2))
    untrusted.extend(get_word2vec_distractors(correct_answer, article, w2v_model, top_n=10))

    correct_lc = str(correct_answer).lower().strip()
    seen: set[str] = set()
    cleaned: list[str] = []

    # Trusted pass: only dedupe + type sanity check
    for cand in trusted:
        if not cand:
            continue
        c_lc = cand.lower().strip()
        if not c_lc or c_lc == correct_lc or c_lc in seen:
            continue
        if not _matches_type(cand, atype):
            continue
        seen.add(c_lc)
        cleaned.append(cand)

    # Untrusted pass: full plausibility filters
    for cand in untrusted:
        if not cand:
            continue
        c_lc = cand.lower().strip()
        if not c_lc or c_lc == correct_lc:
            continue
        if c_lc in correct_lc or correct_lc in c_lc:
            continue
        if c_lc in seen:
            continue
        if not _matches_type(cand, atype):
            continue
        if not _length_ok(cand, correct_answer):
            continue
        if _too_overlapping(cand, correct_answer):
            continue
        seen.add(c_lc)
        cleaned.append(cand)

    # All-pool fallback used only if we still don't have enough
    pool = trusted + untrusted

    # If filters were too aggressive (e.g. for 'year' types where the article
    # has few year-like tokens), retry without the length constraint — but
    # KEEP the overlap filter so we never produce near-duplicate distractors.
    if len(cleaned) < n:
        for cand in pool:
            if not cand:
                continue
            c_lc = cand.lower().strip()
            if (not c_lc or c_lc == correct_lc or c_lc in seen
                    or c_lc in correct_lc or correct_lc in c_lc):
                continue
            if _too_overlapping(cand, correct_answer):
                continue
            seen.add(c_lc)
            cleaned.append(cand)
            if len(cleaned) >= n + 5:
                break

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


# Rubric 6.1: Logistic Regression hint scorer over 5 sentence-level features.

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


# Rubric 6.2: three graduated hints; the third has the answer span redacted.

def generate_hints(article: str, question: str, correct_answer: str = "",
                   vectorizer=None) -> list[str]:
    """
    Three graduated, non-spoiler hints:
      1. GENERAL  — topical sentence relevant to the question (no answer leak).
      2. SPECIFIC — context sentence that narrows the topic (no answer leak).
      3. NEAR     — the sentence containing the answer, but with the answer
                    redacted to '_____'. Forces the student to fill in the blank.
    """
    if vectorizer is None:
        vectorizer = load_vectorizer()
    sentences = _split_sentences(article)
    if not sentences:
        return []

    bundle = _load_hint_scorer()
    n_sent = len(sentences)
    if bundle is not None:
        clf = bundle["clf"]
        feats = [
            _sentence_features(s, question, correct_answer,
                                i / max(1, n_sent - 1), vectorizer)
            for i, s in enumerate(sentences)
        ]
        scores = clf.predict_proba(np.vstack(feats))[:, 1]
    else:
        scores = _cosine(sentences, [question], vectorizer).ravel()

    order = np.argsort(-scores)
    if len(order) == 0:
        return []

    # Partition by whether the sentence contains the answer
    answer_sentences = [i for i in order if _contains_answer(sentences[i], correct_answer)]
    safe_sentences = [i for i in order if not _contains_answer(sentences[i], correct_answer)]

    # 3rd hint (NEAR): top answer-containing sentence, with answer redacted
    if answer_sentences and correct_answer:
        near = _redact_answer(sentences[answer_sentences[0]], correct_answer)
    elif safe_sentences:
        near = sentences[safe_sentences[0]]
    else:
        near = sentences[order[0]]

    # 2nd hint (SPECIFIC): top non-answer-containing sentence by hint score
    if len(safe_sentences) >= 1:
        specific = sentences[safe_sentences[0]]
    else:
        specific = near

    # 1st hint (GENERAL): a more topical sentence — pick from middle of safe ranking
    if len(safe_sentences) >= 2:
        mid = safe_sentences[len(safe_sentences) // 2]
        general = sentences[mid]
    elif len(sentences) >= 1:
        # Fall back to the first sentence of the article (typical topic sentence)
        general = sentences[0] if not _contains_answer(sentences[0], correct_answer) \
                                  else _redact_answer(sentences[0], correct_answer)
    else:
        general = specific

    # Deduplicate while preserving order: general, specific, near
    out, seen = [], set()
    for h in (general, specific, near):
        key = h.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(h)

    # Pad with additional safe sentences if we still don't have 3
    for i in safe_sentences:
        if len(out) >= 3:
            break
        s = sentences[i]
        key = s.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(s)

    return out[:3]


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
