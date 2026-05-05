"""
model_b_train.py  —  RACE Reading Comprehension System  (AL2002 Lab Project)
=============================================================================
Model B: Distractor Generation + Extractive Hint Ranking

Overview
--------
This module implements two core functions that together power the RACE
reading-comprehension UI:

    generate_distractors(article, question, correct_answer, vectorizer)
        Extracts candidate phrases from the article using sliding n-gram
        windows and word-frequency counting (no NLP libraries), then ranks
        them by TF-IDF cosine similarity to the correct answer.  A diversity
        penalty ensures the three returned distractors are distinct from each
        other.  A phrase-inflation post-processor ensures single-word
        candidates are expanded into readable noun phrases from the article.

    generate_hints(article, question, vectorizer, top_k=3)
        Splits the article into sentences, scores each sentence by its
        TF-IDF cosine similarity to the question, and returns the top-k
        sentences as ranked extractive hints.

Both functions use the pre-fitted TfidfVectorizer from preprocessing.py
via .transform() only — the vocabulary and IDF weights are never modified.

Usage
-----
    python model_b_train.py          # runs demo on a random test.csv row
    python model_b_train.py --row 42 # demo on a specific row index
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
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR   = Path("data") / "processed"
MODEL_DIR       = Path("models") / "model_a"

TEST_CSV        = PROCESSED_DIR / "test.csv"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
NGRAM_SIZES: list[int] = [1, 2, 3]
MIN_FREQ: int = 1
MIN_PHRASE_LEN: int = 3
MAX_PHRASE_LEN: int = 80
DIVERSITY_PENALTY: float = 0.6
N_DISTRACTORS: int = 3
N_HINTS: int = 3
MIN_SENTENCE_WORDS: int = 5

# Phrase inflation tunables
_INFLATE_MIN_WORDS: int = 2   # inflate any distractor shorter than this
_INFLATE_MAX_WORDS: int = 10  # cap the window size when pulling context
_PHRASE_TARGET_WORDS: int = 3 # minimum words in the final inflated phrase


# ===========================================================================
# Shared utility — TF-IDF cosine similarity
# ===========================================================================

def _tfidf_cosine(texts_a: list[str], texts_b: list[str], vectorizer) -> np.ndarray:
    """
    Return a (len(texts_a), len(texts_b)) cosine-similarity matrix computed
    from TF-IDF vectors.
    """
    vec_a = vectorizer.transform(texts_a)
    vec_b = vectorizer.transform(texts_b)
    return cosine_similarity(vec_a, vec_b)


# ===========================================================================
# Part 0 — Phrase Inflation (NEW)
# ===========================================================================

_INFLATE_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "have", "has", "had", "that", "this", "it", "its", "s",
})


def _split_sentences_simple(text: str) -> list[str]:
    """Fast sentence splitter (no NLTK required)."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def _is_strippable_stopword(token: str) -> bool:
    """
    Return True only if the token is a stopword AND is NOT a proper noun
    (i.e. not capitalised).  This prevents stripping words like "Nobel",
    "Paris", "Marie" from the edges of a phrase.
    """
    clean = token.rstrip(".,;:\"'()")
    # Preserve capitalised words — they are likely proper nouns
    if clean and clean[0].isupper():
        return False
    return clean.lower() in _INFLATE_STOPWORDS


def _inflate_keyword(keyword: str, article: str, correct_answer: str) -> str:
    """
    Given a bare keyword (e.g. "prize"), find the tightest readable phrase
    in the article that:
      1. Contains the keyword
      2. Reaches _PHRASE_TARGET_WORDS after stopword-stripping
      3. Doesn't start/end with a (non-proper-noun) stopword
      4. Doesn't overlap heavily with the correct answer

    Falls back to a 2-word prefix grab, then to the capitalised keyword.
    """
    kw_lower = keyword.strip().lower()
    correct_tokens = set(correct_answer.lower().split())
    sentences = _split_sentences_simple(article)

    best: Optional[str] = None
    best_len = 9999

    for sent in sentences:
        if kw_lower not in sent.lower():
            continue

        # Skip sentences that mostly restate the correct answer
        sent_tokens = set(sent.lower().split())
        overlap = len(sent_tokens & correct_tokens) / max(len(sent_tokens), 1)
        if overlap > 0.5:
            continue

        tokens = sent.split()
        kw_idx = next(
            (i for i, t in enumerate(tokens) if kw_lower in t.lower()), None
        )
        if kw_idx is None:
            continue

        # Slide windows around the keyword — start from up to MAX_WORDS before it
        for left in range(max(0, kw_idx - (_INFLATE_MAX_WORDS - 1)), kw_idx + 1):
            for right in range(
                kw_idx + 1,
                min(len(tokens), left + _INFLATE_MAX_WORDS) + 1,
            ):
                chunk = list(tokens[left:right])

                # Strip leading/trailing stopwords — but preserve proper nouns
                while chunk and _is_strippable_stopword(chunk[0]):
                    chunk = chunk[1:]
                while chunk and _is_strippable_stopword(chunk[-1]):
                    chunk = chunk[:-1]

                if len(chunk) < _PHRASE_TARGET_WORDS:
                    continue

                phrase = " ".join(chunk).strip(".,;:\"'()")

                if len(chunk) < best_len:
                    best = phrase
                    best_len = len(chunk)

    if best:
        return best[0].upper() + best[1:]

    # ── Fallback 1: grab the word immediately before the keyword in any
    #   sentence (catches "Nobel Prize" → "Nobel Prize")
    for sent in sentences:
        if kw_lower not in sent.lower():
            continue
        tokens = sent.split()
        kw_idx = next(
            (i for i, t in enumerate(tokens) if kw_lower in t.lower()), None
        )
        if kw_idx is not None and kw_idx > 0:
            phrase = " ".join(tokens[kw_idx - 1 : kw_idx + 1]).strip(".,;:\"'()")
            if phrase:
                return phrase[0].upper() + phrase[1:]

    # ── Fallback 2: capitalised keyword as-is
    return keyword[0].upper() + keyword[1:]


def _inflate_distractors(
    distractors: list[dict],
    article: str,
    correct_answer: str,
) -> list[dict]:
    """
    Post-process the MMR-selected distractors.

    Any distractor whose text is shorter than _INFLATE_MIN_WORDS words is
    expanded to a readable phrase by pulling context from the article.
    Distractors that are already phrase-length are returned unchanged.

    Parameters
    ----------
    distractors    : list of {"distractor": str, "similarity": float}
    article        : the RACE passage
    correct_answer : used to avoid producing options that echo the answer

    Returns
    -------
    Same list structure with distractor strings potentially expanded.
    """
    inflated = []
    seen_phrases: set[str] = set()

    for d in distractors:
        text = d["distractor"]
        word_count = len(text.split())

        if word_count < _PHRASE_TARGET_WORDS:
            expanded = _inflate_keyword(text, article, correct_answer)
        else:
            expanded = text[0].upper() + text[1:]   # just capitalise

        # Deduplicate across inflated results
        if expanded.lower() in seen_phrases:
            # Try again with the original raw keyword capitalised as fallback
            expanded = text[0].upper() + text[1:]

        seen_phrases.add(expanded.lower())
        inflated.append({"distractor": expanded, "similarity": d["similarity"]})

    return inflated


# ===========================================================================
# Part 1 — Distractor Generation
# ===========================================================================

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "that",
    "this", "these", "those", "it", "its", "they", "them", "their",
    "he", "she", "his", "her", "we", "our", "you", "your", "i", "my",
    "not", "no", "so", "if", "up", "out", "about", "into", "than",
    "then", "there", "here", "when", "where", "which", "who", "whom",
    "what", "how", "all", "also", "just", "more", "such", "very",
    "s", "t", "re", "ve", "ll", "d",
})


def _tokenise(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _extract_candidate_phrases(
    article: str,
    correct_answer: str,
    ngram_sizes: list[int] = NGRAM_SIZES,
    min_freq: int = MIN_FREQ,
    min_len: int = MIN_PHRASE_LEN,
    max_len: int = MAX_PHRASE_LEN,
) -> list[str]:
    """
    Extract candidate distractor phrases from the article using sliding
    n-gram windows and word-frequency counting (zero NLP dependencies).
    """
    tokens      = _tokenise(article)
    correct_tok = _tokenise(correct_answer)
    correct_str = " ".join(correct_tok)

    seen_phrases: set[str] = set()
    counter: Counter = Counter()

    for n in ngram_sizes:
        for i in range(len(tokens) - n + 1):
            gram   = tokens[i : i + n]
            phrase = " ".join(gram)

            if n == 1 and gram[0] in _STOPWORDS:
                continue

            counter[phrase] += 1

    candidates: list[str] = []
    for phrase, freq in counter.items():
        if freq < min_freq:
            continue
        if not (min_len <= len(phrase) <= max_len):
            continue
        if phrase == correct_str:
            continue
        if phrase in correct_str or correct_str in phrase:
            continue
        if phrase not in seen_phrases:
            seen_phrases.add(phrase)
            candidates.append(phrase)

    candidates.sort(key=lambda p: counter[p], reverse=True)
    return candidates


def _mmr_select(
    scores: np.ndarray,
    inter_sim: np.ndarray,
    n: int,
    diversity_penalty: float = DIVERSITY_PENALTY,
) -> list[int]:
    """
    Maximal Marginal Relevance (MMR) greedy selection.
    """
    remaining = list(range(len(scores)))
    selected: list[int] = []

    for _ in range(min(n, len(remaining))):
        if not selected:
            best = max(remaining, key=lambda i: scores[i])
        else:
            def mmr_score(i: int) -> float:
                redundancy = max(inter_sim[i, j] for j in selected)
                return scores[i] - diversity_penalty * redundancy

            best = max(remaining, key=mmr_score)

        selected.append(best)
        remaining.remove(best)

    return selected


def generate_distractors(
    article: str,
    question: str,
    correct_answer: str,
    vectorizer,
    n: int = N_DISTRACTORS,
    diversity_penalty: float = DIVERSITY_PENALTY,
    pool_size: int = 200,
) -> list[dict]:
    """
    Generate *n* plausible-but-wrong distractor phrases for a RACE question.

    Algorithm
    ---------
    1. Extract candidate phrases from the article (n-grams, frequency-filtered).
    2. Keep the top *pool_size* candidates by frequency to limit computation.
    3. Compute TF-IDF cosine similarity between each candidate and the
       correct answer.
    4. Use MMR to pick *n* candidates that are similar to the correct answer
       but dissimilar to each other.
    5. Inflate any bare keywords into readable phrases using article context.
    6. Return selected phrases with their relevance scores.

    Returns
    -------
    List of dicts: [{"distractor": str, "similarity": float}, ...]
    """
    candidates = _extract_candidate_phrases(article, correct_answer)

    if not candidates:
        log.warning("No candidate phrases found in article.")
        return []

    pool = candidates[:pool_size]

    sim_to_answer = _tfidf_cosine(pool, [correct_answer], vectorizer).squeeze(axis=1)
    inter_sim     = _tfidf_cosine(pool, pool, vectorizer)

    selected_idx = _mmr_select(
        scores=sim_to_answer,
        inter_sim=inter_sim,
        n=n,
        diversity_penalty=diversity_penalty,
    )

    raw_results = [
        {"distractor": pool[i], "similarity": float(sim_to_answer[i])}
        for i in selected_idx
    ]

    # ── Phrase inflation: expand bare keywords into readable phrases ────────
    results = _inflate_distractors(raw_results, article, correct_answer)

    return results


# ===========================================================================
# Part 2 — Extractive Hint Generation
# ===========================================================================

def _split_sentences(text: str) -> list[str]:
    """
    Split *text* into sentences using punctuation heuristics.
    """
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [
        s.strip()
        for s in raw
        if len(s.split()) >= MIN_SENTENCE_WORDS
    ]
    return sentences


def generate_hints(
    article: str,
    question: str,
    vectorizer,
    top_k: int = N_HINTS,
) -> list[dict]:
    """
    Extract the top-*top_k* sentences from the article that are most relevant
    to the question, returned as ranked extractive hints.
    """
    sentences = _split_sentences(article)

    if not sentences:
        log.warning("No sentences extracted from article.")
        return []

    sims = _tfidf_cosine(sentences, [question], vectorizer).squeeze(axis=1)

    scored = [
        {"position": i, "sentence": s, "similarity": float(sims[i])}
        for i, s in enumerate(sentences)
    ]

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    top = scored[:top_k]

    for rank, item in enumerate(top, start=1):
        item["rank"] = rank

    return top


# ===========================================================================
# Utilities
# ===========================================================================

def load_vectorizer(path: Path = VECTORIZER_PATH):
    """Load and return the pre-fitted TfidfVectorizer."""
    if not path.exists():
        log.error(
            "Vectorizer not found: %s\n"
            "  Run preprocessing.py first.", path
        )
        sys.exit(1)
    vec = joblib.load(path)
    log.info("Loaded vectorizer  (vocab size: %d)", len(vec.vocabulary_))
    return vec


def load_test_row(
    path: Path = TEST_CSV,
    row_index: Optional[int] = None,
) -> pd.Series:
    """Return a single row from the test CSV."""
    if not path.exists():
        log.error("Test CSV not found: %s", path)
        sys.exit(1)

    df = pd.read_csv(path)

    if row_index is None:
        row = df.sample(1, random_state=None).iloc[0]
        log.info("Sampled random row  id=%s", row.get("id", "?"))
    else:
        if row_index >= len(df):
            log.error("Row index %d out of range (max %d).", row_index, len(df) - 1)
            sys.exit(1)
        row = df.iloc[row_index]
        log.info("Loaded row index=%d  id=%s", row_index, row.get("id", "?"))

    return row


def _banner(title: str, width: int = 65) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ===========================================================================
# __main__ demo
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model B demo — distractor & hint generation on one test row."
    )
    parser.add_argument("--row",         type=int, default=None, metavar="N")
    parser.add_argument("--vectorizer",  type=str, default=str(VECTORIZER_PATH))
    parser.add_argument("--n-distractors", type=int, default=N_DISTRACTORS)
    parser.add_argument("--n-hints",     type=int, default=N_HINTS)
    args = parser.parse_args()

    vectorizer = load_vectorizer(Path(args.vectorizer))
    row        = load_test_row(row_index=args.row)

    article  = str(row.get("article",  "")).strip()
    question = str(row.get("question", "")).strip()
    answer   = str(row.get("answer",   "")).strip()
    opt_map  = {k: str(row.get(k, "")) for k in "ABCD"}
    correct_text = opt_map.get(answer, "")

    _banner("CONTEXT")
    print(f"Question  : {question}")
    print(f"Answer    : ({answer}) {correct_text}")
    print(f"\nArticle snippet (first 400 chars):\n{article[:400]} ...")

    _banner("DISTRACTOR GENERATION")
    distractors = generate_distractors(
        article=article, question=question,
        correct_answer=correct_text, vectorizer=vectorizer,
        n=args.n_distractors,
    )
    if distractors:
        for i, d in enumerate(distractors, start=1):
            print(f"  Distractor {i}  (sim={d['similarity']:.4f}) : {d['distractor']}")
    else:
        print("  [No distractors generated — article may be too short.]")

    _banner("HINT GENERATION")
    hints = generate_hints(
        article=article, question=question,
        vectorizer=vectorizer, top_k=args.n_hints,
    )
    if hints:
        for h in hints:
            print(f"  Hint #{h['rank']}  (sim={h['similarity']:.4f},  "
                  f"sentence position={h['position']})")
            print(f"    \"{h['sentence']}\"\n")
    else:
        print("  [No hints generated — article may be too short.]")

    _banner("DONE")


if __name__ == "__main__":
    main()