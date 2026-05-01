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
        other.

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
# n-gram sizes to extract as candidate phrases from the article
NGRAM_SIZES: list[int] = [1, 2, 3]

# Minimum raw frequency an n-gram must appear in the article to be a candidate
MIN_FREQ: int = 1

# Minimum character length of a candidate phrase (filters single letters, etc.)
MIN_PHRASE_LEN: int = 3

# Maximum character length of a candidate phrase
MAX_PHRASE_LEN: int = 80

# How much to penalise a candidate for being similar to an already-chosen
# distractor.  Range [0, 1]; 1.0 = fully subtract similarity to chosen set.
DIVERSITY_PENALTY: float = 0.6

# Number of distractors to return
N_DISTRACTORS: int = 3

# Number of hint sentences to return
N_HINTS: int = 3

# Minimum words a sentence must have to be considered as a hint
MIN_SENTENCE_WORDS: int = 5


# ===========================================================================
# Shared utility — TF-IDF cosine similarity
# ===========================================================================

def _tfidf_cosine(texts_a: list[str], texts_b: list[str], vectorizer) -> np.ndarray:
    """
    Return a (len(texts_a), len(texts_b)) cosine-similarity matrix computed
    from TF-IDF vectors.

    Both lists are transformed with the pre-fitted vectorizer (.transform()
    only — no refitting).  The resulting sparse matrices are kept sparse
    until cosine_similarity() densifies them internally.
    """
    vec_a = vectorizer.transform(texts_a)
    vec_b = vectorizer.transform(texts_b)
    return cosine_similarity(vec_a, vec_b)


# ===========================================================================
# Part 1 — Distractor Generation
# ===========================================================================

# English stopwords (subset) used to filter trivial single-word candidates
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

    Steps
    -----
    1. Tokenise the article into lowercase words.
    2. Build n-grams of sizes in *ngram_sizes*.
    3. Count raw frequency of each n-gram string.
    4. Filter:
         - frequency < min_freq → discard
         - phrase length outside [min_len, max_len] → discard
         - all tokens are stopwords → discard (unigrams only)
         - phrase == correct answer (case-insensitive) → discard
         - phrase is a substring of the correct answer → discard
    5. Return unique phrases sorted by descending frequency (most common
       first, so cosine-similarity ranking has a rich pool to reorder).
    """
    tokens      = _tokenise(article)
    correct_tok = _tokenise(correct_answer)
    correct_str = " ".join(correct_tok)

    seen_phrases: set[str] = set()
    counter: Counter = Counter()

    for n in ngram_sizes:
        for i in range(len(tokens) - n + 1):
            gram     = tokens[i : i + n]
            phrase   = " ".join(gram)

            # Skip trivial single-word stopword-only phrases
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

    # Sort by frequency descending so the most article-prominent phrases
    # appear first before cosine-similarity re-ranking
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

    Selects *n* indices that maximise:
        score[i]  -  diversity_penalty * max(sim(i, already_selected))

    Parameters
    ----------
    scores          : 1-D array of relevance scores for each candidate
    inter_sim       : 2-D (n_candidates, n_candidates) similarity matrix
    n               : number of items to select
    diversity_penalty: weight of the redundancy term (0 = no penalty)

    Returns
    -------
    List of selected indices in selection order.
    """
    remaining = list(range(len(scores)))
    selected: list[int] = []

    for _ in range(min(n, len(remaining))):
        if not selected:
            # First pick: pure relevance
            best = max(remaining, key=lambda i: scores[i])
        else:
            # Subsequent picks: relevance minus diversity penalty
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
       correct answer.  High similarity → topically related (plausible).
    4. Use MMR (Maximal Marginal Relevance) to pick *n* candidates that are
       similar to the correct answer but dissimilar to each other.
    5. Return selected phrases with their relevance scores.

    Parameters
    ----------
    article         : the RACE passage text
    question        : the question string (used to filter the article context)
    correct_answer  : the gold answer text
    vectorizer      : pre-fitted TfidfVectorizer
    n               : number of distractors to return (default 3)
    diversity_penalty: MMR lambda — higher = more diverse output
    pool_size       : max candidate phrases to score with cosine similarity

    Returns
    -------
    List of dicts: [{"distractor": str, "similarity": float}, ...]
    """
    # Step 1 — candidate extraction
    candidates = _extract_candidate_phrases(article, correct_answer)

    if not candidates:
        log.warning("No candidate phrases found in article.")
        return []

    # Step 2 — limit pool for efficiency
    pool = candidates[:pool_size]

    # Step 3 — cosine similarity to correct answer
    # Shape: (len(pool), 1) -> squeeze to 1-D
    sim_to_answer = _tfidf_cosine(pool, [correct_answer], vectorizer).squeeze(axis=1)

    # Step 4 — inter-candidate similarity matrix (for MMR diversity penalty)
    inter_sim = _tfidf_cosine(pool, pool, vectorizer)

    # Step 5 — MMR greedy selection
    selected_idx = _mmr_select(
        scores=sim_to_answer,
        inter_sim=inter_sim,
        n=n,
        diversity_penalty=diversity_penalty,
    )

    results = [
        {
            "distractor": pool[i],
            "similarity": float(sim_to_answer[i]),
        }
        for i in selected_idx
    ]
    return results


# ===========================================================================
# Part 2 — Extractive Hint Generation
# ===========================================================================

def _split_sentences(text: str) -> list[str]:
    """
    Split *text* into sentences using punctuation heuristics.

    Uses a regex that splits on  .  !  ?  followed by whitespace + capital,
    which works well for RACE passages without requiring NLTK.
    """
    # Split on sentence-ending punctuation followed by whitespace
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    # Filter very short fragments
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

    Algorithm
    ---------
    1. Split the article into sentences.
    2. Compute TF-IDF cosine similarity between each sentence and the question.
    3. Rank sentences by similarity (descending).
    4. Return the top *top_k* as hint dicts, preserving their original
       position in the article so the UI can display them in context order
       if desired.

    Parameters
    ----------
    article    : the RACE passage text
    question   : the question string
    vectorizer : pre-fitted TfidfVectorizer
    top_k      : number of hint sentences to return

    Returns
    -------
    List of dicts (sorted by similarity descending):
        [{"rank": int, "sentence": str, "similarity": float,
          "position": int}, ...]
    """
    sentences = _split_sentences(article)

    if not sentences:
        log.warning("No sentences extracted from article.")
        return []

    # Cosine similarity: (n_sentences, 1) -> squeeze to 1-D
    sims = _tfidf_cosine(sentences, [question], vectorizer).squeeze(axis=1)

    # Pair each sentence with its similarity score and original position
    scored = [
        {"position": i, "sentence": s, "similarity": float(sims[i])}
        for i, s in enumerate(sentences)
    ]

    # Sort by similarity descending, take top_k
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    top = scored[:top_k]

    # Add rank (1 = most relevant)
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
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        metavar="N",
        help="Row index in test.csv to use.  Default: random.",
    )
    parser.add_argument(
        "--vectorizer",
        type=str,
        default=str(VECTORIZER_PATH),
        metavar="PATH",
        help=f"Path to fitted vectorizer.  Default: {VECTORIZER_PATH}",
    )
    parser.add_argument(
        "--n-distractors",
        type=int,
        default=N_DISTRACTORS,
        metavar="N",
        help=f"Number of distractors to generate.  Default: {N_DISTRACTORS}",
    )
    parser.add_argument(
        "--n-hints",
        type=int,
        default=N_HINTS,
        metavar="N",
        help=f"Number of hint sentences to surface.  Default: {N_HINTS}",
    )
    args = parser.parse_args()

    # ── Load assets ─────────────────────────────────────────────────────
    vectorizer = load_vectorizer(Path(args.vectorizer))
    row        = load_test_row(row_index=args.row)

    article  = str(row.get("article",  "")).strip()
    question = str(row.get("question", "")).strip()
    answer   = str(row.get("answer",   "")).strip()         # letter: A/B/C/D
    opt_map  = {
        "A": str(row.get("A", "")),
        "B": str(row.get("B", "")),
        "C": str(row.get("C", "")),
        "D": str(row.get("D", "")),
    }
    correct_text = opt_map.get(answer, "")

    # ── Print context ────────────────────────────────────────────────────
    _banner("CONTEXT")
    print(f"Question  : {question}")
    print(f"Answer    : ({answer}) {correct_text}")
    print(f"\nArticle snippet (first 400 chars):\n{article[:400]} ...")

    # ── Run distractor generation ─────────────────────────────────────────
    _banner("DISTRACTOR GENERATION")
    print(
        "Algorithm : n-gram candidate extraction → TF-IDF cosine similarity "
        "→ MMR diversity selection\n"
    )

    distractors = generate_distractors(
        article        = article,
        question       = question,
        correct_answer = correct_text,
        vectorizer     = vectorizer,
        n              = args.n_distractors,
    )

    if distractors:
        for i, d in enumerate(distractors, start=1):
            print(f"  Distractor {i}  (sim={d['similarity']:.4f}) : {d['distractor']}")
    else:
        print("  [No distractors generated — article may be too short.]")

    # ── Run hint generation ───────────────────────────────────────────────
    _banner("HINT GENERATION")
    print(
        "Algorithm : sentence splitting → TF-IDF cosine similarity to "
        "question → top-k ranking\n"
    )

    hints = generate_hints(
        article    = article,
        question   = question,
        vectorizer = vectorizer,
        top_k      = args.n_hints,
    )

    if hints:
        for h in hints:
            print(f"  Hint #{h['rank']}  (sim={h['similarity']:.4f},  "
                  f"sentence position={h['position']})")
            print(f"    \"{h['sentence']}\"\n")
    else:
        print("  [No hints generated — article may be too short.]")

    _banner("DONE")
    print(
        "Tip: integrate generate_distractors() and generate_hints() into "
        "src/inference.py for the UI."
    )


if __name__ == "__main__":
    main()