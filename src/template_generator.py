"""
src/template_generator.py
=========================
Template-based question generator for Model A.
Pure traditional ML — NO neural networks.

Pipeline:
  1. Extract candidate sentences via OHE-cosine overlap with the correct answer.
  2. Apply 25 Wh-word templates (5 per Wh) to each top sentence.
  3. Rank candidates with a trained LinearSVC ranker (positive = real RACE Q,
     negative = template-generated Q).

CLI:
    python src/template_generator.py
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import LinearSVC

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
OHE_PATH = MODEL_A_TRAD / "ohe_vectorizer.pkl"
RANKER_PATH = MODEL_A_TRAD / "question_ranker.pkl"

WH_WORDS = ("what", "who", "where", "when", "why")
STOPWORDS_LEAD = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "this", "that", "these", "those", "it",
}

TEMPLATES = {
    "What": [
        "What {verb_phrase} in the passage?",
        "What does the author say about {noun_phrase}?",
        "What is the main idea related to {noun_phrase}?",
        "What happened when {clause}?",
        "What is meant by {noun_phrase} in the passage?",
    ],
    "Who": [
        "Who {verb_phrase} according to the passage?",
        "Who is responsible for {noun_phrase}?",
        "Who does the passage say {clause}?",
        "Who mentioned {noun_phrase}?",
        "Who is described as {adj_phrase} in the passage?",
    ],
    "Where": [
        "Where did {clause}?",
        "Where is {noun_phrase} mentioned?",
        "Where does {noun_phrase} take place?",
        "Where was {noun_phrase} according to the author?",
        "Where can {noun_phrase} be found based on the passage?",
    ],
    "When": [
        "When did {clause}?",
        "When is {noun_phrase} mentioned in the passage?",
        "When does {noun_phrase} occur according to the text?",
        "When was {noun_phrase} first described?",
        "When does the author suggest {clause}?",
    ],
    "Why": [
        "Why did {clause}?",
        "Why is {noun_phrase} important according to the passage?",
        "Why does the author mention {noun_phrase}?",
        "Why was {noun_phrase} significant?",
        "Why does {clause} according to the text?",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [s.strip() for s in raw if s and len(s.split()) >= 3]


def _strip_lead_stopwords(words: list[str]) -> list[str]:
    out = list(words)
    while out and out[0].lower().strip(".,;:!?\"'") in STOPWORDS_LEAD:
        out = out[1:]
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 — Candidate sentence extraction (OHE cosine)
# ══════════════════════════════════════════════════════════════════════════════

def extract_candidate_sentences(article: str, correct_answer: str,
                                vectorizer, top_k: int = 5) -> list[tuple[str, float, int]]:
    """Return top_k (sentence_text, similarity, position_index)."""
    sentences = _split_sentences(article)
    if not sentences:
        return []
    sent_vecs = vectorizer.transform(sentences)
    ans_vec = vectorizer.transform([str(correct_answer)])
    sims = cosine_similarity(sent_vecs, ans_vec).ravel()

    order = np.argsort(-sims)[:top_k]
    return [(sentences[i], float(sims[i]), int(i)) for i in order]


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 — Template filling
# ══════════════════════════════════════════════════════════════════════════════

_NOUN_PHRASE_RE = re.compile(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*")
_ADJ_RE = re.compile(r"\b\w+(?:ing|ed|ful|ous|al)\b", re.IGNORECASE)


def fill_template(template: str, sentence: str, answer: str) -> str:
    """Substitute {noun_phrase}, {verb_phrase}, {clause}, {adj_phrase} with passage content."""
    # Noun phrase: longest run of capitalized tokens, fallback to first 3 answer words
    matches = _NOUN_PHRASE_RE.findall(sentence)
    if matches:
        noun_phrase = max(matches, key=len)
    else:
        ans_words = str(answer).split()
        noun_phrase = " ".join(ans_words[:3]) if ans_words else "this topic"

    tokens = sentence.split()

    # Verb phrase: words 2-4 (after subject)
    verb_phrase = " ".join(tokens[1:4]) if len(tokens) >= 4 else (tokens[0] if tokens else "occurs")

    # Clause: first 8 words minus leading stopwords
    clause = " ".join(_strip_lead_stopwords(tokens[:8])) or sentence[:60]

    # Adjective phrase
    adj_match = _ADJ_RE.search(sentence)
    adj_phrase = adj_match.group(0) if adj_match else "notable"

    try:
        out = template.format(
            noun_phrase=noun_phrase.strip().rstrip(".,;:!?"),
            verb_phrase=verb_phrase.strip().rstrip(".,;:!?"),
            clause=clause.strip().rstrip(".,;:!?"),
            adj_phrase=adj_phrase.strip().rstrip(".,;:!?"),
        )
    except (KeyError, IndexError):
        out = template
    out = re.sub(r"\s+", " ", out).strip()
    return out


def generate_questions(article: str, correct_answer: str, vectorizer,
                       n: int = 5) -> list[dict]:
    """Generate up to 5 Wh templates × n top sentences = ~5n candidate questions."""
    candidates = extract_candidate_sentences(article, correct_answer, vectorizer, top_k=n)
    out = []
    for sent, sim, pos in candidates:
        for wh, templates in TEMPLATES.items():
            for tpl in templates:
                q = fill_template(tpl, sent, correct_answer)
                out.append({
                    "question": q,
                    "source_sentence": sent,
                    "wh_word": wh.lower(),
                    "sentence_similarity": sim,
                    "sentence_position": pos,
                })
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 — Trained SVM ranker
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "cos_q_article", "cos_q_answer", "q_length", "has_wh",
    "answer_overlap", "sentence_position",
]


def _question_features(question: str, article: str, answer: str,
                        vectorizer, sentence_position: float = 0.0) -> np.ndarray:
    q_vec = vectorizer.transform([question])
    a_vec = vectorizer.transform([article])
    ans_vec = vectorizer.transform([answer])
    cos_qa = float(cosine_similarity(q_vec, a_vec).ravel()[0])
    cos_qans = float(cosine_similarity(q_vec, ans_vec).ravel()[0])
    q_words = question.lower().split()
    has_wh = 1.0 if q_words and q_words[0] in WH_WORDS else 0.0
    ans_words = set(str(answer).lower().split())
    answer_overlap = float(sum(1 for w in q_words if w in ans_words))
    return np.array([
        cos_qa, cos_qans, float(len(q_words)), has_wh,
        answer_overlap, float(sentence_position),
    ], dtype=np.float32)


def train_question_ranker(train_df: pd.DataFrame, vectorizer,
                          save_path: Path = RANKER_PATH,
                          max_rows: int = 5000) -> tuple[LinearSVC, list[str]]:
    """Train LinearSVC: pos = real RACE question, neg = template-generated."""
    log.info("Training question ranker on up to %d rows ...", max_rows)
    train_df = train_df.head(max_rows).reset_index(drop=True)

    X, y = [], []
    for _, row in train_df.iterrows():
        article = str(row.get("article", "")).strip()
        question_real = str(row.get("question", "")).strip()
        correct_label = str(row.get("answer", "A")).strip().upper()
        answer_text = str(row.get(correct_label, "")).strip()
        if not article or not question_real or not answer_text:
            continue

        # Approximate sentence position for the real question (use 0.5)
        X.append(_question_features(question_real, article, answer_text,
                                     vectorizer, sentence_position=0.5))
        y.append(1)

        # Generate 4 negatives via templates
        cands = generate_questions(article, answer_text, vectorizer, n=2)
        if not cands:
            continue
        n_sentences = max(1, len(_split_sentences(article)))
        for cand in cands[:4]:
            pos_norm = cand["sentence_position"] / n_sentences
            X.append(_question_features(cand["question"], article, answer_text,
                                         vectorizer, sentence_position=pos_norm))
            y.append(0)

    if not X:
        log.error("No training rows produced for ranker.")
        sys.exit(1)

    X_arr = np.vstack(X)
    y_arr = np.array(y, dtype=np.int8)
    log.info("Ranker training set: %d samples (%d pos / %d neg)",
             len(y_arr), int(y_arr.sum()), int((y_arr == 0).sum()))

    clf = LinearSVC(random_state=42, max_iter=3000, class_weight="balanced")
    clf.fit(X_arr, y_arr)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "feature_names": FEATURE_NAMES}, save_path)
    log.info("Saved question ranker -> %s", save_path)
    return clf, FEATURE_NAMES


def load_ranker(path: Path = RANKER_PATH):
    if not path.exists():
        return None
    return joblib.load(path)


def rank_questions(questions: list[dict], article: str, correct_answer: str,
                   vectorizer, ranker) -> list[dict]:
    """Score each candidate; return list sorted descending by SVC decision_function."""
    if not questions:
        return []
    n_sentences = max(1, len(_split_sentences(article)))
    feats = []
    for q in questions:
        pos_norm = q.get("sentence_position", 0) / n_sentences
        feats.append(_question_features(q["question"], article, correct_answer,
                                         vectorizer, sentence_position=pos_norm))
    X = np.vstack(feats)
    if ranker is None:
        scores = np.zeros(len(questions))
    else:
        clf = ranker["clf"] if isinstance(ranker, dict) else ranker
        scores = clf.decision_function(X)

    out = []
    for q, s in zip(questions, scores):
        q2 = dict(q)
        q2["score"] = float(s)
        out.append(q2)
    return sorted(out, key=lambda d: d["score"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point — train ranker
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train template-based question ranker.")
    parser.add_argument("--max-rows", type=int, default=5000,
                        help="Number of training rows to use.")
    args = parser.parse_args()

    if not OHE_PATH.exists():
        log.error("OHE vectorizer not found: %s. Run preprocessing.py first.", OHE_PATH)
        sys.exit(1)
    vectorizer = joblib.load(OHE_PATH)

    train_csv = PROCESSED_DIR / "train.csv"
    if not train_csv.exists():
        log.error("Train CSV not found: %s. Run preprocessing.py first.", train_csv)
        sys.exit(1)
    train_df = pd.read_csv(train_csv)

    train_question_ranker(train_df, vectorizer, save_path=RANKER_PATH,
                           max_rows=args.max_rows)


if __name__ == "__main__":
    main()
