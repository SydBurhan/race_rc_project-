"""
app.py  —  RACE Reading Comprehension System  (AL2002 Lab Project)
==================================================================
Streamlit UI that wires Model A (Soft-Voting Ensemble: LR + Calibrated SVM)
and Model B (Distractor + Hint Generator) into a polished 4-screen interface.

Run:
    streamlit run ui/app.py
"""

import random
import sys
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config — must be the very first Streamlit call ────────────────────
st.set_page_config(
    layout="wide",
    page_title="RACE Comprehension System",
    page_icon="📖",
)

# ── Resolve src/ on the path so model_b_train imports work ─────────────────
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_b_train import generate_distractors, generate_hints  # noqa: E402

# ---------------------------------------------------------------------------
# Paths (Updated to resolve from the new ui/ folder location)
# ---------------------------------------------------------------------------
MODEL_DIR       = PROJECT_ROOT / "models" / "model_a"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
CLASSIFIER_PATH = MODEL_DIR / "lr_classifier.joblib"
SVM_PATH        = MODEL_DIR / "svm_model.joblib"

# ---------------------------------------------------------------------------
# Global CSS — refined editorial aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Google Fonts ──────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Root palette ──────────────────────────────────────────── */
    :root {
        --ink:      #0f0f0f;
        --paper:    #f7f4ee;
        --accent:   #c0392b;
        --muted:    #7a7269;
        --border:   #d6d0c4;
        --success:  #1a6b3c;
        --error:    #c0392b;
        --card-bg:  #ffffff;
    }

    /* ── Base ──────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--paper);
        color: var(--ink);
    }
    .main .block-container { padding-top: 2rem; max-width: 1100px; }

    /* ── Display headings ──────────────────────────────────────── */
    h1, h2, h3 { font-family: 'Playfair Display', serif; }

    /* ── Sidebar ───────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: var(--ink);
        color: #f7f4ee;
    }
    [data-testid="stSidebar"] * { color: #f7f4ee !important; }
    [data-testid="stSidebar"] .stRadio label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.95rem;
        letter-spacing: 0.04em;
        padding: 0.35rem 0;
    }

    /* ── Cards ─────────────────────────────────────────────────── */
    .rc-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 2px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.25rem;
        box-shadow: 2px 2px 0 var(--border);
    }

    /* ── Article display ───────────────────────────────────────── */
    .rc-article {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.97rem;
        line-height: 1.85;
        color: #1a1a1a;
        background: #fffdf8;
        border-left: 3px solid var(--accent);
        padding: 1.2rem 1.6rem;
        border-radius: 0 2px 2px 0;
        max-height: 320px;
        overflow-y: auto;
    }

    /* ── Question banner ───────────────────────────────────────── */
    .rc-question {
        font-family: 'Playfair Display', serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--ink);
        border-bottom: 2px solid var(--accent);
        padding-bottom: 0.5rem;
        margin: 1.2rem 0 1rem;
    }

    /* ── Distractor badges ─────────────────────────────────────── */
    .rc-distractor {
        display: inline-block;
        background: #fff3f3;
        border: 1px solid #f5c6c6;
        color: #7a2020;
        border-radius: 2px;
        padding: 0.25rem 0.75rem;
        font-size: 0.875rem;
        margin: 0.25rem 0.25rem 0 0;
    }

    /* ── Hint cards ────────────────────────────────────────────── */
    .rc-hint {
        background: #f0f7ff;
        border-left: 3px solid #2563eb;
        padding: 0.9rem 1.2rem;
        border-radius: 0 2px 2px 0;
        font-size: 0.93rem;
        line-height: 1.7;
        color: #1a1a2e;
    }

    /* ── Stat cards (Analytics) ────────────────────────────────── */
    .rc-stat {
        text-align: center;
        padding: 1.5rem 1rem;
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-top: 3px solid var(--accent);
        border-radius: 0 0 2px 2px;
        box-shadow: 2px 2px 0 var(--border);
    }
    .rc-stat .stat-value {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        font-weight: 900;
        color: var(--accent);
        line-height: 1.1;
    }
    .rc-stat .stat-label {
        font-size: 0.8rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
        margin-top: 0.3rem;
    }

    /* ── Submit / Generate buttons ─────────────────────────────── */
    .stButton > button {
        background: var(--ink) !important;
        color: #f7f4ee !important;
        border: none !important;
        border-radius: 2px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 0.6rem 1.8rem !important;
        transition: background 0.2s !important;
    }
    .stButton > button:hover {
        background: var(--accent) !important;
    }

    /* ── Success / Error messages ──────────────────────────────── */
    .stSuccess, .stError { border-radius: 2px !important; }

    /* ── Progress bar colour ───────────────────────────────────── */
    .stProgress > div > div { background-color: var(--accent) !important; }

    /* ── Expander headers ──────────────────────────────────────── */
    [data-testid="stExpander"] summary {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
    }

    /* ── Divider ───────────────────────────────────────────────── */
    .rc-rule { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Resource loading — cached so models load once per session
# ===========================================================================

@st.cache_resource(show_spinner="Loading TF-IDF vectorizer …")
def load_vectorizer():
    if not VECTORIZER_PATH.exists():
        return None
    return joblib.load(VECTORIZER_PATH)


@st.cache_resource(show_spinner="Loading Logistic Regression classifier …")
def load_classifier():
    if not CLASSIFIER_PATH.exists():
        return None
    return joblib.load(CLASSIFIER_PATH)


@st.cache_resource(show_spinner="Loading Calibrated SVM classifier …")
def load_svm():
    if not SVM_PATH.exists():
        return None
    return joblib.load(SVM_PATH)


# ===========================================================================
# Model A inference helpers — Soft-Voting Ensemble
# ===========================================================================

def _get_proba(model, X) -> np.ndarray:
    """Return P(correct) for a single transformed sample."""
    return float(model.predict_proba(X)[0][1])


def verify_option_ensemble(
    article: str,
    question: str,
    option: str,
    vectorizer,
    lr_model,
    svm_model,
) -> float:
    """
    Soft-vote P(correct) by averaging LR and SVM probabilities.
    Falls back to whichever model is available if only one is loaded.
    """
    corpus = article + " " + article + " " + question + " " + option
    X = vectorizer.transform([corpus])

    probs = []
    if lr_model is not None:
        probs.append(_get_proba(lr_model, X))
    if svm_model is not None:
        probs.append(_get_proba(svm_model, X))

    return float(np.mean(probs)) if probs else 0.0


def select_best_answer(
    article: str,
    question: str,
    options: list[str],
    vectorizer,
    lr_model,
    svm_model,
) -> tuple[int, list[float]]:
    """
    Score all options with the Soft-Voting Ensemble and return
    (best_index, all_averaged_probabilities).
    """
    probs = [
        verify_option_ensemble(
            article, question, opt, vectorizer, lr_model, svm_model
        )
        for opt in options
    ]
    return int(np.argmax(probs)), probs


def _ensemble_label(lr_model, svm_model) -> str:
    """Human-readable label describing which models are active."""
    active = []
    if lr_model is not None:
        active.append("LR")
    if svm_model is not None:
        active.append("SVM")
    if len(active) == 2:
        return "Ensemble (LR + SVM)"
    return active[0] if active else "unavailable"


# ===========================================================================
# Session-state initialisation
# ===========================================================================

def _init_state() -> None:
    defaults = {
        "page":            "📝 Article Input",
        "article":         "",
        "question":        "",
        "correct_answer":  "",
        "distractors":     [],
        "hints":           [],
        "quiz_options":    [],
        "correct_idx":     -1,
        "answer_checked":  False,
        "selected_option": None,
        "questions_generated": 0,
        "correct_attempts":    0,
        "total_attempts":      0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ===========================================================================
# Sidebar navigation
# ===========================================================================

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            "<h2 style='font-family:Playfair Display,serif; font-size:1.4rem;"
            "margin-bottom:0.2rem'>RACE System</h2>"
            "<p style='font-size:0.78rem;opacity:0.6;margin-top:0;"
            "letter-spacing:0.06em;text-transform:uppercase'>"
            "Reading Comprehension</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='border-color:#333;margin:0.8rem 0'>",
                    unsafe_allow_html=True)

        pages = [
            "📝 Article Input",
            "📚 Quiz View",
            "💡 Hint Panel",
            "📊 Analytics Dashboard",
        ]
        page = st.radio("Navigation", pages, label_visibility="collapsed")

        st.markdown("<hr style='border-color:#333;margin:1.5rem 0 0.8rem'>",
                    unsafe_allow_html=True)

        st.markdown(
            f"<div style='font-size:0.78rem;opacity:0.55;line-height:1.9'>"
            f"Questions generated: <b>{st.session_state.questions_generated}</b><br>"
            f"Attempts: <b>{st.session_state.total_attempts}</b><br>"
            f"Correct: <b>{st.session_state.correct_attempts}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    return page


# ===========================================================================
# Screen 1 — Article Input
# ===========================================================================

def screen_article_input(vectorizer, lr_model, svm_model) -> None:
    st.markdown(
        "<h1 style='font-size:2.2rem;margin-bottom:0.2rem'>Article Input</h1>"
        "<p style='color:#7a7269;font-size:0.95rem'>Paste an article, enter a "
        "question, and generate a quiz with AI-ranked distractors and hints.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='rc-rule'>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("**Article**")
        article = st.text_area(
            "article_input",
            value=st.session_state.article,
            height=280,
            placeholder="Paste the reading passage here …",
            label_visibility="collapsed",
        )

    with col_right:
        st.markdown("**Question**")
        question = st.text_input(
            "question_input",
            value=st.session_state.question,
            placeholder="What is the main idea of the passage?",
            label_visibility="collapsed",
        )

        st.markdown("**Correct Answer**")
        correct_answer = st.text_input(
            "answer_input",
            value=st.session_state.correct_answer,
            placeholder="Type the correct answer text …",
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        generate_clicked = st.button("⚡  Generate Quiz", use_container_width=True)

    if generate_clicked:
        if not article.strip():
            st.error("Please paste an article before generating.")
        elif not question.strip():
            st.error("Please enter a question.")
        elif not correct_answer.strip():
            st.error("Please enter the correct answer.")
        elif vectorizer is None:
            st.error(
                "Vectorizer not found. Run `preprocessing.py` first to generate "
                f"`{VECTORIZER_PATH}`."
            )
        else:
            with st.spinner("Running Model B — generating distractors & hints …"):
                distractors = generate_distractors(
                    article=article,
                    question=question,
                    correct_answer=correct_answer,
                    vectorizer=vectorizer,
                    n=3,
                )
                hints = generate_hints(
                    article=article,
                    question=question,
                    vectorizer=vectorizer,
                    top_k=3,
                )

            distractor_texts = [d["distractor"] for d in distractors]
            while len(distractor_texts) < 3:
                distractor_texts.append(f"[No distractor {len(distractor_texts)+1} found]")

            quiz_pool = [correct_answer] + distractor_texts
            random.shuffle(quiz_pool)
            correct_idx = quiz_pool.index(correct_answer)

            st.session_state.article         = article
            st.session_state.question        = question
            st.session_state.correct_answer  = correct_answer
            st.session_state.distractors     = distractors
            st.session_state.hints           = hints
            st.session_state.quiz_options    = quiz_pool
            st.session_state.correct_idx     = correct_idx
            st.session_state.answer_checked  = False
            st.session_state.selected_option = None
            st.session_state.questions_generated += 1

            st.success(
                f"Quiz generated — {len(distractors)} distractors, "
                f"{len(hints)} hints. Navigate to **Quiz View** to start!"
            )

            if distractors:
                st.markdown("**Generated distractors:**")
                badges = "".join(
                    f"<span class='rc-distractor'>{d['distractor']} "
                    f"<span style='opacity:0.5'>({d['similarity']:.3f})</span></span>"
                    for d in distractors
                )
                st.markdown(badges, unsafe_allow_html=True)


# ===========================================================================
# Screen 2 — Quiz View
# ===========================================================================

def screen_quiz_view(vectorizer, lr_model, svm_model) -> None:
    st.markdown(
        "<h1 style='font-size:2.2rem;margin-bottom:0.2rem'>Quiz View</h1>"
        "<p style='color:#7a7269;font-size:0.95rem'>Read the passage, answer the "
        "question, and Model A will verify your choice.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='rc-rule'>", unsafe_allow_html=True)

    if not st.session_state.article:
        st.info("No quiz loaded yet. Go to **Article Input** and click Generate Quiz.")
        return

    st.markdown("**Passage**")
    st.markdown(
        f"<div class='rc-article'>{st.session_state.article}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='rc-question'>{st.session_state.question}</div>",
        unsafe_allow_html=True,
    )

    labels = ["A", "B", "C", "D"]
    options = st.session_state.quiz_options
    radio_labels = [f"({labels[i]})  {opt}" for i, opt in enumerate(options)]

    selected = st.radio(
        "Choose your answer:",
        radio_labels,
        index=None,
        label_visibility="visible",
    )

    col_submit, col_reset = st.columns([1, 5])
    submit_clicked = col_submit.button("Submit Answer", use_container_width=True)

    if submit_clicked:
        if selected is None:
            st.warning("Please select an answer before submitting.")
        else:
            selected_idx  = radio_labels.index(selected)
            selected_text = options[selected_idx]
            is_correct    = (selected_idx == st.session_state.correct_idx)

            st.session_state.total_attempts += 1
            if is_correct:
                st.session_state.correct_attempts += 1

            st.session_state.answer_checked  = True
            st.session_state.selected_option = selected_text

            # ── Model A Ensemble inference ───────────────────────────────
            model_a_available = vectorizer is not None and (
                lr_model is not None or svm_model is not None
            )

            if model_a_available:
                best_idx, probs = select_best_answer(
                    article    = st.session_state.article,
                    question   = st.session_state.question,
                    options    = options,
                    vectorizer = vectorizer,
                    lr_model   = lr_model,
                    svm_model  = svm_model,
                )
                model_pick   = options[best_idx]
                active_label = _ensemble_label(lr_model, svm_model)
            else:
                model_pick   = None
                active_label = "unavailable"

            # ── User feedback ─────────────────────────────────────────────
            if is_correct:
                st.success(
                    f"✅ **Correct!** '{selected_text}' is the right answer."
                )
            else:
                st.error(
                    f"❌ **Incorrect.** You chose: '{selected_text}'.\n\n"
                    f"The correct answer was: "
                    f"**'{st.session_state.correct_answer}'**"
                )

            if model_a_available:
                if model_pick == st.session_state.correct_answer:
                    st.markdown(
                        f"<div style='font-size:0.85rem;color:#1a6b3c;margin-top:0.5rem'>"
                        f"🤖 Model A ({active_label}) selected: <b>{model_pick}</b> ✓</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='font-size:0.85rem;color:#7a7269;margin-top:0.5rem'>"
                        f"🤖 Model A ({active_label}) selected: <b>{model_pick}</b> "
                        f"(ensemble val accuracy: ~32%)</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption(
                    "Model A classifier not found — run `model_a_train.py` "
                    "and `model_a_supervised_svm.py` to enable verification."
                )

            # ── Per-option confidence bars ─────────────────────────────────
            if model_a_available:
                st.markdown("<hr class='rc-rule'>", unsafe_allow_html=True)
                st.markdown(
                    f"**Model A ({active_label}) confidence scores:**"
                )
                for i, (opt, prob) in enumerate(zip(options, probs)):
                    bar_col, label_col = st.columns([4, 1])
                    bar_col.progress(float(prob), text=f"({labels[i]}) {opt[:60]}")
                    label_col.markdown(
                        f"<div style='padding-top:0.6rem;font-size:0.85rem;"
                        f"color:#7a7269'>{prob:.3f}</div>",
                        unsafe_allow_html=True,
                    )


# ===========================================================================
# Screen 3 — Hint Panel  (unchanged)
# ===========================================================================

def screen_hint_panel() -> None:
    st.markdown(
        "<h1 style='font-size:2.2rem;margin-bottom:0.2rem'>Hint Panel</h1>"
        "<p style='color:#7a7269;font-size:0.95rem'>Top-ranked sentences from the "
        "passage, extracted by TF-IDF cosine similarity to the question.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='rc-rule'>", unsafe_allow_html=True)

    if not st.session_state.hints:
        st.info("No hints generated yet. Go to **Article Input** and click Generate Quiz.")
        return

    st.markdown(
        f"<div style='font-size:0.9rem;color:#7a7269;margin-bottom:1rem'>"
        f"Question: <em>{st.session_state.question}</em></div>",
        unsafe_allow_html=True,
    )

    for hint in st.session_state.hints:
        rank      = hint["rank"]
        sentence  = hint["sentence"]
        sim_score = hint["similarity"]
        position  = hint.get("position", "?")

        label = (
            f"Hint #{rank}  —  relevance {sim_score:.4f}  "
            f"(sentence {position} in passage)"
        )
        with st.expander(label, expanded=(rank == 1)):
            st.markdown(
                f"<div class='rc-hint'>{sentence}</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Cosine similarity to question: {sim_score:.4f}  ·  "
                f"Original position in article: sentence #{position}"
            )

    st.markdown("<hr class='rc-rule'>", unsafe_allow_html=True)
    st.caption(
        "Hints are extractive — they are exact sentences from the passage, "
        "ranked by TF-IDF cosine similarity.  Hint #1 is the most relevant."
    )


# ===========================================================================
# Screen 4 — Analytics Dashboard
# ===========================================================================

def screen_analytics(lr_model, svm_model) -> None:
    st.markdown(
        "<h1 style='font-size:2.2rem;margin-bottom:0.2rem'>Analytics Dashboard</h1>"
        "<p style='color:#7a7269;font-size:0.95rem'>Session statistics and model "
        "performance summary.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='rc-rule'>", unsafe_allow_html=True)

    # ── Session stats ────────────────────────────────────────────────────
    total     = st.session_state.total_attempts
    correct   = st.session_state.correct_attempts
    user_acc  = (correct / total * 100) if total > 0 else 0.0
    generated = st.session_state.questions_generated

    st.markdown("#### Session Performance")
    c1, c2, c3, c4 = st.columns(4)

    def stat_card(col, value: str, label: str) -> None:
        col.markdown(
            f"<div class='rc-stat'>"
            f"<div class='stat-value'>{value}</div>"
            f"<div class='stat-label'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    stat_card(c1, str(generated),     "Quizzes Generated")
    stat_card(c2, str(total),         "Answers Submitted")
    stat_card(c3, str(correct),       "Correct Answers")
    stat_card(c4, f"{user_acc:.1f}%", "User Accuracy")

    # ── Model A — individual & ensemble accuracy ─────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Model A — Answer Verifier  *(val set, answer-selection)*")

    m1, m2, m3, m4 = st.columns(4)
    stat_card(m1, "31.70%",   "LR Accuracy")
    stat_card(m2, "31.10%",   "SVM Accuracy")
    stat_card(m3, "~32%",     "Ensemble Accuracy")
    stat_card(m4, "15 000",   "TF-IDF Features")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.85rem;color:#7a7269;margin-bottom:0.5rem'>"
        "Validation accuracy vs. random baseline (25 %)</p>",
        unsafe_allow_html=True,
    )
    col_gauge, _ = st.columns([2, 3])
    with col_gauge:
        st.progress(0.32,   text="Ensemble  — ~32.00 %")
        st.progress(0.3170, text="Logistic Regression — 31.70 %")
        st.progress(0.3110, text="Linear SVM — 31.10 %")
        st.progress(0.25,   text="Random baseline — 25.00 %")

    # ── Model status badges ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    lr_status  = "✅ Loaded" if lr_model  is not None else "⚠️ Not found"
    svm_status = "✅ Loaded" if svm_model is not None else "⚠️ Not found"
    st.markdown(
        f"<div style='font-size:0.85rem;color:#7a7269'>"
        f"LR model: <b>{lr_status}</b> &nbsp;·&nbsp; "
        f"SVM model: <b>{svm_status}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Model B summary ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Model B — Distractor & Hint Generator")

    b1, b2, b3 = st.columns(3)
    stat_card(b1, "MMR",    "Distractor Ranker")
    stat_card(b2, "Cosine", "Hint Scorer")
    stat_card(b3, "3",      "Hints per Question")

    # ── Technical stack ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Technical Stack")

    info_rows = [
        ("Vectorizer",      "TF-IDF  |  stop_words='english'  |  sublinear_tf=True  |  max_features=15 000"),
        ("Classifier",      "Soft-Voting Ensemble (Logistic Regression + Calibrated LinearSVC)  |  averaged predict_proba"),
        ("LR config",       "solver=saga  |  class_weight=balanced  |  C=1.0"),
        ("SVM config",      "LinearSVC(C=1.0)  wrapped in CalibratedClassifierCV(cv=3)"),
        ("Distractor algo", "n-gram extraction → TF-IDF cosine similarity → MMR diversity selection"),
        ("Hint algo",       "Sentence splitting → TF-IDF cosine similarity → top-k ranking"),
        ("Framework",       "Streamlit  ·  scikit-learn  ·  scipy.sparse  ·  joblib"),
        ("Dataset",         "RACE (Lai et al., 2017) — Reading Comprehension from Examinations"),
    ]

    for label, detail in info_rows:
        col_l, col_d = st.columns([1, 4])
        col_l.markdown(
            f"<div style='font-size:0.82rem;font-weight:500;color:#7a7269;"
            f"padding-top:0.65rem;text-transform:uppercase;letter-spacing:0.05em'>"
            f"{label}</div>",
            unsafe_allow_html=True,
        )
        col_d.markdown(
            f"<div style='font-size:0.88rem;padding-top:0.6rem;color:#1a1a1a'>"
            f"{detail}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='rc-rule'>", unsafe_allow_html=True)
    st.caption(
        "AL2002 Lab Project — FAST School of Computing, 2026.  "
        "Model accuracy is computed on the RACE validation split."
    )


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    _init_state()

    vectorizer = load_vectorizer()
    lr_model   = load_classifier()
    svm_model  = load_svm()

    page = render_sidebar()
    st.session_state.page = page

    if page == "📝 Article Input":
        screen_article_input(vectorizer, lr_model, svm_model)
    elif page == "📚 Quiz View":
        screen_quiz_view(vectorizer, lr_model, svm_model)
    elif page == "💡 Hint Panel":
        screen_hint_panel()
    elif page == "📊 Analytics Dashboard":
        screen_analytics(lr_model, svm_model)


if __name__ == "__main__":
    main()