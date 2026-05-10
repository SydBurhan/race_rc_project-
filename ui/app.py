"""
ui/app.py
=========
Streamlit UI — RACE Reading Comprehension & Quiz Generation System.
Traditional ML only (no neural networks).
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Path bootstrap so `src.*` resolves ────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import (
    predict_answer,
    generate_question,
    generate_distractors,
    generate_hints,
)

PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics_test.json"

# ══════════════════════════════════════════════════════════════════════════════
#  Page config — must be first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RACE RC Quiz System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  Global CSS & Font Injection
# ══════════════════════════════════════════════════════════════════════════════
# NOTE: Streamlit 1.42+ silently strips <link> tags from st.markdown even with
# unsafe_allow_html=True, which can leave the following <style> block half
# parsed and dump raw CSS text onto the page. Importing the Google Font from
# inside the <style> block via @import avoids that and keeps the styling
# self contained in a single safe HTML payload.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ── Base Reset ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stApp {
    background-color: #f5f7fa !important;
}

/* ── Hide default Streamlit chrome ──────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px !important;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0d2b24 !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * {
    color: #d1fae5 !important;
}
[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    padding: 0.55rem 0.85rem !important;
    margin-bottom: 6px !important;
    display: flex !important;
    align-items: center !important;
    transition: background 0.2s ease !important;
    color: #a7f3d0 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(16,185,129,0.18) !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #6ee7b7 !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
}
[data-testid="stSidebar"] h1 {
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
}

/* ── Page Title Cards ─────────────────────────────────────── */
.page-header {
    background: linear-gradient(135deg, #0d2b24 0%, #134e3a 100%);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 4px 20px rgba(13,43,36,0.18);
}
.page-header .icon {
    font-size: 2rem;
    background: rgba(255,255,255,0.12);
    border-radius: 12px;
    width: 56px; height: 56px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.page-header h1 {
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.2 !important;
}
.page-header p {
    font-size: 0.85rem !important;
    color: #6ee7b7 !important;
    margin: 0 !important;
    font-weight: 500 !important;
}

/* ── Content Cards ────────────────────────────────────────── */
.card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 1.2rem;
    border: 1px solid #e8f5f0;
}
.card-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #059669;
    margin-bottom: 0.6rem;
}

/* ── Quiz Answer Options ─────────────────────────────────── */
.answer-option {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: #ffffff;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.65rem;
    cursor: pointer;
    transition: all 0.18s ease;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.answer-option:hover {
    border-color: #10b981;
    background: #f0fdf9;
    box-shadow: 0 3px 12px rgba(16,185,129,0.12);
    transform: translateY(-1px);
}
.answer-option.selected {
    border-color: #10b981;
    background: #ecfdf5;
}
.answer-option.correct {
    border-color: #16a34a;
    background: #dcfce7;
}
.answer-option.wrong {
    border-color: #dc2626;
    background: #fef2f2;
}
.option-badge {
    width: 36px; height: 36px;
    border-radius: 8px;
    background: #f1f5f9;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
    color: #475569;
    flex-shrink: 0;
    transition: all 0.18s;
}
.answer-option.selected .option-badge {
    background: #10b981;
    color: #ffffff;
}
.answer-option.correct .option-badge {
    background: #16a34a;
    color: #ffffff;
}
.answer-option.wrong .option-badge {
    background: #dc2626;
    color: #ffffff;
}
.option-text {
    font-size: 0.92rem;
    font-weight: 500;
    color: #1e293b;
    flex: 1;
}
.option-check {
    color: #10b981;
    font-size: 1.1rem;
    opacity: 0;
}
.answer-option.selected .option-check { opacity: 1; }

/* ── Hint Boxes ────────────────────────────────────────────── */
.hint-card {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.75rem;
}
.hint-card .hint-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #b45309;
    margin-bottom: 0.3rem;
}
.hint-card .hint-text { color: #1f2937; font-size: 0.9rem; }
.hint-card.blurred { filter: blur(5px); user-select: none; }

/* ── Badges ─────────────────────────────────────────────────── */
.ai-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: #fef3c7; color: #92400e; font-size: 0.72rem;
    padding: 4px 10px; border-radius: 20px; font-weight: 700;
    border: 1px solid #fde68a;
}
.branch-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: #e0e7ff; color: #3730a3; font-size: 0.72rem;
    padding: 4px 10px; border-radius: 20px; font-weight: 700;
}
.correct-badge {
    display: flex; align-items: center; gap: 8px;
    background: #dcfce7; color: #15803d;
    padding: 0.85rem 1.1rem; border-radius: 10px;
    font-weight: 700; font-size: 1rem;
    border: 1px solid #bbf7d0;
    margin: 0.75rem 0;
}
.wrong-badge {
    display: flex; align-items: center; gap: 8px;
    background: #fef2f2; color: #dc2626;
    padding: 0.85rem 1.1rem; border-radius: 10px;
    font-weight: 700; font-size: 1rem;
    border: 1px solid #fecaca;
    margin: 0.75rem 0;
}

/* ── Passage Excerpt Card ────────────────────────────────── */
.passage-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: #475569;
    line-height: 1.7;
    margin-bottom: 1rem;
    border-left: 4px solid #10b981;
}

/* ── Question Display ─────────────────────────────────────── */
.question-display {
    font-size: 1.2rem;
    font-weight: 700;
    color: #0f172a;
    margin: 1rem 0 1.4rem 0;
    line-height: 1.5;
}

/* ── Word Count Bar ───────────────────────────────────────── */
.wc-bar {
    display: flex; align-items: center; gap: 10px;
    font-size: 0.82rem; color: #64748b; font-weight: 500;
    margin-top: 0.4rem;
}
.wc-pill {
    background: #dcfce7; color: #166534;
    padding: 2px 10px; border-radius: 20px;
    font-weight: 700; font-size: 0.78rem;
}

/* ── Section Dividers ─────────────────────────────────────── */
.section-divider {
    border: none;
    border-top: 2px dashed #e2e8f0;
    margin: 1.2rem 0;
}

/* ── Metric Card Override ─────────────────────────────────── */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e8f5f0 !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}
[data-testid="stMetricValue"] {
    font-weight: 800 !important;
    color: #0f172a !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #64748b !important;
}

/* ── Primary Button ───────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #059669, #0d9488) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    padding: 0.6rem 1.4rem !important;
    box-shadow: 0 4px 14px rgba(5,150,105,0.3) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.02em !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(5,150,105,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Secondary Buttons ────────────────────────────────────── */
.stButton > button[kind="secondary"] {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.18s ease !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #10b981 !important;
    color: #059669 !important;
    background: #f0fdf9 !important;
}
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── Text Area ────────────────────────────────────────────── */
.stTextArea textarea {
    border-radius: 12px !important;
    border: 2px solid #e2e8f0 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.9rem !important;
    color: #1e293b !important;
    line-height: 1.7 !important;
    padding: 0.85rem 1rem !important;
    transition: border-color 0.2s ease !important;
}
.stTextArea textarea:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.12) !important;
}

/* ── File Uploader ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #f8fafc !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* ── Info/Warning/Success ─────────────────────────────────── */
.stAlert {
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

/* ── Spinner ──────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #10b981 !important; }

/* ── Dataframe ────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
}

/* ── Caption ──────────────────────────────────────────────── */
.stCaption {
    font-size: 0.78rem !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
}

/* ── Analytics subheaders ─────────────────────────────────── */
.analytics-section {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 1.4rem;
    border: 1px solid #e8f5f0;
}
.analytics-section h3 {
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: #059669 !important;
    margin-bottom: 1rem !important;
}

/* ── Verifier result card ─────────────────────────────────── */
.verifier-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-top: 0.75rem;
    font-size: 0.88rem;
    color: #475569;
}
.verifier-card strong { color: #0f172a; }

/* ── Quick Load Panel ─────────────────────────────────────── */
.quickload-panel {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #e8f5f0;
    height: 100%;
}
.quickload-panel h4 {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #059669 !important;
    margin-bottom: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Cached loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading RACE test samples …")
def load_race_samples(n: int = 200) -> list[dict]:
    """Load test samples from local processed CSV."""
    csv = PROCESSED_DIR / "test.csv"
    if not csv.exists():
        return []
    df = pd.read_csv(csv).head(n)
    out = []
    for _, r in df.iterrows():
        out.append({
            "article": r.get("article", ""),
            "question": r.get("question", ""),
            "answer": str(r.get("answer", "A")).upper(),
            "A": r.get("A", ""), "B": r.get("B", ""),
            "C": r.get("C", ""), "D": r.get("D", ""),
        })
    return out


@st.cache_data(show_spinner=False)
def load_test_metrics() -> dict | None:
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text())
        except Exception:
            return None
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Session state
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "screen": "input",
        "passage": "",
        "pipeline_result": None,
        "user_choice": None,
        "answer_checked": False,
        "hints_revealed": 0,
        "session_log": [],
        "race_samples": [],
        "current_sample": None,
        "verifier_log": [],
        "latency_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════

SCREEN_LABELS = {
    "input":     "🏠 Article Input",
    "quiz":      "❓ Quiz View",
    "hints":     "💡 Hint Panel",
    "analytics": "📊 Analytics",
}
LABEL_TO_KEY = {v: k for k, v in SCREEN_LABELS.items()}

with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 0.25rem 0;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.3rem;">
            <div style="background:linear-gradient(135deg,#10b981,#0d9488);
                        border-radius:10px;width:38px;height:38px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:1.2rem;flex-shrink:0;">📚</div>
            <div>
                <div style="font-size:1rem;font-weight:800;color:#ffffff;line-height:1.2;">RACE RC Quiz</div>
                <div style="font-size:0.72rem;color:#6ee7b7;font-weight:500;">ML Pipeline System</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    if "pending_screen" in st.session_state:
        st.session_state["nav_radio"] = SCREEN_LABELS.get(
            st.session_state.pending_screen, SCREEN_LABELS["input"]
        )
        del st.session_state["pending_screen"]

    nav = st.radio(
        "Navigation",
        list(SCREEN_LABELS.values()),
        key="nav_radio",
    )
    st.session_state.screen = LABEL_TO_KEY[nav]
    st.divider()

    # Session stats
    quiz_count = len(st.session_state.session_log)
    verifier_count = len(st.session_state.verifier_log)
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;gap:6px;">
        <div style="background:rgba(255,255,255,0.07);border-radius:8px;
                    padding:0.5rem 0.75rem;display:flex;justify-content:space-between;
                    align-items:center;">
            <span style="font-size:0.78rem;color:#6ee7b7;font-weight:500;">Quizzes Generated</span>
            <span style="font-size:0.88rem;font-weight:800;color:#ffffff;">{quiz_count}</span>
        </div>
        <div style="background:rgba(255,255,255,0.07);border-radius:8px;
                    padding:0.5rem 0.75rem;display:flex;justify-content:space-between;
                    align-items:center;">
            <span style="font-size:0.78rem;color:#6ee7b7;font-weight:500;">Answers Checked</span>
            <span style="font-size:0.88rem;font-weight:800;color:#ffffff;">{verifier_count}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Pipeline runner: routes a passage through Branch A (RACE), B (regex), or C (template).

def _extract_qa_from_passage(passage: str) -> tuple[str, str] | None:
    import re
    passage = re.sub(r"\s+", " ", str(passage)).strip()
    APPROX = r"(?:over|nearly|about|more than|around|roughly|approximately)"
    def _clean_subject(s: str) -> str:
        s = s.strip().rstrip(",;")
        if s.lower().startswith("the ") and len(s.split()) >= 2:
            s = s[4:]
        return s
    VERB_BASE = {
        "stretches": "stretch", "stretched": "stretch",
        "extends":   "extend",  "extended":  "extend",
        "spans":     "span",    "spanned":   "span",
        "reaches":   "reach",   "reached":   "reach",
        "measures":  "measure", "measured":  "measure",
        "is":        "be",      "are":       "be",
        "was":       "be",      "were":      "be",
        "moved":     "move",    "arrived":   "arrive",
        "enrolled":  "enroll",  "earned":    "earn",
        "returned":  "return",  "died":      "die",
        "published": "publish", "wrote":     "write",
        "built":     "build",   "founded":   "found",
        "signed":    "sign",    "invented":  "invent",
        "discovered":"discover","opened":    "open",
        "won":       "win",     "received":  "receive",
        "joined":    "join",    "graduated": "graduate",
    }
    def _base(verb: str) -> str:
        return VERB_BASE.get(verb.lower(), verb.lower())
    PRONOUNS = {"he", "she", "it", "they", "we", "you", "i", "him", "her",
                "them", "us", "his", "hers", "their", "our", "your", "my"}
    def _leading_proper_noun(text: str) -> str:
        m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", text)
        if m:
            return m.group(1)
        m2 = re.search(r"\b([A-Z][a-z]+)\b", text)
        return m2.group(1) if m2 else "the subject"
    def _resolve_subject(raw: str) -> str:
        cleaned = _clean_subject(raw)
        toks = cleaned.split()
        if toks and toks[0].lower() in PRONOUNS:
            return _leading_proper_noun(passage)
        if len(toks) == 1 and toks[0].lower() in PRONOUNS:
            return _leading_proper_noun(passage)
        return cleaned
    patterns = [
        (
            rf"(?P<subj>[A-Z][^.\n]{{0,80}}?)\s+(?P<verb>stretches|extends|spans|reaches|measures)\s+(?:for\s+)?(?:{APPROX}\s+)?(?P<num>\d[\d,\.]*)\s+(?P<unit>kilometers?|km|miles?|meters?|feet|ft)\b",
            lambda m: (f"How long does the {_clean_subject(m['subj'])} {_base(m['verb'])}?", f"{m['num']} {m['unit']}"),
        ),
        (
            rf"(?P<subj>[A-Z][^.\n]{{0,80}}?)\s+(?:is|are)\s+home to\s+(?:{APPROX}\s+)?(?P<num>\d[\d,\.]*)\s+(?P<unit>(?:species|types|kinds)(?:\s+of\s+\w+)?)",
            lambda m: (f"How many {m['unit']} is {_clean_subject(m['subj'])} home to?", f"{m['num']} {m['unit']}"),
        ),
        (
            r"(?P<subj>[A-Z][^.\n]{0,80}?)\s+(?:was|were)\s+(?P<verb>designated|established|created|founded|built|published|formed|launched|opened|discovered|awarded|elected|named|appointed|crowned|invented|introduced|signed|completed|destroyed)\b[^.\n]{0,80}?\bin\s+(?P<year>\d{4})\b",
            lambda m: (f"When was {_resolve_subject(m['subj'])} {m['verb']}?", m["year"]),
        ),
        (
            r"(?P<subj>[A-Z][^.\n]{0,80}?)\s+(?P<verb>won|received|earned|joined|graduated|enrolled|moved|arrived|returned|died|published|wrote|built|founded|signed|invented|discovered|opened)\b[^.\n]{0,80}?\bin\s+(?P<year>\d{4})\b",
            lambda m: (f"When did {_resolve_subject(m['subj'])} {_base(m['verb'])}?", m["year"]),
        ),
        (
            r"(?P<subj>[A-Z][^.\n]{0,80}?)\s+(?:was\s+)?[Bb]orn\s+in\s+(?P<place>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+in\s+(?P<year>\d{4})\b",
            lambda m: (f"When was {_resolve_subject(m['subj'])} born?", m["year"]),
        ),
        (
            r"\b[Bb]orn\s+in\s+(?P<place>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+in\s+(?P<year>\d{4})\b",
            lambda m: (f"Where was {_leading_proper_noun(passage)} born?", m["place"]),
        ),
        (
            rf"(?P<subj>[A-Z][^.\n]{{0,80}}?)\s+(?:is|are)\s+composed of\s+(?:{APPROX}\s+)?(?P<num>\d[\d,\.]*)\s+(?P<unit>[a-z]+(?:\s+[a-z]+){{0,2}})",
            lambda m: (f"How many {m['unit']} is {_clean_subject(m['subj'])} composed of?", f"{m['num']} {m['unit']}"),
        ),
        (
            r"(?P<subj>[A-Z][^.\n]{0,80}?),?\s+located\s+(?:off\s+the\s+|in\s+the\s+|in\s+|on\s+the\s+|on\s+|near\s+the\s+|near\s+)(?P<place>(?:\w+\s+){0,4}(?:coast|coastline|shore|region|country|state|province|city|island|continent|peninsula)\s+of\s+[A-Z][a-zA-Z]+|[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})",
            lambda m: (f"Where is {_clean_subject(m['subj'])} located?", m["place"].strip().rstrip(",;.!?")),
        ),
    ]
    for pat, builder in patterns:
        m = re.search(pat, passage)
        if not m:
            continue
        try:
            q, a = builder(m)
            q = re.sub(r"\s+", " ", q).strip()
            a = re.sub(r"\s+", " ", a).strip()
            if q and a:
                return q, a
        except Exception:
            continue
    return None


def _pick_answer_from_passage(passage: str) -> str:
    import re
    STOP = {
        "The", "This", "That", "These", "Those", "There", "Their", "They",
        "When", "Where", "Why", "How", "What", "Which", "While", "However",
        "Although", "Because", "Also", "Many", "Some", "Most", "Several",
        "Scientists", "Marine",
    }
    nums = re.findall(r"\b\d[\d,\.]*\b", passage)
    nums = [n for n in nums if len(n) >= 2]
    if nums:
        return nums[0]
    multi = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", passage)
    multi = [m for m in multi if m.split()[0] not in STOP]
    if multi:
        from collections import Counter
        return Counter(multi).most_common(1)[0][0]
    sentences = re.split(r"[.!?]\s+", passage)
    candidates = []
    for s in sentences:
        toks = s.split()
        for tok in toks[1:]:
            clean = re.sub(r"[^A-Za-z]", "", tok)
            if clean and clean[0].isupper() and clean not in STOP and len(clean) > 2:
                candidates.append(clean)
    if candidates:
        from collections import Counter
        return Counter(candidates).most_common(1)[0][0]
    words = [w.strip(".,!?;:\"'") for w in passage.split()]
    words = [w for w in words if len(w) > 4 and w.lower() not in {
        "about", "their", "there", "which", "while", "where", "because",
        "according", "another"
    }]
    return words[0] if words else "the topic"


def _run_pipeline(passage: str, ground_truth_answer: str | None = None,
                   gt_options: dict | None = None,
                   reference_question: str | None = None) -> dict:
    if reference_question and gt_options and ground_truth_answer:
        _branch_used = "A: RACE sample (real human Q + real options)"
        question = reference_question
        answer_text = ground_truth_answer
        latency_q_ms = 0.0
        latency_d_ms = 0.0
        st.session_state.latency_log.append(latency_q_ms)
        option_map = {lbl: gt_options[lbl] for lbl in ("A", "B", "C", "D")}
        correct_label = next(
            (lbl for lbl, txt in option_map.items()
             if str(txt).strip() == str(answer_text).strip()),
            "A",
        )
        distractors = [option_map[l] for l in ("A", "B", "C", "D")
                       if l != correct_label]
    else:
        extracted = _extract_qa_from_passage(passage) if not ground_truth_answer else None
        if extracted is not None:
            _branch_used = "B: Relation extractor pattern matched"
            question, answer_text = extracted
            latency_q_ms = 0.0
            st.session_state.latency_log.append(latency_q_ms)
        else:
            _branch_used = "C: Template fallback (no pattern matched)"
            answer_text = ground_truth_answer or _pick_answer_from_passage(passage)
            t0 = time.perf_counter()
            question = generate_question(passage, answer_text)
            latency_q_ms = (time.perf_counter() - t0) * 1000
            st.session_state.latency_log.append(latency_q_ms)
        t0 = time.perf_counter()
        distractors = generate_distractors(passage, question, answer_text)
        latency_d_ms = (time.perf_counter() - t0) * 1000
        st.session_state.latency_log.append(latency_d_ms)
        options_list = list(distractors[:3]) + [answer_text]
        rng = random.Random(42)
        rng.shuffle(options_list)
        correct_label = ["A", "B", "C", "D"][options_list.index(answer_text)]
        option_map = dict(zip(["A", "B", "C", "D"], options_list))

    t0 = time.perf_counter()
    hints = generate_hints(passage, question, answer_text)
    latency_h_ms = (time.perf_counter() - t0) * 1000
    st.session_state.latency_log.append(latency_h_ms)
    nlp_metrics = {}
    if reference_question:
        from src.evaluate_nlp import score_single
        nlp_metrics = score_single(question, reference_question)
    return {
        "branch_used": _branch_used,
        "passage": passage,
        "question": question,
        "answer": answer_text,
        "correct_label": correct_label,
        "option_map": option_map,
        "distractors": distractors,
        "hints": hints,
        "latency_q_ms": round(latency_q_ms, 2),
        "latency_d_ms": round(latency_d_ms, 2),
        "latency_h_ms": round(latency_h_ms, 2),
        "nlp_metrics": nlp_metrics,
        "has_race_reference": reference_question is not None,
    }


# Rubric 7.1: Screen 1 - paste, upload, or load a random RACE sample.

def screen_input():
    st.markdown("""
    <div class="page-header">
        <div class="icon">🏠</div>
        <div>
            <h1>Article Input</h1>
            <p>Paste a passage, upload a .txt file, or load a random RACE sample</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 1], gap="medium")

    with col_right:
        st.markdown("""
        <div class="quickload-panel">
            <h4>⚡ Quick Load</h4>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎲 Random RACE Sample", use_container_width=True):
            if not st.session_state.race_samples:
                st.session_state.race_samples = load_race_samples()
            if st.session_state.race_samples:
                sample = random.choice(st.session_state.race_samples)
                st.session_state["passage_area"] = sample["article"]
                st.session_state.passage = sample["article"]
                st.session_state.current_sample = sample
                st.rerun()
            else:
                st.warning("No RACE samples available. Run preprocessing.py first.")

        st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload .txt", type=["txt"])
        if uploaded:
            text = uploaded.read().decode("utf-8")
            st.session_state["passage_area"] = text
            st.session_state.passage = text
            st.session_state.current_sample = None
            st.rerun()

        if st.session_state.current_sample:
            st.markdown("""
            <div style="background:#ecfdf5;border:1px solid #a7f3d0;border-radius:10px;
                        padding:0.7rem 0.9rem;margin-top:0.75rem;">
                <div style="font-size:0.72rem;font-weight:700;color:#065f46;
                            text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.2rem;">
                    ✅ RACE Sample Loaded
                </div>
                <div style="font-size:0.78rem;color:#047857;">
                    Ground-truth Q&A available
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_left:
        st.markdown('<div class="card-title">📄 Reading Passage</div>', unsafe_allow_html=True)
        passage = st.text_area(
            "Reading Passage",
            height=300,
            placeholder="Paste a reading passage of at least 30 words …",
            key="passage_area",
            label_visibility="collapsed",
        )
        st.session_state.passage = passage

        word_count = len(passage.split()) if passage.strip() else 0
        wc_color = "#dcfce7" if word_count >= 30 else "#fee2e2"
        wc_text_color = "#166534" if word_count >= 30 else "#991b1b"
        st.markdown(f"""
        <div class="wc-bar">
            <span>Word count:</span>
            <span style="background:{wc_color};color:{wc_text_color};
                         padding:2px 10px;border-radius:20px;font-weight:700;font-size:0.78rem;">
                {word_count} words
            </span>
            {'<span style="color:#10b981;font-size:0.8rem;">✓ Ready to generate</span>' if word_count >= 30 else '<span style="color:#ef4444;font-size:0.8rem;">Need at least 30 words</span>'}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col_btn, col_info = st.columns([2, 3])
    with col_btn:
        if st.button("🚀 Generate Quiz", type="primary",
                     disabled=word_count < 30, use_container_width=True):
            sample = st.session_state.get("current_sample")
            gt_answer = sample[sample["answer"]] if sample else None
            gt_options = (
                {"A": sample["A"], "B": sample["B"], "C": sample["C"], "D": sample["D"]}
                if sample else None
            )
            ref_q = sample["question"] if sample else None

            with st.spinner("Generating quiz with traditional ML pipeline …"):
                try:
                    result = _run_pipeline(passage, gt_answer, gt_options, ref_q)
                except FileNotFoundError as e:
                    st.error(f"Models missing: {e}")
                    st.info("Run `python src/preprocessing.py` then "
                            "`python src/model_a_train.py`, `python src/template_generator.py`, "
                            "and `python src/model_b_train.py` to train all components.")
                    return
                except ValueError as e:
                    st.error(str(e))
                    return
                except Exception as e:
                    import traceback
                    st.error(f"Unexpected error during generation: {type(e).__name__}: {e}")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())
                    return

            st.session_state.pipeline_result = result
            st.session_state.user_choice = None
            st.session_state.answer_checked = False
            st.session_state.hints_revealed = 1
            st.session_state.pop("_verifier_cache", None)
            st.session_state.session_log.append({
                "passage": passage[:80],
                "question": result["question"],
                "answer": result["answer"],
                "latency_q_ms": result["latency_q_ms"],
                "latency_d_ms": result["latency_d_ms"],
                "latency_h_ms": result["latency_h_ms"],
            })
            st.session_state.pending_screen = "quiz"
            st.rerun()

    with col_info:
        if word_count < 30:
            st.markdown("""
            <div style="background:#fef9ee;border:1px solid #fde68a;border-radius:10px;
                        padding:0.7rem 1rem;font-size:0.83rem;color:#78350f;font-weight:500;">
                💬 Please provide at least 30 words to generate a quiz.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                        padding:0.7rem 1rem;font-size:0.83rem;color:#14532d;font-weight:500;">
                ✅ Passage looks good! Click Generate Quiz to continue.
            </div>
            """, unsafe_allow_html=True)


# Rubric 7.2: Screen 2 - generated question, four options, Check Answer, colour coding.

def screen_quiz():
    st.markdown("""
    <div class="page-header">
        <div class="icon">❓</div>
        <div>
            <h1>Quiz View</h1>
            <p>Read the question carefully and select the best answer</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    result = st.session_state.pipeline_result
    if not result:
        st.warning("No quiz generated yet. Go to Article Input first.")
        return

    col_main, col_side = st.columns([2.2, 1], gap="large")

    with col_main:
        # Passage excerpt
        excerpt = result['passage'][:250]
        st.markdown(f"""
        <div class="card-title">📖 Passage Excerpt</div>
        <div class="passage-card">
            {excerpt}…
        </div>
        """, unsafe_allow_html=True)

        # Branch badge
        if result.get("branch_used"):
            st.markdown(
                f'<span class="branch-badge">🛠️ {result["branch_used"]}</span>',
                unsafe_allow_html=True
            )
            st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)

        # Question
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Question</div>
            <div class="question-display">{result['question']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card-title" style="margin-bottom:0.5rem;">Choose Your Answer</div>',
                    unsafe_allow_html=True)

        # Answer options. NOTE: Streamlit reruns automatically when a button
        # is clicked, so an explicit st.rerun() here would cause a SECOND
        # rerun and visibly double the latency on every click. Removed.
        for label in ["A", "B", "C", "D"]:
            text = result["option_map"][label]
            if st.button(f"{label}. {text}", key=f"opt_{label}",
                         use_container_width=True):
                st.session_state.user_choice = label
                st.session_state.answer_checked = False
                # Invalidate any cached verifier result for the previous choice
                st.session_state.pop("_verifier_cache", None)

        # Selected indicator
        if st.session_state.user_choice and not st.session_state.answer_checked:
            chosen_text = result["option_map"][st.session_state.user_choice]
            st.markdown(f"""
            <div style="background:#ecfdf5;border:1px solid #a7f3d0;border-radius:10px;
                        padding:0.65rem 1rem;margin-top:0.5rem;font-size:0.85rem;
                        color:#065f46;font-weight:600;">
                Selected: <strong>{st.session_state.user_choice}</strong> — {chosen_text}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        if st.session_state.user_choice:
            col_check, col_hint = st.columns([1, 1])
            with col_check:
                if st.button("✅ Check Answer", type="primary", use_container_width=True):
                    st.session_state.answer_checked = True
            with col_hint:
                if st.button("💡 Need a Hint?", use_container_width=True):
                    st.session_state.pending_screen = "hints"
                    st.rerun()

        # Answer result
        if st.session_state.answer_checked and st.session_state.user_choice:
            chosen = st.session_state.user_choice
            correct = result["correct_label"]

            # Cache the verifier prediction so we don't re-run the ensemble
            # (and don't pollute verifier_log with duplicate rows) on every
            # subsequent rerun while the user stays on this page.
            cache_key = (result["question"], chosen)
            cached = st.session_state.get("_verifier_cache")
            if not cached or cached.get("key") != cache_key:
                try:
                    verifier_label, conf = predict_answer(
                        result["passage"], result["question"], result["option_map"]
                    )
                except Exception as e:
                    verifier_label, conf = correct, 0.0
                    st.caption(f"Verifier unavailable ({e}).")

                st.session_state.verifier_log.append({
                    "y_true": 1 if chosen == correct else 0,
                    "y_pred": 1 if verifier_label == correct else 0,
                    "confidence": float(conf),
                })
                st.session_state["_verifier_cache"] = {
                    "key": cache_key,
                    "verifier_label": verifier_label,
                    "conf": float(conf),
                }
            else:
                verifier_label = cached["verifier_label"]
                conf = cached["conf"]

            if chosen == correct:
                st.markdown(
                    f'<div class="correct-badge">✅ Correct! &nbsp; '
                    f'<span style="font-weight:500">The answer is <strong>{correct}</strong>: {result["option_map"][correct]}</span></div>',
                    unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(
                    f'<div class="wrong-badge">❌ Incorrect &nbsp; '
                    f'<span style="font-weight:500">You chose <strong>{chosen}</strong>. '
                    f'Correct: <strong>{correct}</strong>: {result["option_map"][correct]}</span></div>',
                    unsafe_allow_html=True)

            st.markdown(f"""
            <div class="verifier-card">
                <strong>🤖 Model A Verifier</strong> (LR+SVM+NB ensemble)<br>
                Predicted <code>{verifier_label}</code> with confidence
                <strong style="color:#059669;">{conf:.2%}</strong>
            </div>
            """, unsafe_allow_html=True)

    with col_side:
        # Latency stats card
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div class="card-title" style="text-align:left;">⏱ Generation Stats</div>
            <div style="display:flex;flex-direction:column;gap:8px;margin-top:0.5rem;">
                <div style="background:#f8fafc;border-radius:8px;padding:0.6rem 0.8rem;
                            display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:0.78rem;color:#64748b;font-weight:500;">Question</span>
                    <span style="font-size:0.85rem;font-weight:700;color:#0f172a;">
                        {result['latency_q_ms']:.0f} ms
                    </span>
                </div>
                <div style="background:#f8fafc;border-radius:8px;padding:0.6rem 0.8rem;
                            display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:0.78rem;color:#64748b;font-weight:500;">Distractors</span>
                    <span style="font-size:0.85rem;font-weight:700;color:#0f172a;">
                        {result['latency_d_ms']:.0f} ms
                    </span>
                </div>
                <div style="background:#f8fafc;border-radius:8px;padding:0.6rem 0.8rem;
                            display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:0.78rem;color:#64748b;font-weight:500;">Hints</span>
                    <span style="font-size:0.85rem;font-weight:700;color:#0f172a;">
                        {result['latency_h_ms']:.0f} ms
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Answer options mini-list
        st.markdown("""
        <div class="card">
            <div class="card-title">📋 Answer Options</div>
        """, unsafe_allow_html=True)
        for lbl in ["A", "B", "C", "D"]:
            chosen = st.session_state.user_choice
            checked = st.session_state.answer_checked
            correct_lbl = result["correct_label"]
            if checked and chosen:
                if lbl == correct_lbl:
                    dot_color = "#10b981"
                elif lbl == chosen:
                    dot_color = "#ef4444"
                else:
                    dot_color = "#e2e8f0"
            elif chosen == lbl:
                dot_color = "#10b981"
            else:
                dot_color = "#e2e8f0"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;padding:0.4rem 0;
                        border-bottom:1px solid #f1f5f9;">
                <div style="width:26px;height:26px;border-radius:6px;
                            background:{dot_color};display:flex;align-items:center;
                            justify-content:center;font-size:0.75rem;font-weight:700;
                            color:{'#fff' if dot_color != '#e2e8f0' else '#94a3b8'};
                            flex-shrink:0;">{lbl}</div>
                <span style="font-size:0.78rem;color:#475569;font-weight:500;
                             white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                             max-width:140px;">{result['option_map'][lbl][:40]}{'…' if len(result['option_map'][lbl]) > 40 else ''}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # NLP metrics if available
        if result.get("nlp_metrics"):
            nm = result["nlp_metrics"]
            st.markdown("""
            <div class="card">
                <div class="card-title">📈 NLP Metrics</div>
            """, unsafe_allow_html=True)
            for k, v in nm.items():
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:0.3rem 0;
                            border-bottom:1px solid #f1f5f9;font-size:0.8rem;">
                    <span style="color:#64748b;font-weight:500;">{k}</span>
                    <span style="font-weight:700;color:#0f172a;">{v:.4f}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# Rubric 7.3: Screen 3 - graduated hints, progressive reveal, gated Reveal Answer.

def screen_hints():
    st.markdown("""
    <div class="page-header">
        <div class="icon">💡</div>
        <div>
            <h1>Hint Panel</h1>
            <p>Reveal graduated hints to guide your thinking</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    result = st.session_state.pipeline_result
    if not result:
        st.warning("No quiz generated yet. Go to Article Input first.")
        return

    # Question recap
    st.markdown(f"""
    <div class="card">
        <div class="card-title">🔍 Question</div>
        <div style="font-size:1.05rem;font-weight:600;color:#0f172a;line-height:1.5;">
            {result['question']}
        </div>
        <div style="margin-top:0.5rem;font-size:0.78rem;color:#94a3b8;font-weight:500;">
            Hint generation latency: {result['latency_h_ms']:.0f} ms
        </div>
    </div>
    """, unsafe_allow_html=True)

    hints = result.get("hints", [])
    if not hints:
        st.info("No hints generated.")
        return

    labels = ["Hint 1 — General Clue", "Hint 2 — More Specific", "Hint 3 — Near-Explicit"]
    revealed = st.session_state.hints_revealed

    icons = ["🌱", "🌿", "🎯"]

    for i, (lbl, hint) in enumerate(zip(labels, hints)):
        if i < revealed:
            st.markdown(f"""
            <div class="hint-card">
                <div class="hint-label">{icons[i]} {lbl}</div>
                <div class="hint-text">{hint}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="hint-card blurred">
                <div class="hint-label">{icons[i]} {lbl}</div>
                <div class="hint-text">{hint}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        if revealed < len(hints):
            if st.button("🔓 Show Next Hint", type="primary", use_container_width=True):
                st.session_state.hints_revealed += 1
                st.rerun()
        else:
            st.markdown("""
            <div style="background:#ecfdf5;border:1px solid #a7f3d0;border-radius:10px;
                        padding:0.65rem 1rem;font-size:0.83rem;color:#065f46;
                        font-weight:600;text-align:center;">
                🎉 All hints revealed!
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        if st.button("🏳️ Reveal Answer",
                     disabled=(revealed < len(hints)),
                     use_container_width=True):
            st.markdown(f"""
            <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
                        padding:0.85rem 1.1rem;font-size:0.95rem;color:#1e40af;font-weight:700;">
                📌 Correct Answer: <span style="color:#1d4ed8;">{result['answer']}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_c:
        if st.button("↩ Back to Quiz", use_container_width=True):
            st.session_state.pending_screen = "quiz"
            st.rerun()


# Rubric 7.4: Screen 4 - verifier metrics, corpus BLEU/ROUGE/METEOR, latency, CSV export.

def screen_analytics():
    import plotly.graph_objects as go

    st.markdown("""
    <div class="page-header">
        <div class="icon">📊</div>
        <div>
            <h1>Analytics Dashboard</h1>
            <p>Session performance metrics and corpus-level NLP evaluation</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────
    # Section helper: opens a styled card that ACTUALLY wraps its contents.
    # Streamlit wraps every st.markdown call in its own DOM container, so a
    # naive "open <div>" + "close </div>" pattern leaves the inner widgets
    # outside the card. Using st.container() with a sentinel class lets us
    # target it from CSS via :has() or by writing the title inline.
    # ──────────────────────────────────────────────────────────────────────
    def section_title(title: str):
        st.markdown(
            f"""
            <div style="background:#ffffff;border-radius:14px 14px 0 0;
                        padding:1.1rem 1.4rem 0.4rem 1.4rem;
                        border:1px solid #e8f5f0;border-bottom:none;
                        margin-top:1.4rem;">
                <div style="font-size:0.85rem;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.07em;
                            color:#059669;">
                    {title}
                </div>
            </div>
            <div style="background:#ffffff;border:1px solid #e8f5f0;
                        border-top:none;border-radius:0 0 14px 14px;
                        padding:1rem 1.4rem 1.2rem 1.4rem;
                        box-shadow:0 2px 12px rgba(0,0,0,0.06);
                        margin-bottom:1.2rem;">
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Verifier classification metrics ──────────────────────────────────────
    section_title("🤖 Model A — Verifier Performance (this session)")

    verifier_log = st.session_state.verifier_log
    if not verifier_log:
        st.info("Use **Check Answer** on Quiz View to accumulate verifier predictions.")
    else:
        from sklearn.metrics import (accuracy_score, precision_score,
                                       recall_score, f1_score, confusion_matrix)
        y_true = np.array([e["y_true"] for e in verifier_log])
        y_pred = np.array([e["y_pred"] for e in verifier_log])
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("Precision (macro)", f"{precision:.3f}")
        c3.metric("Recall (macro)", f"{recall:.3f}")
        c4.metric("F1 (macro)", f"{f1:.3f}")

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            st.markdown(
                """
                <div style="font-size:0.78rem;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.07em;
                            color:#64748b;margin:1rem 0 0.4rem 0;">
                    Confusion Matrix
                </div>
                """,
                unsafe_allow_html=True,
            )
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("TP", int(tp)); cc2.metric("TN", int(tn))
            cc3.metric("FP", int(fp)); cc4.metric("FN", int(fn))

    # ── Corpus-level NLP metrics ──────────────────────────────────────────────
    section_title("📚 Corpus-level NLP Metrics (BLEU · ROUGE-L · METEOR)")

    metrics = load_test_metrics()
    if metrics is None:
        st.info("Run `python src/evaluate_nlp.py` to generate metrics_test.json.")
    else:
        for section_key, section_subtitle in (
            ("model_a", "Model A — Question Generation"),
            ("model_b_distractors", "Model B — Distractors"),
            ("model_b_hints", "Model B — Hints"),
        ):
            block = metrics.get(section_key, {})
            if not block:
                continue
            st.markdown(
                f"""
                <div style="font-size:0.85rem;font-weight:600;color:#475569;
                            margin:0.6rem 0 0.4rem 0;">
                    {section_subtitle}
                    <span style="background:#f1f5f9;border-radius:6px;
                                 padding:2px 8px;font-size:0.72rem;
                                 color:#64748b;margin-left:6px;">
                        n={block.get('num_samples', 'n/a')}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            cols = st.columns(4)
            cols[0].metric("BLEU-1", f"{block.get('bleu_1', 0):.4f}")
            cols[1].metric("BLEU-4", f"{block.get('bleu_4', 0):.4f}")
            cols[2].metric("ROUGE-L F1", f"{block.get('rouge_l_f1', 0):.4f}")
            cols[3].metric("METEOR", f"{block.get('meteor', 0):.4f}")

    # ── Latency tracking ─────────────────────────────────────────────────────
    section_title("⏱ Inference Latency (per quiz)")

    session_log = st.session_state.session_log
    if not session_log:
        st.info("Generate quizzes to track inference latency.")
    else:
        df_lat = pd.DataFrame(session_log)
        x_idx = list(range(1, len(df_lat) + 1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_idx, y=df_lat["latency_q_ms"],
            name="Question",
            mode="lines+markers",
            line=dict(color="#10b981", width=2.5),
            marker=dict(size=8, color="#059669",
                        line=dict(color="#ffffff", width=2)),
            hovertemplate="Quiz #%{x}<br>Question: %{y:.1f} ms<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x_idx, y=df_lat["latency_d_ms"],
            name="Distractors",
            mode="lines+markers",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=8, color="#1d4ed8",
                        line=dict(color="#ffffff", width=2)),
            hovertemplate="Quiz #%{x}<br>Distractors: %{y:.1f} ms<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=x_idx, y=df_lat["latency_h_ms"],
            name="Hints",
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2.5),
            marker=dict(size=8, color="#d97706",
                        line=dict(color="#ffffff", width=2)),
            hovertemplate="Quiz #%{x}<br>Hints: %{y:.1f} ms<extra></extra>",
        ))
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=40),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(family="Plus Jakarta Sans", size=11, color="#475569"),
            xaxis=dict(
                title="Quiz #",
                gridcolor="#f1f5f9",
                zeroline=False,
                dtick=1 if len(x_idx) <= 15 else None,
            ),
            yaxis=dict(
                title="Latency (ms)",
                gridcolor="#f1f5f9",
                zeroline=False,
                rangemode="tozero",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right",  x=1,
                bgcolor="rgba(0,0,0,0)",
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True,
                         config={"displayModeBar": False})

        # Aggregate stats
        df_lat["total_ms"] = (df_lat["latency_q_ms"]
                              + df_lat["latency_d_ms"]
                              + df_lat["latency_h_ms"])
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Quizzes", f"{len(df_lat)}")
        s2.metric("Mean Total", f"{df_lat['total_ms'].mean():.0f} ms")
        s3.metric("Max Total",  f"{df_lat['total_ms'].max():.0f} ms")
        s4.metric("Min Total",  f"{df_lat['total_ms'].min():.0f} ms")

    # ── Session log ─────────────────────────────────────────────────────────
    if st.session_state.session_log:
        section_title("📋 Session Log")
        df = pd.DataFrame(st.session_state.session_log)
        # Round latency columns for cleaner display
        for col in ("latency_q_ms", "latency_d_ms", "latency_h_ms"):
            if col in df.columns:
                df[col] = df[col].round(1)
        st.dataframe(df, use_container_width=True, height=240)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Session Log (CSV)",
            csv,
            file_name="session_log.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Router
# ══════════════════════════════════════════════════════════════════════════════

screen = st.session_state.screen

_router = {
    "input":     screen_input,
    "quiz":      screen_quiz,
    "hints":     screen_hints,
    "analytics": screen_analytics,
}
try:
    _router.get(screen, screen_input)()
except Exception as e:
    import traceback
    st.error(f"Error rendering screen '{screen}': {type(e).__name__}: {e}")
    with st.expander("Traceback"):
        st.code(traceback.format_exc())
    st.info("Use the sidebar to return to 🏠 Article Input.")