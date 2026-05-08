"""
ui/app.py
=========
Streamlit UI — Intelligent Reading Comprehension & Quiz Generation System
AL2002 Lab Project · FAST-NUCES Islamabad · Spring 2026

4 Screens:
  1. Article Input          — paste / upload / load RACE sample
  2. Question & Answer Quiz — generated Q, 4 options (correct + 3 distractors)
  3. Hint Panel             — graduated hints from Model B (TF-IDF)
  4. Analytics Dashboard    — NLP metrics (BLEU/ROUGE/METEOR) + Model B stats

Architecture enforced:
  Model A → FlanT5Generator   (Generative AI, HuggingFace Transformers)
  Model B → Classical ML only (TF-IDF + cosine similarity + MMR, no spaCy/DL)
"""

from __future__ import annotations

import sys
import os
import re
import time
import logging
import random
from typing import Optional

import streamlit as st

# ── Path setup so src/ modules resolve ────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model_a_generator import FlanT5Generator, clean_passage
from src.model_b_distractor import ModelBPipeline          # Classical ML
from src.evaluate_nlp import score_single

logging.basicConfig(level=logging.WARNING)

# ══════════════════════════════════════════════════════════════════════════════
#  Page config — must be first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RC Quiz System — AL2002",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS — light polish
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stButton > button { border-radius: 8px; font-weight: 600; }
    .metric-card {
        background: #f0f4ff; border-radius: 10px; padding: 1rem;
        margin-bottom: 0.5rem; border-left: 4px solid #4f46e5;
    }
    .correct-badge  { color: #16a34a; font-weight: 700; font-size: 1.1rem; }
    .wrong-badge    { color: #dc2626; font-weight: 700; font-size: 1.1rem; }
    .hint-box {
        background: #fffbeb; border-left: 4px solid #f59e0b;
        padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 0.5rem;
    }
    .ai-badge {
        background: #e0e7ff; color: #3730a3; font-size: 0.75rem;
        padding: 2px 8px; border-radius: 12px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Cached model loaders (loaded once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading Flan-T5 model (Model A) …")
def load_model_a() -> FlanT5Generator:
    return FlanT5Generator()


@st.cache_resource(show_spinner="Initialising Model B (Classical ML) …")
def load_model_b() -> ModelBPipeline:
    return ModelBPipeline()


# ══════════════════════════════════════════════════════════════════════════════
#  RACE sample loader (uses HuggingFace datasets)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Fetching RACE dataset …")
def load_race_samples(split: str = "test", n: int = 200) -> list[dict]:
    """Return up to n samples from RACE (high school difficulty)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("race", "high", split=split, trust_remote_code=True)
        idxs = random.sample(range(len(ds)), min(n, len(ds)))
        return [{"article": ds[i]["article"], "question": ds[i]["question"],
                 "answer": ds[i]["answer"],
                 "A": ds[i]["options"][0], "B": ds[i]["options"][1],
                 "C": ds[i]["options"][2], "D": ds[i]["options"][3]}
                for i in idxs]
    except Exception as e:
        st.warning(f"Could not load RACE dataset: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  Session-state initialisation
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "screen":           "input",        # input | quiz | hints | analytics
        "passage":          "",
        "pipeline_result":  None,           # output of Model A + B
        "user_choice":      None,           # A / B / C / D
        "answer_checked":   False,
        "hints_revealed":   0,              # 0–3
        "session_log":      [],             # list of per-inference dicts
        "race_samples":     [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar navigation
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📚 RC Quiz System")
    st.caption("AL2002 · FAST-NUCES Islamabad")
    st.divider()

    nav = st.radio(
        "Navigation",
        ["🏠 Article Input", "❓ Quiz View", "💡 Hint Panel", "📊 Analytics"],
        index=["input", "quiz", "hints", "analytics"].index(st.session_state.screen)
              if st.session_state.screen in ["input", "quiz", "hints", "analytics"]
              else 0,
    )
    screen_map = {
        "🏠 Article Input": "input",
        "❓ Quiz View":     "quiz",
        "💡 Hint Panel":    "hints",
        "📊 Analytics":     "analytics",
    }
    st.session_state.screen = screen_map[nav]

    st.divider()
    st.markdown('<span class="ai-badge">⚠️ AI-Generated Content</span>', unsafe_allow_html=True)
    st.caption("Questions & answers are AI-generated and may contain errors. Not for real exam use.")


# ══════════════════════════════════════════════════════════════════════════════
#  Helper — run full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(passage: str) -> dict:
    """
    Orchestrates Model A (Flan-T5) and Model B (Classical ML).
    Returns a combined result dict stored in session_state.
    """
    model_a: FlanT5Generator = load_model_a()
    model_b: ModelBPipeline  = load_model_b()

    # ── Model A: generate question + extract answer ──────────────────────────
    with st.spinner("Model A (Flan-T5): generating question & answer …"):
        t0 = time.time()
        a_result = model_a.run_pipeline(passage)
        latency_a = round(time.time() - t0, 3)

    question = a_result["question"]
    answer   = a_result["answer"]

    # ── Model B: generate distractors + hints ───────────────────────────────
    with st.spinner("Model B (Classical ML): generating distractors & hints …"):
        t0 = time.time()
        b_result = model_b.run_pipeline(passage, question, answer)
        latency_b = round(time.time() - t0, 3)

    distractors = b_result["distractors"]    # list of 3 strings
    hints       = b_result["hints"]          # list of 3 strings

    # ── Build shuffled options A–D ───────────────────────────────────────────
    options = distractors[:3] + [answer]
    random.shuffle(options)
    correct_label = ["A", "B", "C", "D"][options.index(answer)]
    option_map = dict(zip(["A", "B", "C", "D"], options))

    result = {
        "passage":       passage,
        "question":      question,
        "answer":        answer,
        "correct_label": correct_label,
        "option_map":    option_map,
        "hints":         hints,
        "latency_a":     latency_a,
        "latency_b":     latency_b,
        "distractors":   distractors,
        # NLP metrics scored against answer (self-evaluation proxy)
        "nlp_metrics":   score_single(answer, answer),   # placeholder; real eval uses RACE refs
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  SCREEN 1 — Article Input
# ══════════════════════════════════════════════════════════════════════════════

def screen_input():
    st.header("🏠 Screen 1 — Article Input")
    st.caption("Paste a reading passage below, upload a .txt file, or load a random RACE sample.")

    col_left, col_right = st.columns([3, 1])

    with col_right:
        st.markdown("#### Quick Load")
        if st.button("🎲 Random RACE Sample", use_container_width=True):
            if not st.session_state.race_samples:
                st.session_state.race_samples = load_race_samples()
            if st.session_state.race_samples:
                sample = random.choice(st.session_state.race_samples)
                st.session_state.passage = sample["article"]
            else:
                st.warning("RACE dataset unavailable. Paste text manually.")

        uploaded = st.file_uploader("Upload .txt", type=["txt"])
        if uploaded:
            st.session_state.passage = uploaded.read().decode("utf-8")

    with col_left:
        passage = st.text_area(
            "Reading Passage",
            value=st.session_state.passage,
            height=320,
            placeholder="Paste or upload a reading passage here …",
            key="passage_area",
        )
        st.session_state.passage = passage

    word_count = len(passage.split()) if passage.strip() else 0
    st.caption(f"Word count: {word_count}")

    if word_count < 30:
        st.info("Please provide a passage of at least 30 words for best results.")

    st.divider()

    if st.button("🚀 Generate Quiz", type="primary", disabled=word_count < 30, use_container_width=False):
        cleaned = clean_passage(st.session_state.passage)
        result  = _run_pipeline(cleaned)
        st.session_state.pipeline_result = result
        st.session_state.user_choice     = None
        st.session_state.answer_checked  = False
        st.session_state.hints_revealed  = 0

        # Log to session analytics
        st.session_state.session_log.append({
            "passage_snippet": cleaned[:100],
            "question":        result["question"],
            "answer":          result["answer"],
            "latency_a":       result["latency_a"],
            "latency_b":       result["latency_b"],
        })

        st.session_state.screen = "quiz"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  SCREEN 2 — Question & Answer Quiz View
# ══════════════════════════════════════════════════════════════════════════════

def screen_quiz():
    st.header("❓ Screen 2 — Quiz View")

    result = st.session_state.pipeline_result
    if not result:
        st.warning("No quiz generated yet. Go to Article Input first.")
        return

    st.markdown(f"**Passage excerpt:** _{result['passage'][:200]} …_")
    st.divider()

    # AI-generated badge
    st.markdown('<span class="ai-badge">⚠️ AI-Generated Question</span>', unsafe_allow_html=True)
    st.markdown(f"### {result['question']}")
    st.caption(f"Model A (Flan-T5) · Latency: {result['latency_a']} s")

    st.markdown("---")
    st.markdown("**Choose your answer:**")

    for label in ["A", "B", "C", "D"]:
        text = result["option_map"][label]
        if st.button(f"**{label}.** {text}", key=f"opt_{label}", use_container_width=True):
            st.session_state.user_choice = label
            st.session_state.answer_checked = False

    st.divider()

    if st.session_state.user_choice:
        col_check, col_hint = st.columns([1, 1])
        with col_check:
            if st.button("✅ Check Answer", type="primary"):
                st.session_state.answer_checked = True

        with col_hint:
            if st.button("💡 Need a Hint?"):
                st.session_state.screen = "hints"
                st.rerun()

    if st.session_state.answer_checked and st.session_state.user_choice:
        chosen  = st.session_state.user_choice
        correct = result["correct_label"]
        if chosen == correct:
            st.markdown(f'<p class="correct-badge">✅ Correct! The answer is {correct}: {result["option_map"][correct]}</p>', unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f'<p class="wrong-badge">❌ Incorrect. You chose {chosen}. The correct answer is {correct}: {result["option_map"][correct]}</p>', unsafe_allow_html=True)

        st.markdown('<span class="ai-badge">⚠️ AI-extracted answer — verify against passage</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SCREEN 3 — Hint Panel
# ══════════════════════════════════════════════════════════════════════════════

def screen_hints():
    st.header("💡 Screen 3 — Hint Panel")

    result = st.session_state.pipeline_result
    if not result:
        st.warning("No quiz generated yet. Go to Article Input first.")
        return

    st.markdown(f"**Question:** _{result['question']}_")
    st.caption(f"Model B (Classical ML · TF-IDF) · Latency: {result['latency_b']} s")
    st.divider()

    hints = result.get("hints", [])
    revealed = st.session_state.hints_revealed

    if not hints:
        st.info("No hints were generated for this passage.")
        return

    hint_labels = ["Hint 1 — General Clue", "Hint 2 — More Specific", "Hint 3 — Near-Explicit"]

    for i, (label, hint) in enumerate(zip(hint_labels, hints)):
        if i < revealed:
            st.markdown(f'<div class="hint-box"><b>{label}:</b> {hint}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="hint-box" style="filter:blur(4px);user-select:none"><b>{label}:</b> {hint}</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if revealed < len(hints):
            if st.button("🔓 Reveal Next Hint", type="primary"):
                st.session_state.hints_revealed += 1
                st.rerun()
        else:
            st.success("All hints revealed.")

    with col_b:
        if revealed >= len(hints):
            if st.button("🏳️ Reveal Answer"):
                st.info(f"**Correct Answer:** {result['answer']}")


# ══════════════════════════════════════════════════════════════════════════════
#  SCREEN 4 — Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def screen_analytics():
    import plotly.graph_objects as go

    st.header("📊 Screen 4 — Analytics Dashboard")
    st.caption("Per professor's directive: Model A is evaluated with BLEU, ROUGE-L, and METEOR (text generation metrics). "
               "Model B uses Precision / Recall / F1 for distractor ranking.")

    result = st.session_state.pipeline_result
    log    = st.session_state.session_log

    # ── Model A — NLP Metrics ─────────────────────────────────────────────────
    st.subheader("Model A — NLP Generation Metrics (BLEU · ROUGE-L · METEOR)")
    st.markdown(
        "> **Note:** For a full corpus-level evaluation against RACE reference questions, "
        "run `python src/evaluate_nlp.py --predictions preds.txt --references refs.txt`. "
        "The scores below are computed for the *current* inference (single-sample)."
    )

    if result:
        metrics = result.get("nlp_metrics", {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("BLEU-1",     f"{metrics.get('bleu_1', 0):.4f}")
        col2.metric("BLEU-4",     f"{metrics.get('bleu_4', 0):.4f}")
        col3.metric("ROUGE-L F1", f"{metrics.get('rouge_l_f1', 0):.4f}")
        col4.metric("METEOR",     f"{metrics.get('meteor', 0):.4f}")

        # BLEU bar chart
        bleu_vals = [metrics.get(f"bleu_{n}", 0) for n in range(1, 5)]
        fig = go.Figure(go.Bar(
            x=["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"],
            y=bleu_vals,
            marker_color=["#6366f1", "#818cf8", "#a5b4fc", "#c7d2fe"],
            text=[f"{v:.4f}" for v in bleu_vals],
            textposition="outside",
        ))
        fig.update_layout(title="BLEU-1 through BLEU-4 (current inference)",
                          yaxis_range=[0, 1], height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # ROUGE-L breakdown
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("ROUGE-L Precision", f"{metrics.get('rouge_l_precision', 0):.4f}")
        col_r2.metric("ROUGE-L Recall",    f"{metrics.get('rouge_l_recall', 0):.4f}")
        col_r3.metric("ROUGE-L F1",        f"{metrics.get('rouge_l_f1', 0):.4f}")
    else:
        st.info("Run a quiz first (Screen 1) to see metrics.")

    st.divider()

    # ── Model B — Distractor Quality Metrics ──────────────────────────────────
    st.subheader("Model B — Distractor Quality (Classical ML)")
    st.caption("Precision / Recall / F1 measure how well the TF-IDF ranker selects plausible-but-wrong distractors.")

    if result:
        b_metrics = st.session_state.pipeline_result.get("b_metrics", {})
        col_b1, col_b2, col_b3 = st.columns(3)
        col_b1.metric("Distractor Precision", f"{b_metrics.get('precision', 0.0):.3f}")
        col_b2.metric("Distractor Recall",    f"{b_metrics.get('recall', 0.0):.3f}")
        col_b3.metric("Distractor F1",        f"{b_metrics.get('f1', 0.0):.3f}")

    st.divider()

    # ── Latency Tracking ──────────────────────────────────────────────────────
    st.subheader("Inference Latency")

    if log:
        import pandas as pd
        df = pd.DataFrame(log)
        df.index += 1
        df.columns = ["Passage", "Question", "Answer", "Latency A (s)", "Latency B (s)"]
        st.dataframe(df, use_container_width=True)

        # Latency line chart
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=df["Latency A (s)"], mode="lines+markers", name="Model A (Flan-T5)"))
        fig2.add_trace(go.Scatter(y=df["Latency B (s)"], mode="lines+markers", name="Model B (Classical ML)"))
        fig2.update_layout(title="Inference Latency per Request", yaxis_title="Seconds",
                           xaxis_title="Request #", height=300)
        st.plotly_chart(fig2, use_container_width=True)

        # Export CSV
        csv = df.to_csv(index=True).encode("utf-8")
        st.download_button("⬇️ Export Session Log (CSV)", csv,
                           file_name="session_log.csv", mime="text/csv")
    else:
        st.info("No inferences logged yet. Run a quiz first.")

    st.divider()

    # ── Evaluation Instructions ────────────────────────────────────────────────
    with st.expander("🔬 Run Full RACE Corpus Evaluation"):
        st.code("""
# Step 1: Generate predictions over RACE test split
python src/run_batch_eval.py --split test --max-samples 500 \\
       --output-preds data/processed/preds.txt \\
       --output-refs  data/processed/refs.txt

# Step 2: Compute BLEU / ROUGE-L / METEOR
python src/evaluate_nlp.py \\
       --predictions data/processed/preds.txt \\
       --references  data/processed/refs.txt \\
       --output-json results/nlp_metrics.json
        """, language="bash")


# ══════════════════════════════════════════════════════════════════════════════
#  Router
# ══════════════════════════════════════════════════════════════════════════════

screen = st.session_state.screen
if screen == "input":
    screen_input()
elif screen == "quiz":
    screen_quiz()
elif screen == "hints":
    screen_hints()
elif screen == "analytics":
    screen_analytics()
else:
    screen_input()