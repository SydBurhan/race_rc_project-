"""
EDA & Preprocessing Visualizations
RACE Reading Comprehension Dataset — FAST-NUCES AI Lab Project

Generates 5 high-quality figures saved to reports/figures/
Run from project root: python eda_visualizations.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — no GUI needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter
from pathlib import Path
import re
import warnings
warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
ACCENT   = "#4C72B0"
FIG_DIR  = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv("data/processed/train.csv")
print(f"  Rows loaded: {len(df):,}")
print(f"  Columns    : {list(df.columns)}\n")

# ── Helper ─────────────────────────────────────────────────────────────────
def word_count(text):
    return len(str(text).split())

def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

# ══════════════════════════════════════════════════════════════════════════
# Chart 1 — Article / Passage Length Distribution
# ══════════════════════════════════════════════════════════════════════════
print("Chart 1 — Article length distribution")
df["article_len"] = df["article"].apply(word_count)

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df["article_len"], bins=60, kde=True, color=ACCENT, ax=ax)
ax.axvline(df["article_len"].median(), color="#e05c5c", linestyle="--",
           linewidth=1.5, label=f"Median = {df['article_len'].median():.0f} words")
ax.set_title("Distribution of Article / Passage Lengths", fontsize=15, fontweight="bold")
ax.set_xlabel("Word Count")
ax.set_ylabel("Number of Passages")
ax.legend()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
save(fig, "01_article_length_dist.png")

# ══════════════════════════════════════════════════════════════════════════
# Chart 2 — Question Length Distribution
# ══════════════════════════════════════════════════════════════════════════
print("Chart 2 — Question length distribution")
df["question_len"] = df["question"].apply(word_count)

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df["question_len"], bins=40, kde=True, color="#55A868", ax=ax)
ax.axvline(df["question_len"].median(), color="#e05c5c", linestyle="--",
           linewidth=1.5, label=f"Median = {df['question_len'].median():.0f} words")
ax.set_title("Distribution of Question Lengths", fontsize=15, fontweight="bold")
ax.set_xlabel("Word Count")
ax.set_ylabel("Number of Questions")
ax.legend()
save(fig, "02_question_length_dist.png")

# ══════════════════════════════════════════════════════════════════════════
# Chart 3 — Correct Answer Distribution (A / B / C / D)
# ══════════════════════════════════════════════════════════════════════════
print("Chart 3 — Correct answer distribution")
answer_counts = df["answer"].str.upper().value_counts().reindex(["A","B","C","D"], fill_value=0)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(answer_counts.index, answer_counts.values,
              color=[ACCENT, "#55A868", "#C44E52", "#8172B2"], edgecolor="white", width=0.55)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + answer_counts.max() * 0.015,
            f"{int(bar.get_height()):,}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Distribution of Correct Answer Options", fontsize=15, fontweight="bold")
ax.set_xlabel("Answer Option")
ax.set_ylabel("Count")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_ylim(0, answer_counts.max() * 1.12)
save(fig, "03_answer_distribution.png")

# ══════════════════════════════════════════════════════════════════════════
# Chart 4 — Top-20 Most Frequent Content Words (articles)
# ══════════════════════════════════════════════════════════════════════════
print("Chart 4 — Top-20 frequent words")

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","was","are","were","be","been","being","have","has","had","do",
    "does","did","will","would","could","should","may","might","shall",
    "that","this","these","those","it","its","he","she","they","we","i",
    "his","her","their","our","my","by","from","up","about","into","than",
    "then","so","if","as","not","no","also","just","more","one","can","all",
    "which","who","what","when","where","said","s","t","re","ve","d","ll",
}

all_text = " ".join(df["article"].dropna().astype(str))
tokens   = re.findall(r"\b[a-z]{3,}\b", all_text.lower())
top20    = Counter(t for t in tokens if t not in STOPWORDS).most_common(20)
words, freqs = zip(*top20)

fig, ax = plt.subplots(figsize=(11, 6))
colors = sns.color_palette("Blues_d", len(words))
ax.barh(list(reversed(words)), list(reversed(freqs)), color=list(reversed(colors)), edgecolor="white")
ax.set_title("Top-20 Most Frequent Content Words in Articles", fontsize=15, fontweight="bold")
ax.set_xlabel("Frequency")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
save(fig, "04_top20_words.png")

# ══════════════════════════════════════════════════════════════════════════
# Chart 5 — Bonus: Article vs Question length scatter (sample)
# ══════════════════════════════════════════════════════════════════════════
print("Chart 5 — Article vs Question length scatter (bonus)")
sample = df.sample(min(3000, len(df)), random_state=42)

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(sample["article_len"], sample["question_len"],
           alpha=0.25, s=12, color=ACCENT)
ax.set_title("Article Length vs Question Length (sample n=3,000)", fontsize=14, fontweight="bold")
ax.set_xlabel("Article Word Count")
ax.set_ylabel("Question Word Count")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
save(fig, "05_article_vs_question_scatter.png")

# ── Summary stats ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("SUMMARY STATISTICS")
print("=" * 55)
stats = df[["article_len", "question_len"]].describe().round(1)
stats.columns = ["Article Length (words)", "Question Length (words)"]
print(stats.to_string())
print("\nAnswer distribution:")
print(answer_counts.to_string())
print("\nAll figures saved to:", FIG_DIR.resolve())