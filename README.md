# 📖 RACE Reading Comprehension System
**AL2002 Artificial Intelligence Lab Project (Spring 2026)**  
**Institution:** FAST School of Computing, NUCES Islamabad  
**Authors:** Syed Burhan Ahmad (i230757) · Mushahid Hussain (23i-0541)

---

## 📌 Project Overview
An end-to-end Machine Learning pipeline built on the **RACE (Reading Comprehension from Examinations)** dataset. The system provides a complete educational interface capable of verifying multiple-choice answers, extracting contextual hints, and generating plausible distractors dynamically from reading passages.

Two core intelligent agents power the system:

- **Model A (Answer Verifier):** A soft-voting ensemble classifier that evaluates MCQ options and predicts the correct answer using TF-IDF features.
- **Model B (Quiz Generator):** An extractive NLP engine that generates ranked hints and context-aware distractors for automated quiz generation.

---

## ⚙️ Technical Architecture

### Model A — Answer Verification
| Component | Detail |
|---|---|
| Feature Engineering | TF-IDF Vectorization (`max_features=15000`, `sublinear_tf=True`) |
| Unsupervised | K-Means Clustering — Purity: 0.75, Silhouette: 0.0512 *(expected range for high-dimensional sparse text)* |
| Logistic Regression | `solver=saga`, `class_weight=balanced` — Val Accuracy: **31.70%** |
| Calibrated LinearSVC | `C=1.0`, wrapped in `CalibratedClassifierCV(cv=3)` — Val Accuracy: **31.10%** |
| **Soft-Voting Ensemble** | Averaged `predict_proba` — Val Accuracy: **~32%** *(vs. 25% random baseline)* |

### Model B — Hint & Distractor Generation
- **Extractive Hints:** Splits passages into sentences and ranks by TF-IDF cosine similarity to the question. Returns top-3 most relevant clues with similarity scores.
- **Distractor Generation (MMR + Contextual Inflator):**
  1. Extracts n-gram candidates and re-ranks using **Maximal Marginal Relevance (MMR)** for semantic diversity.
  2. Passes candidates through a custom **Contextual Inflator** — a sliding word-window that maps keywords back to source sentences, strips stopwords (preserving proper nouns), and expands bare tokens into readable noun phrases (e.g. `"win"` → `"First woman to win"`).
  3. Evaluated with Precision, Recall, F1, and a pseudo-Confusion Matrix using unigram token overlap.

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/race_rc_project.git
cd race_rc_project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the RACE dataset
Download from the [official RACE page](http://www.cs.cmu.edu/~glai1/data/race/) and extract into `data/raw/` so the structure looks like:
```
data/raw/
├── train/
├── dev/
└── test/
```

### 4. Run the full training pipeline (in order)
```bash
python src/preprocessing.py            # builds processed CSVs + vectorizer
python src/eda_visualizations.py       # generates EDA charts → reports/figures/
python src/model_a_train.py            # trains Logistic Regression
python src/model_a_supervised_svm.py   # trains Calibrated SVM
python src/model_a_unsupervised.py     # runs K-Means clustering
python src/model_b_evaluate.py         # evaluates distractor generation
```

### 5. Launch the Streamlit app
```bash
streamlit run ui/app.py
```
Open `http://localhost:8501` in your browser.

---

## 🖥️ System Interface (Streamlit)
| Screen | Description |
|---|---|
| 📝 Article Input | Paste a passage, enter a question and correct answer, trigger AI generation |
| 📚 Quiz View | Take the quiz; Model A Ensemble verifies your choice with confidence scores |
| 💡 Hint Panel | View top-3 extractive hints ranked by cosine similarity to the question |
| 📊 Analytics Dashboard | Session stats, model accuracy comparison, and full technical stack |

---

## 📂 Repository Structure
```text
race_rc_project/
├── data/
│   ├── raw/                        # Original RACE dataset files
│   └── processed/                  # Tokenized, feature-engineered CSVs
├── models/
│   └── model_a/                    # Saved joblib models (LR, SVM, Vectorizer)
├── notebooks/
│   ├── EDA.ipynb                   # Exploratory Data Analysis & visualisations
│   └── experiments.ipynb           # Experiment tracking log
├── reports/
│   ├── figures/                    # PNG charts generated from EDA
│   └── final_report.pdf            # Methodology and evaluation document
├── src/
│   ├── preprocessing.py            # Dataset loading & TF-IDF pipeline
│   ├── eda_visualizations.py       # EDA chart generation script
│   ├── model_a_train.py            # Logistic Regression training
│   ├── model_a_supervised_svm.py   # Calibrated SVM training
│   ├── model_a_unsupervised.py     # K-Means clustering
│   ├── model_b_train.py            # Distractor/Hint generator (MMR + Inflator)
│   └── model_b_evaluate.py         # Distractor evaluation (F1/Precision/Recall)
├── ui/
│   └── app.py                      # Streamlit entry point
├── requirements.txt                # Pinned Python dependencies
└── README.md
```

---

## 📊 Model Results Summary
| Model | Val Accuracy | Notes |
|---|---|---|
| Random Baseline | 25.00% | 4-class uniform |
| Logistic Regression | 31.70% | TF-IDF, saga solver |
| Calibrated LinearSVC | 31.10% | CalibratedClassifierCV |
| **Soft-Voting Ensemble** | **~32%** | Averaged predict_proba |
| K-Means (Unsupervised) | Purity: 0.75 | Silhouette: 0.0512 |

---

## 📚 References
- Lai, G. et al. (2017). *RACE: Large-scale ReAding Comprehension Dataset From Examinations.* EMNLP 2017.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL 2019.
- Guo, Q. et al. (2016). *Generating Distractors for Reading Comprehension Questions.* AAAI 2016.