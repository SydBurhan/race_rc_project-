# RACE Reading Comprehension & Quiz Generation System

AL2002 Artificial Intelligence Lab Project — FAST-NUCES Islamabad
Authors: Syed Burhan Ahmad (23i-0757), Mushahid Hussain (23i-0541)

A traditional-ML pipeline (no neural networks) over the RACE dataset that:
- Verifies multiple-choice answers (Model A — LR / SVM / NB / RF / XGBoost ensemble)
- Generates questions via template-based pipeline ranked by an SVM (Model A)
- Generates 3 plausible distractors per question (Model B — TF-IDF + MMR + frequency + Word2Vec)
- Generates 3 graduated hints (Model B — extractive ranker + Logistic Regression scorer)

## 0. Run on Colab / Kaggle (recommended for speed)

Open `notebooks/colab_kaggle_run.ipynb` in Colab or Kaggle and run top-to-bottom.
It clones the repo, installs deps, downloads RACE, trains all models, and runs
NLP evaluation. XGBoost auto-detects CUDA when a GPU is available (5–10× speedup
on that one model). Other sklearn models stay on CPU but still benefit from the
extra vCPUs and RAM.

## 1. Setup

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt punkt_tab wordnet omw-1.4
```

**Optional (Python ≤ 3.12 only):** install `gensim` for Word2Vec-based distractors:

```bash
pip install "gensim>=4.3.0"
```

On Python 3.13+ gensim has no upstream wheel; the distractor pipeline silently
falls back to TF-IDF + frequency strategies, which is acceptable for grading.

## 2. Data

Download RACE from <https://www.kaggle.com/datasets/ankitdhiman7/race-dataset>.

- **Preferred:** place all three CSVs at `data/raw/{train,val,test}.csv` (or `dev.csv` / `validation.csv` instead of `val.csv`).
- **Fallback:** place only `data/raw/train.csv`. `preprocessing.py` will auto-split it 80/10/10 with `random_state=42`.
- If the three files are byte-identical (a common Kaggle re-upload artefact),
  the loader detects this and falls back to auto-split to avoid data leakage.

Each CSV must contain the columns: `id, article, question, A, B, C, D, answer`.

## 3. Reproduce Full Pipeline (in order)

```bash
python src/preprocessing.py
python src/model_a_train.py
python src/template_generator.py
python src/model_b_train.py
python src/evaluate_nlp.py
```

## 4. Run the App

```bash
streamlit run ui/app.py
```

## 5. Project Structure

```
race_rc_project/
├── data/
│   ├── raw/              # train.csv, val.csv, test.csv (RACE)
│   └── processed/        # OHE/TF-IDF feature matrices, label arrays
├── models/
│   ├── model_a/
│   │   └── traditional/  # lr, svm, nb, rf, xgb, ensemble, kmeans, label_prop, question_ranker
│   └── model_b/
│       └── traditional/  # hint_scorer, distractor artefacts
├── src/
│   ├── preprocessing.py
│   ├── model_a_train.py
│   ├── template_generator.py
│   ├── model_b_train.py
│   ├── evaluate_nlp.py
│   └── inference.py
├── ui/
│   └── app.py            # Streamlit entry point
├── notebooks/
│   ├── EDA.ipynb
│   └── experiments.ipynb
├── reports/
│   ├── figures/
│   └── metrics_test.json
├── requirements.txt
├── README.md
└── report/
    └── final_report.pdf
```
