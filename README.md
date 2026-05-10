# RACE Reading Comprehension & Quiz Generation System

AL2002 Artificial Intelligence Lab Project — FAST-NUCES Islamabad
Authors: Syed Burhan Ahmad (23i-0757), Mushahid Hussain (23i-0541)

A traditional-ML pipeline (no neural networks) over the RACE dataset that:
- Verifies multiple-choice answers (Model A — LR / SVM / NB / RF / XGBoost ensemble)
- Generates questions via template-based pipeline ranked by an SVM (Model A)
- Generates 3 plausible distractors per question (Model B — TF-IDF + MMR + frequency + Word2Vec)
- Generates 3 graduated hints (Model B — extractive ranker + Logistic Regression scorer)

## 1. Setup

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt wordnet omw-1.4
```

## 2. Data

```
Download from: https://www.kaggle.com/datasets/ankitdhiman7/race-dataset
Place files at: data/raw/train.csv, data/raw/val.csv, data/raw/test.csv
```

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
