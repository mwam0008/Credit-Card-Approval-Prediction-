# Credit Card Approval Prediction
### CST2216 — Machine Learning 2 | Algonquin College | Capstone Project

**Group Members:**
- Fritz Mwambo
- Hisham Khraibah
- Jinchuan Liu
- Youssra Ibrahim

---

## Project Purpose

This capstone project predicts credit card approval outcomes using real applicant
demographic and financial data. The project demonstrates the full machine learning
engineering lifecycle: data cleaning, feature engineering, model training, evaluation,
and deployment as a live interactive web application.

**Two prediction tasks are implemented:**
- **Regression** — predict an applicant's continuous credit SCORE
- **Classification** — predict whether an applicant is Approved or Not Approved
  (threshold: top 25% of SCORE, i.e. SCORE ≥ Q3)

---

## Live App

👉 **[Open the Streamlit App](https://your-app-url.streamlit.app)**  
*(replace with your deployed URL after connecting to Streamlit Cloud)*

---

## Project Structure

```
capstone_credit_approval/
├── app.py                        ← Streamlit UI only (no ML logic)
├── model.py                      ← All ML logic: loading, cleaning, training, evaluation, prediction
├── utils.py                      ← All charts and visualizations
├── AC_Capstone_topic_2.csv       ← Raw dataset (36,457 applicant records)
├── requirements.txt              ← Python dependencies
└── README.md                     ← This file
```

---

## Dataset

| Property        | Value |
|-----------------|-------|
| File            | AC_Capstone_topic_2.csv |
| Raw records     | 36,457 |
| Cleaned records | ~27,918 (after deduplication) |
| Features        | 18 (before encoding) |
| Target (Regression) | SCORE (continuous, range 316–10,000) |
| Target (Classification) | APPROVAL (binary: 1 = top 25% SCORE, 0 = otherwise) |

### Feature Descriptions

| Column | Description |
|--------|-------------|
| CODE_GENDER | Applicant gender (M / F) |
| FLAG_OWN_CAR | Owns a car (Y / N) |
| FLAG_OWN_REALTY | Owns real estate (Y / N) |
| CNT_CHILDREN | Number of children |
| AMT_INCOME_TOTAL | Annual income |
| NAME_INCOME_TYPE | Income source type |
| NAME_EDUCATION_TYPE | Highest education level |
| NAME_FAMILY_STATUS | Marital status |
| NAME_HOUSING_TYPE | Housing situation |
| BIRTHDAY | Date of birth (converted to AGE) |
| EMPLOYED_DATE | Employment start date (converted to IS_EMPLOYED) |
| OCCUPATION_TYPE | Occupation category (31.1% missing — treated as separate category) |
| CNT_FAM_MEMBERS | Number of family members |
| SCORE | Credit score (target for regression) |

---

## Methodology

### Data Cleaning
- Remove duplicate rows (8,539 removed)
- Drop `ID` column (unique identifier only)
- Create `IS_EMPLOYED` from `NAME_INCOME_TYPE` (Pensioner → 0, all others → 1), drop `EMPLOYED_DATE`
- Convert `BIRTHDAY` to `AGE` in years, drop `BIRTHDAY`
- Create `AGE_GROUP` for EDA only (dropped before modelling)
- Drop low-variance binary flags: `FLAG_MOBIL`, `FLAG_WORK_PHONE`, `FLAG_PHONE`, `FLAG_EMAIL`
- Clean `CNT_CHILDREN`: remove non-integer values
- Apply IQR Winsorization to cap outliers in all numeric columns

### Feature Engineering & Encoding
- Binary map: `CODE_GENDER` (M=1, F=0), `FLAG_OWN_CAR` (Y=1, N=0), `FLAG_OWN_REALTY` (Y=1, N=0)
- One-hot encode `OCCUPATION_TYPE` with `dummy_na=True` (preserves missing as a category)
- One-hot encode all remaining categoricals

### Models — Regression (predicting SCORE)
| Model | Key Parameters |
|-------|----------------|
| Linear Regression | MinMaxScaled features |
| Decision Tree Regressor | max_depth=10, min_samples_leaf=10, min_samples_split=20 |
| Random Forest Regressor | n_estimators=300, max_depth=20, min_samples_leaf=5, max_features='sqrt' |

### Models — Classification (predicting APPROVAL)
| Model | Key Parameters |
|-------|----------------|
| Logistic Regression | max_iter=1000, class_weight='balanced', MinMaxScaled |
| Random Forest Classifier | n_estimators=300, max_depth=20, class_weight='balanced' |

### Train/Test Split
- 80% training / 20% testing, `random_state=42`
- MinMaxScaler fitted on training data only (no data leakage)
- APPROVAL threshold: Q3 of SCORE distribution

---

## Key Concepts Used

- **Separation of Concerns**: ML logic (model.py), visualizations (utils.py), UI (app.py)
- **IQR Winsorization**: Capping extreme values at Q1 − 1.5×IQR and Q3 + 1.5×IQR
- **Class imbalance**: Handled with `class_weight='balanced'` in classification models
- **Data leakage prevention**: Scaler fitted on `X_train` only, applied to `X_test`
- **ROC-AUC**: Used alongside accuracy to evaluate classifiers under class imbalance
- **Cross-validation**: 5-fold CV R² and RMSE for all regression models

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/capstone_credit_approval.git
cd capstone_credit_approval

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The dataset `AC_Capstone_topic_2.csv` must be in the same directory as `app.py`.

---

## Streamlit Cloud Deployment

1. Push this repository to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repository, branch `main`, entry file `app.py`
4. Click **Deploy** — Streamlit Cloud installs dependencies automatically

> **Note:** Streamlit Cloud runs Python 3.14. All libraries in `requirements.txt`
> are compatible. Do not add TensorFlow or PyTorch.

---

## Limitations

- Dataset sizes after deduplication (~27,918 rows) are sufficient but below production scale
- SCORE has near-zero correlation with available features → regression R² is very low (~0.03)
- No automated hyperparameter search (GridSearchCV/RandomizedSearchCV) was performed
- Streamlit Cloud free tier does not persist model files between sessions — models retrain on each visit
- Minimal feature engineering; domain-specific features (income-to-family ratio, etc.) were not explored
