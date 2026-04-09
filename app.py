"""
app.py — Streamlit UI for Credit Card Approval Prediction
CST2216 Machine Learning 2 | Algonquin College
Group: Fritz Mwambo · Hisham Khraibah · Jinchuan Liu · Youssra Ibrahim
"""

import os
import logging
import warnings

import numpy as np
import pandas as pd
import streamlit as st

import model as m
import utils as u

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Approval Prediction",
    page_icon="💳",
    layout="wide",
)

DATASET_PATH = "AC_Capstone_topic_2.csv"

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — load and cache data + models
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset…")
def get_data():
    raw   = m.load_data(DATASET_PATH)
    clean = m.clean_data(raw)
    enc   = m.encode_features(clean)
    q3    = float(raw["SCORE"].quantile(0.75)) if "SCORE" in raw.columns else float(enc["SCORE"].quantile(0.75))
    return raw, clean, enc, q3


@st.cache_resource(show_spinner="Training models…")
def get_trained_models():
    raw, clean, enc, q3 = get_data()

    (X_train_r, X_test_r, y_train_r, y_test_r,
     X_train_r_sc, X_test_r_sc,
     scaler_r, feat_cols_r) = m.prepare_regression(enc)

    lr_model   = m.train_linear_regression(X_train_r_sc, y_train_r)
    dt_model   = m.train_decision_tree_regressor(X_train_r, y_train_r)
    rf_r_model = m.train_random_forest_regressor(X_train_r, y_train_r)

    lr_metrics   = m.evaluate_regression(lr_model,   X_test_r_sc, y_test_r, "Linear Regression")
    dt_metrics   = m.evaluate_regression(dt_model,   X_test_r,    y_test_r, "Decision Tree")
    rf_r_metrics = m.evaluate_regression(rf_r_model, X_test_r,    y_test_r, "Random Forest")

    lr_cv   = m.cross_validate_regression(lr_model,   X_train_r_sc, y_train_r)
    dt_cv   = m.cross_validate_regression(dt_model,   X_train_r,    y_train_r)
    rf_r_cv = m.cross_validate_regression(rf_r_model, X_train_r,    y_train_r)

    (X_train_c, X_test_c, y_train_c, y_test_c,
     X_train_c_sc, X_test_c_sc,
     scaler_c, feat_cols_c, q3_val) = m.prepare_classification(enc)

    log_model  = m.train_logistic_regression(X_train_c_sc, y_train_c)
    rf_c_model = m.train_random_forest_classifier(X_train_c, y_train_c)

    log_metrics  = m.evaluate_classification(log_model,  X_test_c_sc, y_test_c, "Logistic Regression")
    rf_c_metrics = m.evaluate_classification(rf_c_model, X_test_c,    y_test_c, "Random Forest Classifier")

    return {
        "lr_model": lr_model,     "dt_model": dt_model,     "rf_r_model": rf_r_model,
        "lr_metrics": lr_metrics, "dt_metrics": dt_metrics, "rf_r_metrics": rf_r_metrics,
        "lr_cv": lr_cv,           "dt_cv": dt_cv,           "rf_r_cv": rf_r_cv,
        "scaler_r": scaler_r,     "feat_cols_r": feat_cols_r,
        "X_train_r": X_train_r,   "y_train_r": y_train_r,
        "log_model": log_model,   "rf_c_model": rf_c_model,
        "log_metrics": log_metrics, "rf_c_metrics": rf_c_metrics,
        "scaler_c": scaler_c,     "feat_cols_c": feat_cols_c,
        "q3": q3_val,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.image(
    "https://www.algonquincollege.com/wp-content/uploads/2022/05/AC_RGB_horiz_pos.png",
    use_column_width=True,
)
st.sidebar.title("💳 Credit Card Approval")
st.sidebar.markdown("**CST2216 · ML2 Capstone**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Data Overview", "📈 Regression", "🎯 Classification", "🔮 Predict"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Group Members**\n"
    "- Fritz Mwambo\n"
    "- Hisham Khraibah\n"
    "- Jinchuan Liu\n"
    "- Youssra Ibrahim"
)

raw, clean, enc, q3 = get_data()
trained = get_trained_models()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — DATA OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

if page == "📊 Data Overview":
    st.title("📊 Data Overview")
    st.markdown(
        "Exploratory analysis of the **AC_Capstone_topic_2** dataset containing "
        f"**{raw.shape[0]:,} applicant records** and **{raw.shape[1]} features** "
        "before and after cleaning."
    )

    st.subheader("Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Raw Rows",           f"{raw.shape[0]:,}")
    col2.metric("Cleaned Rows",       f"{clean.shape[0]:,}")
    col3.metric("Features",           f"{raw.shape[1] - 1}")
    col4.metric("Duplicates Removed", f"{raw.shape[0] - clean.shape[0]:,}")

    with st.expander("Preview raw data (first 10 rows)"):
        st.dataframe(raw.head(10))

    with st.expander("Cleaned dataset info"):
        st.dataframe(clean.describe(include="all").T)

    st.subheader("Missing Values (Raw Dataset)")
    st.pyplot(u.plot_missing_values(raw))

    st.subheader("SCORE Distribution")
    st.pyplot(u.plot_score_distribution(clean))
    st.info(
        f"📌 Most scores are concentrated between **8,500 and 10,000**. "
        f"The distribution is left-skewed — mean ≈ **{clean['SCORE'].mean():.0f}**, "
        f"median ≈ **{clean['SCORE'].median():.0f}**."
    )

    st.subheader("Approval Distribution")
    st.pyplot(u.plot_approval_distribution(clean, q3))
    st.info(
        f"📌 Approval threshold = Q3 of SCORE = **{q3:.0f}**. "
        f"Top 25% of applicants are classified as **Approved**."
    )

    st.subheader("Credit Score by Age Group")
    st.pyplot(u.plot_age_group_vs_score(clean))

    st.subheader("Occupation Type Distribution")
    st.pyplot(u.plot_occupation_distribution(clean))
    st.warning(
        "⚠️ OCCUPATION_TYPE has **31.1% missing values** in the raw data. "
        "Missing values are treated as a separate category ('Unknown / Not Provided') "
        "to preserve all records."
    )

    st.subheader("Feature Correlation Heatmap (Encoded)")
    st.pyplot(u.plot_correlation_heatmap(enc))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📈 Regression":
    st.title("📈 Regression — Predicting Credit Score")
    st.markdown(
        "Three regression models are trained to predict the **continuous SCORE** variable. "
        "Note: SCORE has near-zero correlation with the available features, "
        "so low R² values are expected and are an honest reflection of the data's limitations."
    )

    model_choice = st.selectbox(
        "Select model to inspect",
        ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"],
    )

    key_map = {
        "Linear Regression":       ("lr_model",   "lr_metrics",   "lr_cv"),
        "Decision Tree Regressor": ("dt_model",   "dt_metrics",   "dt_cv"),
        "Random Forest Regressor": ("rf_r_model", "rf_r_metrics", "rf_r_cv"),
    }
    mk, mek, cvk = key_map[model_choice]
    mdl    = trained[mk]
    met    = trained[mek]
    cv     = trained[cvk]
    feat_r = trained["feat_cols_r"]

    st.subheader(f"{model_choice} — Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test R²",        f"{met['R2']:.4f}")
    c2.metric("Adjusted R²",    f"{met['Adjusted_R2']:.4f}")
    c3.metric("Test RMSE",      f"{met['RMSE']:.2f}")
    c4.metric("Test MAE",       f"{met['MAE']:.2f}")

    c5, c6 = st.columns(2)
    c5.metric("CV R² (mean)",   f"{cv['CV_R2_mean']:.4f}")
    c6.metric("CV RMSE (mean)", f"{cv['CV_RMSE_mean']:.2f}")

    if met["R2"] < 0.05:
        st.warning(
            "⚠️ R² is very close to zero, indicating that SCORE cannot be reliably predicted "
            "from the available demographic and financial features. This is consistent with the "
            "notebook's findings that feature-SCORE correlations are near zero throughout."
        )

    st.subheader("Diagnostic Plots")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.pyplot(u.plot_actual_vs_predicted(
            met["y_test"], met["y_pred"], model_choice, met["R2"]))
    with col_b:
        st.pyplot(u.plot_residual_distribution(
            met["y_test"], met["y_pred"], model_choice))
    with col_c:
        st.pyplot(u.plot_residuals_vs_predicted(
            met["y_test"], met["y_pred"], model_choice))

    if hasattr(mdl, "feature_importances_") or hasattr(mdl, "coef_"):
        st.subheader("Feature Importance")
        st.pyplot(u.plot_regression_feature_importance(mdl, feat_r, model_choice))

    st.subheader("All Models — Comparison")
    metrics_summary = {
        "Linear Regression":       trained["lr_metrics"],
        "Decision Tree":           trained["dt_metrics"],
        "Random Forest Regressor": trained["rf_r_metrics"],
    }
    st.pyplot(u.plot_regression_comparison(metrics_summary))

    rows = []
    for name, met_d in metrics_summary.items():
        cv_d = trained[{"Linear Regression": "lr_cv",
                         "Decision Tree": "dt_cv",
                         "Random Forest Regressor": "rf_r_cv"}[name]]
        rows.append({
            "Model":          name,
            "Test R²":        met_d["R2"],
            "Adjusted R²":    met_d["Adjusted_R2"],
            "Test RMSE":      met_d["RMSE"],
            "Test MAE":       met_d["MAE"],
            "CV R² (mean)":   cv_d["CV_R2_mean"],
            "CV RMSE (mean)": cv_d["CV_RMSE_mean"],
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🎯 Classification":
    st.title("🎯 Classification — Predicting Credit Card Approval")
    st.markdown(
        f"Applicants with SCORE ≥ **{trained['q3']:.0f}** (Q3) are classified as **Approved**. "
        "Two classifiers are compared: Logistic Regression and Random Forest."
    )

    model_choice_c = st.selectbox(
        "Select classifier to inspect",
        ["Logistic Regression", "Random Forest Classifier"],
    )

    key_map_c = {
        "Logistic Regression":      ("log_model",  "log_metrics"),
        "Random Forest Classifier": ("rf_c_model", "rf_c_metrics"),
    }
    cmk, cmek = key_map_c[model_choice_c]
    cmet  = trained[cmek]
    cmdl  = trained[cmk]
    feat_c = trained["feat_cols_c"]

    st.subheader(f"{model_choice_c} — Performance Metrics")
    cc1, cc2 = st.columns(2)
    cc1.metric("Accuracy", f"{cmet['Accuracy']:.4f}")
    if cmet["ROC_AUC"]:
        cc2.metric("ROC-AUC", f"{cmet['ROC_AUC']:.4f}")

    with st.expander("Classification Report"):
        st.dataframe(pd.DataFrame(cmet["Report"]).T.round(4))

    st.subheader("Confusion Matrix")
    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.pyplot(u.plot_confusion_matrix(cmet["Confusion_Matrix"], model_choice_c))
    with col_roc:
        if cmet["y_proba"] is not None:
            st.pyplot(u.plot_roc_curve(
                cmet["y_test"], cmet["y_proba"], model_choice_c, cmet["ROC_AUC"]))

    if hasattr(cmdl, "feature_importances_"):
        st.subheader("Feature Importance")
        st.pyplot(u.plot_classification_feature_importance(cmdl, feat_c, model_choice_c))

    st.subheader("Feature Correlation with APPROVAL Target")
    st.pyplot(u.plot_correlation_with_approval(enc, trained["q3"]))

    st.subheader("Logistic Regression vs Random Forest — Comparison")
    comp_data = {
        "Model":    ["Logistic Regression", "Random Forest Classifier"],
        "Accuracy": [trained["log_metrics"]["Accuracy"], trained["rf_c_metrics"]["Accuracy"]],
        "ROC-AUC":  [trained["log_metrics"]["ROC_AUC"],  trained["rf_c_metrics"]["ROC_AUC"]],
    }
    st.dataframe(pd.DataFrame(comp_data).set_index("Model"))

    if (trained["log_metrics"]["y_proba"] is not None and
            trained["rf_c_metrics"]["y_proba"] is not None):
        st.pyplot(u.plot_roc_comparison(
            trained["log_metrics"]["y_test"],
            trained["log_metrics"]["y_proba"],
            trained["rf_c_metrics"]["y_proba"],
            trained["log_metrics"]["ROC_AUC"],
            trained["rf_c_metrics"]["ROC_AUC"],
        ))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — PREDICT
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔮 Predict":
    st.title("🔮 Predict — Credit Score & Approval")
    st.markdown(
        "Enter an applicant's details below and choose whether to predict their "
        "**Credit Score** or **Approval Status**."
    )

    pred_mode = st.radio(
        "Prediction type",
        ["💰 Predict Credit Score", "✅ Predict Approval Status"],
        horizontal=True,
    )

    st.markdown("---")
    st.subheader("Applicant Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender       = st.radio("Gender", ["Male", "Female"], horizontal=True)
        own_car      = st.radio("Owns a Car?", ["Yes", "No"], horizontal=True)
        own_realty   = st.radio("Owns Real Estate?", ["Yes", "No"], horizontal=True)
        cnt_children = st.slider("Number of Children", 0, 10, 0)

    with col2:
        income = st.number_input(
            "Annual Income (CAD)", min_value=27000, max_value=1575000,
            value=150000, step=5000, format="%d",
        )
        cnt_fam     = st.slider("Family Members", 1, 10, 2)
        income_type = st.selectbox(
            "Income Type",
            ["Working", "Commercial associate", "State servant", "Pensioner", "Student"],
        )
        education = st.selectbox(
            "Education Level",
            ["Secondary / secondary special", "Higher education",
             "Incomplete higher", "Academic degree", "Lower secondary"],
        )

    with col3:
        family_status = st.selectbox(
            "Family Status",
            ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        )
        housing = st.selectbox(
            "Housing Type",
            ["House / apartment", "Rented apartment", "With parents",
             "Municipal apartment", "Co-op apartment", "Office apartment"],
        )
        birthday   = st.date_input("Date of Birth", value=pd.Timestamp("1985-01-01"))
        occupation = st.selectbox(
            "Occupation Type",
            ["Unknown / Not Provided", "Accountants", "Cleaning staff", "Cooking staff",
             "Core staff", "Drivers", "HR staff", "High skill tech staff", "IT staff",
             "Laborers", "Low-skill Laborers", "Managers", "Medicine staff",
             "Private service staff", "Realty agents", "Sales staff", "Secretaries",
             "Security staff", "Waiters/barmen staff"],
        )

    input_dict = {
        "CODE_GENDER":         "M" if gender == "Male" else "F",
        "FLAG_OWN_CAR":        "Y" if own_car == "Yes" else "N",
        "FLAG_OWN_REALTY":     "Y" if own_realty == "Yes" else "N",
        "CNT_CHILDREN":        cnt_children,
        "AMT_INCOME_TOTAL":    income,
        "CNT_FAM_MEMBERS":     cnt_fam,
        "NAME_INCOME_TYPE":    income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS":  family_status,
        "NAME_HOUSING_TYPE":   housing,
        "OCCUPATION_TYPE":     None if occupation == "Unknown / Not Provided" else occupation,
        "BIRTHDAY":            str(birthday),
    }

    st.markdown("---")

    if st.button("🔮 Run Prediction", type="primary"):
        try:
            if pred_mode == "💰 Predict Credit Score":
                score = m.predict_score(
                    input_dict, trained["scaler_r"],
                    trained["rf_r_model"], trained["feat_cols_r"],
                )
                st.success(f"### Predicted Credit Score: **{score:,.0f}**")
                st.caption(
                    f"Random Forest Regressor · Test RMSE ≈ {trained['rf_r_metrics']['RMSE']:.0f} points · "
                    f"R² ≈ {trained['rf_r_metrics']['R2']:.4f}"
                )
                if score >= trained["q3"]:
                    st.info(
                        f"💡 This score ({score:,.0f}) is above the Q3 threshold "
                        f"({trained['q3']:.0f}), suggesting likely **Approval**."
                    )
                else:
                    st.info(
                        f"💡 This score ({score:,.0f}) is below the Q3 threshold "
                        f"({trained['q3']:.0f}), suggesting likely **Not Approved**."
                    )
            else:
                result = m.predict_approval(
                    input_dict, trained["scaler_c"],
                    trained["rf_c_model"], trained["feat_cols_c"], trained["q3"],
                )
                label = result["label"]
                prob  = result["probability"]
                if label == "Approved":
                    st.success(f"### ✅ {label}  —  Confidence: {prob:.1%}")
                else:
                    st.error(f"### ❌ {label}  —  Confidence: {1 - prob:.1%}")
                st.caption(
                    f"Random Forest Classifier · "
                    f"ROC-AUC = {trained['rf_c_metrics']['ROC_AUC']:.4f} · "
                    f"Approval threshold: SCORE ≥ {trained['q3']:.0f} (Q3)"
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            logging.error(f"Prediction error: {e}")
