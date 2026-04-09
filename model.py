"""
model.py — All ML logic for Credit Card Approval Prediction Capstone
CST2216 Machine Learning 2 | Algonquin College
"""

import logging
import warnings
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV dataset into a DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Raw DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: duplicates, missing values, outliers, feature engineering.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame ready for encoding.
    """
    try:
        df = df.copy()

        # Drop ID — unique identifier, not a feature
        if "ID" in df.columns:
            df.drop(columns=["ID"], inplace=True)
            logging.info("Dropped ID column.")

        # Remove duplicate rows
        before = len(df)
        df.drop_duplicates(inplace=True)
        logging.info(f"Removed {before - len(df)} duplicate rows. Remaining: {len(df)}.")

        # IS_EMPLOYED: Pensioners are not employed
        df["IS_EMPLOYED"] = (df["NAME_INCOME_TYPE"] != "Pensioner").astype(int)
        logging.info("Created IS_EMPLOYED feature.")

        # Drop EMPLOYED_DATE — replaced by IS_EMPLOYED
        if "EMPLOYED_DATE" in df.columns:
            df.drop(columns=["EMPLOYED_DATE"], inplace=True)

        # Convert BIRTHDAY → AGE
        df["BIRTHDAY"] = pd.to_datetime(df["BIRTHDAY"], errors="coerce")
        today = pd.Timestamp.today()
        df["AGE"] = (today - df["BIRTHDAY"]).dt.days // 365
        df.drop(columns=["BIRTHDAY"], inplace=True)
        logging.info("Converted BIRTHDAY to AGE.")

        # AGE_GROUP for EDA only — dropped before modelling
        df["AGE_GROUP"] = pd.cut(
            df["AGE"],
            bins=[18, 30, 40, 50, 60, 70],
            labels=["18-30", "31-40", "41-50", "51-60", "61-70"],
        )

        # Drop low-variance binary flags
        low_var = ["FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL"]
        df.drop(columns=[c for c in low_var if c in df.columns], inplace=True)
        logging.info("Dropped low-variance flag columns.")

        # Fix CNT_CHILDREN — keep only integer values
        df["CNT_CHILDREN"] = pd.to_numeric(df["CNT_CHILDREN"], errors="coerce")
        df = df[df["CNT_CHILDREN"].notna()]
        df = df[df["CNT_CHILDREN"] == df["CNT_CHILDREN"].astype(int)]
        df["CNT_CHILDREN"] = df["CNT_CHILDREN"].astype(int)
        logging.info("Cleaned CNT_CHILDREN column.")

        # IQR Winsorization on all numeric columns
        df = _cap_outliers_iqr(df)
        logging.info("Applied IQR winsorization to numeric columns.")

        logging.info(f"Cleaning complete. Final shape: {df.shape}")
        return df

    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        raise


def _cap_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers using IQR method (Winsorization).

    Args:
        df: DataFrame with numeric columns.

    Returns:
        DataFrame with capped outlier values.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode all categorical features. Must be called after clean_data().

    Args:
        df: Cleaned DataFrame.

    Returns:
        Fully numeric DataFrame ready for modelling.
    """
    try:
        df = df.copy()

        # Drop EDA-only columns before encoding
        df.drop(columns=["AGE", "AGE_GROUP"], inplace=True, errors="ignore")

        # Binary maps
        df["CODE_GENDER"]    = df["CODE_GENDER"].map({"M": 1, "F": 0})
        df["FLAG_OWN_CAR"]   = df["FLAG_OWN_CAR"].map({"Y": 1, "N": 0})
        df["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].map({"Y": 1, "N": 0})

        # OCCUPATION_TYPE: treat NaN as its own category
        df = pd.get_dummies(df, columns=["OCCUPATION_TYPE"], dummy_na=True)

        # Remaining categoricals
        df = pd.get_dummies(df, drop_first=False)

        # Convert all bool columns to int
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)

        logging.info(f"Encoding complete. Final shape: {df.shape}, columns: {df.shape[1]}")
        return df

    except Exception as e:
        logging.error(f"Error during encoding: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE / TARGET SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def prepare_regression(df: pd.DataFrame):
    """Prepare data for regression (predicting SCORE).

    Returns:
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_columns
    """
    try:
        y = df["SCORE"]
        X = df.drop(columns=["SCORE"])
        feature_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        logging.info(f"Regression split — Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_columns

    except Exception as e:
        logging.error(f"Error in prepare_regression: {e}")
        raise


def prepare_classification(df: pd.DataFrame):
    """Prepare data for classification (predicting APPROVAL = top 25% SCORE).

    Returns:
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled,
        scaler, feature_columns, q3_threshold
    """
    try:
        q3 = df["SCORE"].quantile(0.75)
        df = df.copy()
        df["APPROVAL"] = (df["SCORE"] >= q3).astype(int)

        y = df["APPROVAL"]
        X = df.drop(columns=["SCORE", "APPROVAL"])
        feature_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        logging.info(f"Classification split — Q3={q3:.0f}, Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_columns, q3

    except Exception as e:
        logging.error(f"Error in prepare_classification: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 5. REGRESSION MODELS
# ─────────────────────────────────────────────────────────────────────────────

def train_linear_regression(X_train_scaled, y_train) -> LinearRegression:
    """Train a Multiple Linear Regression model on scaled features.

    Args:
        X_train_scaled: Scaled training features.
        y_train: Training target (SCORE).

    Returns:
        Fitted LinearRegression model.
    """
    try:
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        logging.info("Linear Regression trained.")
        return model
    except Exception as e:
        logging.error(f"Error training Linear Regression: {e}")
        raise


def train_decision_tree_regressor(X_train, y_train) -> DecisionTreeRegressor:
    """Train a Decision Tree Regressor.

    Args:
        X_train: Unscaled training features.
        y_train: Training target (SCORE).

    Returns:
        Fitted DecisionTreeRegressor model.
    """
    try:
        model = DecisionTreeRegressor(
            max_depth=10,
            min_samples_leaf=10,
            min_samples_split=20,
            random_state=42,
        )
        model.fit(X_train, y_train)
        logging.info(f"Decision Tree trained — depth={model.get_depth()}, leaves={model.get_n_leaves()}")
        return model
    except Exception as e:
        logging.error(f"Error training Decision Tree Regressor: {e}")
        raise


def train_random_forest_regressor(X_train, y_train) -> RandomForestRegressor:
    """Train a Random Forest Regressor.

    Args:
        X_train: Unscaled training features.
        y_train: Training target (SCORE).

    Returns:
        Fitted RandomForestRegressor model.
    """
    try:
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        logging.info(f"Random Forest Regressor trained — {model.n_estimators} trees.")
        return model
    except Exception as e:
        logging.error(f"Error training Random Forest Regressor: {e}")
        raise


def evaluate_regression(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Evaluate a regression model and return key metrics.

    Args:
        model: Fitted regression model.
        X_test: Test features (scaled or unscaled, matching how model was trained).
        y_test: True target values.
        model_name: Label for logging.

    Returns:
        Dictionary with R2, Adjusted_R2, RMSE, MAE.
    """
    try:
        y_pred = model.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)

        n, p = X_test.shape
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        metrics = {"R2": round(r2, 4), "Adjusted_R2": round(adj_r2, 4),
                   "RMSE": round(rmse, 2), "MAE": round(mae, 2),
                   "y_pred": y_pred, "y_test": y_test}

        logging.info(f"{model_name} — R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")
        return metrics

    except Exception as e:
        logging.error(f"Error evaluating regression model: {e}")
        raise


def cross_validate_regression(model, X_train, y_train, cv: int = 5) -> dict:
    """Run k-fold cross-validation for a regression model.

    Args:
        model: Fitted or unfitted sklearn regressor.
        X_train: Training features.
        y_train: Training target.
        cv: Number of folds.

    Returns:
        Dictionary with CV R2 mean/std and CV RMSE mean/std.
    """
    try:
        cv_r2   = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        cv_rmse = -cross_val_score(model, X_train, y_train, cv=cv,
                                   scoring="neg_root_mean_squared_error", n_jobs=-1)
        return {
            "CV_R2_mean":   round(cv_r2.mean(), 4),
            "CV_R2_std":    round(cv_r2.std(), 4),
            "CV_RMSE_mean": round(cv_rmse.mean(), 2),
            "CV_RMSE_std":  round(cv_rmse.std(), 2),
        }
    except Exception as e:
        logging.error(f"Error during cross-validation: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLASSIFICATION MODELS
# ─────────────────────────────────────────────────────────────────────────────

def train_logistic_regression(X_train_scaled, y_train) -> LogisticRegression:
    """Train a Logistic Regression classifier on scaled features.

    Args:
        X_train_scaled: Scaled training features.
        y_train: Binary target (APPROVAL).

    Returns:
        Fitted LogisticRegression model.
    """
    try:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X_train_scaled, y_train)
        logging.info("Logistic Regression trained.")
        return model
    except Exception as e:
        logging.error(f"Error training Logistic Regression: {e}")
        raise


def train_random_forest_classifier(X_train, y_train) -> RandomForestClassifier:
    """Train a Random Forest Classifier.

    Args:
        X_train: Unscaled training features.
        y_train: Binary target (APPROVAL).

    Returns:
        Fitted RandomForestClassifier model.
    """
    try:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        logging.info(f"Random Forest Classifier trained — {model.n_estimators} trees.")
        return model
    except Exception as e:
        logging.error(f"Error training Random Forest Classifier: {e}")
        raise


def evaluate_classification(model, X_test, y_test, model_name: str = "Model",
                             predict_proba: bool = True) -> dict:
    """Evaluate a classification model and return key metrics.

    Args:
        model: Fitted classification model.
        X_test: Test features.
        y_test: True binary labels.
        model_name: Label for logging.
        predict_proba: Whether to compute ROC-AUC (requires predict_proba support).

    Returns:
        Dictionary with accuracy, report, confusion matrix, ROC-AUC, predictions.
    """
    try:
        y_pred  = model.predict(X_test)
        acc     = accuracy_score(y_test, y_pred)
        report  = classification_report(y_test, y_pred,
                                        target_names=["Not Approved", "Approved"],
                                        output_dict=True)
        cm      = confusion_matrix(y_test, y_pred)

        auc = None
        y_proba = None
        if predict_proba and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = round(roc_auc_score(y_test, y_proba), 4)

        logging.info(f"{model_name} — Accuracy={acc:.4f}, AUC={auc}")
        return {"Accuracy": round(acc, 4), "Report": report,
                "Confusion_Matrix": cm, "ROC_AUC": auc,
                "y_pred": y_pred, "y_proba": y_proba, "y_test": y_test}

    except Exception as e:
        logging.error(f"Error evaluating classification model: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 7. PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def predict_score(input_dict: dict, scaler, model,
                  feature_columns: list) -> float:
    """Predict credit SCORE for a single applicant.

    Args:
        input_dict: Raw applicant feature dictionary (pre-encoding).
        scaler: Fitted MinMaxScaler for regression.
        model: Fitted regression model.
        feature_columns: Ordered list of feature column names from training.

    Returns:
        Predicted score as float.
    """
    try:
        row = _encode_input(input_dict, feature_columns)
        row_scaled = scaler.transform(row)
        prediction = model.predict(row_scaled)[0]
        logging.info(f"Predicted SCORE: {prediction:.0f}")
        return float(prediction)
    except Exception as e:
        logging.error(f"Error in predict_score: {e}")
        raise


def predict_approval(input_dict: dict, scaler, model,
                     feature_columns: list, q3_threshold: float) -> dict:
    """Predict credit card APPROVAL for a single applicant.

    Args:
        input_dict: Raw applicant feature dictionary (pre-encoding).
        scaler: Fitted MinMaxScaler for classification (used only for LogReg).
        model: Fitted classification model.
        feature_columns: Ordered list of feature column names from training.
        q3_threshold: Q3 SCORE value used to define the APPROVAL boundary.

    Returns:
        Dictionary with 'label' (Approved/Not Approved) and 'probability'.
    """
    try:
        row = _encode_input(input_dict, feature_columns)

        # Use scaled input only for Logistic Regression
        if isinstance(model, LogisticRegression):
            row_input = scaler.transform(row)
        else:
            row_input = row.values

        prob  = model.predict_proba(row_input)[0][1]
        label = "Approved" if prob >= 0.5 else "Not Approved"
        logging.info(f"Approval prediction: {label} ({prob:.2%})")
        return {"label": label, "probability": round(float(prob), 4),
                "q3_threshold": q3_threshold}
    except Exception as e:
        logging.error(f"Error in predict_approval: {e}")
        raise


def _encode_input(input_dict: dict, feature_columns: list) -> pd.DataFrame:
    """Encode a single applicant input dict into a model-ready DataFrame row.

    Args:
        input_dict: Keys are raw feature names, values are raw user inputs.
        feature_columns: Full list of columns expected by the model.

    Returns:
        Single-row DataFrame aligned to feature_columns.
    """
    # Binary maps
    row = {}
    row["CODE_GENDER"]    = 1 if input_dict.get("CODE_GENDER") == "M" else 0
    row["FLAG_OWN_CAR"]   = 1 if input_dict.get("FLAG_OWN_CAR") == "Y" else 0
    row["FLAG_OWN_REALTY"] = 1 if input_dict.get("FLAG_OWN_REALTY") == "Y" else 0
    row["CNT_CHILDREN"]   = int(input_dict.get("CNT_CHILDREN", 0))
    row["AMT_INCOME_TOTAL"] = float(input_dict.get("AMT_INCOME_TOTAL", 0))
    row["CNT_FAM_MEMBERS"] = int(input_dict.get("CNT_FAM_MEMBERS", 2))
    row["IS_EMPLOYED"]    = 1 if input_dict.get("NAME_INCOME_TYPE") != "Pensioner" else 0

    # One-hot: OCCUPATION_TYPE (with NaN category)
    occ = input_dict.get("OCCUPATION_TYPE", None)
    all_occ = [
        "Accountants", "Cleaning staff", "Cooking staff", "Core staff", "Drivers",
        "HR staff", "High skill tech staff", "IT staff", "Laborers",
        "Low-skill Laborers", "Managers", "Medicine staff", "Private service staff",
        "Realty agents", "Sales staff", "Secretaries", "Security staff",
        "Waiters/barmen staff",
    ]
    for o in all_occ:
        row[f"OCCUPATION_TYPE_{o}"] = 1 if occ == o else 0
    row["OCCUPATION_TYPE_nan"] = 1 if (occ is None or occ == "Unknown / Not Provided") else 0

    # One-hot: NAME_INCOME_TYPE
    for v in ["Commercial associate", "Pensioner", "State servant", "Student", "Working"]:
        row[f"NAME_INCOME_TYPE_{v}"] = 1 if input_dict.get("NAME_INCOME_TYPE") == v else 0

    # One-hot: NAME_EDUCATION_TYPE
    for v in ["Academic degree", "Higher education", "Incomplete higher",
              "Lower secondary", "Secondary / secondary special"]:
        row[f"NAME_EDUCATION_TYPE_{v}"] = 1 if input_dict.get("NAME_EDUCATION_TYPE") == v else 0

    # One-hot: NAME_FAMILY_STATUS
    for v in ["Civil marriage", "Married", "Separated", "Single / not married", "Widow"]:
        row[f"NAME_FAMILY_STATUS_{v}"] = 1 if input_dict.get("NAME_FAMILY_STATUS") == v else 0

    # One-hot: NAME_HOUSING_TYPE
    for v in ["Co-op apartment", "House / apartment", "Municipal apartment",
              "Office apartment", "Rented apartment", "With parents"]:
        row[f"NAME_HOUSING_TYPE_{v}"] = 1 if input_dict.get("NAME_HOUSING_TYPE") == v else 0

    # Build DataFrame aligned to training columns
    df_row = pd.DataFrame([row])
    for col in feature_columns:
        if col not in df_row.columns:
            df_row[col] = 0
    df_row = df_row[feature_columns]
    return df_row


# ─────────────────────────────────────────────────────────────────────────────
# 8. MODEL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_models(objects: dict, path: str = MODELS_DIR) -> None:
    """Pickle all model objects to disk.

    Args:
        objects: Dictionary of {filename_stem: object} to save.
        path: Directory to save into.
    """
    try:
        os.makedirs(path, exist_ok=True)
        for name, obj in objects.items():
            filepath = os.path.join(path, f"{name}.pkl")
            with open(filepath, "wb") as f:
                pickle.dump(obj, f)
            logging.info(f"Saved: {filepath}")
    except Exception as e:
        logging.error(f"Error saving models: {e}")
        raise


def load_models(path: str = MODELS_DIR) -> dict:
    """Load all pickled model objects from disk.

    Args:
        path: Directory to load from.

    Returns:
        Dictionary of {filename_stem: object}.
    """
    try:
        objects = {}
        for fname in os.listdir(path):
            if fname.endswith(".pkl"):
                key = fname.replace(".pkl", "")
                with open(os.path.join(path, fname), "rb") as f:
                    objects[key] = pickle.load(f)
                logging.info(f"Loaded: {fname}")
        return objects
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        raise
