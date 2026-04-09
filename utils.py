"""
utils.py — All charts and visualizations for Credit Card Approval Prediction
CST2216 Machine Learning 2 | Algonquin College
"""

import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.tree import plot_tree

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

# ── Shared style ──────────────────────────────────────────────
PALETTE = {
    "primary":   "#1B6CA8",
    "secondary": "#02C39A",
    "orange":    "#E07B00",
    "light":     "#AEC7E8",
    "gray":      "#5A6E82",
    "red":       "#EF4444",
}
sns.set_theme(style="whitegrid", palette="muted")


# ─────────────────────────────────────────────────────────────────────────────
# DATA OVERVIEW CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram + KDE of the SCORE target variable with mean/median lines.

    Args:
        df: Cleaned DataFrame containing SCORE column.

    Returns:
        Matplotlib Figure.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df["SCORE"], bins=30, kde=True, color=PALETTE["primary"],
                     edgecolor="white", ax=ax)
        mean_val   = df["SCORE"].mean()
        median_val = df["SCORE"].median()
        ax.axvline(mean_val,   color=PALETTE["red"],    linestyle="--", lw=1.8,
                   label=f"Mean = {mean_val:.0f}")
        ax.axvline(median_val, color=PALETTE["orange"], linestyle="--", lw=1.8,
                   label=f"Median = {median_val:.0f}")
        ax.set_title("Distribution of Credit Score (SCORE)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        logging.info("plot_score_distribution rendered.")
        return fig
    except Exception as e:
        logging.error(f"Error in plot_score_distribution: {e}")
        raise


def plot_approval_distribution(df: pd.DataFrame, q3: float) -> plt.Figure:
    """Count plot of Approved vs Not Approved applicants.

    Args:
        df: DataFrame containing SCORE column.
        q3: Q3 threshold used to define APPROVAL.

    Returns:
        Matplotlib Figure.
    """
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        labels = df["SCORE"].apply(lambda s: "Approved" if s >= q3 else "Not Approved")
        order  = ["Not Approved", "Approved"]
        colors = [PALETTE["gray"], PALETTE["secondary"]]
        sns.countplot(x=labels, order=order, palette=colors, ax=ax, edgecolor="white")
        for container in ax.containers:
            ax.bar_label(container, fontsize=11)
        ax.set_title(f"Applicant Approval Distribution\n(Threshold: SCORE ≥ {q3:.0f})",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Application Status")
        ax.set_ylabel("Number of Applicants")
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error in plot_approval_distribution: {e}")
        raise


def plot_missing_values(df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart showing missing value counts per column.

    Args:
        df: Raw DataFrame before cleaning.

    Returns:
        Matplotlib Figure.
    """
    try:
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=True)
        if missing.empty:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "No missing values found.", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.axis("off")
            return fig

        fig, ax = plt.subplots(figsize=(8, max(3, len(missing) * 0.5)))
        colors = [PALETTE["orange"] if v > 5000 else PALETTE["primary"] for v in missing.values]
        missing.plot(kind="barh", color=colors, edgecolor="white", ax=ax)
        for i, v in enumerate(missing.values):
            ax.text(v + 50, i, f"{v:,}  ({v / len(df) * 100:.1f}%)", va="center", fontsize=9)
        ax.set_title("Missing Values per Column", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Missing Values")
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error in plot_missing_values: {e}")
        raise


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap of numeric feature correlations.

    Args:
        df: Encoded numeric DataFrame.

    Returns:
        Matplotlib Figure.
    """
    try:
        corr = df.select_dtypes(include="number").corr()
        fig, ax = plt.subplots(figsize=(14, 11))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm",
                    linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error in plot_correlation_heatmap: {e}")
        raise


def plot_occupation_distribution(df: pd.DataFrame) -> plt.Figure:
    """Count plot of OCCUPATION_TYPE including missing values.

    Args:
        df: Cleaned DataFrame (before encoding) with OCCUPATION_TYPE column.

    Returns:
        Matplotlib Figure.
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 5))
        counts = df["OCCUPATION_TYPE"].fillna("Unknown / Not Provided").value_counts()
        colors = [PALETTE["orange"] if k == "Unknown / Not Provided"
                  else PALETTE["primary"] for k in counts.index]
        counts.sort_values().plot(kind="barh", color=colors[::-1], edgecolor="white", ax=ax)
        ax.set_title("Occupation Type Distribution (incl. Missing)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Count")
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error in plot_occupation_distribution: {e}")
        raise


def plot_age_group_vs_score(df: pd.DataFrame) -> plt.Figure:
    """Box plot of SCORE by AGE_GROUP.

    Args:
        df: Cleaned DataFrame with AGE and SCORE columns.

    Returns:
        Matplotlib Figure.
    """
    try:
        df = df.copy()
        df["AGE_GROUP"] = pd.cut(df["AGE"], bins=[18, 30, 40, 50, 60, 70],
                                 labels=["18-30", "31-40", "41-50", "51-60", "61-70"])
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x="AGE_GROUP", y="SCORE", data=df, palette="Blues", ax=ax)
        ax.set_title("Credit Score Distribution by Age Group", fontsize=13, fontweight="bold")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Credit Score")
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error in plot_age_group_vs_score: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(y_test, y_pred, model_name: str, r2: float) -> plt.Figure:
    """Scatter plot of actual vs predicted SCORE values.

    Args:
        y_test: True SCORE values.
        y_pred: Predicted SCORE values.
        model_name: Label for plot title.
        r2: R² score for display.

    Returns:
        Matplotlib Figure.
    """
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_test, y_pred, alpha=0.25, s=8, color=PALETTE["primary"])
        rng = [float(min(y_test)), float(max(y_test))]
        ax.plot(rng, rng, "r--", lw=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Predicted Score")
        ax.set_title(f"{model_name} — Actual vs Predicted\n(Test R² = {r2:.4f})",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error in plot_actual_vs_predicted: {e}")
        raise


def plot_residual_distribution(y_test, y_pred, model_name: str) -> plt.Figure:
    """Histogram of prediction residuals.

    Args:
        y_test: True values.
        y_pred: Predicted values.
        model_name: Label for title.

    Returns:
        Matplotlib Figure.
    """
    try:
        residuals = np.array(y_test) - np.array(y_pred)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(residuals, bins=50, color=PALETTE["primary"], edgecolor="white")
        ax.axvline(0, color=PALETTE["red"], linestyle="--", lw=1.5)
        ax.set_title(f"{model_name} — Residual Distribution\n"
                     f"Mean={residuals.mean():.1f}  Std={residuals.std():.1f}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Residual (Actual − Predicted)")
        ax.set_ylabel("Count")
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error in plot_residual_distribution: {e}")
        raise


def plot_residuals_vs_predicted(y_test, y_pred, model_name: str) -> plt.Figure:
    """Scatter plot of residuals vs predicted values.

    Args:
        y_test: True values.
        y_pred: Predicted values.
        model_name: Label for title.

    Returns:
        Matplotlib Figure.
    """
    try:
        residuals = np.array(y_test) - np.array(y_pred)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(y_pred, residuals, alpha=0.25, s=8, color=PALETTE["primary"])
        ax.axhline(0, color=PALETTE["red"], linestyle="--", lw=1.5)
        ax.set_xlabel("Predicted Score")
        ax.set_ylabel("Residual")
        ax.set_title(f"{model_name} — Residuals vs Predicted",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error in plot_residuals_vs_predicted: {e}")
        raise


def plot_regression_feature_importance(model, feature_names: list,
                                       model_name: str, top_n: int = 15) -> plt.Figure:
    """Horizontal bar chart of regression feature importances or coefficients.

    Args:
        model: Fitted regression model.
        feature_names: List of feature column names.
        model_name: Label for title.
        top_n: Number of top features to display.

    Returns:
        Matplotlib Figure.
    """
    try:
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_names)
            xlabel = "Importance Score (Mean Decrease in Impurity)"
        else:
            imp = pd.Series(np.abs(model.coef_), index=feature_names)
            xlabel = "|Coefficient|"

        top = imp.sort_values(ascending=False).head(top_n).sort_values()
        colors = [PALETTE["primary"] if i >= len(top) - 5 else PALETTE["light"]
                  for i in range(len(top))]
        fig, ax = plt.subplots(figsize=(10, 6))
        top.plot(kind="barh", color=colors[::-1], edgecolor="white", ax=ax)
        ax.set_title(f"Top {top_n} Feature Importances — {model_name}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error in plot_regression_feature_importance: {e}")
        raise


def plot_regression_comparison(metrics_dict: dict) -> plt.Figure:
    """Side-by-side bar chart comparing regression model R² and RMSE.

    Args:
        metrics_dict: {model_name: {R2, RMSE, ...}} for each model.

    Returns:
        Matplotlib Figure.
    """
    try:
        names  = list(metrics_dict.keys())
        r2s    = [metrics_dict[m]["R2"]   for m in names]
        rmses  = [metrics_dict[m]["RMSE"] for m in names]
        colors = [PALETTE["primary"], PALETTE["orange"], PALETTE["secondary"]]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # R²
        bars = axes[0].bar(names, r2s, color=colors[: len(names)], edgecolor="white")
        axes[0].set_title("Test R² Comparison (Higher = Better)", fontweight="bold")
        axes[0].set_ylabel("R²")
        axes[0].axhline(0, color="black", lw=0.8, linestyle="--")
        for bar, val in zip(bars, r2s):
            axes[0].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.001, f"{val:.4f}",
                         ha="center", va="bottom", fontsize=9)

        # RMSE
        bars2 = axes[1].bar(names, rmses, color=colors[: len(names)], edgecolor="white")
        axes[1].set_title("Test RMSE Comparison (Lower = Better)", fontweight="bold")
        axes[1].set_ylabel("RMSE (score points)")
        for bar, val in zip(bars2, rmses):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 1, f"{val:.1f}",
                         ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error in plot_regression_comparison: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, model_name: str) -> plt.Figure:
    """Heatmap of a confusion matrix.

    Args:
        cm: Confusion matrix array.
        model_name: Label for title.

    Returns:
        Matplotlib Figure.
    """
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        labels = ["Not Approved", "Approved"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error in plot_confusion_matrix: {e}")
        raise


def plot_roc_curve(y_test, y_proba, model_name: str, auc: float) -> plt.Figure:
    """ROC curve for a single classification model.

    Args:
        y_test: True binary labels.
        y_proba: Predicted probabilities for the positive class.
        model_name: Label for legend.
        auc: ROC-AUC score.

    Returns:
        Matplotlib Figure.
    """
    try:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color=PALETTE["primary"], lw=2,
                label=f"{model_name} (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], color=PALETTE["gray"], lw=1.5,
                linestyle="--", label="Random chance (AUC = 0.50)")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {model_name}", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error in plot_roc_curve: {e}")
        raise


def plot_roc_comparison(y_test, lr_proba, rf_proba,
                        lr_auc: float, rf_auc: float) -> plt.Figure:
    """Overlaid ROC curves for Logistic Regression vs Random Forest Classifier.

    Args:
        y_test: True binary labels.
        lr_proba: Logistic Regression predicted probabilities.
        rf_proba: Random Forest predicted probabilities.
        lr_auc: Logistic Regression AUC score.
        rf_auc: Random Forest AUC score.

    Returns:
        Matplotlib Figure.
    """
    try:
        fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
        fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr_lr, tpr_lr, color=PALETTE["orange"], lw=2,
                label=f"Logistic Regression (AUC = {lr_auc:.4f})")
        ax.plot(fpr_rf, tpr_rf, color=PALETTE["primary"], lw=2,
                label=f"Random Forest (AUC = {rf_auc:.4f})")
        ax.plot([0, 1], [0, 1], color=PALETTE["gray"], lw=1.5,
                linestyle="--", label="Random chance (AUC = 0.50)")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — Logistic Regression vs Random Forest",
                     fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error in plot_roc_comparison: {e}")
        raise


def plot_classification_feature_importance(model, feature_names: list,
                                           model_name: str, top_n: int = 15) -> plt.Figure:
    """Horizontal bar chart of Random Forest Classifier feature importances.

    Args:
        model: Fitted RandomForestClassifier.
        feature_names: List of feature column names.
        model_name: Label for title.
        top_n: Number of top features to display.

    Returns:
        Matplotlib Figure.
    """
    try:
        imp = pd.Series(model.feature_importances_, index=feature_names)
        top = imp.sort_values(ascending=False).head(top_n).sort_values()
        colors = [PALETTE["primary"] if i >= len(top) - 5 else PALETTE["light"]
                  for i in range(len(top))]
        fig, ax = plt.subplots(figsize=(10, 6))
        top.plot(kind="barh", color=colors[::-1], edgecolor="white", ax=ax)
        ax.set_title(f"Top {top_n} Feature Importances — {model_name}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance Score (Mean Decrease in Impurity)")
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error in plot_classification_feature_importance: {e}")
        raise


def plot_correlation_with_approval(df: pd.DataFrame, q3: float) -> plt.Figure:
    """Bar chart of feature correlations with the APPROVAL target.

    Args:
        df: Encoded numeric DataFrame with SCORE column.
        q3: Q3 threshold for defining APPROVAL.

    Returns:
        Matplotlib Figure.
    """
    try:
        df = df.copy()
        df["APPROVAL"] = (df["SCORE"] >= q3).astype(int)
        corr = df.corr(numeric_only=True)["APPROVAL"].drop(["APPROVAL", "SCORE"],
                                                             errors="ignore")
        corr = corr.sort_values()
        colors = [PALETTE["secondary"] if v > 0 else PALETTE["red"] for v in corr.values]
        fig, ax = plt.subplots(figsize=(9, max(5, len(corr) * 0.22)))
        corr.plot(kind="barh", color=colors, edgecolor="white", ax=ax)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title("Feature Correlation with Approval Target",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Correlation Coefficient")
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error in plot_correlation_with_approval: {e}")
        raise
