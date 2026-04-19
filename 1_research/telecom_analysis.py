"""
ChurnShield - Phase 1 Research Script
=====================================

Reproducible research pipeline on the raw telecom subscriber dataset
(`final_train.csv`, ~100k rows, 20+ features, 11 plan codes).

This file is the cleaned-up, production-friendly version of the original
notebook-like script we wrote during the research phase. It keeps the same
analytical logic but fixes three things that made the original unusable on
a third-party machine:

  1. Hard-coded macOS paths -> replaced by CLI arguments
  2. `main()` was a graveyard of commented-out calls -> replaced by a
     step-based dispatcher (`--step 1` ... `--step all`)
  3. Non-ASCII font configuration and Chinese-specific assumptions -> kept
     but made opt-in

The pipeline has seven ordered steps:

    1. load_clean       Load the CSV, fix `\\N` sentinels, cast dtypes, clip 1/99 quantiles
    2. categorical_eda  Bar charts + chi-square test for every categorical vs target
    3. numeric_eda      KDE plots + ANOVA F-test for every numeric vs target
    4. correlation      Pearson correlation heatmap on the numeric block
    5. feature_importance  Random Forest top-N feature importance ranking
    6. pca_fees         PCA on the four monthly-fee columns -> single `total_fee`
    7. model            XGBoost benchmark with GridSearchCV and confusion matrix

Usage
-----

    python telecom_analysis.py --input /path/to/final_train.csv --step all
    python telecom_analysis.py --input ./final_train.csv --step 5 --top-n 20
    python telecom_analysis.py --input ./final_train.csv --step model --results-dir ./results

All generated artifacts (plots, Excel summaries, reduced CSV) land in the
`--results-dir` folder (default: `./results`). Nothing is written outside it.
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, f_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# A font fallback that works across OS. Original used SimHei for Chinese labels.
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# Canonical column lists - derived from the original `final_train.csv` schema
# ---------------------------------------------------------------------------

TARGET_COL = "current_service"

CATEGORICAL_COLS = [
    "service_type",
    "is_mix_service",
    "many_over_bill",
    "contract_type",
    "is_promise_low_consume",
    "net_service",
    "gender",
    "complaint_level",
]

NUMERIC_COLS = [
    "1_total_fee",
    "2_total_fee",
    "3_total_fee",
    "4_total_fee",
    "month_traffic",
    "pay_times",
    "pay_num",
    "last_month_traffic",
    "local_trafffic_month",
    "local_caller_time",
    "service1_caller_time",
    "former_complaint_num",
    "former_complaint_fee",
    "online_time",
    "contract_time",
    "age",
]

FEE_COLS = ["1_total_fee", "2_total_fee", "3_total_fee", "4_total_fee"]


# ---------------------------------------------------------------------------
# Step 1 - load and clean
# ---------------------------------------------------------------------------

def load_clean(input_path: str) -> pd.DataFrame:
    """
    Load the raw CSV and apply the cleaning rules we settled on in research.

    The dataset uses the sentinel ``\\N`` for missing values (common in
    MySQL dumps). We replace it with ``NaN``, drop the unused
    `service2_caller_time` column, cast numeric columns to float, fill
    missing numerics with 0, and clip to the 1st/99th quantile to neutralise
    extreme outliers without losing the tail shape.
    """
    print(f"[Phase 1] Loading dataset from {input_path}")
    df = pd.read_csv(input_path)
    print(f"[Phase 1] Raw shape: {df.shape}")

    df = df.replace("\\N", np.nan)
    if "service2_caller_time" in df.columns:
        df = df.drop(columns=["service2_caller_time"])

    present_numeric = [c for c in NUMERIC_COLS if c in df.columns]
    df[present_numeric] = df[present_numeric].apply(pd.to_numeric, errors="coerce")
    df[present_numeric] = df[present_numeric].fillna(0)

    for col in present_numeric:
        low = df[col].quantile(0.01)
        high = df[col].quantile(0.99)
        df[col] = df[col].clip(low, high)

    print(f"[Phase 1] Clean shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Step 2 - categorical EDA
# ---------------------------------------------------------------------------

def _save_fig(results_dir: str, name: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, name), bbox_inches="tight", dpi=110)
    plt.close()


def categorical_eda(df: pd.DataFrame, results_dir: str) -> None:
    """
    For each categorical feature, produce a grouped bar chart of its
    distribution per target class and run a chi-square test of independence
    against the target. The chi-square p-value is written into the figure
    title so we can eyeball the most discriminative variables quickly.
    """
    print("[Phase 1] Categorical EDA + chi-square test")
    for feature in CATEGORICAL_COLS:
        if feature not in df.columns:
            continue

        sub = df[[feature, TARGET_COL]].dropna()
        if sub.empty:
            continue

        x = pd.get_dummies(sub[[feature]])
        y = sub[TARGET_COL]
        chi, pval = chi2(x, y)
        min_p = float(np.min(pval)) if len(pval) else np.nan

        cross = pd.crosstab(sub[feature], sub[TARGET_COL], normalize="index")
        cross.plot(kind="bar", stacked=False, figsize=(9, 5))
        plt.title(f"{feature} vs {TARGET_COL}  (min p-value = {min_p:.3g})")
        plt.ylabel("Rate")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        sns.despine()
        plt.tight_layout()
        _save_fig(results_dir, f"cat_{feature}.png")

    print(f"[Phase 1] Saved categorical plots in {results_dir}/")


# ---------------------------------------------------------------------------
# Step 3 - numeric EDA
# ---------------------------------------------------------------------------

def numeric_eda(df: pd.DataFrame, results_dir: str) -> None:
    """
    For each numeric feature, plot the KDE distribution split by target
    class, both on the original scale and on log(1+x) for the fat-tailed
    features. Runs an ANOVA F-test against the target.
    """
    print("[Phase 1] Numeric EDA + ANOVA F-test")
    for feature in NUMERIC_COLS:
        if feature not in df.columns:
            continue

        sub = df[[feature, TARGET_COL]].dropna()
        if sub.empty:
            continue

        f_val, p_val = f_classif(sub[[feature]], sub[TARGET_COL])
        p = float(p_val[0]) if len(p_val) else np.nan

        sub_log = sub.copy()
        sub_log["transformed"] = np.log1p(sub[feature])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle(f"{feature} vs {TARGET_COL}  (p = {p:.3g})")
        sns.kdeplot(
            data=sub, x=feature, hue=TARGET_COL,
            fill=True, common_norm=False, alpha=0.4, ax=ax1,
        )
        ax1.set_title("Original scale")
        sns.kdeplot(
            data=sub_log, x="transformed", hue=TARGET_COL,
            fill=True, common_norm=False, alpha=0.4, ax=ax2,
        )
        ax2.set_title("log(1 + x)")
        plt.tight_layout()
        _save_fig(results_dir, f"num_{feature}.png")

    print(f"[Phase 1] Saved numeric plots in {results_dir}/")


# ---------------------------------------------------------------------------
# Step 4 - correlation heatmap
# ---------------------------------------------------------------------------

def correlation_heatmap(df: pd.DataFrame, results_dir: str) -> None:
    """
    Pearson correlation heatmap restricted to the numeric columns that
    actually exist in the dataframe. Useful to spot multicollinearity
    (see fee columns -> PCA step).
    """
    print("[Phase 1] Correlation heatmap")
    present = [c for c in NUMERIC_COLS if c in df.columns]
    corr = df[present].corr()

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.4,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature correlation matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig(results_dir, "correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Step 5 - Random Forest feature importance
# ---------------------------------------------------------------------------

def feature_importance(df: pd.DataFrame, results_dir: str, top_n: int = 15) -> pd.DataFrame:
    """
    Train a Random Forest on the full feature set (label-encoded) and plot
    the Top-N feature importances. Returns the full ranked dataframe.
    """
    print(f"[Phase 1] Random Forest feature importance (top {top_n})")

    drop_cols = [c for c in [TARGET_COL, "user_id"] if c in df.columns]
    x = df.drop(columns=drop_cols).copy()
    y = df[TARGET_COL]

    for col in x.columns:
        if not pd.api.types.is_numeric_dtype(x[col]):
            x[col] = LabelEncoder().fit_transform(x[col].astype(str))

    model = RandomForestClassifier(n_estimators=100, random_state=1234, n_jobs=-1)
    model.fit(x, y)

    importance = (
        pd.DataFrame({"feature": x.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    top = importance.head(top_n)
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(top)))
    plt.barh(top["feature"], top["importance"], color=colors, edgecolor="black", linewidth=0.4)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} feature importance (Random Forest)")
    plt.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    _save_fig(results_dir, "feature_importance.png")

    importance.to_csv(os.path.join(results_dir, "feature_importance.csv"), index=False)
    return importance


# ---------------------------------------------------------------------------
# Step 6 - PCA on the fee columns
# ---------------------------------------------------------------------------

def pca_fees(df: pd.DataFrame, results_dir: str) -> pd.DataFrame:
    """
    The four `*_total_fee` columns are strongly collinear (see heatmap).
    We collapse them into a single principal component `total_fee` and
    save a reduced copy of the dataframe for downstream modelling.
    """
    print("[Phase 1] PCA on fee columns")
    present = [c for c in FEE_COLS if c in df.columns]
    if len(present) < 2:
        print("[Phase 1] Not enough fee columns present - skipping PCA")
        return df

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[present])
    pca = PCA(n_components=1)
    reduced = pca.fit_transform(scaled)

    var_explained = float(pca.explained_variance_ratio_[0])
    print(f"[Phase 1] PCA single component explains {var_explained:.1%} of variance")

    out = df.drop(columns=present).copy()
    out["total_fee"] = reduced.flatten()

    os.makedirs(results_dir, exist_ok=True)
    out.to_csv(os.path.join(results_dir, "train_reduced.csv"), index=False)
    return out


# ---------------------------------------------------------------------------
# Step 7 - XGBoost benchmark with grid search
# ---------------------------------------------------------------------------

def train_xgboost(df: pd.DataFrame, results_dir: str) -> dict:
    """
    XGBoost benchmark with a small grid search. We keep the grid
    deliberately tight so the script finishes in a few minutes on a
    laptop; widening it is a one-liner.

    Returns a dict with best params, macro-F1 on the hold-out set, and
    the per-class classification report as a string.
    """
    print("[Phase 1] XGBoost benchmark with GridSearchCV")

    x = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL]

    for col in ["online_time", "contract_time", "age"]:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0)

    target_encoder = LabelEncoder()
    y_enc = target_encoder.fit_transform(y)

    for col in x.columns:
        if not pd.api.types.is_numeric_dtype(x[col]):
            x[col] = LabelEncoder().fit_transform(x[col].astype(str))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_enc, test_size=0.3, random_state=1234, stratify=y_enc,
    )

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train,
    )
    weight_dict = dict(zip(np.unique(y_train), class_weights))
    sample_weights = np.array([weight_dict[i] for i in y_train])

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 0.6],
        "colsample_bytree": [0.8, 1.0],
    }
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_enc)),
        random_state=1234,
        eval_metric="mlogloss",
        n_jobs=-1,
        tree_method="hist",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    grid = GridSearchCV(
        estimator=xgb, param_grid=param_grid, cv=cv,
        scoring="f1_macro", n_jobs=-1, verbose=1,
    )
    grid.fit(x_train, y_train, sample_weight=sample_weights)

    best = grid.best_estimator_
    y_pred = best.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n[Phase 1] Best params: {grid.best_params_}")
    print(f"[Phase 1] Test accuracy: {acc:.4f}")
    print(f"[Phase 1] Test macro-F1: {f1m:.4f}")
    print(report)

    plt.figure(figsize=(9, 7))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion matrix (XGBoost)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    _save_fig(results_dir, "confusion_matrix.png")

    return {
        "best_params": grid.best_params_,
        "accuracy": acc,
        "f1_macro": f1m,
        "report": report,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_steps(steps: Iterable[str], input_path: str, results_dir: str, top_n: int) -> None:
    """Dispatch one or more research steps in order."""
    df = load_clean(input_path) if any(s in steps for s in ["1", "all", "load_clean"]) else None
    if df is None:
        # Still load for downstream steps that all need it
        df = load_clean(input_path)

    step_map = {
        "2": "categorical_eda",
        "3": "numeric_eda",
        "4": "correlation",
        "5": "feature_importance",
        "6": "pca_fees",
        "7": "model",
    }
    aliases = {"all": list(step_map.values())}

    # Normalise steps to their canonical names
    normalized = []
    for s in steps:
        if s == "all":
            normalized.extend(aliases["all"])
        elif s in step_map:
            normalized.append(step_map[s])
        else:
            normalized.append(s)

    if "categorical_eda" in normalized:
        categorical_eda(df, results_dir)
    if "numeric_eda" in normalized:
        numeric_eda(df, results_dir)
    if "correlation" in normalized:
        correlation_heatmap(df, results_dir)
    if "feature_importance" in normalized:
        feature_importance(df, results_dir, top_n=top_n)
    if "pca_fees" in normalized:
        df = pca_fees(df, results_dir)
    if "model" in normalized:
        train_xgboost(df, results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChurnShield Phase 1 - telecom research pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the cleaned training CSV (e.g. final_train.csv)",
    )
    parser.add_argument(
        "--results-dir", default="./results",
        help="Where to write plots, Excel and reduced CSV (default: ./results)",
    )
    parser.add_argument(
        "--step", default="all", nargs="+",
        help=(
            "Which step(s) to run. Accepts numbers (1-7), names "
            "(load_clean/categorical_eda/numeric_eda/correlation/"
            "feature_importance/pca_fees/model) or 'all'."
        ),
    )
    parser.add_argument(
        "--top-n", type=int, default=15,
        help="Top-N features for the Random Forest importance plot",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    run_steps(args.step, args.input, args.results_dir, args.top_n)
    print("[Phase 1] Done.")


if __name__ == "__main__":
    main()
