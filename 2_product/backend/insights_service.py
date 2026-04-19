"""
ChurnShield - Insights Service (automated EDA).

Builds a structured JSON payload for the analyst dashboard. Input is
the uploaded dataframe plus the current feature mapping; output is a
bundle of numbers and strings the frontend can render as tables and
cards without any additional computation.

What's in here
--------------

- Overview : rows, columns, missing rate, completeness
- Quality  : missing cells per column, cardinality, constant columns
- Distributions : mean / median / std / p05 / p95 / outlier rate per numeric
- Correlations  : top correlated pairs (Pearson)
- Multicollinearity : pairs with |corr| > 0.85 - flagged for PCA candidate
- Segments      : per-plan aggregates (size, avg usage, ...)
- Anomalies     : top-N z-score outliers across usage features

Statistical tests (Phase-1-derived):

- Chi-square : every categorical column vs the target plan column
- ANOVA F    : every numeric column vs the target plan column
- Random Forest feature importance : top-N on label-encoded columns

Explicitly *not* included:

- Confusion matrix / per-class accuracy reports -> that belongs in the
  research phase, not in a product EDA view. The product reports model
  accuracy once, next to the misalignment results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import LabelEncoder

# Hard caps so the payload never explodes on a fat dataset.
MAX_NUMERIC_FOR_STATS = 30
MAX_CATEGORICAL_FOR_STATS = 20
MULTICOLLINEARITY_THRESHOLD = 0.85


class InsightsService:
    """Produces the full insights payload from a raw dataframe + mapping."""

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            num = float(value)
            if np.isnan(num) or np.isinf(num):
                return default
            return num
        except (TypeError, ValueError):
            return default

    def _iqr_outlier_rate(self, series: pd.Series) -> float:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean) < 5:
            return 0.0
        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return 0.0
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rate = ((clean < lower) | (clean > upper)).mean()
        return round(self._safe_float(rate) * 100, 2)

    # ------------------------------------------------------------------
    # Strategy cards (plain-English recommendations)
    # ------------------------------------------------------------------

    def _build_strategy_cards(self, df: pd.DataFrame, numeric_columns: list[str]) -> list[dict]:
        cards: list[dict] = []
        total_rows = len(df)
        total_cols = len(df.columns)
        total_cells = max(total_rows * total_cols, 1)
        missing_cells = int(df.isnull().sum().sum())
        missing_rate = round(missing_cells / total_cells * 100, 2)

        if missing_rate >= 8:
            cards.append({
                "title": "Launch data quality remediation",
                "priority": "High",
                "insight": f"{missing_rate}% of dataset cells are missing, "
                           "which can bias segmentation and model confidence.",
                "action": "Prioritise mandatory-field controls on ingestion and "
                          "enrich the top missing columns before the next campaign.",
            })

        if total_rows < 500:
            cards.append({
                "title": "Increase sample size before full rollout",
                "priority": "Medium",
                "insight": "Dataset is relatively small for robust strategic segmentation.",
                "action": "Aggregate additional periods or sources to improve stability of recommendations.",
            })

        if numeric_columns:
            outlier_cols = []
            for col in numeric_columns:
                rate = self._iqr_outlier_rate(df[col])
                if rate >= 4:
                    outlier_cols.append((col, rate))
            if outlier_cols:
                top_col, top_rate = sorted(outlier_cols, key=lambda x: x[1], reverse=True)[0]
                cards.append({
                    "title": "Treat extreme-value segments separately",
                    "priority": "Medium",
                    "insight": f"Column '{top_col}' has {top_rate}% statistical outliers.",
                    "action": "Create dedicated retention / upsell plays for extreme users "
                              "instead of one-size-fits-all targeting.",
                })

        if not cards:
            cards.append({
                "title": "Dataset readiness is strong",
                "priority": "Low",
                "insight": "No major data quality blockers detected for initial strategy design.",
                "action": "Move to plan-level experimentation and track uplift by priority segment.",
            })
        return cards[:4]

    # ------------------------------------------------------------------
    # Statistical tests (Phase-1-derived)
    # ------------------------------------------------------------------

    def _chi_square_tests(self, df: pd.DataFrame, target_col: str,
                         categorical_cols: list[str]) -> list[dict]:
        """Chi-square independence test: every categorical vs target."""
        results: list[dict] = []
        if not target_col or target_col not in df.columns:
            return results
        y = df[target_col].astype(str)
        for col in categorical_cols[:MAX_CATEGORICAL_FOR_STATS]:
            if col == target_col:
                continue
            try:
                sub = df[[col, target_col]].dropna()
                if sub.empty or sub[col].nunique() < 2:
                    continue
                x = pd.get_dummies(sub[col].astype(str), drop_first=False)
                chi, pvals = chi2(x, sub[target_col].astype(str))
                min_p = float(np.min(pvals)) if len(pvals) else 1.0
                total_chi = float(np.sum(chi)) if len(chi) else 0.0
                results.append({
                    "feature": col,
                    "chi2": round(total_chi, 3),
                    "min_p_value": round(min_p, 6),
                    "significant": bool(min_p < 0.05),
                })
            except Exception:
                continue
        results.sort(key=lambda x: x["min_p_value"])
        return results

    def _anova_tests(self, df: pd.DataFrame, target_col: str,
                     numeric_cols: list[str]) -> list[dict]:
        """ANOVA F-test: every numeric vs target."""
        results: list[dict] = []
        if not target_col or target_col not in df.columns:
            return results
        y = df[target_col].astype(str)
        for col in numeric_cols[:MAX_NUMERIC_FOR_STATS]:
            if col == target_col:
                continue
            try:
                x = pd.to_numeric(df[col], errors="coerce").fillna(0).values.reshape(-1, 1)
                if np.nanstd(x) == 0:
                    continue
                f_val, p_val = f_classif(x, y)
                results.append({
                    "feature": col,
                    "f_statistic": round(float(f_val[0]), 3),
                    "p_value": round(float(p_val[0]), 6),
                    "significant": bool(p_val[0] < 0.05),
                })
            except Exception:
                continue
        results.sort(key=lambda x: x["p_value"])
        return results

    def _rf_feature_importance(self, df: pd.DataFrame, target_col: str,
                               top_n: int) -> list[dict]:
        """Random Forest feature importance on label-encoded columns."""
        if not target_col or target_col not in df.columns:
            return []
        try:
            y = df[target_col].astype(str)
            if y.nunique() < 2:
                return []
            x = df.drop(columns=[target_col]).copy()
            for col in x.columns:
                if not pd.api.types.is_numeric_dtype(x[col]):
                    x[col] = LabelEncoder().fit_transform(x[col].astype(str))
                else:
                    x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0)

            # Keep runtime predictable on fat files.
            if len(x) > 20000:
                sample = x.sample(20000, random_state=42)
                y_sample = y.loc[sample.index]
            else:
                sample, y_sample = x, y

            model = RandomForestClassifier(
                n_estimators=80,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(sample, y_sample)

            importance = pd.DataFrame({
                "feature": sample.columns,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False).head(top_n)

            return [
                {"feature": row["feature"], "importance": round(float(row["importance"]), 4)}
                for _, row in importance.iterrows()
            ]
        except Exception:
            return []

    @staticmethod
    def _multicollinearity_pairs(corr_df: pd.DataFrame) -> list[dict]:
        """All pairs with |corr| > MULTICOLLINEARITY_THRESHOLD."""
        pairs: list[dict] = []
        cols = list(corr_df.columns)
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                val = corr_df.loc[c1, c2]
                if pd.isna(val):
                    continue
                if abs(float(val)) >= MULTICOLLINEARITY_THRESHOLD:
                    pairs.append({
                        "left": c1,
                        "right": c2,
                        "correlation": round(float(val), 3),
                        "recommendation": "Candidate for PCA or drop.",
                    })
        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return pairs

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def generate_insights(self, df: pd.DataFrame, feature_mapping: dict | None = None,
                          top_n: int = 8) -> dict:
        if df is None or len(df) == 0:
            raise ValueError("Dataset is empty. Upload a file first.")

        feature_mapping = feature_mapping or {}
        target_col = feature_mapping.get("current_plan")

        rows = int(len(df))
        cols = int(len(df.columns))
        duplicate_rows = int(df.duplicated().sum())
        total_cells = max(rows * cols, 1)
        missing_cells = int(df.isnull().sum().sum())
        missing_rate = round(missing_cells / total_cells * 100, 2)

        numeric_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        categorical_columns = [c for c in df.columns if c not in numeric_columns]

        completeness = round(100 - missing_rate, 2)
        uniqueness_mean = round(self._safe_float(df.nunique(dropna=True).mean()), 2)

        # --- Quality -------------------------------------------------
        null_by_column = sorted(
            [
                {
                    "column": col,
                    "missing_count": int(df[col].isnull().sum()),
                    "missing_rate": round(self._safe_float(df[col].isnull().mean()) * 100, 2),
                }
                for col in df.columns
            ],
            key=lambda x: x["missing_count"],
            reverse=True,
        )

        cardinality = sorted(
            [
                {
                    "column": col,
                    "unique_count": int(df[col].nunique(dropna=True)),
                    "unique_rate": round(
                        self._safe_float(df[col].nunique(dropna=True) / max(rows, 1)) * 100, 2,
                    ),
                }
                for col in df.columns
            ],
            key=lambda x: x["unique_count"],
            reverse=True,
        )

        constant_columns = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

        # --- Distributions ------------------------------------------
        distributions = []
        for col in numeric_columns[:20]:
            clean = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(clean) == 0:
                continue
            distributions.append({
                "column": col,
                "mean": round(self._safe_float(clean.mean()), 3),
                "median": round(self._safe_float(clean.median()), 3),
                "std": round(self._safe_float(clean.std()), 3),
                "p05": round(self._safe_float(clean.quantile(0.05)), 3),
                "p95": round(self._safe_float(clean.quantile(0.95)), 3),
                "outlier_rate": self._iqr_outlier_rate(clean),
            })

        # --- Correlations + multicollinearity -----------------------
        correlations: list[dict] = []
        multicollinearity: list[dict] = []
        if len(numeric_columns) >= 2:
            corr_df = df[numeric_columns].corr(numeric_only=True)
            pairs = []
            for i, c1 in enumerate(corr_df.columns):
                for c2 in corr_df.columns[i + 1:]:
                    val = corr_df.loc[c1, c2]
                    if pd.isna(val):
                        continue
                    pairs.append({
                        "left": c1,
                        "right": c2,
                        "correlation": round(self._safe_float(val), 3),
                        "strength": round(abs(self._safe_float(val)), 3),
                    })
            correlations = sorted(pairs, key=lambda x: x["strength"], reverse=True)[: max(top_n, 1)]
            multicollinearity = self._multicollinearity_pairs(corr_df)

        # --- Per-plan segmentation ----------------------------------
        segments = []
        if target_col and target_col in df.columns:
            for plan, group in df.groupby(target_col, dropna=False):
                seg = {
                    "segment": str(plan),
                    "customers": int(len(group)),
                    "share_pct": round(self._safe_float(len(group) / rows) * 100, 2),
                }
                for metric in ["monthly_cost", "usage_primary", "usage_secondary",
                               "complaints", "tenure"]:
                    mapped = feature_mapping.get(metric)
                    if mapped and mapped in group.columns:
                        s = pd.to_numeric(group[mapped], errors="coerce")
                        seg[f"avg_{metric}"] = round(self._safe_float(s.mean()), 2)
                segments.append(seg)
            segments.sort(key=lambda x: x["customers"], reverse=True)

        # --- Anomalies ---------------------------------------------
        anomaly_scores = []
        for key in ["usage_primary", "usage_secondary", "usage_tertiary",
                    "monthly_cost", "complaints"]:
            col = feature_mapping.get(key)
            if col and col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                z = (series - series.mean()) / (series.std() + 1e-6)
                anomaly_scores.append(z.abs())
        anomalies: list[dict] = []
        if anomaly_scores:
            combined = pd.concat(anomaly_scores, axis=1).mean(axis=1)
            id_col = feature_mapping.get("customer_id")
            top_idx = combined.nlargest(min(max(top_n, 1), len(df))).index
            for idx in top_idx:
                row = df.loc[idx]
                anomalies.append({
                    "customer_id": str(row.get(id_col, f"ROW-{idx}")) if id_col else f"ROW-{idx}",
                    "current_plan": str(row.get(target_col, "Unknown")) if target_col else "Unknown",
                    "anomaly_score": round(self._safe_float(combined.loc[idx]), 3),
                })

        # --- Phase-1-derived statistical layer ----------------------
        chi_square = self._chi_square_tests(df, target_col, categorical_columns)
        anova = self._anova_tests(df, target_col, numeric_columns)
        rf_importance = self._rf_feature_importance(df, target_col, top_n=max(top_n, 8))

        strategy_cards = self._build_strategy_cards(df, numeric_columns)

        return {
            "overview": {
                "rows": rows,
                "columns": cols,
                "numeric_columns": len(numeric_columns),
                "categorical_columns": len(categorical_columns),
                "duplicate_rows": duplicate_rows,
                "missing_cells": missing_cells,
                "missing_rate": missing_rate,
                "completeness": completeness,
                "avg_uniqueness": uniqueness_mean,
            },
            "quality": {
                "null_by_column": null_by_column[:20],
                "cardinality": cardinality[:20],
                "constant_columns": constant_columns[:20],
            },
            "distributions": distributions,
            "correlations": correlations,
            "multicollinearity": multicollinearity,
            "segments": segments,
            "anomalies": anomalies,
            "statistical_tests": {
                "chi_square": chi_square,
                "anova": anova,
                "target_column": target_col,
            },
            "feature_importance": rf_importance,
            "strategy_cards": strategy_cards,
        }
