"""
ChurnShield - Misalignment Engine
=================================

The heart of the product. Given a dataframe with a plan column and a
handful of usage features, it answers two questions:

  1. Which customers are on the wrong plan?
  2. For each of them, which plan *in the same catalog* would fit them best?

Pipeline
--------

  Step A - Unsupervised per-plan anomaly detection
      For each plan present in the data, fit an IsolationForest on the
      usage features of its subscribers. A customer is a "misalignment
      candidate" if the per-plan IsolationForest flags them as an outlier
      within their own plan.

  Step B - Supervised refinement with XGBoost
      Take the IsolationForest candidates as positive pseudo-labels,
      everyone else as negatives, and train a binary XGBoost on the full
      feature matrix (usage + plan-relative z-scores + cost efficiency +
      complaint rate). This smooths the noisy per-plan signal and gives
      us calibrated confidence scores.

  Step C - Data-relative plan recommendation
      Build a profile of the mean usage vector for every plan in the
      dataset. For each flagged customer, recommend the plan whose
      profile minimises the standardised distance to them. No external
      knowledge, no hardcoded thresholds - everything is relative to the
      catalog the user uploaded.

Why not just train a multi-class classifier on `current_plan`?
--------------------------------------------------------------

That's what Phase 1 did. But in a product context, `current_plan` is by
definition what the customer *is* on, not what they *should* be on. A
classifier trained to predict the current plan is therefore trained to
reproduce the status quo, which is the opposite of what we want. Using
anomalies within each plan as pseudo-labels targets the "this person
doesn't look like the rest of their cohort" signal directly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from model_service import XGBoostModelService

# Features the engine always tries to use when they are mapped.
USAGE_FEATURES = ["usage_primary", "usage_secondary", "usage_tertiary"]
ALL_NUMERIC_FEATURES = USAGE_FEATURES + ["monthly_cost", "complaints", "tenure", "extra_usage"]


class MisalignmentEngine:
    """Per-plan anomaly detection + supervised refinement + reco."""

    def __init__(self, anomaly_rate: float = 0.15, random_state: int = 42) -> None:
        self.anomaly_rate = anomaly_rate
        self.random_state = random_state

        self.model_service = XGBoostModelService()
        self.label_encoder = LabelEncoder()

        self.feature_mapping: dict = {}
        self.feature_columns: list[str] = []
        self.plan_profiles: pd.DataFrame | None = None  # index=plan, cols=mean/std per feature
        self.available_plans: list[str] = []
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    def _prepare_features(self, df: pd.DataFrame, fit_encoder: bool) -> pd.DataFrame:
        """
        Build the model-ready feature matrix from the mapped columns.

        - Numeric features are coerced and NaN-filled.
        - The plan column is label-encoded (refit only at training time).
        - Per-plan z-scores and a few ratios are added on top so the
          supervised model sees both absolute and relative signals.
        """
        mapping = self.feature_mapping
        features = pd.DataFrame(index=df.index)

        # Numeric features
        for feat in ALL_NUMERIC_FEATURES:
            col = mapping.get(feat)
            if col and col in df.columns:
                features[feat] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                features[feat] = 0.0

        # Plan column: encode + compute per-plan z-scores and ratios
        plan_col = mapping.get("current_plan")
        if plan_col and plan_col in df.columns:
            plans = df[plan_col].astype(str)
            if fit_encoder:
                self.available_plans = sorted(plans.unique().tolist())
                self.label_encoder.fit(plans)

            # Fallback for unseen plans at predict time: clamp to most frequent known.
            known = set(self.label_encoder.classes_.tolist())
            safe_plans = plans.where(plans.isin(known), other=next(iter(known), "Unknown"))
            features["plan_encoded"] = self.label_encoder.transform(safe_plans)

            # Per-plan mean / std, aligned back to each row
            combo = features[USAGE_FEATURES].copy()
            combo["__plan__"] = plans.values
            plan_means = combo.groupby("__plan__")[USAGE_FEATURES].transform("mean")
            plan_stds = combo.groupby("__plan__")[USAGE_FEATURES].transform("std").replace(0, 1).fillna(1)

            for feat in USAGE_FEATURES:
                features[f"{feat}_zscore"] = (features[feat] - plan_means[feat]) / plan_stds[feat]

            features["cost_efficiency"] = features["usage_primary"] / features["monthly_cost"].replace(0, 1)
            features["complaint_rate"] = features["complaints"] / features["tenure"].replace(0, 1)

        # Sanitise
        features = features.replace([np.inf, -np.inf], 0).fillna(0)

        if fit_encoder:
            self.feature_columns = list(features.columns)
        return features

    # ------------------------------------------------------------------
    # Step A - unsupervised per-plan labels
    # ------------------------------------------------------------------

    def _generate_anomaly_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        For each plan present in the data, fit an IsolationForest on the
        usage vectors of its subscribers. Flag the outliers as misaligned.

        Guarantees at least one positive and one negative so downstream
        training never blows up on a degenerate fold.
        """
        plan_col = self.feature_mapping.get("current_plan")
        labels = pd.Series(0, index=df.index, dtype=int)

        # Feature subset IsolationForest looks at
        usage_cols = [c for c in USAGE_FEATURES if c in features.columns]
        if not usage_cols:
            usage_cols = [c for c in ALL_NUMERIC_FEATURES if c in features.columns]

        if plan_col and plan_col in df.columns:
            plans = df[plan_col].astype(str)
            for plan, idx in plans.groupby(plans).groups.items():
                idx = list(idx)
                if len(idx) < 20:
                    # Not enough data to model a plan - skip it safely.
                    continue
                x = features.loc[idx, usage_cols].values
                iso = IsolationForest(
                    n_estimators=150,
                    contamination=self.anomaly_rate,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                preds = iso.fit_predict(x)  # -1 = outlier
                labels.loc[idx] = (preds == -1).astype(int)
        else:
            # No plan column -> fall back to a global IsolationForest.
            x = features[usage_cols].values
            iso = IsolationForest(
                n_estimators=200,
                contamination=self.anomaly_rate,
                random_state=self.random_state,
                n_jobs=-1,
            )
            preds = iso.fit_predict(x)
            labels[:] = (preds == -1).astype(int)

        # Guarantee both classes are present.
        unique = set(labels.unique().tolist())
        n = len(labels)
        if unique == {0}:
            k = max(1, int(round(n * self.anomaly_rate)))
            labels.iloc[:k] = 1
        elif unique == {1}:
            labels.iloc[: max(1, n - 1)] = 0

        return labels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_feature_mapping(self, mapping: dict) -> None:
        self.feature_mapping = mapping

    def fit(self, df: pd.DataFrame) -> dict:
        """Run step A (pseudo-labels) then step B (XGBoost refinement)."""
        features = self._prepare_features(df, fit_encoder=True)
        labels = self._generate_anomaly_labels(df, features)

        # Build the per-plan profile once, for the reco step C.
        plan_col = self.feature_mapping.get("current_plan")
        if plan_col and plan_col in df.columns:
            usage_block = features[[c for c in USAGE_FEATURES if c in features.columns]].copy()
            usage_block["__plan__"] = df[plan_col].astype(str).values
            agg = usage_block.groupby("__plan__").agg(["mean", "std"])
            self.plan_profiles = agg

        x = features.values
        y = labels.values

        can_stratify = pd.Series(y).value_counts().min() >= 2
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y if can_stratify else None,
        )
        accuracy, report = self.model_service.train(x_train, y_train, x_test, y_test)

        self.is_trained = True
        return {
            "accuracy": round(float(accuracy) * 100, 1),
            "misaligned_count": int(labels.sum()),
            "report": report,
        }

    # ------------------------------------------------------------------
    # Step C - data-relative recommendation
    # ------------------------------------------------------------------

    def _plan_mean_cost(self, df: pd.DataFrame, plan: str) -> float:
        """Average monthly cost observed in the dataset for a given plan."""
        cost_col = self.feature_mapping.get("monthly_cost")
        plan_col = self.feature_mapping.get("current_plan")
        if not (cost_col and plan_col and cost_col in df.columns and plan_col in df.columns):
            return 0.0
        subset = df[df[plan_col].astype(str) == plan]
        if subset.empty:
            return 0.0
        return float(pd.to_numeric(subset[cost_col], errors="coerce").fillna(0).mean())

    def _recommend_plan(self, row_features: pd.Series, current_plan: str) -> str:
        """
        Pick the plan from the dataset catalog whose mean usage profile
        minimises the z-score distance to the customer.
        """
        if self.plan_profiles is None or not self.available_plans:
            return current_plan

        usage_cols = [c for c in USAGE_FEATURES if c in row_features.index]
        if not usage_cols:
            return current_plan

        best_plan = current_plan
        best_distance = float("inf")

        for plan in self.available_plans:
            if plan not in self.plan_profiles.index:
                continue
            distance = 0.0
            for feat in usage_cols:
                plan_mean = self.plan_profiles.loc[plan, (feat, "mean")]
                plan_std = self.plan_profiles.loc[plan, (feat, "std")]
                if pd.isna(plan_std) or plan_std == 0:
                    plan_std = 1.0
                distance += abs(float(row_features[feat]) - float(plan_mean)) / float(plan_std)
            if distance < best_distance:
                best_distance = distance
                best_plan = plan

        return best_plan

    def predict(
        self,
        df: pd.DataFrame,
        target_plans: Optional[list[str]] = None,
        min_confidence: int = 70,
        max_results: int = 500,
        upsell_only: bool = False,
    ) -> pd.DataFrame:
        """
        Score every row and return a ranked dataframe of misaligned
        customers with their recommended plan.

        Parameters
        ----------
        target_plans
            If set, only keep recommendations towards plans in this list.
        min_confidence
            Confidence floor (0-100). Rows under the threshold are dropped.
        max_results
            Hard cap on the number of rows returned.
        upsell_only
            If True, drop recommendations whose mean plan cost is lower
            than the customer's current plan cost (i.e. downsells).
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        features = self._prepare_features(df, fit_encoder=False)
        x = features.values

        predictions = self.model_service.predict(x)
        probas_raw = self.model_service.predict_proba(x)
        classes = list(self.model_service.classes_)

        # Pick the "misaligned" probability column robustly.
        if probas_raw.ndim == 1:
            probas = probas_raw
        elif probas_raw.shape[1] == 1:
            only_class = classes[0] if classes else 0
            probas = np.ones(len(df)) if only_class == 1 else np.zeros(len(df))
        else:
            try:
                mis_idx = classes.index(1)
            except ValueError:
                mis_idx = probas_raw.shape[1] - 1
            probas = probas_raw[:, mis_idx]

        plan_col = self.feature_mapping.get("current_plan")
        id_col = self.feature_mapping.get("customer_id")
        phone_col = self.feature_mapping.get("phone")

        # Pre-compute mean plan costs once (used for upsell filter).
        mean_cost_by_plan = {p: self._plan_mean_cost(df, p) for p in self.available_plans}

        rows = []
        for i in range(len(df)):
            if predictions[i] != 1:
                continue
            confidence = round(float(probas[i]) * 100, 1)
            if confidence < min_confidence:
                continue

            current_plan = str(df.iloc[i].get(plan_col, "Unknown")) if plan_col else "Unknown"
            customer_id = str(df.iloc[i].get(id_col, f"ROW-{i}")) if id_col else f"ROW-{i}"
            phone = str(df.iloc[i].get(phone_col, "N/A")) if phone_col else "N/A"

            reco = self._recommend_plan(features.iloc[i], current_plan)
            if reco == current_plan:
                continue
            if target_plans and reco not in target_plans:
                continue
            if upsell_only:
                if mean_cost_by_plan.get(reco, 0.0) <= mean_cost_by_plan.get(current_plan, 0.0):
                    continue

            if confidence >= 85:
                priority = "High"
            elif confidence >= 70:
                priority = "Medium"
            else:
                priority = "Low"

            rows.append({
                "Customer ID": customer_id,
                "Phone": phone,
                "Current Plan": current_plan,
                "Recommended Plan": reco,
                "Confidence (%)": confidence,
                "Priority": priority,
            })

        results = pd.DataFrame(rows)
        if not results.empty:
            results = (
                results.sort_values("Confidence (%)", ascending=False)
                .head(max_results)
                .reset_index(drop=True)
            )
        return results
