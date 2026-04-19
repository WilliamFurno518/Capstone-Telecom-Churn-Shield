"""
Thin wrapper around XGBoost for binary misalignment classification.

Kept separate from the orchestrator so we can swap the backend (LightGBM,
CatBoost, ...) without touching the rest of the code.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


class XGBoostModelService:
    """Fit / predict / probas, with a stable interface for the engine."""

    def __init__(self) -> None:
        self.model: XGBClassifier | None = None

    def train(self, x_train, y_train, x_test, y_test):
        """Fit on the training fold, return (accuracy, text_report)."""
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        self.model.fit(x_train, y_train)

        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            labels=[0, 1],
            target_names=["Aligned", "Misaligned"],
            zero_division=0,
        )
        return accuracy, report

    def predict(self, x):
        if self.model is None:
            raise RuntimeError("Model is not initialised. Train it first.")
        return self.model.predict(x)

    def predict_proba(self, x):
        if self.model is None:
            raise RuntimeError("Model is not initialised. Train it first.")
        return self.model.predict_proba(x)

    @property
    def classes_(self):
        if self.model is None:
            return np.array([])
        return self.model.classes_

    @property
    def feature_importances_(self):
        if self.model is None:
            return np.array([])
        return self.model.feature_importances_
