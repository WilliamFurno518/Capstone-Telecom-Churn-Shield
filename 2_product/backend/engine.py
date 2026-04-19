"""
ChurnShield - Orchestration Engine
==================================

A thin coordinator. All the real work lives in three dedicated bricks:

  * MappingService     -> column auto-detection (LLM + heuristic)
  * InsightsService    -> automated EDA payload (stats, segments, ...)
  * MisalignmentEngine -> IsolationForest + XGBoost + reco

Keeping this file small and dumb on purpose. If you find yourself adding
feature-engineering or labelling logic here, it belongs in one of the
bricks above.
"""

from __future__ import annotations

import os

import pandas as pd

from features import CORE_FEATURES
from insights_service import InsightsService
from mapping_service import MappingService
from misalignment_engine import MisalignmentEngine


class ChurnShieldEngine:
    """Top-level facade used by the API blueprint."""

    def __init__(self) -> None:
        self.mapping_service = MappingService(CORE_FEATURES)
        self.insights_service = InsightsService()
        self.misalignment_engine = MisalignmentEngine()

        self.feature_mapping: dict = {}
        self.confidence_score: float = 0.0

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load a CSV or Excel file into a DataFrame."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(filepath)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use CSV or XLSX.")
        print(f"[ChurnShield] Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    @staticmethod
    def get_column_summary(df: pd.DataFrame) -> list[dict]:
        """Light per-column summary used to render the mapping UI."""
        summary = []
        for col in df.columns:
            info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "sample_values": df[col].dropna().head(3).tolist(),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique()),
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                info["min"] = float(df[col].min())
                info["max"] = float(df[col].max())
                info["mean"] = round(float(df[col].mean()), 2)
            summary.append(info)
        return summary

    @staticmethod
    def get_plan_values(df: pd.DataFrame, plan_col: str) -> list[str]:
        """Unique plan values present in a column (used to populate checkboxes)."""
        if not plan_col or plan_col not in df.columns:
            return []
        values = df[plan_col].dropna().astype(str).unique().tolist()
        return sorted(values)

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    def auto_suggest_mapping(self, df: pd.DataFrame) -> dict:
        return self.mapping_service.auto_suggest_mapping(df)

    def set_feature_mapping(self, mapping: dict) -> dict:
        self.feature_mapping = mapping or {}
        self.misalignment_engine.set_feature_mapping(self.feature_mapping)

        mapped = sum(1 for v in self.feature_mapping.values() if v)
        total = len(CORE_FEATURES)
        self.confidence_score = mapped / total if total else 0.0
        return {
            "mapped": mapped,
            "total": total,
            "confidence": round(self.confidence_score * 100, 1),
        }

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def analyze(
        self,
        df: pd.DataFrame,
        target_plans: list[str] | None = None,
        min_confidence: int = 70,
        max_results: int = 500,
        upsell_only: bool = False,
    ) -> dict:
        """Fit + predict in one shot. Returns the API payload."""
        train_stats = self.misalignment_engine.fit(df)
        results = self.misalignment_engine.predict(
            df,
            target_plans=target_plans,
            min_confidence=min_confidence,
            max_results=max_results,
            upsell_only=upsell_only,
        )
        return {
            "train_stats": train_stats,
            "results": results,
            "available_plans": self.misalignment_engine.available_plans,
        }

    # ------------------------------------------------------------------
    # Insights
    # ------------------------------------------------------------------

    def generate_insights(self, df: pd.DataFrame, top_n: int = 8) -> dict:
        return self.insights_service.generate_insights(df, self.feature_mapping, top_n=top_n)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_results(self, results_df: pd.DataFrame,
                       output_path: str = "churnshield_results.xlsx") -> str:
        """Write the misaligned customers + summary sheet to an .xlsx."""
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name="Misaligned Customers", index=False)
            summary = pd.DataFrame({
                "Metric": [
                    "Total flagged",
                    "High priority",
                    "Medium priority",
                    "Low priority",
                    "Average confidence",
                ],
                "Value": [
                    len(results_df),
                    int((results_df["Priority"] == "High").sum()) if len(results_df) else 0,
                    int((results_df["Priority"] == "Medium").sum()) if len(results_df) else 0,
                    int((results_df["Priority"] == "Low").sum()) if len(results_df) else 0,
                    f"{results_df['Confidence (%)'].mean():.1f}%" if len(results_df) else "N/A",
                ],
            })
            summary.to_excel(writer, sheet_name="Summary", index=False)
        print(f"[ChurnShield] Results exported to {output_path}")
        return output_path


if __name__ == "__main__":
    import json
    import sys

    engine = ChurnShieldEngine()
    filepath = sys.argv[1] if len(sys.argv) > 1 else "../data/sample_telecom_demo.csv"

    print("=" * 60)
    print(" CHURNSHIELD - end-to-end smoke run")
    print("=" * 60)

    df = engine.load_data(filepath)
    mapping = engine.auto_suggest_mapping(df)
    print(f"Auto mapping: {json.dumps(mapping, indent=2)}")
    engine.set_feature_mapping(mapping)

    payload = engine.analyze(df, min_confidence=60, max_results=500)
    print(f"Accuracy: {payload['train_stats']['accuracy']}%")
    print(f"Flagged : {len(payload['results'])}")
    print(payload["results"].head(10).to_string(index=False))

    out = engine.export_results(payload["results"], "../data/churnshield_results.xlsx")
    print(f"Exported to {out}")
