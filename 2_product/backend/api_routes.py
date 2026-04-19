"""
ChurnShield - Flask routes.

Endpoints
---------

  GET  /api/health         service heartbeat
  GET  /api/features       canonical feature schema (for the mapping UI)
  POST /api/upload         upload a CSV / XLSX, get column summary + auto-mapping
  POST /api/mapping        set or update the feature mapping (manual)
  POST /api/mapping/auto   re-run auto-mapping on the current dataset
  GET  /api/plan-values    unique values in the current plan column
  POST /api/analyze        fit + predict, return misaligned customers + insights
  GET  /api/insights       return the insights payload alone
  GET  /api/export         download the latest results as an .xlsx
"""

from __future__ import annotations

import os
import traceback

from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.utils import secure_filename

from engine import ChurnShieldEngine
from features import CORE_FEATURES

api_bp = Blueprint("api", __name__)

# One engine per session id. Keeps state (uploaded df, trained model,
# last results) between calls without hitting a database.
_engines: dict[str, ChurnShieldEngine] = {}


def _get_engine(session_id: str = "default") -> ChurnShieldEngine:
    if session_id not in _engines:
        _engines[session_id] = ChurnShieldEngine()
    return _engines[session_id]


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ChurnShield API"})


@api_bp.route("/features", methods=["GET"])
def get_features():
    return jsonify({"features": CORE_FEATURES})


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@api_bp.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        engine = _get_engine()
        df = engine.load_data(filepath)

        # Cache the dataframe on the engine so subsequent calls don't need it.
        engine._current_filepath = filepath
        engine._current_df = df

        summary = engine.get_column_summary(df)
        suggestions = engine.auto_suggest_mapping(df)
        stats = engine.set_feature_mapping(suggestions)

        # Pre-compute the list of plans for the frontend checkboxes.
        suggested_plan_col = suggestions.get("current_plan")
        plan_values = engine.get_plan_values(df, suggested_plan_col) if suggested_plan_col else []

        return jsonify({
            "success": True,
            "filename": filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "column_summary": summary,
            "preview": df.head(5).to_dict(orient="records"),
            "auto_mapping": suggestions,
            "auto_mapping_applied": True,
            "auto_mapping_stats": stats,
            "suggested_plan_column": suggested_plan_col,
            "plan_values": plan_values,
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

@api_bp.route("/mapping", methods=["POST"])
def set_mapping():
    data = request.json or {}
    mapping = data.get("mapping", {})
    engine = _get_engine()
    result = engine.set_feature_mapping(mapping)

    # Always refresh the plan values when the plan column changes.
    plan_values = []
    if hasattr(engine, "_current_df") and mapping.get("current_plan"):
        plan_values = engine.get_plan_values(engine._current_df, mapping["current_plan"])

    return jsonify({
        "success": True,
        "mode": "manual",
        "mapping": mapping,
        "mapped": result["mapped"],
        "total": result["total"],
        "confidence": result["confidence"],
        "plan_values": plan_values,
    })


@api_bp.route("/mapping/auto", methods=["POST"])
def set_auto_mapping():
    engine = _get_engine()
    if not hasattr(engine, "_current_df") or engine._current_df is None:
        return jsonify({"error": "No dataset loaded. Upload a file first."}), 400

    df = engine._current_df
    mapping = engine.auto_suggest_mapping(df)
    result = engine.set_feature_mapping(mapping)
    plan_values = engine.get_plan_values(df, mapping.get("current_plan")) if mapping.get("current_plan") else []

    return jsonify({
        "success": True,
        "mode": "auto",
        "mapping": mapping,
        "mapped": result["mapped"],
        "total": result["total"],
        "confidence": result["confidence"],
        "plan_values": plan_values,
    })


# ---------------------------------------------------------------------------
# Plan values (for checkboxes on the analyze step)
# ---------------------------------------------------------------------------

@api_bp.route("/plan-values", methods=["GET"])
def get_plan_values():
    engine = _get_engine()
    if not hasattr(engine, "_current_df") or engine._current_df is None:
        return jsonify({"error": "No dataset loaded. Upload a file first."}), 400
    plan_col = request.args.get("column") or engine.feature_mapping.get("current_plan")
    if not plan_col:
        return jsonify({"plan_values": [], "plan_column": None})
    values = engine.get_plan_values(engine._current_df, plan_col)
    return jsonify({"plan_values": values, "plan_column": plan_col})


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

@api_bp.route("/analyze", methods=["POST"])
def analyze():
    data = request.json or {}
    target_plans = data.get("target_plans") or None
    min_confidence = int(data.get("min_confidence", 70))
    max_results = int(data.get("max_results", 500))
    upsell_only = bool(data.get("upsell_only", False))

    engine = _get_engine()
    if not hasattr(engine, "_current_df") or engine._current_df is None:
        return jsonify({"error": "No dataset loaded. Upload a file first."}), 400

    try:
        df = engine._current_df
        payload = engine.analyze(
            df,
            target_plans=target_plans,
            min_confidence=min_confidence,
            max_results=max_results,
            upsell_only=upsell_only,
        )
        results = payload["results"]
        train_stats = payload["train_stats"]

        total_customers = len(df)
        flagged = len(results)
        misaligned_rate = round(flagged / total_customers * 100, 1) if total_customers else 0.0
        avg_conf = round(results["Confidence (%)"].mean(), 1) if flagged else 0.0
        high = int((results["Priority"] == "High").sum()) if flagged else 0
        medium = int((results["Priority"] == "Medium").sum()) if flagged else 0
        low = int((results["Priority"] == "Low").sum()) if flagged else 0

        engine._results = results
        insights = engine.generate_insights(df)

        return jsonify({
            "success": True,
            "metrics": {
                "total_customers": total_customers,
                "flagged": flagged,
                "misaligned_rate": misaligned_rate,
                "avg_confidence": avg_conf,
                "high_priority": high,
                "medium_priority": medium,
                "low_priority": low,
                "model_accuracy": train_stats["accuracy"],
            },
            "results": results.to_dict(orient="records"),
            "available_plans": payload["available_plans"],
            "insights": insights,
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------

@api_bp.route("/insights", methods=["GET"])
def get_insights():
    engine = _get_engine()
    if not hasattr(engine, "_current_df") or engine._current_df is None:
        return jsonify({"error": "No dataset loaded. Upload a file first."}), 400
    try:
        top_n = request.args.get("top_n", default=8, type=int)
        top_n = max(3, min(top_n, 20))
        insights = engine.generate_insights(engine._current_df, top_n=top_n)
        return jsonify({"success": True, "insights": insights})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@api_bp.route("/export", methods=["GET"])
def export_results():
    engine = _get_engine()
    if not hasattr(engine, "_results") or engine._results is None:
        return jsonify({"error": "No results to export. Run analysis first."}), 400

    output_path = os.path.join(current_app.config["UPLOAD_FOLDER"], "churnshield_results.xlsx")
    engine.export_results(engine._results, output_path)
    return send_file(
        output_path,
        as_attachment=True,
        download_name="churnshield_results.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
