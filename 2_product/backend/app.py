"""
ChurnShield - Flask API server entrypoint.

Loads `.env` if present so the LLM mapping can pick up its API key
without any manual export. Logs the LLM status at startup so you know
at a glance whether auto-mapping will go through the model or fall
back to the heuristic.
"""

from __future__ import annotations

import os
import tempfile

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

from api_routes import api_bp

# Load .env from the repo root (two levels up from this file), then
# fall back to the current working directory for good measure.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT_ENV = os.path.abspath(os.path.join(_HERE, "..", "..", ".env"))
if os.path.exists(_ROOT_ENV):
    load_dotenv(_ROOT_ENV)
else:
    load_dotenv()  # cwd / nearest .env


def _llm_status() -> str:
    enabled = os.getenv("CHURNSHIELD_ENABLE_LLM_MAPPING", "true").lower() in {"1", "true", "yes", "on"}
    key = os.getenv("CHURNSHIELD_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not enabled:
        return "disabled via CHURNSHIELD_ENABLE_LLM_MAPPING"
    if not key:
        return "no API key -> heuristic fallback only"
    model = os.getenv("CHURNSHIELD_LLM_MODEL", "gpt-4o-mini")
    return f"enabled (model={model})"


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    upload_folder = tempfile.mkdtemp(prefix="churnshield_")
    app.config["UPLOAD_FOLDER"] = upload_folder
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

    app.register_blueprint(api_bp, url_prefix="/api")
    return app


app = create_app()


if __name__ == "__main__":
    print("=" * 56)
    print(" ChurnShield API Server")
    print("=" * 56)
    print(f" LLM auto-mapping : {_llm_status()}")
    print(f" Upload folder    : {app.config['UPLOAD_FOLDER']}")
    print(" Listening on     : http://0.0.0.0:5000")
    print("=" * 56)
    app.run(host="0.0.0.0", port=5000, debug=True)
