"""
Automatic column -> canonical feature mapping.

Strategy:
  1. Ask an OpenAI-compatible LLM to return a JSON mapping.
  2. Whatever the LLM didn't cover, fill with a deterministic heuristic
     that scores each (feature, column) pair on name match + value profile.

The LLM call is optional. If `CHURNSHIELD_ENABLE_LLM_MAPPING` is false or
no API key is available, we go straight to the heuristic path.
"""

from __future__ import annotations

import json
import os
import re
from urllib import request
from urllib.error import HTTPError, URLError

import pandas as pd


class MappingService:
    """LLM-assisted column mapping with heuristic backup."""

    def __init__(self, core_features: dict) -> None:
        self.core_features = core_features

        # Keyword catalog used by the heuristic. Kept deliberately small
        # and obvious - the heuristic is a safety net, not the main path.
        self.keyword_map = {
            "customer_id":     ["customer id", "customerid", "client id", "userid", "user id",
                                "subscriber id", "account id", "id"],
            "phone":           ["phone", "mobile", "msisdn", "tel", "telephone",
                                "contact number", "contact"],
            "current_plan":    ["current plan", "plan", "package", "product",
                                "subscription", "tariff", "offer"],
            "monthly_cost":    ["monthly bill", "monthly fee", "bill", "fee",
                                "cost", "charge", "price", "amount", "arpu"],
            "usage_primary":   ["data usage", "data", "usage", "balance",
                                "consumption", "volume", "gb"],
            "usage_secondary": ["call", "calls", "minutes", "transaction",
                                "transactions", "voice"],
            "usage_tertiary":  ["sms", "login", "digital", "app", "session", "activity"],
            "complaints":      ["complaint", "complaints", "ticket", "tickets",
                                "support", "issue", "incident"],
            "tenure":          ["tenure", "age", "duration", "account age", "lifetime"],
            "extra_usage":     ["intl", "international", "roaming", "addon",
                                "extra", "transfer"],
        }

    # --------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------

    def auto_suggest_mapping(self, df: pd.DataFrame) -> dict:
        """
        Merge LLM suggestions with heuristic suggestions. LLM wins when
        both disagree; heuristic only fills gaps.
        """
        if df is None or len(df.columns) == 0:
            return {}

        heuristic = self._heuristic_suggest_mapping(df)
        llm = self._llm_suggest_mapping(df)

        suggestions: dict = {}
        used: set = set()

        for feature in self.core_features.keys():
            llm_col = llm.get(feature)
            if llm_col in df.columns and llm_col not in used:
                suggestions[feature] = llm_col
                used.add(llm_col)
                continue

            heuristic_col = heuristic.get(feature)
            if heuristic_col in df.columns and heuristic_col not in used:
                suggestions[feature] = heuristic_col
                used.add(heuristic_col)

        return suggestions

    # --------------------------------------------------------------------
    # LLM path
    # --------------------------------------------------------------------

    def _llm_suggest_mapping(self, df: pd.DataFrame) -> dict:
        """POST a structured prompt to an OpenAI-compatible chat endpoint."""
        enabled = os.getenv("CHURNSHIELD_ENABLE_LLM_MAPPING", "true").lower() in {"1", "true", "yes", "on"}
        api_key = os.getenv("CHURNSHIELD_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

        if not enabled or not api_key:
            return {}

        model = os.getenv("CHURNSHIELD_LLM_MODEL", "gpt-4o-mini")
        base_url = os.getenv("CHURNSHIELD_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        endpoint = f"{base_url}/chat/completions"

        columns_profile = []
        for col in df.columns:
            columns_profile.append({
                "name": str(col),
                "dtype": str(df[col].dtype),
                "sample_values": df[col].dropna().head(3).astype(str).tolist(),
                "unique_count": int(df[col].nunique(dropna=True)),
            })

        system_prompt = (
            "You map raw dataset columns to canonical churn features. "
            "Return a valid JSON object only, with no markdown fences or commentary. "
            "Keys are canonical feature names; values are exact column names from the input."
        )
        user_prompt = {
            "task": "Map dataset columns to canonical features for churn / misalignment modelling.",
            "canonical_features": self.core_features,
            "dataset_columns": columns_profile,
            "constraints": [
                "Do not invent column names.",
                "Use each input column at most once.",
                "If uncertain, omit the key.",
                "Return JSON only, no prose.",
            ],
        }

        payload = {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
        }

        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=18) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            return {}

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return {}

        parsed = self._parse_json_content(content)
        if not isinstance(parsed, dict):
            return {}

        valid_cols = set(df.columns.tolist())
        clean: dict = {}
        used_cols: set = set()
        for feature in self.core_features.keys():
            col = parsed.get(feature)
            if isinstance(col, str) and col in valid_cols and col not in used_cols:
                clean[feature] = col
                used_cols.add(col)
        return clean

    @staticmethod
    def _parse_json_content(content) -> dict:
        if not content:
            return {}
        text = str(content).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Model returned prose around the JSON - extract the object span.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}

    # --------------------------------------------------------------------
    # Heuristic path
    # --------------------------------------------------------------------

    def _heuristic_suggest_mapping(self, df: pd.DataFrame) -> dict:
        """Score every (feature, column) pair and do a greedy assignment."""
        if df is None or len(df.columns) == 0:
            return {}

        def normalize_name(name: str):
            normalized = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()
            tokens = {t for t in normalized.split() if t}
            return normalized, tokens

        def name_score(feature: str, col_name: str) -> float:
            normalized, tokens = normalize_name(col_name)
            score = 0.0
            for kw in self.keyword_map.get(feature, []):
                kw_norm, kw_tokens = normalize_name(kw)
                if normalized == kw_norm:
                    score = max(score, 8.0)
                    continue
                if kw_norm and kw_norm in normalized:
                    score = max(score, 6.0)
                if kw_tokens:
                    overlap = len(tokens & kw_tokens)
                    if overlap > 0:
                        score = max(score, 3.0 + overlap)
            return score

        def profile(series: pd.Series) -> dict:
            clean = series.dropna()
            total = max(len(series), 1)
            unique_ratio = clean.nunique() / max(len(clean), 1)
            non_null_ratio = len(clean) / total
            is_numeric = pd.api.types.is_numeric_dtype(series)

            phone_like_ratio = 0.0
            integer_like_ratio = 0.0
            positive_ratio = 0.0
            median_val = 0.0
            std_val = 0.0

            if len(clean) > 0:
                if is_numeric:
                    num = pd.to_numeric(clean, errors="coerce").dropna()
                    if len(num) > 0:
                        integer_like_ratio = float(((num % 1) == 0).mean())
                        positive_ratio = float((num >= 0).mean())
                        median_val = float(num.median())
                        std_val = float(num.std()) if len(num) > 1 else 0.0
                else:
                    txt = clean.astype(str).str.replace(r"\D", "", regex=True)
                    lengths = txt.str.len()
                    phone_like_ratio = float(((lengths >= 8) & (lengths <= 15)).mean())

            return {
                "is_numeric": is_numeric,
                "unique_ratio": unique_ratio,
                "non_null_ratio": non_null_ratio,
                "phone_like_ratio": phone_like_ratio,
                "integer_like_ratio": integer_like_ratio,
                "positive_ratio": positive_ratio,
                "median": median_val,
                "std": std_val,
                "nunique": int(clean.nunique()),
            }

        def feature_score(feature: str, col_name: str, stats: dict) -> float:
            score = name_score(feature, col_name)
            col_norm = str(col_name).lower()
            is_num = stats["is_numeric"]
            unique_ratio = stats["unique_ratio"]
            nunique = stats["nunique"]

            if feature == "customer_id":
                if unique_ratio >= 0.85:
                    score += 4.0
                if not is_num:
                    score += 0.5

            elif feature == "phone":
                if stats["phone_like_ratio"] >= 0.55:
                    score += 5.0
                if unique_ratio >= 0.6:
                    score += 1.0

            elif feature == "current_plan":
                if not is_num:
                    score += 2.0
                if 2 <= nunique <= 30:
                    score += 2.0
                if unique_ratio < 0.4:
                    score += 1.0

            elif feature == "monthly_cost":
                if is_num:
                    score += 2.0
                if stats["positive_ratio"] >= 0.8:
                    score += 0.8
                if 1 <= stats["median"] <= 5000:
                    score += 1.0

            elif feature in {"usage_primary", "usage_secondary", "usage_tertiary", "extra_usage"}:
                if is_num:
                    score += 2.0
                if stats["std"] > 0:
                    score += 0.8
                if stats["positive_ratio"] >= 0.7:
                    score += 0.6
                # Nudge usage_secondary when the name mentions calls / transactions.
                if feature == "usage_secondary" and any(
                    x in col_norm for x in ["transaction", "call", "minutes", "voice"]
                ):
                    score += 1.3

            elif feature == "complaints":
                if is_num:
                    score += 2.0
                if stats["integer_like_ratio"] >= 0.8:
                    score += 1.0
                if stats["median"] <= 10:
                    score += 0.8

            elif feature == "tenure":
                if is_num:
                    score += 2.0
                if stats["integer_like_ratio"] >= 0.8:
                    score += 1.0
                if 1 <= stats["median"] <= 240:
                    score += 1.2
                # Penalise obvious non-tenure numerics that leak into the match.
                if any(x in col_norm for x in ["transaction", "call", "sms", "login",
                                               "balance", "usage", "fee", "bill"]):
                    score -= 2.2

            score += 0.2 * stats["non_null_ratio"]
            return score

        col_stats = {col: profile(df[col]) for col in df.columns}

        all_candidates = []
        for feature in self.core_features.keys():
            for col in df.columns:
                all_candidates.append((
                    feature_score(feature, col, col_stats[col]),
                    feature,
                    col,
                ))
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        suggestions: dict = {}
        used_columns: set = set()
        for score, feature, col in all_candidates:
            if feature in suggestions or col in used_columns:
                continue
            if score < 3.2:
                continue
            suggestions[feature] = col
            used_columns.add(col)

        # Fill numerical gaps greedily with the leftover numeric columns
        # of highest variance. Identifiers / plan are never auto-filled here.
        numeric_left = [c for c in df.columns
                        if c not in used_columns and col_stats[c]["is_numeric"]]
        numeric_left.sort(key=lambda c: col_stats[c]["std"], reverse=True)
        for feature in ["monthly_cost", "usage_primary", "usage_secondary",
                        "usage_tertiary", "complaints", "tenure", "extra_usage"]:
            if feature not in suggestions and numeric_left:
                suggestions[feature] = numeric_left.pop(0)

        return suggestions
