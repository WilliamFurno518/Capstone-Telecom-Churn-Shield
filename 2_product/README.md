# Phase 2 — The Product

> Upload any subscriber file. Get a ranked list of customers who are on the wrong plan.

This folder is the self-contained SaaS side of ChurnShield: a Flask API,
a static React frontend (no build step), and two demo fixtures. The
product does not reuse the Phase 1 model — it retrains itself on every
uploaded file so it can adapt to any plan catalog in any language.

---

## The three bricks

The entire product is orchestrated by a thin `engine.py` that delegates
to three independent services.

### Brick 1 — Insights Engine (`insights_service.py`)

Automated EDA in JSON. Given the uploaded dataframe and the feature
mapping, it produces:

- **Overview** — row/column counts, missing rate, completeness
- **Quality** — null counts, cardinality, constant columns
- **Distributions** — mean/median/std/p05/p95 + IQR outlier rate per numeric
- **Correlations** — top correlated pairs (Pearson)
- **Multicollinearity** — pairs with |corr| ≥ 0.85, flagged as PCA candidates
- **Segments** — per-plan aggregates (size, avg usage, avg cost, …)
- **Anomalies** — top-N z-score outliers across usage features
- **Statistical tests** — chi² for every categorical vs the plan column, ANOVA F for every numeric vs the plan column
- **Feature importance** — Random Forest ranking on label-encoded columns
- **Strategy cards** — plain-English recommendations derived from the stats

The same statistical tools we used during Phase 1 research, now packaged
as a dashboard payload.

### Brick 2 — Misalignment Engine (`misalignment_engine.py`)

The heart of the product, a three-step pipeline:

```
  Step A   IsolationForest fitted per plan
           -> flag outliers within each plan's cohort
  Step B   XGBoost refines those pseudo-labels on the full feature matrix
           (usage + per-plan z-scores + cost_efficiency + complaint_rate)
           -> calibrated confidence scores
  Step C   For each flagged customer, recommend the plan in the uploaded
           catalog whose mean usage profile minimises z-score distance
           -> data-relative recommendations, no external reference
```

Why this and not a multi-class classifier on `current_plan` (like Phase 1 did)?

Because a classifier trained to predict the current plan is trained to
reproduce the status quo, which is the opposite of what a plan-misalignment
product needs to do. IsolationForest within each plan targets the "this
person doesn't look like the rest of their cohort" signal directly, then
XGBoost smooths and calibrates it.

### Brick 3 — Mapping Service (`mapping_service.py`)

Translates the arbitrary column names of an uploaded file into the
canonical feature schema the rest of the pipeline understands.

- **LLM first** (if `CHURNSHIELD_LLM_API_KEY` is set). Any OpenAI-compatible
  endpoint works — GPT-4o-mini, Gemini, Claude, a local Ollama model.
  The service POSTs a structured prompt with column metadata and asks for
  a JSON mapping.
- **Heuristic fallback**. Keyword match + value-profile scoring per feature.
  Robust enough on its own that the product works with zero configuration.

The user only has to confirm one thing: which column holds the customer's
plan. Everything else is inferred.

---

## Architecture

```
                          ┌──────────────────┐
                          │    index.html    │  static React (no build)
                          │  (frontend/)     │  drop file -> pick plan col
                          └────────┬─────────┘  -> tune params -> results
                                   │  JSON over HTTP
                          ┌────────▼─────────┐
                          │     app.py       │  loads .env, registers routes
                          │   (Flask API)    │
                          └────────┬─────────┘
                                   │
                          ┌────────▼─────────┐
                          │  api_routes.py   │  /upload /mapping /analyze
                          └────────┬─────────┘  /insights /export /plan-values
                                   │
                          ┌────────▼─────────┐
                          │    engine.py     │  thin orchestrator
                          └─┬───────┬──────┬─┘
                            │       │      │
               ┌────────────▼──┐ ┌──▼───┐ ┌▼─────────────────┐
               │ Mapping       │ │Ins.  │ │ Misalignment     │
               │ Service       │ │Serv. │ │ Engine           │
               │ (LLM + rules) │ │      │ │ (IForest + XGB)  │
               └───────────────┘ └──────┘ └──────────────────┘
```

---

## API reference

All endpoints live under `/api` (the blueprint prefix) and return JSON
unless noted otherwise.

| Method  | Path                | Description                                           |
|---------|---------------------|-------------------------------------------------------|
| `GET`   | `/api/health`       | Service heartbeat                                     |
| `GET`   | `/api/features`     | Canonical feature schema (for the mapping UI)         |
| `POST`  | `/api/upload`       | Upload CSV / XLSX, get auto-mapping + plan values     |
| `POST`  | `/api/mapping`      | Set / update the feature mapping manually             |
| `POST`  | `/api/mapping/auto` | Re-run auto-mapping on the current dataset            |
| `GET`   | `/api/plan-values`  | Unique values of the plan column                      |
| `POST`  | `/api/analyze`      | Fit + predict, returns misaligned customers + insights |
| `GET`   | `/api/insights`     | Return only the insights payload                      |
| `GET`   | `/api/export`       | Download the latest results as `.xlsx`                |

### `/api/analyze` payload

```json
{
  "target_plans": ["Premium", "Unlimited"],   // optional, null = no filter
  "min_confidence": 70,                        // 0-100
  "max_results": 500,                          // hard cap on rows returned
  "upsell_only": true                          // drop recommendations where
                                               // recommended plan is cheaper
                                               // than current plan on average
}
```

---

## Running the product

From the repo root, after installing `requirements.txt`:

```bash
# Optionally enable LLM auto-mapping
cp ../.env.example ../.env
# -> fill in CHURNSHIELD_LLM_API_KEY

# Start the API
cd 2_product/backend
python app.py
```

Then open `2_product/frontend/index.html` directly in a browser. It talks
to the API on `localhost:5000` — CORS is wide open for local development.

### First-run demo (takes ~20 seconds end to end)

1. Drag-drop `2_product/data/sample_telecom_demo.csv` onto the upload zone.
2. The plan column dropdown auto-fills with `current_plan`. Click **Continue**.
3. Review the auto-mapping on the Mapping step. Click **Validate mapping**.
4. On Configure:
   - All four plans are pre-checked. Uncheck `Basic` if you want to focus on upmarket.
   - Flip **Upsell focus only** on.
   - Leave confidence at 70% and max results at 500.
5. Click **Run analysis**. After a few seconds you get a ranked table of
   misaligned customers with their recommended plan, a priority badge, and
   a confidence score.
6. Click **Export to Excel** to download the results.

### Demo expectations

On the shipped telecom fixture (5 000 rows, ~15 % misaligned by design):

| Metric                  | Expected value           |
|-------------------------|--------------------------|
| Model accuracy          | ≈ 98 % (hold-out fold)   |
| Misaligned flagged      | 400–500                  |
| With `upsell_only: true`| ~330 (downsells filtered)|

---

## Frontend notes

- **No build step.** React is loaded from a CDN as UMD, Babel compiles JSX
  at runtime. The tradeoff is a slower first load; the benefit is that a
  grader can open the file with no tooling at all.
- **Four steps, one state atom.** `App` owns everything — `fileData`,
  `mapping`, `config`, `results`. Each step component gets exactly the
  props it needs. No router, no context, no Redux.
- **CORS.** Wide open via `flask-cors`. Fine for local demo, should be
  tightened for any real deployment.

---

## Known limitations

- **Cold start on every upload.** There is no persistent model. Every
  request to `/analyze` retrains the pipeline on the uploaded dataset.
  That's a feature, not a bug — it keeps the product data-agnostic — but
  it also means a 500 MB upload would take a few minutes.
- **In-memory session state.** `_engines` in `api_routes.py` is a dict
  keyed by session id. Multi-user production would need Redis or similar.
- **LLM mapping is best-effort.** If the model returns malformed JSON,
  we silently fall back to the heuristic. There is no retry / repair.
- **Single server process.** Flask's dev server is fine for the demo.
  For a real deployment, wrap with gunicorn behind a reverse proxy.
