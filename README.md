# ChurnShield

**A self-service SaaS platform that surfaces customers who are on the wrong plan.**

ChurnShield turns a raw subscriber file into a ranked list of actionable upsell
and downsell opportunities, without the user having to train a model, write
SQL, or even know which column is which.

---

## The story in one paragraph

We started from a real telecom dataset — 100 000 rows of messy Chinese subscriber
records — and spent Phase 1 answering a research question: *can we predict which
package a customer belongs to, given their usage pattern?* After cleaning,
EDA, PCA, and a Random Forest vs. XGBoost benchmark, XGBoost won on macro-F1.
That validated the core idea: usage patterns carry a strong signal about
plan-fit. But a locked-in research model, hard-coded to 11 specific Chinese
plan codes, is not a product. Phase 2 takes the same idea and flips it into a
**data-agnostic product**: upload any subscriber file, in any language, with any
plan catalog, and ChurnShield retrains itself on your data and hands you back a
list of misaligned customers you can call tomorrow morning.

---

## Repository layout

```
ChurnShield/
├── 1_research/                    Phase 1 — the R&D story
│   ├── telecom_analysis.py        Cleaned, reproducible research script
│   ├── README.md                  What we did, what we found, why it matters
│   └── results/                   Generated plots and Excel outputs (gitignored)
│
├── 2_product/                     Phase 2 — the SaaS product
│   ├── backend/                   Flask API + ML engine (3 bricks)
│   │   ├── app.py                 Entrypoint
│   │   ├── api_routes.py          REST endpoints
│   │   ├── engine.py              Thin orchestrator
│   │   ├── insights_service.py    Brick 1 — automated EDA
│   │   ├── misalignment_engine.py Brick 2 — IsolationForest + XGBoost
│   │   ├── mapping_service.py     Brick 3 — LLM + heuristic auto-mapping
│   │   ├── model_service.py       XGBoost wrapper
│   │   └── features.py            Canonical feature schema
│   ├── frontend/                  Static React SPA (served as-is)
│   │   ├── index.html
│   │   ├── app.js
│   │   └── style.css
│   ├── data/                      Demo fixtures
│   │   ├── sample_telecom_demo.csv
│   │   ├── sample_banking_demo.csv
│   │   └── generate_demo_samples.py
│   └── README.md                  How the product is built + how to run it
│
├── requirements.txt               One file for both phases
├── .env.example                   LLM config template
├── .gitignore
└── README.md                      You are here
```

---

## Architecture of the product

```
            ┌───────────────────────────────────────────────┐
            │                    FRONTEND                   │
            │         upload  →  confirm  →  analyze        │
            └───────────────────────┬───────────────────────┘
                                    │  JSON over HTTP
            ┌───────────────────────▼───────────────────────┐
            │                 Flask API (app.py)            │
            └───────────────────────┬───────────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │  engine.py        │   (orchestrator only)
                          └─────────┬─────────┘
                  ┌─────────────────┼─────────────────┐
                  ▼                 ▼                 ▼
        ┌──────────────────┐ ┌────────────────┐ ┌──────────────────┐
        │ InsightsService  │ │  Misalignment  │ │ MappingService   │
        │ (automated EDA)  │ │     Engine     │ │ (LLM + heuristic)│
        │                  │ │                │ │                  │
        │ chi², ANOVA, RF, │ │ IsolationForest│ │ Column → feature │
        │ multicolinearity │ │  → XGBoost     │ │  auto-detection  │
        │                  │ │ z-score reco   │ │                  │
        └──────────────────┘ └────────────────┘ └──────────────────┘
```

Three things make this different from a Kaggle notebook:

1. **No pre-trained model.** The engine fits itself on every uploaded file.
   Your plan catalog can be `{Basic, Premium}` or `{卡包A, 卡包B, 卡包C}` — same
   code path.
2. **Data-relative recommendations.** When we flag a misaligned customer, the
   recommended plan is the plan from *your* catalog whose average usage
   profile minimises the z-score distance to that customer. No external
   reference, no hard-coded thresholds.
3. **Invisible mapping.** The user uploads a file, a LLM silently guesses
   which column is the plan / usage / tenure, the user confirms one dropdown
   (the plan column) and moves on. Heuristic fallback if no API key.

---

## Running the project

### Prerequisites

- Python 3.10 or newer
- ~500 MB free disk space (Phase 1 dataset is optional)

### Install

```bash
git clone <repo-url> ChurnShield
cd ChurnShield
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # edit to add your LLM key (optional)
```

### Run the product (Phase 2)

```bash
cd 2_product/backend
python app.py
```

Then open `2_product/frontend/index.html` in a browser — it talks to the API
on `localhost:5000`.

First-time checklist:
1. Drag-drop `2_product/data/sample_telecom_demo.csv`.
2. Confirm the plan column in the dropdown (should be pre-filled).
3. Hit **Analyze** — you should see misaligned customers within a few seconds.

### Run the research script (Phase 1, optional)

Phase 1 uses the original 100 k-row telecom dataset, which is not shipped in
this repo (Chinese source, licensing). If you have your own `final_train.csv`:

```bash
cd 1_research
python telecom_analysis.py --input path/to/final_train.csv --step all
```

See `1_research/README.md` for per-step flags.

---

## Tech stack

| Layer             | Choice                              | Why                                           |
|-------------------|-------------------------------------|-----------------------------------------------|
| Language          | Python 3.10+                        | Standard for ML, good library ecosystem       |
| ML                | scikit-learn + XGBoost              | XGBoost won the Phase 1 benchmark             |
| Anomaly detection | IsolationForest (per plan)          | Unsupervised, no labels needed on upload      |
| API               | Flask + flask-cors                  | Minimal, fits a single-machine SaaS demo      |
| Frontend          | React (CDN UMD) + plain HTML/CSS    | No build step, grader can open `index.html`   |
| LLM mapping       | OpenAI-compatible chat completions  | Works with GPT-4o-mini, Gemini, Claude, local |

---

## Limitations and ethical notes

- **Small plan catalogs hurt.** If the uploaded dataset has < 2 plans, the
  per-plan anomaly logic degenerates to a single global model.
- **Synthetic fixtures for the demo.** The two CSVs under `2_product/data/`
  are synthetic. The research signal in Phase 1 came from a real dataset, but
  we don't redistribute it.
- **LLM mapping is a convenience, not a contract.** The user confirms the plan
  column manually. Everything else has a heuristic fallback.
- **No carbon-intensive training.** The product fits small models (< 5 s on
  10 k rows). No GPU, no persistent artifacts.

---

## Team

SKEMA Business School — MSc AIBT — Capstone 2025/2026.
