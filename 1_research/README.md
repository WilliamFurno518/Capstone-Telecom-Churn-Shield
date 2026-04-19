# Phase 1 — Research

> Can we predict which telecom package a customer belongs to, given their usage pattern?

This folder holds the research work that led to ChurnShield the product.
It is intentionally separated from `2_product/`: the code here is
exploratory, tied to one specific dataset, and runs as a one-shot script —
nothing about it is meant to live inside the SaaS.

---

## The dataset

- **Source.** A public Chinese telecom subscriber dataset (`final_train.csv`),
  ~100 000 rows sampled from a larger dump.
- **Columns.** 20+ numeric and categorical features — monthly fees over four
  months, data traffic, call durations, contract type, complaints, gender,
  age, tenure. The target is `current_service`, the subscribed plan code.
- **Target distribution.** 11 plan codes, heavy long tail — top 3 plans
  cover ~66 % of the user base.
- **Why not a toy dataset?** Precisely because it is messy. `\\N` sentinels
  instead of `NaN`, misspelled column names (`local_trafffic_month`), fat
  tails on every numeric column, and severe class imbalance on almost all
  categorical features. That's the point.
- **Not shipped in this repo.** Licensing. If you have your own copy, drop
  it anywhere and pass the path to `--input`.

---

## The pipeline

The script `telecom_analysis.py` runs seven ordered steps. They are the
same steps we explored interactively in the research phase, now wrapped in
a single reproducible CLI.

| # | Step                   | What it does                                           | Output                          |
|---|------------------------|--------------------------------------------------------|---------------------------------|
| 1 | `load_clean`           | Load CSV, replace `\\N`, drop unused columns, clip 1/99 | (in-memory DataFrame)           |
| 2 | `categorical_eda`      | Bar chart + chi-square per categorical feature         | `cat_*.png`                     |
| 3 | `numeric_eda`          | KDE (raw + log) + ANOVA F-test per numeric feature     | `num_*.png`                     |
| 4 | `correlation`          | Pearson heatmap of numeric columns                     | `correlation_heatmap.png`       |
| 5 | `feature_importance`   | Random Forest top-N feature importance                 | `feature_importance.png`, `.csv` |
| 6 | `pca_fees`             | PCA on the four `*_total_fee` columns -> `total_fee`   | `train_reduced.csv`             |
| 7 | `model`                | XGBoost + GridSearchCV, confusion matrix, full report  | `confusion_matrix.png`          |

---

## Key findings

- **Multicollinearity on fees.** The four monthly-fee columns correlate at
  >0.85. PCA collapses them into a single `total_fee` with ~90 % of the
  variance preserved.
- **Chi-square + ANOVA agree.** The most discriminative variables are
  `service_type`, `contract_type`, `total_fee`, `online_time`,
  `month_traffic`, `local_caller_time`, `pay_num`. Features like `gender`
  and `complaint_level` are weak.
- **Random Forest vs. XGBoost.** Both models were tuned with 5-fold
  stratified CV on macro-F1. XGBoost won — better handling of the long
  tail, faster with `tree_method="hist"` on 100k rows.
- **Class imbalance matters.** Without class weights, the model collapses
  onto the top-3 plans and ignores the rest. `compute_class_weight("balanced")`
  on the training fold fixed that.

---

## How to run

```bash
# From the repo root, after installing requirements.txt
cd 1_research
python telecom_analysis.py --input /path/to/final_train.csv --step all
```

Run only a subset:

```bash
python telecom_analysis.py --input ./final_train.csv --step 1 5 7
python telecom_analysis.py --input ./final_train.csv --step feature_importance --top-n 20
```

Outputs land in `./results/` by default. Override with `--results-dir`.

---

## Why this belongs in the repo

Two reasons.

1. **Proof the core idea works.** Phase 2 is built on the assumption that
   usage patterns predict plan-fit. Phase 1 is the empirical evidence for
   that assumption, on real data with real mess.
2. **A place where ML lives honestly.** Chi-square, ANOVA, PCA, RF
   importance, grid-search, confusion matrices — those all belong in a
   research notebook, not in a SaaS UI. The product reuses the same
   statistical techniques (see `2_product/backend/insights_service.py`),
   but packaged for a non-technical user.

---

## Limitations

- **One-shot script.** Not meant to be deployed, not meant to be unit-tested,
  not meant to scale. It's a reproducible research artifact.
- **Chinese-specific assumptions.** The 11 plan codes are specific to the
  source dataset. That's exactly why the product had to throw them away
  and learn from whatever plan catalog the user uploads.
- **No carbon impact reporting.** The grid is small (2 × 2 × 2 × 2 × 2 = 32
  combos, 5-fold CV, `tree_method="hist"`), but we have not instrumented it.
  Phase 2 fits smaller models on smaller files and is effectively free.
