# Final Implementation Report

This records the initial upgrade. See [PORTFOLIO_REVIEW.md](PORTFOLIO_REVIEW.md) for the subsequent presentation polish, final checks and exact demo sequence.

## Outcome

The original Streamlit project was refactored into a nine-page portfolio application with reproducible model selection, one saved preprocessing/model pipeline, documented threshold selection, batch scoring, explainability, tests and deployment configuration. Existing analysis and legacy artifacts were preserved.

The previous audit chat was not available. Findings were reconstructed from the actual code and notebook and recorded in [AUDIT.md](AUDIT.md).

## Files created

| File | Purpose |
|---|---|
| `.gitignore` | Exclude environments, caches, logs, secrets and IDE files |
| `.streamlit/config.toml` | Professional light theme, upload limit and server settings |
| `.github/workflows/tests.yml` | Python 3.14 Linux dependency installation and regression checks on push/PR |
| `requirements-notebook.txt` | Optional notebook plotting and Jupyter dependencies |
| `src/__init__.py` | Importable shared package |
| `src/config.py` | Portable paths, field schema, categorical values and meaningful labels |
| `src/data_processing.py` | Dataset loading, CSV parsing and common input validation |
| `src/feature_engineering.py` | Train-fitted outlier caps and prior-contact indicator |
| `src/train.py` | Pipeline, modest tuning, selection, threshold analysis and persistence |
| `src/evaluate.py` | Actual classification metrics, ROC/PR curves and confusion matrix |
| `src/predict.py` | Saved artifact loading, inference and individual sensitivity |
| `src/visualization.py` | Shared Plotly styling, rate charts and dynamic descriptive findings |
| `models/pipeline.joblib` | Complete current fitted pipeline |
| `models/evaluation.json` | Metrics, selection, parameters, thresholds, curves, importance and provenance |
| `models/test_predictions.csv` | Recomputable held-out prediction evidence |
| `models/example_customers.csv` | Five-row valid input example |
| `tests/test_pipeline.py` | Shared validation, inference and artifact regression checks |
| `tests/test_app.py` | Streamlit page, EDA, manual prediction and batch workflow checks |
| `docs/AUDIT.md` | Evidence-based original findings and resolutions |
| `docs/IMPLEMENTATION_REPORT.md` | This report, run instructions, limits and demo/checklist |
| `docs/screenshots/01-dashboard.png` | Real application dashboard capture |
| `docs/screenshots/02-performance.png` | Real model-comparison and evaluation-chart capture |
| `docs/screenshots/03-prediction.png` | Real manual prediction and sensitivity capture |

## Files modified

| File | Changes |
|---|---|
| `app.py` | Expanded existing Streamlit interface into nine clear pages, shared helpers, validated scoring, downloads, model evidence and readable error states |
| `train_model.py` | Converted to a small explicit training entry point that calls the shared workflow |
| `requirements.txt` | Pinned the eight existing runtime dependencies to versions actually used for training/testing |
| `README.md` | Replaced brief app notes with project story, measured results, methodology, structure, installation, usage, limitations and deployment instructions |

No existing user files were deleted. Generated caches and the pre-existing venv remain local and are ignored by Git.

## Features added

- Executive KPIs calculated from the actual dataset and current evaluation artifact.
- Organized, filterable EDA with useful existing chart concepts retained and misleading rate displays corrected.
- Computed business findings with group sizes and explicit association/causation distinction.
- A complete 15-field customer form with meaningful labels and shared validation.
- Saved-threshold class prediction, probability display, progress visualization and CSV download.
- Individual one-feature replacement sensitivity, clearly distinguished from SHAP.
- Batch CSV parsing, schema validation, bounded size, scoring preview and full results download.
- Actual model comparisons, baseline, six metrics, confusion matrix, ROC and PR curves.
- Global held-out permutation importance with variability over five shuffles.
- Model card, feature dictionary, tuning details, package versions and dataset fingerprint.
- Downloadable cohort data, analysis summary, comparison table, example input and predictions.
- Lightweight event logging without logging customer input values.
- Tests and a GitHub Actions configuration.

## Existing features preserved

`Bank_Marketing_Full_Project.ipynb`, `bank.csv`, `bank_clean.csv`, `model.pkl`, `columns.pkl`, `metrics.pkl` and `feature_importances.csv` are unchanged. The notebook retains the full original EDA, visualizations, preprocessing, model fitting and narrative.

The application retains occupation conversion, prior-outcome analysis, age and balance distributions, occupation/marital heatmap, monthly analysis, correlation, customer prediction, model comparison and importance concepts. The new model pipeline keeps the original prior-contact feature and balance/campaign capping intent, with leakage corrected. Logistic Regression, Random Forest and XGBoost are all retained. No correct existing tuning was removed; no hyperparameter search was found in the original code.

## Model pipeline and selection

1. Read and validate the raw dataset; exclude duration and target from all predictors.
2. Make an 80/20 stratified split with seed 42: 8,929 training and 2,233 held-out rows.
3. Fit capping limits inside each CV training fold, derive prior contact, scale numeric fields and one-hot encode categories.
4. Run three-fold GridSearchCV with 13 total candidate configurations across baseline and the three original algorithms. Compare average precision; assess weighting without oversampling.
5. Select XGBoost by training CV average precision (0.7889). Its selected depth is 3, estimators 300, learning rate 0.05 and scale_pos_weight 1.0.
6. Generate training out-of-fold predictions. Choose threshold 0.375 by maximum F1 on the documented grid. This is exploratory and not nested or cost-optimized.
7. Evaluate the frozen choices on the holdout; compute curves and permutation importance.
8. Save the complete pipeline, matching run metadata, held-out predictions and example CSV. The app loads once per process through resource caching and never fits on startup or upload.

| Held-out metric at threshold 0.375 | Result |
|---|---:|
| Accuracy | 0.7165 |
| Precision | 0.6763 |
| Recall | 0.7703 |
| F1 | 0.7203 |
| ROC-AUC | 0.7950 |
| Average precision | 0.7955 |

At the common 0.5 threshold, XGBoost accuracy is 0.7447 and F1 is 0.6952. The app clearly labels that comparison separately from the selected operating threshold. No metrics are fabricated or taken from stale hardcoded tables.

## Application pages

1. **Executive Dashboard** — dataset size, conversion, balance and ranking metric; occupation and previous-outcome rates.
2. **Exploratory Analysis** — occupation/marital/age filters and six analysis focuses, including financial, demographic, historical and campaign views.
3. **Business Insights** — computed best-performing observed groups, denominators, interpretation and summary download.
4. **Customer Prediction** — validated manual input, saved model probability/class, explicit threshold, sensitivity analysis and download.
5. **Batch Prediction** — CSV example/schema, readable validation errors, full-pipeline scoring and result export.
6. **Model Performance** — comparison, held-out metrics, confusion matrix, ROC/PR curves and training threshold analysis.
7. **Model Explainability** — permutation importance in original feature space, variability and interpretation limits.
8. **Model Information** — problem, features, methods, hyperparameters, versions and model limitations.
9. **About Project** — objective, data attribution, technology, workflow and demonstrated skills.

## Exact commands

For the existing environment from the project root:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

For a fresh clone using Python 3.14:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Optional retraining (restart the application afterward):

```powershell
.\.venv\Scripts\python.exe train_model.py
```

Local address: http://localhost:8501. No public deployment has been performed.

## Testing results

Thirteen automated unittest/AppTest checks passed in the supplied Windows Python 3.14.3 environment:

- All nine pages load without Streamlit exceptions.
- All six EDA focuses load; empty cohorts produce a readable state.
- Manual prediction returns a probability, class and download control.
- Batch app path accepts the example and rejects a missing-column file (uploader mocked in AppTest).
- Full batch inference equals row-by-row inference and does not call fit.
- Saved held-out predictions match the persisted model; all six metrics recompute correctly.
- Selection follows CV results rather than test ranking.
- Inference does not modify learned capping limits.
- Invalid categories, blanks, non-numbers, infinity, out-of-range values and fractional integer fields are rejected.
- Empty, malformed, invalid-encoding, duplicate-header, missing-column and oversized CSV input is rejected.
- Result CSV serialization round-trips with the expected rows and output columns.
- Duration and target extra columns never enter scoring.
- Missing model file produces an actionable error.
- Individual sensitivity returns finite effects for all 15 features.

The local Streamlit server health endpoint returned `ok` and browser rendering was inspected. A real browser CSV upload scored five rows; the Download scored CSV button produced a file whose five rows and probability fields were verified. A manual browser submission returned a 60.1% score for the default hypothetical profile. `pip check` reported no broken requirements and `git diff --check` passed. Screenshots show actual application state. Upstream joblib/NumPy deprecation and AppTest bare-context warnings appeared without test failures. Clean-install Linux CI is configured but has not been executed in this local task.

## Remaining limitations and manual work

- Verify dataset provenance and redistribution rights; select an appropriate code license before publication.
- The original notebook already used a similar held-out split, so the current holdout is not a fresh external benchmark.
- Historical sample composition and uncalibrated scores limit deployment claims. Fresh temporal validation and calibration are future work.
- Historical campaign counts may represent later information than a planned early call; real scoring needs a defined as-of-time contract.
- Threshold maximizes training F1, not campaign profit/capacity. No business-cost assumptions were fabricated.
- One-feature sensitivity is not SHAP, not additive and not causal. Correlated-feature importance needs careful interpretation.
- Authentication, monitoring, prospective evaluation and fairness analysis are outside this portfolio implementation.
- Public hosting and fresh Linux dependency installation remain to be validated through deployment/CI.
- Notebook-specific optional dependencies are provided, but the preserved notebook was not rerun or rewritten.

## LinkedIn demo guide

Capture these seven views using hypothetical inputs:

1. **Executive Dashboard:** KPI cards, occupation rates and previous-outcome chart. Explain the business objective.
2. **Exploratory Analysis → Customer demographics:** show the age chart and occupation/marital heatmap; mention cohort filtering.
3. **Business Insights:** show one group rate with its denominator and association caveat.
4. **Customer Prediction form:** show all input groups and that duration is excluded.
5. **Prediction result:** show the score, explicit 0.375 threshold and individual sensitivity; call it a model estimate.
6. **Model Performance:** show the model comparison and ROC/precision–recall curves; explain CV-based selection.
7. **Model Explainability:** show permutation importance and explain that it describes model reliance.

For a recording, optionally finish with the example batch CSV, scored preview and download. Do not claim that 47.4% is a bank-wide conversion rate or that model associations establish causality.

## Final GitHub checklist

- [ ] Review `git diff` and the exact staged file list.
- [ ] Confirm dataset redistribution rights and credit the source accurately.
- [ ] Choose a license for your own code; do not imply it overrides dataset rights.
- [ ] Include the matching `models/` pipeline, JSON, held-out evidence and example CSV.
- [ ] Keep original notebook/data/legacy outputs for provenance, clearly labeled as historical.
- [ ] Verify no venv, cache, credentials, private customer uploads or machine-specific paths are staged.
- [ ] Run `python -m unittest discover -s tests -v` in a clean pinned environment.
- [ ] Let the GitHub Actions Linux checks finish successfully.
- [ ] Confirm README screenshots and artifact links render.
- [ ] Deploy `app.py` with matching Python/dependency versions and smoke-test each page.
- [ ] Add the verified public URL to README/LinkedIn only after deployment works.
