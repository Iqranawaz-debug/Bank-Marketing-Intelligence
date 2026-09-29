# Portfolio review — Bank Marketing Intelligence

Reviewed and polished on 29 September 2026. This is the final presentation review after the initial upgrade; no model was retrained or replaced.

## Final project name

**Bank Marketing Intelligence** — campaign analytics and term-deposit subscription prediction.

## What was improved

The application now leads with its actual purpose, a concise subtitle and four relevant KPIs. Charts use readable field labels. Business insights distinguish finding, evidence and interpretation. The prediction form has three labeled groups, contextual help and a reproducible dataset example. Prediction results use “Estimated model probability,” human-readable outcomes and a percentage threshold. Batch scoring includes positive/negative totals, positive share and an input preview. The performance page prioritizes ranking/F1/recall metrics, highlights the selected model, explains its selection and separates error counts from ROC/PR curves. Technical details remain available on the model card or in expanders.

## Page-by-page review

| Page | Finding and improvement | Value to its audience |
|---|---|---|
| Executive Dashboard | Generic headline and less relevant balance KPI replaced by project purpose, selected model and ROC-AUC; concise skills area added | Recruiter understands the use case quickly; stakeholder sees sample context; data scientist sees properly labeled ranking evidence |
| Exploratory Analysis | Raw field/axis labels improved; retained six coherent analysis focuses rather than adding charts | Recruiter sees interactive analytics; stakeholder reads ordinary terms; data scientist can inspect cohorts and relationships |
| Business Insights | Repetitive disclaimer replaced by Finding / Evidence / Business interpretation structure | Connects descriptive analysis to testable campaign questions without causal claims |
| Customer Prediction | Ungrouped fields and arbitrary defaults replaced by labeled groups, help and the first actual dataset example; result hierarchy strengthened | A stakeholder can complete the form; a recruiter sees an end-to-end workflow; a data scientist can reproduce the score |
| Batch Prediction | Added explicit three-step instructions, validated-input preview, positive/negative totals and positive share | Shows a practical CSV workflow without confusing model outputs with observed conversion |
| Model Performance | Six cramped metric cards and wide comparison simplified into four primary cards, separate accuracy/precision, highlighted compact comparison and full-detail expander | Makes a strong evaluation screenshot while retaining all metrics and CV variability |
| Model Explainability | Raw feature names translated; global behavior separated from individual explanation guidance | Explains model reliance and sensitivity without implying causality or SHAP |
| Model Information | Existing methodology and limitations retained; technical evidence stays in its existing expander | Provides auditability for technical viewers without burdening the landing page |
| About Project | Repository-maintenance wording replaced by the analytical workflow; branding unified | Presents skills and purpose as a portfolio product |

All nine pages were retained. Their tasks are distinct and the flat sidebar fits comfortably on a desktop. Adding another navigation layer would increase clicks without removing duplication. Their conceptual groups remain Analytics (first three), Prediction (next two), Model (next three), and Project (About).

## Files modified in this polish pass

| File | Change |
|---|---|
| `app.py` | Copy, hierarchy, KPI cards, form grouping/help/defaults, result presentation, batch summary/preview, performance layout and explainability labels |
| `src/visualization.py` | Readable category/axis names and consistent chart margins/typography |
| `tests/test_app.py` | Updated label assertion and three regression checks for demo score, displayed metrics and blank upload values |
| `README.md` | Recruiter-first summary, accurate main results, technology stack and Windows/Linux/macOS quick start |
| `.gitignore` | Added local exports, coverage, temporary files and lint caches |
| `docs/IMPLEMENTATION_REPORT.md` | Linked the initial implementation report to this final review |
| `docs/screenshots/01-dashboard.png` | Refreshed current dashboard capture |
| `docs/screenshots/02-performance.png` | Refreshed current performance capture |
| `docs/screenshots/03-prediction.png` | Refreshed current dataset-example prediction capture |

New files: `src/presentation.py` centralizes human-readable labels, input help and the threshold explanation. `docs/PORTFOLIO_REVIEW.md` is this report and capture guide. Additional new captures: `docs/screenshots/04-insights.png`, `05-eda.png`, `06-explainability.png`, and `07-batch.png`. These complete the seven actual page captures; their numbering follows capture order, while the recommended carousel sequence is below.

## ML and technical integrity

Training, feature engineering, preprocessing, model selection and thresholds were not changed. No fitting was performed. The app continues to load the same saved pipeline. Historical notebooks, duplicate historical datasets and legacy model outputs remain deliberately preserved and documented rather than deleted.

The model SHA-256 remains `e3ba7f7bf0992008797da892fad8f3dc66c57892898bf2a237ccc0381935f633`.
The evaluation JSON SHA-256 remains `888c2613683f92f5d6b4185081b04aa94fc8d6d6fb4fc4916228581b2ce28d54`.

| Metric | Verified saved result |
|---|---:|
| ROC-AUC | 0.795020 |
| Average precision | 0.795519 |
| F1 | 0.720283 |
| Recall | 77.0321% |
| Precision | 67.6349% |
| Accuracy | 71.6525% |
| Classification threshold | 37.5% |

ROC-AUC is not accuracy. XGBoost was selected by training cross-validation average precision. The application compares class metrics at a common 0.5 cutoff and separately labels operational metrics at 0.375. Lower thresholds generally increase recall and false-positive workload; no universally optimal business policy is claimed. Scores remain uncalibrated.

## Tests and results

The existing 13 tests were rerun successfully after the presentation changes. **All 16 tests passed** after adding three focused regression checks. The local server was restarted successfully so imported chart helpers also use the new presentation code. Browser verification confirmed the dashboard, EDA, insights, performance cards, global explanation, model card and the 25.2% example prediction. A real five-row CSV upload produced 0 positives / 5 negatives; the browser download was read back and its five scores verified. All seven portfolio views have real local-app screenshots. The tests cover all nine pages, six EDA views, empty filters, model loading, individual/batch parity, saved metrics, no fitting during inference, malformed CSVs, missing fields/values, output round-trips and individual sensitivity.

`pip check` reported no broken requirements. `git diff --check` passed. Source inspection found no Windows/OneDrive paths in application code. No publishable file exceeded 5 MB. Git excludes the existing virtual environment, caches and secrets file. This is a practical repository review, not a guarantee from a dedicated secret scanner.

## Remaining issues

- Confirm original dataset provenance and redistribution rights and choose a license for the project's own code.
- Run the existing Linux CI in a clean environment; a successful local Windows run alone cannot establish host compatibility.
- Scores are uncalibrated; temporal/external validation, fairness analysis, prospective testing and drift monitoring remain future work.
- The original analysis examined a similar split, so current metrics are holdout diagnostics rather than a new external benchmark.
- Contact-time availability of historical campaign counts needs validation before operational use.
- No public deployment was performed. No credentials or cloud architecture were added.

## Deployment readiness: Ready with minor fixes

The app has a portable pathlib-based data/model layout, pinned dependencies, cached model loading, bounded uploads, no retraining at startup, a small model artifact and no required secrets. Entry point: `app.py`.

Before publishing, verify redistribution rights, ensure the matching artifacts and raw dataset are included, and run the Linux CI/fresh-host installation with Python 3.14 and the pinned package versions. Do not use a different Python/dependency environment silently with the saved pickle. If the host lacks matching packages, compatibility work is required before deployment; do not alter the model merely for presentation.

Streamlit documents selecting the Python version in Advanced settings and providing dependencies in requirements.txt: [deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [dependency guidance](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Exactly seven LinkedIn screenshots

Capture at normal desktop width. Keep the sidebar visible, close menus/tooltips and avoid including personal data. The sequence below uses actual dataset/model evidence.

| # | Page and exact state | Inputs and expansion state | Why this screenshot matters |
|---|---|---|---|
| 1 | Executive Dashboard: title, subtitle, customer/sample/model/ROC-AUC cards and occupation/previous-outcome bars | No inputs; start at top | Establishes the business problem, sample scale and selected model immediately |
| 2 | Exploratory Analysis → Customer demographics: age distribution and education rate, then occupation × marital heatmap if the viewport permits | No occupation or marital filters; age 18–95; analysis focus Customer demographics | Demonstrates cohort exploration and readable customer-segment analysis |
| 3 | Business Insights: Customer segments chart with Student finding/evidence/interpretation | No inputs; top section; no expansions | Shows 269/360 = 74.7% with a careful business interpretation, not a causal claim |
| 4 | Customer Prediction: submit the default first historical example, then scroll to Prediction result and individual sensitivity | Age 59; occupation admin.; marital married; education secondary; balance 2343; default no; housing yes; loan no; previous 0; pdays -1; contact unknown; day 5; month may; campaign 1; poutcome unknown. Threshold expander closed | Actual saved-model score 25.1536% (display 25.2%), Unlikely to subscribe, threshold 37.5%; demonstrates a genuine prediction and local explanation |
| 5 | Batch Prediction: upload `models/example_customers.csv`; show summary and scored table | Five supplied rows unchanged; Required input schema and Preview validated input collapsed | Actual result: 5 rows, 0 positive, 5 negative, 0.0% positive share. Shows a usable batch workflow without fabricating successful predictions |
| 6 | Model Performance: top metrics and compact selected-model comparison; scroll separately for ROC/PR if recording | Full comparison and CV variability collapsed; no input | Shows honest metric distinctions, selection evidence and the chosen operating trade-off |
| 7 | Model Explainability: Global model behavior and held-out permutation-importance chart | No inputs; keep variability/causality explanation visible where possible | Shows interpretable model reliance with understandable feature names |

Screenshot 4's sample actually has `deposit=yes` in the historical file, while the model predicts no. It is a real model miss, not a fabricated success. It is useful to explain uncertainty; do not present it as validation of accuracy. Aggregate evaluation is the evidence for performance.

## Recruiter demo flow — approximately 80 seconds

- **0–15 seconds, Home:** “This analyzes historical bank campaigns and estimates subscription before a planned contact. It contains 11,162 records; the sample response rate is not a bank-wide forecast.”
- **15–30 seconds, Insight:** Open Business Insights. Point to the Student group, 269/360 subscriptions, and the distinction between association and a testable campaign hypothesis.
- **30–45 seconds, Prediction:** Open Customer Prediction and submit the prefilled historical example. Show 25.2%, the negative classification and the 37.5% threshold. Explain that probabilities are uncalibrated estimates.
- **45–60 seconds, Explanation:** Show the local sensitivity chart; then global importance if time permits. Explain model reliance rather than causality.
- **60–80 seconds, Performance:** Show XGBoost's CV-based selection, ROC-AUC 0.795, AP 0.796, F1 0.720 and recall 77.0%. Mention that all preprocessing is saved in the pipeline and prediction never retrains.

## GitHub checklist

- [ ] Review the exact staged files and diff; the workspace includes uncommitted work from both implementation phases.
- [ ] Confirm dataset redistribution terms and attribution; add an appropriate code license.
- [ ] Include `bank.csv`, `src/`, `models/`, app.py and requirements.txt.
- [ ] Preserve and clearly identify the original notebook and legacy artifacts as historical.
- [ ] Keep venv, caches, secrets, local exports and temporary outputs out of Git.
- [ ] Run `python -m unittest discover -s tests -v` and `python -m pip check` in a clean environment.
- [ ] Confirm the existing Linux GitHub Actions check passes.
- [ ] Check README images and relative links after pushing.
- [ ] Verify the seven screenshot states against the actual application.
- [ ] Deploy only after authorization and host compatibility checks; add a live URL only once verified.

## Local run

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

The README includes activation and setup for Windows, Linux and macOS. Public deployment remains intentionally unperformed.
