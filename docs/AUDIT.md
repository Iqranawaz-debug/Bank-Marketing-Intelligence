# Implementation audit

The earlier chat audit was not present among accessible recent chats. This audit was reconstructed directly from the original notebook, app, training script, artifacts and actual dataset; no prior findings were invented.

| Finding in the original project | Resolution |
|---|---|
| Useful exploratory notebook, three existing estimators and interactive Plotly charts | Notebook and legacy artifacts preserved; all three model families and useful chart concepts retained |
| Balance/campaign caps computed before the train/test split | Custom sklearn transformer learns caps inside each training fold |
| Manual inference did not apply fitted capping | Complete feature/preprocessing/model pipeline is persisted and shared by all inference paths |
| Model scores and comparisons hardcoded into app text | Current metrics, curves and comparisons loaded from generated JSON |
| No actual hyperparameter search found in notebook | Small reproducible grid added (13 configurations, three folds) |
| Previous outcome unknown relabeled not_contacted | Preserve unknown; derive previous-contact indicator from pdays independently |
| Rates rendered as a pie; monthly rate scaled to volume axis | Rate bars with sample denominators; separate monthly volume and percentage charts |
| Hardcoded day=15 in scoring | Explicit validated planned-day input |
| Unqualified 0.5 decision threshold | Documented training-only F1 threshold search with precision/recall trade-off chart |
| Global importance shown beside one customer | Clearly separated global permutation importance and individual sensitivity |
| Relative file paths and unhandled artifact load errors | Project-root paths, saved-model checks and readable errors |
| No batch scoring, exports, tests or repository ignores | Added bounded CSV workflow, exports, unittest/AppTest checks and .gitignore |

The original EDA outputs remain historical evidence. They are not silently presented as outputs of the refactored model. Raw-data EDA in the app is uncapped; model preprocessing is learned from training rows. The current model artifact is separate from the root-level legacy model.
