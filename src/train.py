"""Reproduce training, selection, tuning, evaluation and complete persistence."""
import hashlib
import importlib.metadata
import json
import logging
import platform
from datetime import datetime, timezone
from uuid import uuid4
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from src.config import CATEGORIES, DATA_PATH, FEATURES, MODEL_DIR, MODEL_PATH, NUMERIC, REPORT_PATH, SEED
from src.data_processing import load_data
from src.evaluate import curves, metrics
from src.feature_engineering import CampaignFeatures


def build_pipeline(estimator):
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC + ["was_contacted_previously"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORIES)),
    ])
    return Pipeline([("features", CampaignFeatures()), ("preprocess", preprocess), ("model", estimator)])


def train():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = load_data()
    X, y = data[FEATURES], data.deposit_flag
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    candidates = {
        "Dummy baseline": (DummyClassifier(strategy="prior"), {}),
        "Logistic Regression": (LogisticRegression(max_iter=1500, random_state=SEED), {"model__C": [.1, 1.], "model__class_weight": [None, "balanced"]}),
        "Random Forest": (RandomForestClassifier(n_estimators=300, max_depth=10, random_state=SEED, n_jobs=2), {"model__max_depth": [10, None], "model__class_weight": [None, "balanced"]}),
        "XGBoost": (XGBClassifier(n_estimators=300, max_depth=5, learning_rate=.05, eval_metric="logloss", random_state=SEED, n_jobs=2), {"model__max_depth": [3, 5], "model__scale_pos_weight": [1., 1.1]}),
    }
    fitted, comparison, tuning = {}, [], []
    for name, (estimator, grid) in candidates.items():
        logging.info("Training and cross-validating %s", name)
        search = GridSearchCV(build_pipeline(estimator), grid, scoring="average_precision", cv=cv, n_jobs=1, error_score="raise")
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        comparison.append({"Model": name, "CV average precision": float(search.best_score_),
                           "CV standard deviation": float(search.cv_results_["std_test_score"][search.best_index_]),
                           **metrics(y_test, search.predict_proba(X_test)[:, 1])})
        for params, mean, std in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"]):
            tuning.append({"Model": name, "parameters": params, "CV average precision": float(mean), "CV standard deviation": float(std)})
    selected = max(comparison, key=lambda row: row["CV average precision"])["Model"]
    model = fitted[selected]
    oof = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    threshold_table = [{"threshold": float(t), **metrics(y_train, oof, t)} for t in np.arange(.1, .901, .025)]
    best_threshold = max(threshold_table, key=lambda row: (row["F1"], row["threshold"]))["threshold"]
    probability = model.predict_proba(X_test)[:, 1]
    logging.info("Selected %s; computing held-out permutation importance", selected)
    importance = permutation_importance(model, X_test, y_test, scoring="average_precision", n_repeats=5, random_state=SEED, n_jobs=1)
    run_id = uuid4().hex
    model.bank_run_id_ = run_id
    reference = {c: (float(X_train[c].median()) if c in NUMERIC else str(X_train[c].mode()[0])) for c in FEATURES}
    report = {"run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(), "selected_model": selected,
              "selection_metric": "3-fold training cross-validation average precision", "threshold": best_threshold,
              "threshold_method": "Maximum F1 on out-of-fold training predictions, grid 0.10-0.90 in 0.025 steps; exploratory, not cost-optimized.",
              "rows": len(data), "train_rows": len(X_train), "test_rows": len(X_test),
              "positive_rate": float(y.mean()), "test_positive_rate": float(y_test.mean()),
              "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(), "seed": SEED,
              "python": platform.python_version(),
              "versions": {p: importlib.metadata.version(p) for p in ["scikit-learn", "xgboost", "numpy", "pandas", "joblib"]},
              "comparison_at_0_5": comparison, "tuning": tuning, "threshold_analysis": threshold_table,
              "metrics": metrics(y_test, probability, best_threshold), "curves": curves(y_test, probability, best_threshold),
              "parameters": {k: (str(v) if isinstance(v, float) and not np.isfinite(v) else v)
                             for k, v in model.named_steps["model"].get_params().items()}, "reference": reference,
              "capping_bounds": model.named_steps["features"].bounds_,
              "importance": [{"feature": f, "mean": float(m), "std": float(s)} for f, m, s in zip(FEATURES, importance.importances_mean, importance.importances_std)]}
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame({"source_row": X_test.index, "actual": y_test.to_numpy(), "probability": probability}).to_csv(MODEL_DIR / "test_predictions.csv", index=False)
    data[FEATURES].head(5).to_csv(MODEL_DIR / "example_customers.csv", index=False)
    logging.info("Saved complete pipeline. Selected=%s, threshold=%.3f, metrics=%s", selected, best_threshold, report["metrics"])
    return report


if __name__ == "__main__":
    train()
