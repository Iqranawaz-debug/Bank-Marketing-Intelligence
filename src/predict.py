"""Load trusted local artifacts and score without fitting any component."""
import json
import logging
import joblib
import numpy as np
import pandas as pd
from src.config import FEATURES, MODEL_PATH, REPORT_PATH
from src.data_processing import validate_inputs
logger = logging.getLogger(__name__)


def load_artifacts():
    if not MODEL_PATH.exists() or not REPORT_PATH.exists():
        raise FileNotFoundError("Model artifacts are missing. Run: python train_model.py")
    model = joblib.load(MODEL_PATH)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if model.bank_run_id_ != report["run_id"]:
        raise ValueError("Model and evaluation files belong to different runs. Regenerate both artifacts.")
    logger.info("Loaded model artifacts: %s", report["selected_model"])
    return model, report


def predict_customers(model, frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between zero and one.")
    validated = validate_inputs(frame)
    probability = model.predict_proba(validated)[:, 1]
    result = validated.copy()
    result["subscription_probability"] = probability
    result["prediction"] = np.where(probability >= threshold, "yes", "no")
    result["decision_threshold"] = threshold
    logger.info("Scored %d customer rows", len(result))
    return result


def local_sensitivity(model, frame: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """One-at-a-time replacement effects, deliberately not labeled SHAP."""
    row = validate_inputs(frame).iloc[[0]]
    base = float(model.predict_proba(row)[0, 1])
    variants = pd.concat([row] * len(FEATURES), ignore_index=True)
    for i, feature in enumerate(FEATURES):
        variants.loc[i, feature] = reference[feature]
    replaced = model.predict_proba(variants)[:, 1]
    return pd.DataFrame({"feature": FEATURES, "difference": base - replaced})
