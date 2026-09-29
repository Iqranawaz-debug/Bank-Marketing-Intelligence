"""Portable paths and public prediction schema."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "bank.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "pipeline.joblib"
REPORT_PATH = MODEL_DIR / "evaluation.json"
SEED = 42
NUMERIC = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORIES = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["divorced", "married", "single"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"], "housing": ["no", "yes"], "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "poutcome": ["failure", "other", "success", "unknown"],
}
FEATURES = NUMERIC + list(CATEGORIES)
LABELS = {"age": "Age (years)", "balance": "Average yearly balance (EUR)", "day": "Planned contact day of month", "campaign": "Contacts this campaign (including planned contact)", "pdays": "Days since previous campaign contact (-1 = never)", "previous": "Contacts before this campaign", "job": "Occupation", "marital": "Marital status", "education": "Education", "default": "Credit in default", "housing": "Housing loan", "loan": "Personal loan", "contact": "Planned contact method", "month": "Planned contact month", "poutcome": "Previous campaign outcome"}
