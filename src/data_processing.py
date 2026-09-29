"""Read raw data and validate manual and batch scoring inputs."""
import csv
import io
import numpy as np
import pandas as pd
from src.config import CATEGORIES, DATA_PATH, FEATURES, NUMERIC


def validate_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject malformed rows; unknown is an explicit valid category."""
    if frame.empty:
        raise ValueError("No customer rows were supplied.")
    if frame.columns.duplicated().any():
        raise ValueError("CSV column names must be unique.")
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    result = frame[FEATURES].copy()
    for column in NUMERIC:
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"{column}: enter finite numbers with no blank cells.")
        if column != "balance" and (values % 1 != 0).any():
            raise ValueError(f"{column}: whole numbers are required.")
        result[column] = values
    for column, choices in CATEGORIES.items():
        values = result[column].astype("string").str.strip().str.lower()
        if values.isna().any() or not values.isin(choices).all():
            raise ValueError(f"{column}: use one of {', '.join(choices)}; blanks are not accepted.")
        result[column] = values.astype(object)
    bounds = {"age": (18, 120), "day": (1, 31), "campaign": (1, 10000), "pdays": (-1, 100000), "previous": (0, 10000), "balance": (-1e9, 1e9)}
    for column, (low, high) in bounds.items():
        if not result[column].between(low, high).all():
            raise ValueError(f"{column}: allowed range is {low:g} to {high:g}.")
    return result


def read_prediction_csv(content: bytes) -> pd.DataFrame:
    """Parse bounded UTF-8 CSV uploads without exposing parser internals."""
    if not content or len(content) > 5 * 1024 * 1024:
        raise ValueError("Upload a non-empty CSV smaller than 5 MB.")
    try:
        decoded = content.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(decoded)))
        if len(header) != len(set(header)):
            raise ValueError("CSV column names must be unique.")
        frame = pd.read_csv(io.StringIO(decoded))
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, StopIteration) as exc:
        raise ValueError("Could not read CSV. Use UTF-8 text with commas and a header row.") from exc
    if len(frame) > 10000:
        raise ValueError("Upload at most 10,000 customers per batch.")
    return validate_inputs(frame)


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH)
    validated = validate_inputs(frame)
    if not frame["deposit"].isin(["yes", "no"]).all():
        raise ValueError("The dataset target must contain yes/no values.")
    validated["deposit"] = frame["deposit"]
    validated["deposit_flag"] = frame["deposit"].eq("yes").astype(int)
    return validated
