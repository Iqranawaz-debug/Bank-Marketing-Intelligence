"""Train-fitted winsorization and deterministic prior-contact feature."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CampaignFeatures(TransformerMixin, BaseEstimator):
    """Learn capping limits exclusively from fit rows."""
    def fit(self, X: pd.DataFrame, y=None):
        self.bounds_ = {"balance": (float(X.balance.quantile(.01)), float(X.balance.quantile(.99))),
                        "campaign": (1., float(X.campaign.quantile(.99)))}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "bounds_")
        result = X.copy()
        for name, (low, high) in self.bounds_.items():
            result[name] = result[name].clip(low, high)
        result["was_contacted_previously"] = result.pdays.ne(-1).astype(int)
        return result
