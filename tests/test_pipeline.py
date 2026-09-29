"""Regression checks for data contracts, saved artifacts and inference."""
import io
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.config import FEATURES, MODEL_DIR
from src.data_processing import load_data, read_prediction_csv, validate_inputs
from src.evaluate import metrics
from src.feature_engineering import CampaignFeatures
from src.predict import load_artifacts, local_sensitivity, predict_customers


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.report = load_artifacts()
        cls.data = load_data()
        cls.sample = cls.data[FEATURES].head(5)

    def test_batch_equals_individual_and_no_fitting(self):
        with patch.object(self.model, "fit", side_effect=AssertionError("Inference must not fit")):
            batch = predict_customers(self.model, self.sample, self.report["threshold"])
            single = [predict_customers(self.model, self.sample.iloc[[i]], self.report["threshold"]).subscription_probability.iloc[0] for i in range(5)]
        np.testing.assert_allclose(batch.subscription_probability, single)
        self.assertTrue(batch.subscription_probability.between(0, 1).all())

    def test_saved_metrics_match_test_predictions(self):
        heldout = pd.read_csv(MODEL_DIR / "test_predictions.csv")
        recomputed = metrics(heldout.actual, heldout.probability, self.report["threshold"])
        for name, value in recomputed.items():
            self.assertAlmostEqual(value, self.report["metrics"][name], places=10)
        probs = self.model.predict_proba(self.data.loc[heldout.source_row, FEATURES])[:, 1]
        np.testing.assert_allclose(probs, heldout.probability, atol=1e-7)

    def test_selection_uses_cv(self):
        selected = max(self.report["comparison_at_0_5"], key=lambda row: row["CV average precision"])["Model"]
        self.assertEqual(selected, self.report["selected_model"])

    def test_caps_do_not_learn_from_inference(self):
        transform = CampaignFeatures().fit(self.sample)
        before = transform.bounds_.copy()
        outlier = self.sample.copy()
        outlier["balance"] = 1e8
        transformed = transform.transform(outlier)
        self.assertEqual(before, transform.bounds_)
        self.assertEqual(transformed.balance.max(), before["balance"][1])
        self.assertTrue((transformed.was_contacted_previously == 0).all())

    def test_invalid_values(self):
        for feature, value in [("age", "oops"), ("age", 12), ("day", 32), ("campaign", 0), ("pdays", -2), ("previous", .5), ("balance", float("inf")), ("job", "astronaut"), ("contact", None)]:
            with self.subTest(feature=feature, value=value):
                bad = self.sample.astype(object).copy()
                bad.loc[bad.index[0], feature] = value
                with self.assertRaises(ValueError):
                    validate_inputs(bad)
        with self.assertRaisesRegex(ValueError, "Missing required"):
            validate_inputs(self.sample.drop(columns="age"))
        with self.assertRaises(ValueError):
            validate_inputs(self.sample.iloc[:0])

    def test_csv_and_download_roundtrip(self):
        parsed = read_prediction_csv(self.sample.to_csv(index=False).encode())
        result = predict_customers(self.model, parsed, self.report["threshold"])
        restored = pd.read_csv(io.StringIO(result.to_csv(index=False)))
        self.assertEqual(len(restored), 5)
        self.assertIn("prediction", restored)
        for content in [b"", b"\xff\xfe", b"age,age\n1,2", b"age\n20", b'age,job\n1,"bad', b" " * (5 * 1024 * 1024 + 1)]:
            with self.subTest(content=content[:20]):
                with self.assertRaises(ValueError):
                    read_prediction_csv(content)

    def test_extra_columns_never_enter_model(self):
        extra = self.sample.assign(duration=999999, deposit="yes")
        np.testing.assert_allclose(predict_customers(self.model, extra, .5).subscription_probability,
                                   predict_customers(self.model, self.sample, .5).subscription_probability)
        self.assertNotIn("duration", validate_inputs(extra))

    def test_missing_model_error(self):
        with patch("src.predict.MODEL_PATH", MODEL_DIR / "absent.joblib"):
            with self.assertRaisesRegex(FileNotFoundError, "train_model.py"):
                load_artifacts()

    def test_local_sensitivity(self):
        result = local_sensitivity(self.model, self.sample.iloc[[0]], self.report["reference"])
        self.assertEqual(set(result.feature), set(FEATURES))
        self.assertTrue(np.isfinite(result.difference).all())


if __name__ == "__main__":
    unittest.main()
