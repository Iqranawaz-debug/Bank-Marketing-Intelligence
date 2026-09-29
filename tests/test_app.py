"""Exercise every Streamlit route, EDA view and prediction form."""
import unittest
import io
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from src.config import ROOT
from src.predict import load_artifacts, predict_customers
from src.data_processing import load_data
from src.config import FEATURES


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    def assert_clean(self):
        self.assertEqual(len(self.app.exception), 0, str(self.app.exception))

    def test_all_pages(self):
        self.assert_clean()
        for page in self.app.sidebar.radio[0].options:
            with self.subTest(page=page):
                self.app.sidebar.radio[0].set_value(page).run()
                self.assert_clean()

    def test_all_eda_views_and_empty_filter(self):
        self.app.sidebar.radio[0].set_value("Exploratory Analysis").run()
        for group in self.app.selectbox[0].options:
            self.app.selectbox[0].set_value(group).run()
            self.assert_clean()
        self.app.multiselect[0].set_value(["student"]).run()
        self.app.slider[0].set_value((90, 95)).run()
        self.assert_clean()
        self.assertTrue(any("No records" in item.value for item in self.app.info))

    def test_manual_prediction_and_download(self):
        self.app.sidebar.radio[0].set_value("Customer Prediction").run()
        self.app.button[0].click().run()
        self.assert_clean()
        self.assertTrue(any(item.label == "Estimated model probability" for item in self.app.metric))
        self.assertGreater(len(self.app.get("download_button")), 0)

    def test_batch_upload_and_invalid_file(self):
        self.app.sidebar.radio[0].set_value("Batch Prediction").run()
        content = (ROOT / "models" / "example_customers.csv").read_bytes()
        with patch("streamlit.file_uploader", return_value=io.BytesIO(content)):
            self.app.run()
            self.assert_clean()
            self.assertTrue(any(item.label == "Rows scored" and item.value == "5" for item in self.app.metric))
            self.assertEqual(len(self.app.get("download_button")), 2)
        with patch("streamlit.file_uploader", return_value=io.BytesIO(b"age\n30")):
            self.app.run()
            self.assert_clean()
            self.assertTrue(any("Missing required columns" in item.value for item in self.app.error))

    def test_example_form_matches_saved_pipeline(self):
        model, report = load_artifacts()
        expected = predict_customers(model, load_data()[FEATURES].iloc[[0]], report["threshold"])
        self.app.sidebar.radio[0].set_value("Customer Prediction").run()
        self.app.button[0].click().run()
        actual = next(item.value for item in self.app.metric if item.label == "Estimated model probability")
        self.assertEqual(actual, f"{expected.subscription_probability.iloc[0]:.1%}")

    def test_performance_cards_match_saved_report(self):
        _, report = load_artifacts()
        self.app.sidebar.radio[0].set_value("Model Performance").run()
        displayed = {item.label: item.value for item in self.app.metric}
        self.assertEqual(displayed["ROC-AUC"], f"{report['metrics']['ROC-AUC']:.3f}")
        self.assertEqual(displayed["Recall"], f"{report['metrics']['Recall']:.1%}")

    def test_batch_blank_values_show_readable_error(self):
        sample = load_data()[FEATURES].head(2).astype(object)
        sample.loc[0, "age"] = None
        self.app.sidebar.radio[0].set_value("Batch Prediction").run()
        with patch("streamlit.file_uploader", return_value=io.BytesIO(sample.to_csv(index=False).encode())):
            self.app.run()
            self.assert_clean()
            self.assertTrue(any("blank" in item.value for item in self.app.error))


if __name__ == "__main__":
    unittest.main()
