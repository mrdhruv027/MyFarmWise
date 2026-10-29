"""
Unit tests for Crop Recommendation Subsystem (Module C).
"""

import unittest
from src.crop.predictor import predict_crop, load_crop_model, FEATURE_NAMES
from src.crop.evaluate import run_crop_evaluation


class TestCropRecommendation(unittest.TestCase):

    def test_model_loading(self):
        model = load_crop_model()
        self.assertIsNotNone(model)
        self.assertEqual(len(model.classes_), 22)
        self.assertEqual(len(model.feature_importances_), len(FEATURE_NAMES))

    def test_predict_crop_rice_envelope(self):
        # Conditions optimal for Rice (High rain, high humidity)
        res = predict_crop(
            nitrogen=90.0,
            phosphorus=42.0,
            potassium=43.0,
            temperature=21.0,
            humidity=82.0,
            ph=6.5,
            rainfall=205.0,
            is_district_estimate=False
        )
        self.assertEqual(res["recommended_crop"].lower(), "rice")
        self.assertGreater(res["confidence_pct"], 50.0)
        self.assertEqual(len(res["top_candidates"]), 3)

    def test_predict_crop_chickpea_envelope(self):
        # Conditions optimal for Chickpea (Moderate NPK, low humidity/rain)
        res = predict_crop(
            nitrogen=40.0,
            phosphorus=60.0,
            potassium=80.0,
            temperature=18.0,
            humidity=17.0,
            ph=7.0,
            rainfall=75.0,
            is_district_estimate=True
        )
        self.assertIn(res["recommended_crop"].lower(), ["chickpea", "kidneybeans", "lentil"])
        self.assertTrue(res["is_district_estimate"])

    def test_crop_evaluation_accuracy(self):
        report = run_crop_evaluation(test_size=0.2, random_state=42)
        # Target criteria in Table 6.1: Retain 99.09% on original split
        self.assertGreaterEqual(report["test_accuracy_pct"], 98.5)
        self.assertGreaterEqual(report["macro_f1_pct"], 98.5)
        self.assertEqual(report["crop_classes_count"], 22)


if __name__ == "__main__":
    unittest.main()
