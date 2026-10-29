"""
Unit tests for Fertilizer Recommendation Subsystem (Module D).
"""

import unittest
from src.fertilizer.recommender import (
    calculate_nutrient_deficit,
    recommend_fertilizer,
    SOIL_TYPES,
)
from config.settings import COMMERCIAL_FERTILIZERS


class TestFertilizerEngine(unittest.TestCase):

    def test_supported_soil_types(self):
        self.assertEqual(len(SOIL_TYPES), 5)
        for stype in ["Sandy", "Loamy", "Black", "Red", "Clayey"]:
            self.assertIn(stype, SOIL_TYPES)

    def test_commercial_fertilizer_grades(self):
        # Synopsis specifies 7 commercial fertilizer formulations
        self.assertEqual(len(COMMERCIAL_FERTILIZERS), 7)
        for fert in ["Urea", "DAP", "14-35-14", "28-28", "17-17-17", "20-20", "10-26-26"]:
            self.assertIn(fert, COMMERCIAL_FERTILIZERS)

    def test_dap_recommendation_on_high_p_deficit(self):
        # High soil N and low soil P should trigger DAP (root/basal phosphate)
        advisory = recommend_fertilizer(
            crop_name="wheat",
            soil_n=350.0,
            soil_p=5.0,
            soil_k=200.0,
            soil_type="Loamy"
        )
        self.assertEqual(advisory["recommended_fertilizer"], "DAP")
        self.assertGreater(advisory["indicative_quantity_kg_ha"], 50.0)
        self.assertLessEqual(advisory["indicative_quantity_kg_ha"], 350.0)

    def test_balanced_np_co_deficiency(self):
        # Both N and P deficient in loamy soil triggers 20-20
        advisory = recommend_fertilizer(
            crop_name="wheat",
            soil_n=150.0,
            soil_p=8.0,
            soil_k=200.0,
            soil_type="Loamy"
        )
        self.assertEqual(advisory["recommended_fertilizer"], "20-20")

    def test_urea_recommendation_on_high_n_deficit(self):
        # Low soil N should trigger Urea
        advisory = recommend_fertilizer(
            crop_name="sugarcane",
            soil_n=40.0,
            soil_p=120.0,
            soil_k=250.0,
            soil_type="Black"
        )
        self.assertEqual(advisory["recommended_fertilizer"], "Urea")
        self.assertIn("Nitrogen", advisory["primary_nutrient_addressed"])

    def test_disclaimer_presence(self):
        advisory = recommend_fertilizer(crop_name="rice", soil_n=100, soil_p=20, soil_k=100)
        self.assertIn("disclaimer", advisory)
        self.assertIn("Soil Health Card", advisory["disclaimer"])


if __name__ == "__main__":
    unittest.main()
