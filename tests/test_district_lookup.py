"""
Unit tests for District-Level Soil and Climate Lookup Subsystem (Module B).
"""

import unittest
from src.data.district_lookup import (
    get_all_states,
    get_districts_by_state,
    lookup_district,
    convert_rating_to_numeric,
)


class TestDistrictLookup(unittest.TestCase):

    def test_get_all_states(self):
        states = get_all_states()
        self.assertIsInstance(states, list)
        self.assertGreater(len(states), 25)
        self.assertIn("MAHARASHTRA", states)
        self.assertIn("UTTAR PRADESH", states)

    def test_get_districts_by_state(self):
        up_districts = get_districts_by_state("UTTAR PRADESH")
        self.assertIsInstance(up_districts, list)
        self.assertGreater(len(up_districts), 50)
        self.assertTrue(any("GAUTAM BUDDHA" in d for d in up_districts))
        self.assertIn("LUCKNOW", up_districts)

    def test_lookup_verified_district(self):
        pune = lookup_district("MAHARASHTRA", "PUNE")
        self.assertTrue(pune["found"])
        self.assertEqual(pune["district"], "PUNE")
        self.assertEqual(pune["soil_type"], "Black")
        self.assertIsNotNone(pune["soil_chemistry"]["nitrogen"]["value"])
        self.assertIsNotNone(pune["climate_normals"]["annual_rainfall_mm"])
        self.assertIn("Verified", pune["metadata"]["status"])

    def test_lookup_bhavnagar_district(self):
        # Specifically cited in project synopsis mKisan survey
        bhav = lookup_district("GUJARAT", "BHAVNAGAR")
        self.assertTrue(bhav["found"])
        self.assertEqual(bhav["soil_type"], "Black")
        self.assertEqual(bhav["soil_chemistry"]["nitrogen"]["rating"], "Low")
        self.assertIn("ICAR", bhav["metadata"]["source_soil"])

    def test_lookup_missing_district(self):
        res = lookup_district("FICTIONAL_STATE", "FICTIONAL_DISTRICT")
        self.assertFalse(res["found"])
        self.assertIn("error", res)

    def test_rating_conversion_rules(self):
        n_low = convert_rating_to_numeric("nitrogen", "Low")
        n_med = convert_rating_to_numeric("nitrogen", "Medium")
        n_high = convert_rating_to_numeric("nitrogen", "High")
        self.assertLess(n_low, n_med)
        self.assertLess(n_med, n_high)

        p_low = convert_rating_to_numeric("phosphorus", "Low")
        self.assertEqual(p_low, 8.5)


if __name__ == "__main__":
    unittest.main()
