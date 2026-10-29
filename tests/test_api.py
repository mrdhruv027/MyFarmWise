"""
Unit and integration tests for Flask REST API endpoints.
"""

import unittest
from backend.app import app


class TestFlaskAPI(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertEqual(json_data["status"], "Operational")
        self.assertIn("29 October 2026", json_data["demonstration_date"])
        self.assertIn("subsystems", json_data)

    def test_district_states_endpoint(self):
        response = self.client.get("/api/districts/states")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data["success"])
        self.assertGreater(json_data["count"], 20)

    def test_district_lookup_endpoint(self):
        response = self.client.get("/api/districts/lookup?state=MAHARASHTRA&district=PUNE")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data["success"])
        self.assertEqual(json_data["data"]["district"], "PUNE")

    def test_crop_recommend_endpoint(self):
        payload = {
            "N": 90.0,
            "P": 42.0,
            "K": 43.0,
            "temperature": 21.0,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 205.0,
            "is_district_estimate": False
        }
        response = self.client.post("/api/crop/recommend", json=payload)
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data["success"])
        self.assertEqual(json_data["data"]["recommended_crop"].lower(), "rice")

    def test_fertilizer_recommend_endpoint(self):
        payload = {
            "crop": "wheat",
            "soil_type": "Loamy",
            "N": 150.0,
            "P": 10.0,
            "K": 180.0
        }
        response = self.client.post("/api/fertilizer/recommend", json=payload)
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data["success"])
        self.assertIn("recommended_fertilizer", json_data["data"])

    def test_disease_architecture_endpoint(self):
        response = self.client.get("/api/disease/architecture")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data["success"])
        self.assertEqual(json_data["data"]["gradcam_target_layer"], "last_se_conv")


if __name__ == "__main__":
    unittest.main()
