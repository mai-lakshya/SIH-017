"""
End-to-End Verification Test for Geospatial Remoteness Diagnostic Module
Validates that the diagnostic takes real Census 2011 geospatial data,
returns accurate nearest settlements, distance calculations, urban tiers,
and integrates smoothly with the /predict pipeline and /remoteness/evaluate endpoint.
"""
import urllib.request
import json
import unittest

BASE_URL = "http://127.0.0.1:8000"

class TestRemotenessE2E(unittest.TestCase):

    def test_01_reference_coordinates(self):
        url = f"{BASE_URL}/reference/coordinates"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))
        
        self.assertIsInstance(data, dict)
        self.assertGreater(len(data), 400)
        self.assertIn("Madhya Pradesh|Betul", data)
        self.assertIn("Rajasthan|Banswara", data)
        self.assertIn("Maharashtra|Pune", data)
        
        betul_coords = data["Madhya Pradesh|Betul"]
        self.assertEqual(len(betul_coords), 2)
        self.assertAlmostEqual(betul_coords[0], 21.8797, places=2)
        self.assertAlmostEqual(betul_coords[1], 77.8754, places=2)

    def test_02_remoteness_evaluate_betul(self):
        url = f"{BASE_URL}/remoteness/evaluate"
        payload = {
            "state": "Madhya Pradesh",
            "district": "Betul",
            "project_type": "Highway"
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode('utf-8'))

        self.assertIn("nearest_settlement", res)
        stl = res["nearest_settlement"]
        self.assertEqual(stl["name"], "Betul Town")
        self.assertEqual(stl["tier"], "Census Town")
        self.assertEqual(stl["population"], 65000)
        self.assertEqual(stl["population_data_vintage"], "Census 2011")
        self.assertIn("top_3_settlements", res)
        self.assertGreaterEqual(len(res["top_3_settlements"]), 2)
        self.assertGreater(res["remoteness_delay_days"], 0.0)

    def test_03_remoteness_evaluate_tribal_banswara(self):
        url = f"{BASE_URL}/remoteness/evaluate"
        payload = {
            "state": "Rajasthan",
            "district": "Banswara",
            "project_type": "Highway"
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode('utf-8'))

        stl = res["nearest_settlement"]
        self.assertEqual(stl["name"], "Banswara Town")
        # Tribal area triggers FRA 2006 penalty (+90d)
        self.assertEqual(res["terrain_type"], "forest_tribal")
        self.assertEqual(res["component_breakdown"]["fra_flat_penalty"], 90.0)
        self.assertGreater(res["remoteness_delay_days"], 90.0)

    def test_04_predict_pipeline_integration(self):
        url = f"{BASE_URL}/predict"
        payload = {
            "project_id": "TEST-REAL-DATA-001",
            "project_type": "Highway",
            "state": "Madhya Pradesh",
            "district": "Betul",
            "terrain_type": "Plain",
            "estimated_cost_inr_crore": 850.0,
            "land_area_hectares": 120.0,
            "sia_approval_status": "Approved",
            "forest_clearance_status": "Not_Required",
            "fund_disbursement_percent": 55.0,
            "affected_families_count": 85,
            "title_dispute_rate_percent": 8.0,
            "compensation_multiplier_demand": 1.3,
            "section_11_notification_days": 60,
            "local_protest_flag": False
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={"Content-Type": "application/json", "X-API-Key": "super-secret-token"}
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode('utf-8'))

        self.assertIn("remoteness_analysis", res)
        rem = res["remoteness_analysis"]
        self.assertEqual(rem["nearest_settlement"]["name"], "Betul Town")
        self.assertAlmostEqual(rem["site"]["lat"], 21.8797, places=2)
        self.assertAlmostEqual(rem["site"]["lon"], 77.8754, places=2)
        self.assertEqual(rem["nearest_settlement"]["population"], 65000)

if __name__ == "__main__":
    unittest.main()
