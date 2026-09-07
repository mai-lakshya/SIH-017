import unittest
import urllib.request
import json

class TestReferenceDistrictsAPI(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:8000"

    def test_get_reference_districts_success(self):
        url = f"{self.BASE_URL}/reference/districts"
        req = urllib.request.Request(url, headers={"User-Agent": "TestClient/1.0"})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode('utf-8'))

        self.assertIsInstance(data, dict)
        self.assertEqual(len(data), 36, "Must return mapping for all 36 Indian states and Union Territories")

        # Verify key states exist
        self.assertIn("Rajasthan", data)
        self.assertIn("West Bengal", data)
        self.assertIn("Maharashtra", data)
        self.assertIn("Uttar Pradesh", data)
        self.assertIn("Jammu and Kashmir", data)
        self.assertIn("Delhi", data)

        # Verify Rajasthan districts
        rj_districts = data["Rajasthan"]
        self.assertIn("Banswara", rj_districts)
        self.assertIn("Jaipur", rj_districts)
        self.assertIn("Jodhpur", rj_districts)
        self.assertEqual(rj_districts, sorted(rj_districts), "Districts must be sorted alphabetically")

        # Verify West Bengal districts
        wb_districts = data["West Bengal"]
        self.assertIn("Paschim Medinipur", wb_districts)
        self.assertIn("Kolkata", wb_districts)
        self.assertEqual(wb_districts, sorted(wb_districts), "Districts must be sorted alphabetically")

        # Verify Uttar Pradesh has all 75 official districts
        self.assertIn("Uttar Pradesh", data)
        up_districts = data["Uttar Pradesh"]
        self.assertEqual(len(up_districts), 75, f"Uttar Pradesh must have all 75 districts, got {len(up_districts)}")
        self.assertIn("Agra", up_districts)
        self.assertIn("Lucknow", up_districts)
        self.assertIn("Varanasi", up_districts)
        self.assertIn("Gautam Buddha Nagar (Noida)", up_districts)
        self.assertEqual(up_districts, sorted(up_districts), "UP districts must be sorted alphabetically")

        # Verify against full CSV directly using pandas
        import os, pandas as pd
        csv_path = "indian_infrastructure_projects_dataset.csv"
        if not os.path.exists(csv_path):
            csv_path = "Revolution-main/indian_infrastructure_projects_dataset.csv"
        
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 13532, "Full CSV must have 13,532 rows (not capped at 200)")

        # Spot-check 4 states against pandas nunique
        spot_checks = ["Maharashtra", "Rajasthan", "West Bengal", "Gujarat"]
        for st in spot_checks:
            expected_cnt = df[df['state'] == st]['district'].dropna().nunique()
            actual_cnt = len(data[st])
            self.assertEqual(actual_cnt, expected_cnt, f"District count for {st} ({actual_cnt}) should match CSV count ({expected_cnt})")

        # Verify all districts are non-empty strings
        for state, districts in data.items():
            self.assertIsInstance(districts, list, f"Districts for {state} must be a list")
            self.assertTrue(len(districts) > 0, f"State {state} should have at least 1 district in dataset")
            for d in districts:
                self.assertIsInstance(d, str)
                self.assertTrue(len(d.strip()) > 0)
                self.assertNotIn(d.lower(), ['nan', 'none', 'unknown', ''])

if __name__ == '__main__':
    unittest.main()
