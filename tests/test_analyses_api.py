import urllib.request
import json
import sys

headers = {
    'X-API-Key': 'super-secret-token',
    'Content-Type': 'application/json'
}

# 1. Test validation error (empty project_name)
bad_payload = json.dumps({
    'project_name': '   ',
    'state': 'Rajasthan',
    'district': 'Banswara',
    'input_payload': {}
}).encode()
req = urllib.request.Request('http://127.0.0.1:8000/analyses/save', data=bad_payload, headers=headers)
try:
    urllib.request.urlopen(req)
    print('FAIL: Expected 400 for empty project_name')
    sys.exit(1)
except urllib.error.HTTPError as e:
    assert e.code == 400
    print('PASS: 400 for empty project_name')

# 2. Test successful save
valid_analysis = {
    'project_name': 'Test Highway Corridor - Banswara',
    'state': 'Rajasthan',
    'district': 'Banswara',
    'input_payload': {
        'project_id': 'NHAI-TEST-001',
        'state': 'Rajasthan',
        'district': 'Banswara',
        'land_area_hectares': 150.0,
        'project_type': 'Highway',
        'terrain_type': 'Plain',
        'estimated_cost_inr_crore': 320.0,
        'affected_families_count': 450,
        'title_dispute_rate_percent': 12.5,
        'local_protest_flag': False,
        'compensation_multiplier_demand': 1.5,
        'sia_approval_status': 'Pending',
        'forest_clearance_status': 'Stage_1_Pending',
        'fund_disbursement_percent': 35.0,
        'section_11_notification_days': 120
    }
}
req = urllib.request.Request('http://127.0.0.1:8000/analyses/save', data=json.dumps(valid_analysis).encode(), headers=headers)
res = urllib.request.urlopen(req)
saved_record = json.loads(res.read().decode())
print('PASS: Saved record created:')
print(json.dumps(saved_record, indent=2))
assert saved_record['project_name'] == 'Test Highway Corridor - Banswara'
assert saved_record['latitude'] is not None
assert saved_record['longitude'] is not None
assert saved_record['risk_tier'] in ['Low', 'Medium', 'High']

# 3. Test GET /analyses
req = urllib.request.Request('http://127.0.0.1:8000/analyses', headers=headers)
res = urllib.request.urlopen(req)
items = json.loads(res.read().decode())
print(f'PASS: GET /analyses returned {len(items)} items')
assert len(items) >= 1
assert items[0]['id'] == saved_record['id']

# 4. Test GET /analyses/{id}
req = urllib.request.Request(f'http://127.0.0.1:8000/analyses/{saved_record["id"]}', headers=headers)
res = urllib.request.urlopen(req)
detail = json.loads(res.read().decode())
print('PASS: GET /analyses/{id} returned detail:', detail['id'], 'input project_id:', detail['input_payload']['project_id'])
assert detail['input_payload']['project_id'] == 'NHAI-TEST-001'

# 5. Test DELETE /analyses/{id}
req = urllib.request.Request(f'http://127.0.0.1:8000/analyses/{saved_record["id"]}', headers=headers, method='DELETE')
res = urllib.request.urlopen(req)
del_res = json.loads(res.read().decode())
print('PASS: DELETE /analyses/{id} returned:', del_res)
assert del_res['status'] == 'deleted'

# 6. Verify 404 after delete
try:
    req = urllib.request.Request(f'http://127.0.0.1:8000/analyses/{saved_record["id"]}', headers=headers)
    urllib.request.urlopen(req)
    print('FAIL: Expected 404 after delete')
    sys.exit(1)
except urllib.error.HTTPError as e:
    assert e.code == 404
    print('PASS: 404 verified after delete')

print('ALL API TESTS PASSED!')
