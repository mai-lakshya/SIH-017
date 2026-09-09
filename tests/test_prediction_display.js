// Verification script for dynamic prediction values, error margins, and model accuracy metrics
const fs = require('fs');
const path = require('path');

async function testPredictionDisplay() {
  console.log('=== TEST: DYNAMIC PREDICTION DISPLAY & MODEL ACCURACY ===\n');

  // 1. Verify HTML Structure in dashboard/index.html
  const htmlPath = path.join(__dirname, '..', 'dashboard', 'index.html');
  if (!fs.existsSync(htmlPath)) {
    throw new Error(`index.html not found at: ${htmlPath}`);
  }
  const html = fs.readFileSync(htmlPath, 'utf-8');

  const requiredElements = [
    'id="out-prob"',
    'id="out-prob-sub"',
    'id="out-confidence-tag"',
    'id="out-crs"',
    'id="out-crs-sub"',
    'id="out-crs-ci"',
    'id="out-delay-days"',
    'id="out-delay-sub"',
    'id="out-delay-ci"',
    'id="out-median-surv"',
    'id="out-surv-human"',
    'id="model-benchmarks-strip"',
    'id="bench-c-index"',
    'id="bench-r2"',
    'id="bench-mae"',
    'id="bench-mape"',
    'id="bench-acc"',
    'id="drawer-delay-prob"',
    'id="drawer-delay-days"',
    'id="drawer-delay-ci"'
  ];

  requiredElements.forEach(id => {
    if (!html.includes(id)) {
      throw new Error(`Missing expected element in index.html: ${id}`);
    }
    console.log(`✔ Found DOM Element: ${id}`);
  });

  // 2. Query live /predict endpoint and verify returned payload schema
  console.log('\n--- Querying Live API: POST http://127.0.0.1:8000/predict ---');
  const payload = {
    project_id: 'NHAI-RJ-2023-0001',
    project_type: 'Highway',
    state: 'Rajasthan',
    district: 'Jaipur',
    terrain_type: 'Plain',
    estimated_cost_inr_crore: 850.0,
    land_area_hectares: 120.0,
    sia_approval_status: 'Approved',
    forest_clearance_status: 'Approved',
    fund_disbursement_percent: 65.0,
    affected_families_count: 145,
    title_dispute_rate_percent: 3.5,
    compensation_multiplier_demand: 1.2,
    section_11_notification_days: 90,
    local_protest_flag: false
  };

  const res = await fetch('http://127.0.0.1:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': 'super-secret-token'
    },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    throw new Error(`API returned HTTP ${res.status}: ${res.statusText}`);
  }

  const data = await res.json();
  console.log(`✔ API returned HTTP 200 OK`);

  const pred = data.predictions;
  if (!pred) throw new Error('Missing predictions object in API response');

  // Verify Predicted Delay (Days and Human-Readable)
  console.log(`\nPredicted Delay Days: ${pred.predicted_delay_days}`);
  console.log(`Human Readable Duration: ${pred.delay_human_readable}`);
  if (typeof pred.predicted_delay_days !== 'number' || pred.predicted_delay_days < 30 || pred.predicted_delay_days > 730) {
    throw new Error(`Invalid predicted_delay_days: ${pred.predicted_delay_days}`);
  }
  if (!pred.delay_human_readable || !pred.delay_human_readable.includes('Months') && !pred.delay_human_readable.includes('Weeks')) {
    throw new Error(`Missing or malformed delay_human_readable: ${pred.delay_human_readable}`);
  }
  console.log(`✔ Valid statutory delay duration & human readable format`);

  // Verify Confidence Score & Probability
  console.log(`Delay Probability: ${pred.delay_probability}%`);
  console.log(`Confidence Score: ${pred.confidence_score}% (${pred.confidence_label})`);
  if (typeof pred.delay_probability !== 'number') {
    throw new Error(`Invalid delay_probability: ${pred.delay_probability}`);
  }
  if (typeof pred.confidence_score !== 'number' || pred.confidence_score < 50 || pred.confidence_score > 100) {
    throw new Error(`Invalid confidence_score: ${pred.confidence_score}`);
  }
  console.log(`✔ Valid calibrated probability & confidence score`);

  // Verify Error Margins
  console.log(`Delay MAE Error Margin: ±${pred.error_margin_days_mae} days`);
  console.log(`Delay Conformal Margin: ±${pred.error_margin_days_conformal} days [${pred.days_p10}d – ${pred.days_p90}d]`);
  console.log(`CRS MAE Error Margin: ±${pred.error_margin_crs_mae} points`);
  if (pred.error_margin_days_mae !== 31.58) {
    throw new Error(`Expected error_margin_days_mae == 31.58, got ${pred.error_margin_days_mae}`);
  }
  if (pred.error_margin_crs_mae !== 0.047) {
    throw new Error(`Expected error_margin_crs_mae == 0.047, got ${pred.error_margin_crs_mae}`);
  }
  console.log(`✔ Valid MAE & Conformal error margins`);

  // Verify Model Diagnostics
  const acc = data.model_accuracy;
  if (!acc) throw new Error('Missing model_accuracy object in API response');
  console.log(`\nModel Diagnostics:`);
  console.log(`- Uno's C-Index: ${acc.uno_c_index} (${acc.c_index_ci})`);
  console.log(`- Timeline R²: ${acc.timeline_r2}`);
  console.log(`- Timeline MAE: ${acc.timeline_mae_days} days`);
  console.log(`- Timeline MAPE: ${acc.timeline_mape_pct}%`);
  console.log(`- Classification Accuracy: ${acc.classification_accuracy}%`);
  console.log(`- Conformal Coverage: ${acc.conformal_coverage_pct}%`);

  if (acc.uno_c_index !== 0.906) throw new Error(`Unexpected c_index: ${acc.uno_c_index}`);
  if (acc.timeline_r2 !== 0.9460) throw new Error(`Unexpected timeline_r2: ${acc.timeline_r2}`);
  if (acc.classification_accuracy !== 100.0) throw new Error(`Unexpected classification_accuracy: ${acc.classification_accuracy}`);

  console.log(`✔ All verified model benchmarks matched evaluation standards!`);

  console.log('\n======================================================');
  console.log('✔ ALL PREDICTION DISPLAY TESTS PASSED SUCCESSFULLY!');
  console.log('======================================================\n');
}

testPredictionDisplay().catch(err => {
  console.error('❌ Test Failed:', err);
  process.exit(1);
});
