/**
 * Verification Test: Model Health & Continuous Learning System
 */
const fs = require('fs');
const path = require('path');

async function runHealthVerification() {
  console.log('=== VERIFYING MODEL HEALTH & CONTINUOUS LEARNING ===\n');

  // 1. Verify DOM Elements in dashboard/index.html
  console.log('1. Checking DOM Elements in dashboard/index.html...');
  const htmlContent = fs.readFileSync(path.join(__dirname, '../dashboard/index.html'), 'utf8');

  const requiredElements = [
    'id="tab-health-btn"',
    'id="view-health"',
    'id="health-version-val"',
    'id="health-cindex-val"',
    'id="health-ece-val"',
    'id="health-auc-val"',
    'id="sch-provider-status"',
    'id="sch-daily-status"',
    'id="sch-weekly-status"',
    'id="sch-dataset-size"',
    'id="btn-manual-retrain"',
    'id="drift-matrix-tbody"'
  ];

  for (const el of requiredElements) {
    if (!htmlContent.includes(el)) {
      throw new Error(`Missing required DOM element: ${el}`);
    }
    console.log(`  [OK] Found DOM Element: ${el}`);
  }

  // 2. Query live /model/health endpoint
  console.log('\n2. Querying Live API: GET http://127.0.0.1:8000/model/health ...');
  const healthRes = await fetch('http://127.0.0.1:8000/model/health');
  if (healthRes.status !== 200) {
    throw new Error(`/model/health returned HTTP ${healthRes.status}`);
  }
  const healthData = await healthRes.json();
  console.log(`  [OK] Current Model Version: ${healthData.current_version}`);
  console.log(`  [OK] Hardware Provider: ${healthData.npu_provider}`);
  console.log(`  [OK] C-Index: ${healthData.c_index} (Gate: >= 0.88)`);
  console.log(`  [OK] ECE: ${healthData.ece} (Gate: <= 0.10)`);
  console.log(`  [OK] AUC: ${healthData.auc} (Gate: >= 0.85)`);

  if (!healthData.current_version) throw new Error('Missing current_version in health response');
  if (healthData.c_index < 0.88) throw new Error(`C-Index ${healthData.c_index} is below gate 0.88`);
  if (healthData.ece > 0.10) throw new Error(`ECE ${healthData.ece} exceeds gate 0.10`);
  if (healthData.auc < 0.85) throw new Error(`AUC ${healthData.auc} is below gate 0.85`);

  // 3. Query live /model/drift endpoint
  console.log('\n3. Querying Live API: GET http://127.0.0.1:8000/model/drift ...');
  const driftRes = await fetch('http://127.0.0.1:8000/model/drift');
  if (driftRes.status !== 200) {
    throw new Error(`/model/drift returned HTTP ${driftRes.status}`);
  }
  const driftData = await driftRes.json();
  console.log(`  [OK] Features Monitored: ${driftData.total_features_monitored}`);
  console.log(`  [OK] Max PSI: ${driftData.max_psi}`);
  console.log(`  [OK] Auto-Retrain Trigger Flag: ${driftData.auto_retrain_triggered}`);

  // 4. Verify Streamlit Model Health Tab implementation
  console.log('\n4. Checking Streamlit dashboard.py implementation...');
  const pyDashboard = fs.readFileSync(path.join(__dirname, '../dashboard.py'), 'utf8');
  if (!pyDashboard.includes('tab_model_health') || !pyDashboard.includes('Model Health')) {
    throw new Error('dashboard.py missing Model Health tab implementation');
  }
  console.log('  [OK] Streamlit Model Health tab verified');

  console.log('\n======================================================');
  console.log('[OK] ALL MODEL HEALTH & CONTINUOUS LEARNING TESTS PASSED!');
  console.log('======================================================\n');
}

runHealthVerification().catch(err => {
  console.error('Test Failed:', err.message);
  process.exit(1);
});
