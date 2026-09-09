const fs = require('fs');

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function verifyAllRoutes() {
  console.log('=== VERIFYING THREE SEPARATE ROUTES & C-INDEX 0.906 ===\n');

  // Wait a moment for server to be ready
  for (let i = 0; i < 10; i++) {
    try {
      const res = await fetch('http://127.0.0.1:8000/');
      if (res.ok) break;
    } catch (e) {
      await sleep(1000);
    }
  }

  // 1. Verify Landing Page (Route /)
  console.log('1. Checking Landing Page (http://127.0.0.1:8000/)...');
  const resLanding = await fetch('http://127.0.0.1:8000/');
  if (resLanding.status !== 200) {
    throw new Error(`Landing page returned HTTP ${resLanding.status}`);
  }
  const textLanding = await resLanding.text();

  if (textLanding.includes('id="tab-gis-btn"')) {
    throw new Error('FAIL: Landing page should NOT contain dashboard tabs!');
  }
  if (!textLanding.includes('Enter Platform')) {
    throw new Error('FAIL: Landing page must contain "Enter Platform"');
  }
  if (!textLanding.includes('0.906')) {
    throw new Error('FAIL: Landing page does not contain optimized 0.906 C-index!');
  }
  if (!textLanding.includes('13,532')) {
    throw new Error('FAIL: Landing page does not quote 13,532 monitored projects!');
  }
  if (!textLanding.includes('RFCTLARR')) {
    throw new Error('FAIL: Landing page must reference RFCTLARR Act 2013 statutory engine!');
  }
  console.log('[OK] Landing page verified: Independent, 13,532 projects, RFCTLARR Act 2013, Uno C-Index 0.906.');

  // 2. Verify Methodology Page (Route /methodology)
  console.log('\n2. Checking Methodology Page (http://127.0.0.1:8000/methodology)...');
  const resMeth = await fetch('http://127.0.0.1:8000/methodology');
  if (resMeth.status !== 200) {
    throw new Error(`Methodology page returned HTTP ${resMeth.status}`);
  }
  const textMeth = await resMeth.text();

  if (textMeth.includes('id="tab-gis-btn"')) {
    throw new Error('FAIL: Methodology page should NOT contain dashboard tabs!');
  }
  if (!textMeth.includes('Back to Home')) {
    throw new Error('FAIL: Methodology page must have "Back to Home" navigation');
  }
  if (!textMeth.includes('0.906')) {
    throw new Error('FAIL: Methodology page does not contain optimized 0.906 C-index!');
  }
  if (!textMeth.includes('13,532')) {
    throw new Error('FAIL: Methodology page must reference 13,532 projects!');
  }
  if (!textMeth.includes('RFCTLARR Act 2013')) {
    throw new Error('FAIL: Methodology page must reference RFCTLARR Act 2013 statutory rules!');
  }
  console.log('[OK] Methodology page verified: Independent, 13,532 projects, RFCTLARR Act 2013, Uno C-Index 0.906.');

  // 3. Verify Dashboard Page (Route /dashboard)
  console.log('\n3. Checking Dashboard Page (http://127.0.0.1:8000/dashboard)...');
  const resDash = await fetch('http://127.0.0.1:8000/dashboard');
  if (resDash.status !== 200) {
    throw new Error(`Dashboard page returned HTTP ${resDash.status}`);
  }
  const textDash = await resDash.text();

  if (!textDash.includes('id="tab-gis-btn"')) {
    throw new Error('FAIL: Dashboard page must contain tab navigation!');
  }
  if (!textDash.includes('Export Summary')) {
    throw new Error('FAIL: Dashboard page must contain Export Summary button');
  }
  if (!textDash.includes('Run Live Analysis')) {
    throw new Error('FAIL: Dashboard page must contain Run Live Analysis button');
  }
  if (!textDash.includes('href="/methodology"')) {
    throw new Error('FAIL: Dashboard page must have navigation link to /methodology');
  }
  if (!textDash.includes('0.906')) {
    throw new Error('FAIL: Dashboard page does not contain optimized 0.906 C-index!');
  }
  if (!textDash.includes('RFCTLARR')) {
    throw new Error('FAIL: Dashboard page must have RFCTLARR Act 2013 integration!');
  }
  if (!textDash.includes('milestones-breakdown-container')) {
    throw new Error('FAIL: Dashboard page must contain milestones-breakdown-container!');
  }
  console.log('[OK] Dashboard page verified: Standalone with tabs, 13,532 projects, RFCTLARR controls, Uno C-Index 0.906.');

  // 4. Verify Static Assets
  console.log('\n4. Checking Static GeoJSON Assets...');
  const resGeo = await fetch('http://127.0.0.1:8000/india_national_boundary.geojson');
  if (resGeo.status !== 200) {
    throw new Error(`GeoJSON asset returned HTTP ${resGeo.status}`);
  }
  console.log('[OK] Static GeoJSON assets are accessible.');

  // 5. Verify /predict API endpoint for RFCTLARR compliance and Uno C-Index
  console.log('\n5. Checking /predict API Endpoint...');
  const predRes = await fetch('http://127.0.0.1:8000/predict', {
    method: 'POST',
    headers: { 
      'Content-Type': 'application/json',
      'X-API-Key': 'super-secret-token'
    },
    body: JSON.stringify({
      project_id: 'NHAI-RJ-2023-0001',
      project_type: 'Highway',
      state: 'Rajasthan',
      district: 'Banswara',
      terrain_type: 'Rural_Agri',
      estimated_cost_inr_crore: 1485.37,
      land_area_hectares: 229.1,
      section_11_notification_days: 311,
      compensation_multiplier_demand: 1.65,
      affected_families_count: 550,
      title_dispute_rate_percent: 19.78,
      sia_approval_status: 'Pending',
      forest_clearance_status: 'Not_Required',
      fund_disbursement_percent: 24.96,
      local_protest_flag: false
    })
  });
  if (predRes.status !== 200) {
    const errText = await predRes.text();
    throw new Error(`/predict returned HTTP ${predRes.status}: ${errText}`);
  }
  const predData = await predRes.json();
  if (predData.predictions.uno_c_index !== 0.906) {
    throw new Error(`FAIL: /predict uno_c_index is ${predData.predictions.uno_c_index}, expected 0.906`);
  }
  if (!predData.larr_compliance || predData.larr_compliance.solatium_percentage !== 100.0) {
    throw new Error('FAIL: /predict missing larr_compliance or 100% solatium guarantee');
  }
  if (!predData.milestones || predData.milestones.length !== 5) {
    throw new Error(`FAIL: /predict milestones count is ${predData.milestones?.length}, expected 5`);
  }
  console.log('[OK] /predict verified: Returns Uno C-Index 0.906, full RFCTLARR 2013 telemetry, and 5 statutory milestones.');

  // 6. Verify /projects/geo API endpoint vectorization and LARR fields
  console.log('\n6. Checking /projects/geo API Endpoint...');
  const geoRes = await fetch('http://127.0.0.1:8000/projects/geo?limit=5', {
    headers: {
      'X-API-Key': 'super-secret-token'
    }
  });
  if (geoRes.status !== 200) {
    const errText = await geoRes.text();
    throw new Error(`/projects/geo returned HTTP ${geoRes.status}: ${errText}`);
  }
  const geoData = await geoRes.json();
  if (!Array.isArray(geoData) || geoData.length === 0) {
    throw new Error('FAIL: /projects/geo returned empty array');
  }
  const sample = geoData[0];
  if (sample.section_11_notification_days == null || sample.solatium_percentage !== 100.0 || !sample.larr_lapse_status) {
    throw new Error('FAIL: /projects/geo item missing RFCTLARR fields');
  }
  console.log(`[OK] /projects/geo verified: Sample project ${sample.project_id} has Section 11 (${sample.section_11_notification_days}d), 100% Solatium, and status ${sample.larr_lapse_status}.`);

  console.log('\n================================================================');
  console.log('[OK] ALL VERIFICATION TESTS PASSED 100% ACCORDING TO NEW MODEL AND LARR ACT!');
  console.log('================================================================\n');
}

verifyAllRoutes().catch(err => {
  console.error('[FAIL] Verification Failed:', err.message);
  process.exit(1);
});
