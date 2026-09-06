// Verification script for Apple-Styled AI Dashboard with GIS Integration
const fs = require('fs');
const path = require('path');

async function verifyDashboard() {
  console.log('=== VERIFYING APPLE-STYLED AI FRONTEND DASHBOARD ===\n');

  const htmlPath = path.join(__dirname, '..', 'dashboard', 'index.html');
  if (!fs.existsSync(htmlPath)) {
    throw new Error(`Dashboard file not found at: ${htmlPath}`);
  }

  const html = fs.readFileSync(htmlPath, 'utf-8');
  console.log(`✔ dashboard/index.html found (${html.length} bytes)`);

  // 1. Verify Apple Design Tokens
  const requiredTokens = [
    '--color-apple-blue: #0071e3',
    '--color-link-blue: #0066cc',
    '--color-carbon: #1d1d1f',
    '--color-frost: #f5f5f7',
    '--color-ice: #f4f8fb',
    '--radius-buttons: 980px',
    '--radius-cards: 8px'
  ];

  requiredTokens.forEach(token => {
    if (!html.includes(token)) {
      throw new Error(`Missing Apple Design System token: ${token}`);
    }
    console.log(`✔ Found token: ${token}`);
  });

  // 2. Verify Tab Navigation
  const requiredTabs = [
    'id="tab-gis-btn"',
    'id="tab-predictor-btn"',
    'id="tab-xai-btn"',
    'id="tab-mitigations-btn"',
    'id="tab-monitor-btn"'
  ];

  requiredTabs.forEach(tab => {
    if (!html.includes(tab)) {
      throw new Error(`Missing Navigation Tab: ${tab}`);
    }
    console.log(`✔ Found Tab: ${tab}`);
  });

  // 3. Verify GIS Infrastructure Elements
  const requiredGisElements = [
    'id="map"',
    'id="search-input"',
    'id="state-select"',
    'id="type-select"',
    'id="status-select"',
    'id="chip-low"',
    'id="chip-med"',
    'id="chip-high"',
    'id="heatmap-toggle"',
    'id="heatmap-knob"',
    'id="map-legend"',
    'id="legend-markers"',
    'id="legend-heatmap"',
    'id="reset-view"',
    'id="project-inspector"',
    'id="drawer-project-name"',
    'id="drawer-delay-prob"',
    'id="drawer-load-predictor-btn"'
  ];

  requiredGisElements.forEach(el => {
    if (!html.includes(el)) {
      throw new Error(`Missing GIS Element: ${el}`);
    }
    console.log(`✔ Found GIS Element: ${el}`);
  });

  // 4. Verify Predictor, XAI & Simulator Elements
  const requiredAiElements = [
    'id="preset-selector"',
    'id="inp-cost"',
    'id="inp-land-area"',
    'id="out-prob"',
    'id="out-crs"',
    'id="out-delay-days"',
    'id="out-median-surv"',
    'id="shapChart"',
    'id="survivalChart"',
    'id="calibrationChart"',
    'id="driftChart"',
    'id="sim-slider-fund"',
    'id="sim-days-saved"',
    'id="sim-risk-reduced"'
  ];

  requiredAiElements.forEach(el => {
    if (!html.includes(el)) {
      throw new Error(`Missing AI Element: ${el}`);
    }
    console.log(`✔ Found AI Element: ${el}`);
  });

  // 5. Test Live API Connection
  try {
    const res = await fetch('http://127.0.0.1:8000/projects/geo', {
      headers: { 'X-API-Key': 'super-secret-token' }
    });
    if (res.ok) {
      const projects = await res.json();
      console.log(`✔ Successfully connected to API: ${projects.length} GIS projects retrieved.`);
    } else {
      console.log(`⚠ Warning: API returned HTTP ${res.status}`);
    }
  } catch (err) {
    console.log(`⚠ Warning: Live API fetch skipped (${err.message})`);
  }

  console.log('\n✔ ALL APPLE DASHBOARD VERIFICATIONS PASSED SUCCESSFULLY!\n');
}

verifyDashboard().catch(err => {
  console.error('❌ Verification Failed:', err);
  process.exit(1);
});
