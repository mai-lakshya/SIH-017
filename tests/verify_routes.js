const fs = require('fs');

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function verifyAllRoutes() {
  console.log('=== VERIFYING THREE SEPARATE ROUTES & C-INDEX 0.667 ===\n');

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
  if (!textLanding.includes('Learn More')) {
    throw new Error('FAIL: Landing page must contain "Learn More"');
  }
  if (textLanding.includes('0.906')) {
    throw new Error('FAIL: Landing page still contains 0.906!');
  }
  if (!textLanding.includes('0.667')) {
    throw new Error('FAIL: Landing page does not contain verified 0.667 C-index!');
  }
  console.log('✔ Landing page is independent, has Learn More & Enter Platform, and verified 0.667 C-index.');

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
  if (textMeth.includes('0.906')) {
    throw new Error('FAIL: Methodology page still contains 0.906!');
  }
  if (!textMeth.includes('0.667')) {
    throw new Error('FAIL: Methodology page does not contain verified 0.667 C-index!');
  }
  console.log('✔ Methodology page is independent, has Back to Home, and verified 0.667 C-index.');

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
  if (textDash.includes('0.906')) {
    throw new Error('FAIL: Dashboard page still contains 0.906!');
  }
  if (!textDash.includes('0.667')) {
    throw new Error('FAIL: Dashboard page does not contain verified 0.667 C-index!');
  }
  console.log('✔ Dashboard page is standalone with tabs, View Methodology link, and verified 0.667 C-index.');

  // 4. Verify Static Assets
  console.log('\n4. Checking Static GeoJSON Assets...');
  const resGeo = await fetch('http://127.0.0.1:8000/india_national_boundary.geojson');
  if (resGeo.status !== 200) {
    throw new Error(`GeoJSON asset returned HTTP ${resGeo.status}`);
  }
  console.log('✔ Static GeoJSON assets are accessible.');

  console.log('\n✔ ALL THREE DISTINCT ROUTES AND C-INDEX VERIFICATIONS PASSED 100%!\n');
}

verifyAllRoutes().catch(err => {
  console.error('❌ Verification Failed:', err.message);
  process.exit(1);
});
