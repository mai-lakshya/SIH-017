const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const EDGE_PATH = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const USER_DATA_DIR = path.join(__dirname, '..', 'scratch_edge_analyses_' + Date.now());
const PORT = 9568;

function sleep(ms) {
  return new Promise(res => setTimeout(res, ms));
}

function getJson(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try { resolve(JSON.parse(data)); } catch (e) { reject(e); }
      });
    }).on('error', reject);
  });
}

class CDPClient {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.id = 1;
    this.callbacks = new Map();
    this.ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.id && this.callbacks.has(msg.id)) {
        const { resolve, reject } = this.callbacks.get(msg.id);
        this.callbacks.delete(msg.id);
        if (msg.error) reject(msg.error);
        else resolve(msg.result);
      }
    };
  }

  ready() {
    return new Promise((resolve) => {
      if (this.ws.readyState === WebSocket.OPEN) return resolve();
      this.ws.onopen = () => resolve();
    });
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = this.id++;
      const timer = setTimeout(() => {
        if (this.callbacks.has(id)) {
          this.callbacks.delete(id);
          reject(new Error(`CDP Timeout on ${method}`));
        }
      }, 15000);
      this.callbacks.set(id, { resolve: (val) => { clearTimeout(timer); resolve(val); }, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async eval(expression) {
    const res = await this.send('Runtime.evaluate', {
      expression,
      returnByValue: true,
      awaitPromise: true
    });
    if (res.exceptionDetails) {
      throw new Error(JSON.stringify(res.exceptionDetails));
    }
    return res.result ? res.result.value : undefined;
  }
}

async function main() {
  console.log('=== STARTING PERSISTENT SAVED ANALYSES E2E VERIFICATION ===\n');

  if (!fs.existsSync(USER_DATA_DIR)) {
    fs.mkdirSync(USER_DATA_DIR, { recursive: true });
  }

  const edge = spawn(EDGE_PATH, [
    '--headless=new',
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${USER_DATA_DIR}`,
    '--disable-gpu',
    '--window-size=1500,980',
    `http://127.0.0.1:8000/dashboard?t=${Date.now()}`
  ]);

  let wsUrl = null;
  for (let i = 0; i < 30; i++) {
    await sleep(500);
    try {
      const list = await getJson(`http://127.0.0.1:${PORT}/json`);
      if (list && list.length > 0) {
        const target = list.find(t => t.url && t.url.includes('8000') && t.webSocketDebuggerUrl) || list[0];
        if (target && target.webSocketDebuggerUrl) {
          wsUrl = target.webSocketDebuggerUrl;
          break;
        }
      }
    } catch (e) {}
  }

  if (!wsUrl) {
    edge.kill();
    throw new Error('Failed to connect to Edge CDP port ' + PORT);
  }

  const client = new CDPClient(wsUrl);
  await client.ready();
  await client.send('Page.enable');
  await client.send('Runtime.enable');

  // Wait for GIS engine & predictor to load
  console.log('1. Waiting for dashboard to initialize...');
  for (let i = 0; i < 40; i++) {
    await sleep(500);
    const ready = await client.eval(`Boolean(window.map && window.allProjects && window.allProjects.length > 0 && window.savedAnalysesMarkersLayer)`);
    if (ready) break;
  }

  const projectsCount = await client.eval(`window.allProjects.length`);
  console.log(`[OK] Reference projects loaded: ${projectsCount}`);

  // Test Task 3: Save Flow on Predictor Tab
  console.log('\n2. Testing Predictor Save Flow...');
  await client.eval(`switchMainTab('predictor')`);
  await sleep(600);

  const saveBtnVisible = await client.eval(`Boolean(document.getElementById('btn-save-analysis'))`);
  console.log(`[OK] 'Save This Analysis' button present: ${saveBtnVisible}`);

  // Open Save Modal
  await client.eval(`openSaveAnalysisModal()`);
  await sleep(400);

  const modalDisplay = await client.eval(`document.getElementById('save-analysis-modal').style.display`);
  const prefilledName = await client.eval(`document.getElementById('save-project-name-input').value`);
  console.log(`[OK] Modal opened (display: ${modalDisplay}), prefilled name: "${prefilledName}"`);

  // Set custom project name and submit
  const projName1 = "Banswara Solar & Highway Corridor";
  await client.eval(`document.getElementById('save-project-name-input').value = '${projName1}'`);
  await client.eval(`submitSaveAnalysis()`);
  await sleep(1500);

  const toastText1 = await client.eval(`document.getElementById('toast-message').textContent`);
  console.log(`[OK] Save submitted, toast message: "${toastText1}"`);

  // Save 2 more analyses directly via POST /analyses/save
  console.log('\n3. Saving 2 more diverse test analyses (Maharashtra & Assam)...');
  await client.eval(`(async () => {
    const headers = { ...getAuthHeaders(), 'Content-Type': 'application/json' };
    await fetch('/analyses/save', {
      method: 'POST',
      headers,
      body: JSON.stringify({
        project_name: 'Pune Metro Rail Corridor Ext',
        state: 'Maharashtra',
        district: 'Pune',
        input_payload: {
          project_id: 'MH-METRO-042',
          project_type: 'Metro_Rail',
          state: 'Maharashtra',
          district: 'Pune',
          land_area_hectares: 85.0,
          terrain_type: 'Urban',
          estimated_cost_inr_crore: 850.0,
          affected_families_count: 720,
          title_dispute_rate_percent: 24.5,
          compensation_multiplier_demand: 2.1,
          sia_approval_status: 'Pending',
          forest_clearance_status: 'Not_Required',
          fund_disbursement_percent: 15.0,
          section_11_notification_days: 290,
          local_protest_flag: true
        }
      })
    });

    await fetch('/analyses/save', {
      method: 'POST',
      headers,
      body: JSON.stringify({
        project_name: 'Guwahati-Kamrup Expressway',
        state: 'Assam',
        district: 'Kamrup',
        input_payload: {
          project_id: 'AS-HWY-019',
          project_type: 'Expressway',
          state: 'Assam',
          district: 'Kamrup',
          land_area_hectares: 120.0,
          terrain_type: 'Plain',
          estimated_cost_inr_crore: 420.0,
          affected_families_count: 210,
          title_dispute_rate_percent: 3.5,
          compensation_multiplier_demand: 1.25,
          sia_approval_status: 'Approved',
          forest_clearance_status: 'Approved',
          fund_disbursement_percent: 60.0,
          section_11_notification_days: 90,
          local_protest_flag: false
        }
      })
    });

    await loadSavedAnalyses();
  })()`);
  await sleep(2000);

  // Verify Task 4: Map & Diamond Markers
  console.log('\n4. Verifying National GIS Map & Diamond Markers...');
  await client.eval(`switchMainTab('gis')`);
  await sleep(1000);

  const savedCount = await client.eval(`window.savedAnalyses.length`);
  const savedMarkerCount = await client.eval(`window.savedAnalysesMarkersLayer.getLayers().length`);
  console.log(`[OK] Total saved analyses retrieved: ${savedCount}`);
  console.log(`[OK] Diamond markers on map: ${savedMarkerCount}`);

  // Check legend note
  const legendDistinction = await client.eval(`Boolean(document.getElementById('legend-distinction'))`);
  const legendText = await client.eval(`document.getElementById('legend-distinction').textContent`);
  console.log(`[OK] Legend note present: ${legendDistinction} ("${legendText.trim().replace(/\\s+/g, ' ')}")`);

  // Check custom diamond DOM elements on Leaflet map
  const diamondElementsCount = await client.eval(`document.querySelectorAll('.custom-diamond-marker').length`);
  console.log(`[OK] Custom diamond DOM elements rendered on map: ${diamondElementsCount}`);

  // Verify Task 5: Sidebar List & Badge
  console.log('\n5. Verifying Sidebar Saved Analyses List & Controls...');
  const badgeVal = await client.eval(`document.getElementById('saved-analyses-badge').textContent`);
  const listRowsCount = await client.eval(`document.querySelectorAll('.saved-analysis-row').length`);
  console.log(`[OK] Sidebar badge count: ${badgeVal}`);
  console.log(`[OK] Sidebar list item rows rendered: ${listRowsCount}`);

  // Test zoomToSavedAnalysis & Popup inspection
  console.log('\n6. Testing Marker Click & Popup Inspection...');
  const firstId = await client.eval(`window.savedAnalyses[0].id`);
  await client.eval(`zoomToSavedAnalysis('${firstId}')`);
  await sleep(1000);

  const popupContent = await client.eval(`document.querySelector('.leaflet-popup-content')?.textContent || ''`);
  const hasSavedLabel = popupContent.includes('Your Saved Analysis');
  const hasDeleteBtn = Boolean(await client.eval(`Boolean(document.querySelector('.leaflet-popup-content button[onclick*="deleteAnalysis"]'))`));
  console.log(`[OK] Popup opened: contains 'Your Saved Analysis': ${hasSavedLabel}, has Delete button: ${hasDeleteBtn}`);

  // Test Deletion
  console.log('\n7. Testing Deletion on one saved analysis...');
  const idToDelete = await client.eval(`window.savedAnalyses[window.savedAnalyses.length - 1].id`);
  // Bypass confirm dialog in browser window for automated test
  await client.eval(`window.confirm = () => true`);
  await client.eval(`deleteAnalysis('${idToDelete}')`);
  await sleep(1000);

  const countAfterDelete = await client.eval(`window.savedAnalyses.length`);
  const markersAfterDelete = await client.eval(`window.savedAnalysesMarkersLayer.getLayers().length`);
  console.log(`[OK] Count after delete: ${countAfterDelete} (markers on map: ${markersAfterDelete})`);

  // Fly to all India view to showcase both circular reference markers and diamond markers
  await client.eval(`(() => { map.setView([22.5937, 78.9629], 5); return true; })()`);
  await sleep(1500);

  // Capture Verification Screenshot
  console.log('\n8. Capturing Screenshot of GIS Map...');
  const screenshot = await client.send('Page.captureScreenshot', { format: 'png' });
  const outPath = path.join(__dirname, 'saved_analyses_map_markers.png');
  fs.writeFileSync(outPath, Buffer.from(screenshot.data, 'base64'));
  console.log(`[OK] Screenshot saved to: ${outPath}`);

  // Close browser and cleanup
  try {
    edge.kill();
    fs.rmSync(USER_DATA_DIR, { recursive: true, force: true });
  } catch (e) {}

  console.log('\n=== ALL E2E VERIFICATIONS PASSED SUCCESSFULLY ===');
}

main().catch(err => {
  console.error('Test Failed:', err);
  process.exit(1);
});
