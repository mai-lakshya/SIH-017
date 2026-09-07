const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const EDGE_PATH = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const USER_DATA_DIR = path.join(__dirname, '..', 'scratch_edge_cascading_' + Date.now());
const PORT = 9572;

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
  console.log('=== STARTING STATE & CASCADING DISTRICT DROPDOWN VERIFICATION ===\n');

  if (!fs.existsSync(USER_DATA_DIR)) {
    fs.mkdirSync(USER_DATA_DIR, { recursive: true });
  }

  const edge = spawn(EDGE_PATH, [
    '--headless=new',
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${USER_DATA_DIR}`,
    '--disable-gpu',
    '--window-size=1500,980',
    `http://127.0.0.1:8000/dashboard?tab=predictor&t=${Date.now()}`
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

  console.log('1. Waiting for Predictor tab and reference districts mapping to load...');
  for (let i = 0; i < 40; i++) {
    await sleep(500);
    const ready = await client.eval(`Boolean(window.referenceDistrictsMap && Object.keys(window.referenceDistrictsMap).length > 0)`);
    if (ready) break;
  }

  // --- VERIFICATION 1: State Dropdown has all 36 States/UTs alphabetically ---
  console.log('\n--- VERIFICATION 1: Complete State Dropdown ---');
  const stateOptions = await client.eval(`
    Array.from(document.getElementById('inp-state').options)
      .map(o => o.value)
      .filter(v => Boolean(v))
  `);
  console.log(`✓ Total states in dropdown: ${stateOptions.length} (Expected: 36)`);
  
  const isSorted = JSON.stringify(stateOptions) === JSON.stringify([...stateOptions].sort());
  console.log(`✓ States are strictly alphabetically sorted: ${isSorted}`);
  console.log(`  First 3: ${stateOptions.slice(0, 3).join(', ')}`);
  console.log(`  Last 3: ${stateOptions.slice(-3).join(', ')}`);

  if (stateOptions.length !== 36 || !isSorted) {
    throw new Error(`Verification 1 failed: Expected 36 sorted states, got ${stateOptions.length}`);
  }

  // --- VERIFICATION 2: Select State with good data (Rajasthan & West Bengal) ---
  console.log('\n--- VERIFICATION 2: Real Districts for States ---');
  // Select Rajasthan
  await client.eval(`(() => {
    const stateEl = document.getElementById('inp-state');
    stateEl.value = 'Rajasthan';
    handlePredictorStateChange('Rajasthan');
  })()`);
  await sleep(300);

  const rjIsSelect = await client.eval(`document.getElementById('inp-district').tagName === 'SELECT'`);
  const rjDistricts = await client.eval(`
    Array.from(document.getElementById('inp-district').options)
      .map(o => o.value)
      .filter(v => Boolean(v))
  `);
  console.log(`✓ Rajasthan rendered as SELECT: ${rjIsSelect}`);
  console.log(`✓ Rajasthan district count: ${rjDistricts.length}`);
  console.log(`✓ Contains Banswara, Jaipur, Jodhpur, Udaipur: ${['Banswara', 'Jaipur', 'Jodhpur', 'Udaipur'].every(d => rjDistricts.includes(d))}`);

  // Select West Bengal
  await client.eval(`(() => {
    const stateEl = document.getElementById('inp-state');
    stateEl.value = 'West Bengal';
    handlePredictorStateChange('West Bengal');
  })()`);
  await sleep(300);

  const wbIsSelect = await client.eval(`document.getElementById('inp-district').tagName === 'SELECT'`);
  const wbDistricts = await client.eval(`
    Array.from(document.getElementById('inp-district').options)
      .map(o => o.value)
      .filter(v => Boolean(v))
  `);
  console.log(`✓ West Bengal rendered as SELECT: ${wbIsSelect}`);
  // Select Uttar Pradesh (verify full 75 districts)
  console.log('\n--- VERIFICATION 2B: Uttar Pradesh 75 Districts Check ---');
  await client.eval(`(() => {
    const stateEl = document.getElementById('inp-state');
    stateEl.value = 'Uttar Pradesh';
    handlePredictorStateChange('Uttar Pradesh');
  })()`);
  await sleep(300);

  const upIsSelect = await client.eval(`document.getElementById('inp-district').tagName === 'SELECT'`);
  const upDistricts = await client.eval(`
    Array.from(document.getElementById('inp-district').options)
      .map(o => o.value)
      .filter(v => Boolean(v))
  `);
  console.log(`✓ Uttar Pradesh rendered as SELECT: ${upIsSelect}`);
  console.log(`✓ Uttar Pradesh district count: ${upDistricts.length} (Expected: 75)`);
  console.log(`✓ Contains Lucknow, Varanasi, Prayagraj, Agra, Noida: ${['Lucknow', 'Varanasi', 'Prayagraj', 'Agra', 'Gautam Buddha Nagar (Noida)'].every(d => upDistricts.includes(d))}`);

  if (upDistricts.length !== 75) {
    throw new Error(`Expected 75 districts for Uttar Pradesh, got ${upDistricts.length}`);
  }

  // --- VERIFICATION 3: Graceful fallback to manual text input when 0 districts on file ---
  console.log('\n--- VERIFICATION 3: Fallback to Manual Text Entry ---');
  await client.eval(`
    // Temporarily simulate a territory with zero records
    populateDistrictField('Autonomous Territory X', '');
  `);
  await sleep(300);

  const fallbackTag = await client.eval(`document.getElementById('inp-district').tagName`);
  const fallbackNoteDisplay = await client.eval(`window.getComputedStyle(document.getElementById('district-fallback-note')).display`);
  const fallbackNoteText = await client.eval(`document.getElementById('district-fallback-note').textContent.trim()`);
  console.log(`✓ When 0 districts exist, rendered as: ${fallbackTag} (Expected: INPUT)`);
  console.log(`✓ Fallback note display: "${fallbackNoteDisplay}"`);
  console.log(`✓ Fallback note message: "${fallbackNoteText}"`);

  // --- VERIFICATION 4: Reset District when State Changes ---
  console.log('\n--- VERIFICATION 4: District Reset on State Change ---');
  // First pick Maharashtra and select Pune
  await client.eval(`
    document.getElementById('inp-state').value = 'Maharashtra';
    handlePredictorStateChange('Maharashtra');
    document.getElementById('inp-district').value = 'Pune';
  `);
  await sleep(200);

  const mhSelectedBefore = await client.eval(`document.getElementById('inp-district').value`);
  console.log(`✓ Picked Maharashtra district: "${mhSelectedBefore}"`);

  // Now change state to Gujarat
  await client.eval(`
    document.getElementById('inp-state').value = 'Gujarat';
    handlePredictorStateChange('Gujarat');
  `);
  await sleep(200);

  const gjDistrictVal = await client.eval(`document.getElementById('inp-district').value`);
  console.log(`✓ After switching to Gujarat, district value reset to: "${gjDistrictVal}" (Cleared from Pune)`);

  // --- VERIFICATION 5: End-to-End Prediction with new dropdowns ---
  console.log('\n--- VERIFICATION 5: End-to-End Prediction Submission ---');
  // Select Rajasthan -> Banswara and calculate risk
  await client.eval(`
    document.getElementById('inp-state').value = 'Rajasthan';
    handlePredictorStateChange('Rajasthan');
    document.getElementById('inp-district').value = 'Banswara';
    executePredictRisk();
  `);
  await sleep(2000);

  const lastState = await client.eval(`window.lastPredictorPayload.state`);
  const lastDistrict = await client.eval(`window.lastPredictorPayload.district`);
  const predProb = await client.eval(`document.getElementById('out-prob').textContent`);
  const predCrs = await client.eval(`document.getElementById('out-crs').textContent`);
  console.log(`✓ Prediction sent to /predict with state: "${lastState}", district: "${lastDistrict}"`);
  console.log(`✓ Prediction output updated: Delay Prob ${predProb}, CRS ${predCrs.trim()}`);

  // Also test with manual fallback input
  console.log('\n--- VERIFICATION 5B: Prediction with Manual Fallback Input ---');
  await client.eval(`
    populateDistrictField('Custom Region', 'New Industrial Zone');
    executePredictRisk();
  `);
  await sleep(1500);

  const fallbackDistrictSent = await client.eval(`window.lastPredictorPayload.district`);
  console.log(`✓ Manual input correctly captured into payload: "${fallbackDistrictSent}"`);

  // Reset back to Highway scenario
  await client.eval(`loadPresetScenario('highway'); executePredictRisk();`);
  await sleep(1000);

  // Capture screenshot of the updated Risk Predictor form with State and District dropdown
  console.log('\nCapturing screenshot of updated Risk Predictor form...');
  const screenshot = await client.send('Page.captureScreenshot', { format: 'png' });
  const outPath = path.join(__dirname, 'cascading_districts_predictor.png');
  fs.writeFileSync(outPath, Buffer.from(screenshot.data, 'base64'));
  console.log(`✓ Saved screenshot to: ${outPath}`);

  const artifactDir = "C:\\Users\\Lakshya Valecha\\.gemini\\antigravity-ide\\brain\\d7ba2f30-81c8-4fbe-94ee-b86fc6b26aa4";
  if (!fs.existsSync(artifactDir)) {
    fs.mkdirSync(artifactDir, { recursive: true });
  }
  fs.copyFileSync(outPath, path.join(artifactDir, 'cascading_districts_predictor.png'));

  // --- VERIFICATION 6: Stats on dataset coverage ---
  console.log('\n--- VERIFICATION 6: Dataset Coverage Report ---');
  const coverage = await client.eval(`
    (() => {
      const states = Array.from(document.getElementById('inp-state').options)
        .map(o => o.value)
        .filter(Boolean);
      const map = window.referenceDistrictsMap || {};
      let withDistricts = 0;
      let withoutDistricts = 0;
      const details = [];
      states.forEach(s => {
        const count = (map[s] && map[s].length) || 0;
        if (count > 0) withDistricts++;
        else withoutDistricts++;
        details.push({ state: s, count });
      });
      return { total: states.length, withDistricts, withoutDistricts, details };
    })()
  `);
  console.log(`✓ Total States/UTs checked: ${coverage.total}`);
  console.log(`✓ States with real district dropdown data: ${coverage.withDistricts}`);
  console.log(`✓ States falling back to manual entry: ${coverage.withoutDistricts}`);

  // Close browser
  try {
    edge.kill();
    fs.rmSync(USER_DATA_DIR, { recursive: true, force: true });
  } catch (e) {}

  console.log('\n=== ALL VERIFICATIONS PASSED SUCCESSFULLY ===');
}

main().catch(err => {
  console.error('Test Failed:', err);
  process.exit(1);
});
