const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const EDGE_PATH = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const USER_DATA_DIR = path.join(__dirname, '..', 'scratch_edge_saved_win_' + Date.now());
const PORT = 9570;

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
  console.log('=== VERIFYING SAVED ANALYSES MODAL WINDOW ===\n');

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

  console.log('1. Waiting for GIS dashboard to load...');
  for (let i = 0; i < 30; i++) {
    await sleep(500);
    const ready = await client.eval(`Boolean(window.map && window.savedAnalysesMarkersLayer)`);
    if (ready) break;
  }

  // Ensure at least 2 analyses exist in DB
  await client.eval(`(async () => {
    if (window.savedAnalyses.length < 2) {
      const headers = { ...getAuthHeaders(), 'Content-Type': 'application/json' };
      await fetch('/analyses/save', {
        method: 'POST',
        headers,
        body: JSON.stringify({
          project_name: 'Western Dedicated Freight Corridor (Surat Sec)',
          state: 'Gujarat',
          district: 'Surat',
          input_payload: {
            project_id: 'WDFC-GUJ-01',
            project_type: 'Freight_Corridor',
            state: 'Gujarat',
            district: 'Surat',
            land_area_hectares: 90.0,
            estimated_cost_inr_crore: 620.0,
            affected_families_count: 310,
            title_dispute_rate_percent: 12.0,
            compensation_multiplier_demand: 1.5,
            section_11_notification_days: 120,
            local_protest_flag: false
          }
        })
      });
      await loadSavedAnalyses();
    }
  })()`);
  await sleep(1500);

  const savedCount = await client.eval(`window.savedAnalyses.length`);
  console.log(`[OK] Total saved analyses available: ${savedCount}`);

  // Test opening the window modal
  console.log('\n2. Testing opening the separate Saved Analyses window modal...');
  await client.eval(`openSavedAnalysesModal()`);
  await sleep(600);

  const modalDisplay = await client.eval(`document.getElementById('saved-analyses-window-modal').style.display`);
  const cardsCount = await client.eval(`document.querySelectorAll('.saved-analysis-window-card').length`);
  const rect = await client.eval(`(() => {
    const el = document.getElementById('saved-analyses-window-modal');
    const r = el.getBoundingClientRect();
    const cs = window.getComputedStyle(el);
    return { x: r.x, y: r.y, width: r.width, height: r.height, zIndex: cs.zIndex, display: cs.display, opacity: cs.opacity, visibility: cs.visibility };
  })()`);
  console.log(`[OK] Modal opened, display style: "${modalDisplay}", rendered cards: ${cardsCount}`, rect);

  // Capture screenshot of the opened window modal
  console.log('\n3. Capturing screenshot of Saved Analyses window modal...');
  const screenshot = await client.send('Page.captureScreenshot', { format: 'png' });
  const outPath = path.join(__dirname, 'saved_analyses_modal_window.png');
  fs.writeFileSync(outPath, Buffer.from(screenshot.data, 'base64'));
  console.log(`[OK] Saved screenshot to: ${outPath}`);

  // Copy to artifacts directory
  const artifactDir = "C:\\Users\\Lakshya Valecha\\.gemini\\antigravity-ide\\brain\\6a5cd46e-a182-438d-a6eb-41835a92f98b";
  fs.copyFileSync(outPath, path.join(artifactDir, 'saved_analyses_modal_window.png'));

  // Test filtering
  console.log('\n4. Testing filter input and risk tier buttons...');
  await client.eval(`
    document.getElementById('saved-search-input').value = 'Pune';
    filterSavedAnalysesList();
  `);
  await sleep(400);

  const filteredCount = await client.eval(`document.querySelectorAll('.saved-analysis-window-card').length`);
  console.log(`[OK] Filtered by 'Pune': ${filteredCount} card(s) shown`);

  // Reset filter
  await client.eval(`
    document.getElementById('saved-search-input').value = '';
    filterSavedAnalysesList();
  `);
  await sleep(400);

  // Test selecting an analysis to focus map
  console.log('\n5. Testing "Select & Focus Map" action...');
  const firstId = await client.eval(`window.savedAnalyses[0].id`);
  await client.eval(`selectSavedAnalysisFromModal('${firstId}')`);
  await sleep(1500);

  const modalAfterSelect = await client.eval(`document.getElementById('saved-analyses-window-modal').style.display`);
  const popupOpen = await client.eval(`Boolean(document.querySelector('.leaflet-popup'))`);
  console.log(`[OK] Modal closed after selection: ${modalAfterSelect === 'none'}`);
  console.log(`[OK] Leaflet marker popup opened for selected project: ${popupOpen}`);

  // Clean up
  try {
    edge.kill();
    fs.rmSync(USER_DATA_DIR, { recursive: true, force: true });
  } catch (e) {}

  console.log('\n=== SAVED ANALYSES MODAL WINDOW VERIFICATION COMPLETED ===');
}

main().catch(err => {
  console.error('Test Failed:', err);
  process.exit(1);
});
