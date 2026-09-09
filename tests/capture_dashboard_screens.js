const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const EDGE_PATH = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const USER_DATA_DIR = path.join(__dirname, '..', 'scratch_edge_fresh_' + Date.now());
const PORT = 9444;
const ARTIFACT_DIR = "C:\\Users\\PRATYUSH\\.gemini\\antigravity-ide\\brain\\1dd2c50f-16d4-4a1d-be9d-3466ecb2b48f";

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
      if (msg.method === 'Runtime.consoleAPICalled') {
        console.log('[BROWSER CONSOLE]', msg.params.type, msg.params.args.map(a => a.value !== undefined ? a.value : (a.description || '')).join(' '));
      }
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
          reject(new Error(`Timeout on ${method}`));
        }
      }, 10000);
      this.callbacks.set(id, {
        resolve: (val) => { clearTimeout(timer); resolve(val); },
        reject: (err) => { clearTimeout(timer); reject(err); }
      });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async eval(expr) {
    const res = await this.send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
    return res.result ? res.result.value : null;
  }
}

async function captureScreen(client, filename) {
  const screenshot = await client.send('Page.captureScreenshot', { format: 'png' });
  const buf = Buffer.from(screenshot.data, 'base64');
  fs.writeFileSync(path.join(ARTIFACT_DIR, filename), buf);
  console.log(`[OK] Saved ${filename} (${buf.length} bytes)`);
}

async function main() {
  console.log(`Starting Edge on port ${PORT}...`);
  if (!fs.existsSync(USER_DATA_DIR)) {
    fs.mkdirSync(USER_DATA_DIR, { recursive: true });
  }

  const edge = spawn(EDGE_PATH, [
    '--headless=new',
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${USER_DATA_DIR}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-gpu',
    '--window-size=1440,900',
    `http://127.0.0.1:8000/?t=${Date.now()}`
  ]);

  let wsUrl = null;
  for (let i = 0; i < 20; i++) {
    await sleep(500);
    try {
      const list = await getJson(`http://127.0.0.1:${PORT}/json`);
      if (list && list.length > 0) {
        // Pick the actual dashboard target on port 8000
        const pageTarget = list.find(t => t.url && t.url.includes('8000') && t.webSocketDebuggerUrl)
          || list.find(t => t.type === 'page' && t.webSocketDebuggerUrl)
          || list[0];
        if (pageTarget && pageTarget.webSocketDebuggerUrl) {
          wsUrl = pageTarget.webSocketDebuggerUrl;
          break;
        }
      }
    } catch (e) {}
  }

  if (!wsUrl) {
    edge.kill();
    throw new Error('Could not find WebSocket debugger URL on port ' + PORT);
  }

  console.log('[OK] Connected to Edge debugger');
  const client = new CDPClient(wsUrl);
  await client.ready();
  await client.send('Page.enable');
  await client.send('Runtime.enable');

  // Wait for initial render and geo data load
  console.log('Waiting for geo data load...');
  for (let i = 0; i < 40; i++) {
    await sleep(500);
    const count = await client.eval('typeof allProjects !== "undefined" ? allProjects.length : (window.allProjects ? window.allProjects.length : 0)');
    if (count >= 1000) {
      console.log(`[OK] Geo projects loaded in DOM: ${count} projects`);
      break;
    }
    if (i % 5 === 0) console.log(`Polling DOM projects count (iteration ${i}): ${count}`);
  }
  await sleep(2000);

  // 1. Capture GIS View
  console.log('Capturing GIS map view...');
  await captureScreen(client, 'apple_gis_map_view.png');

  // 1b. Zoom to high-density district location to demonstrate dots on same location
  console.log('Capturing district level same-location dots in Varanasi (83 projects)...');
  const zoomRes = await client.eval(`
    (() => {
      const vLat = 25.3356, vLon = 83.0076;
      if (window.map) {
        window.map.setView([vLat, vLon], 13, { animate: false });
        if (window.markersLayer && typeof window.markersLayer.eachLayer === 'function') {
          let opened = false;
          window.markersLayer.eachLayer(m => {
            if (!opened && m.getLatLng) {
              const ll = m.getLatLng();
              const dist = Math.hypot(ll.lat - vLat, ll.lng - vLon);
              if (dist < 0.05) {
                m.openPopup();
                opened = true;
              }
            }
          });
        }
      }
      return { zoomed: true, lat: vLat, lon: vLon };
    })()
  `);
  console.log('Zoom result:', zoomRes);
  await sleep(3000);
  await captureScreen(client, 'apple_gis_district_cluster_view.png');

  // Reset map view
  await client.eval('(() => { if (window.map) window.map.setView([22.5937, 78.9629], 5, { animate: false }); return true; })()');
  await sleep(800);

  // 2. Open Project Inspector Drawer
  console.log('Opening project inspector drawer...');
  await client.eval(`
    (() => {
      const drawer = document.getElementById('project-inspector');
      if (drawer) drawer.classList.add('open');
      const id = (typeof allProjects !== 'undefined' && allProjects[0]) ? allProjects[0].project_id : ((window.allProjects && window.allProjects[0]) ? window.allProjects[0].project_id : 'NHAI-MH-001');
      if (typeof window.openInspectorForProject === 'function') {
        window.openInspectorForProject(id);
      }
      return true;
    })()
  `);
  await sleep(1500);
  await captureScreen(client, 'apple_gis_inspector_view.png');

  // 3. Switch to Predictor tab
  console.log('Switching to Risk Predictor tab...');
  await client.eval('(() => { closeInspectorDrawer(); switchMainTab("predictor"); return true; })()');
  await sleep(1000);
  await captureScreen(client, 'apple_risk_predictor_view.png');

  // 4. Switch to XAI tab
  console.log('Switching to XAI & Survival tab...');
  await client.eval('(() => { switchMainTab("xai"); return true; })()');
  await sleep(1200);
  await captureScreen(client, 'apple_xai_survival_view.png');

  // 5. Switch to Prescriptive Mitigations tab
  console.log('Switching to Prescriptive AI tab...');
  await client.eval('(() => { switchMainTab("mitigations"); return true; })()');
  await sleep(1000);
  await captureScreen(client, 'apple_prescriptive_ai_view.png');

  edge.kill();
  console.log('\n[OK] ALL 5 APPLE DASHBOARD VIEWS CAPTURED SUCCESSFULLY!');
}

main().catch(err => {
  console.error('Execution error:', err);
  process.exit(1);
});
