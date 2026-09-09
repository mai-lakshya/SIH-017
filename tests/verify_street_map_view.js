const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const EDGE_PATH = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const USER_DATA_DIR = path.join(__dirname, '..', 'scratch_edge_streetview_' + Date.now());
const PORT = 9560;

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
      }, 10000);
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
  console.log('=== VERIFYING STREET MAP VIEW AND MODAL GROUND TELEMETRY ===\n');

  if (!fs.existsSync(USER_DATA_DIR)) {
    fs.mkdirSync(USER_DATA_DIR, { recursive: true });
  }

  const edge = spawn(EDGE_PATH, [
    '--headless=new',
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${USER_DATA_DIR}`,
    '--disable-gpu',
    '--window-size=1440,900',
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

  // Wait for map and projects
  for (let i = 0; i < 40; i++) {
    await sleep(500);
    const ready = await client.eval(`Boolean(window.map && window.setBasemap && window.openStreetViewModal && window.allProjects && window.allProjects.length > 0)`);
    if (ready) break;
  }

  console.log('1. Testing Street Map View Basemap Switcher...');
  const switchResult = await client.eval(`
    (() => {
      window.setBasemap('streets');
      const activeBtn = document.querySelector('.basemap-btn.active');
      const btnId = activeBtn ? activeBtn.id : null;
      const isStreets = window.map.hasLayer(streetsLayer);
      return { btnId, isStreets };
    })()
  `);
  console.log('[OK] Switched basemap to "streets":', switchResult);
  if (!switchResult.isStreets || switchResult.btnId !== 'basemap-streets-btn') {
    edge.kill();
    throw new Error('Basemap switcher failed to activate streets layer!');
  }

  console.log('\n2. Testing Street View Modal Launch on Project...');
  const modalCheck = await client.eval(`
    (() => {
      const p = window.allProjects[0];
      window.openStreetViewModal(p.project_id);
      const modal = document.getElementById('street-view-modal');
      const isVisible = modal && modal.style.display === 'flex';
      const iframeSrc = document.getElementById('sv-modal-iframe')?.src || '';
      const title = document.getElementById('sv-modal-title')?.textContent || '';
      const extHref = document.getElementById('sv-modal-ext-link')?.href || '';
      return { isVisible, title, iframeHasOsm: iframeSrc.includes('openstreetmap.org'), extHasMaps: extHref.includes('google.com/maps') };
    })()
  `);
  console.log('[OK] Street View Modal Status:', modalCheck);
  if (!modalCheck.isVisible || !modalCheck.iframeHasOsm || !modalCheck.extHasMaps) {
    edge.kill();
    throw new Error('Street View Modal failed to open with correct embeds!');
  }

  console.log('\n3. Testing "Switch Main Map to Street Map View" Action...');
  const focusCheck = await client.eval(`
    (() => {
      window.switchMapToStreetViewAndFocus();
      const modal = document.getElementById('street-view-modal');
      return { modalClosed: modal.style.display === 'none' };
    })()
  `);
  await sleep(1800);
  const currentZoom = await client.eval(`window.map.getZoom()`);
  console.log(`[OK] Modal Closed: ${focusCheck.modalClosed}, Main Map Zoomed to: ${currentZoom}`);
  if (!focusCheck.modalClosed || currentZoom < 14) {
    edge.kill();
    throw new Error('Switch main map to street view and zoom failed!');
  }

  console.log('\n4. Capturing Street Map View Screenshot...');
  const screenshot = await client.send('Page.captureScreenshot', { format: 'png' });
  const buf = Buffer.from(screenshot.data, 'base64');
  const screenshotPath = path.join(__dirname, '..', 'dashboard', 'screens', 'street_map_view_active.png');
  fs.writeFileSync(screenshotPath, buf);
  console.log(`[OK] Saved screenshot to: ${screenshotPath} (${buf.length} bytes)`);

  edge.kill();
  console.log('\n[OK] ALL STREET MAP VIEW INTEGRATIONS VALIDATED 100%!');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
