const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const EDGE_PATH = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const USER_DATA_DIR = path.join(__dirname, '..', 'scratch_edge_larr_' + Date.now());
const PORT = 9596;

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

  async ready() {
    if (this.ws.readyState === WebSocket.OPEN) return;
    return new Promise((resolve, reject) => {
      this.ws.onopen = () => resolve();
      this.ws.onerror = err => reject(err);
    });
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = this.id++;
      this.callbacks.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  close() {
    this.ws.close();
  }
}

async function run() {
  console.log('Testing Predictor and XAI views for RFCTLARR Act 2013 and new model data...');
  const edgeProc = spawn(EDGE_PATH, [
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${USER_DATA_DIR}`,
    '--headless=new',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    'about:blank'
  ]);

  try {
    let version = null;
    for (let i = 0; i < 30; i++) {
      try {
        version = await getJson(`http://127.0.0.1:${PORT}/json/version`);
        break;
      } catch (e) {
        await sleep(200);
      }
    }

    if (!version) throw new Error('Could not connect to Edge debugging port.');

    const targets = await getJson(`http://127.0.0.1:${PORT}/json/list`);
    const pageTarget = targets.find(t => t.type === 'page');
    const client = new CDPClient(pageTarget.webSocketDebuggerUrl);
    await client.ready();

    await client.send('Page.enable');
    await client.send('DOM.enable');
    await client.send('Emulation.setDeviceMetricsOverride', {
      width: 1440,
      height: 1450,
      deviceScaleFactor: 1,
      mobile: false
    });

    // 1. Navigate to Predictor tab
    await client.send('Page.navigate', { url: 'http://127.0.0.1:8000/dashboard?tab=predictor' });
    await sleep(2000);

    const shotPredictor = await client.send('Page.captureScreenshot', { format: 'png' });
    const predImgPath = path.join(__dirname, 'dashboard_predictor_larr.png');
    fs.writeFileSync(predImgPath, Buffer.from(shotPredictor.data, 'base64'));
    console.log(`Saved predictor screenshot to: ${predImgPath}`);

    // 2. Select statutory lapse scenario and calculate risk
    await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          loadPresetScenario('statutory_lapse');
          executePredictRisk();
        })()
      `
    });
    await sleep(1500);

    const shotLapse = await client.send('Page.captureScreenshot', { format: 'png' });
    const lapseImgPath = path.join(__dirname, 'dashboard_lapse_triggered.png');
    fs.writeFileSync(lapseImgPath, Buffer.from(shotLapse.data, 'base64'));
    console.log(`Saved lapse screenshot to: ${lapseImgPath}`);

    // 3. Navigate to XAI tab to verify 5 milestones & C-index
    await client.send('Runtime.evaluate', {
      expression: `switchMainTab('xai')`
    });
    await sleep(1500);

    const shotXai = await client.send('Page.captureScreenshot', { format: 'png' });
    const xaiImgPath = path.join(__dirname, 'dashboard_xai_milestones.png');
    fs.writeFileSync(xaiImgPath, Buffer.from(shotXai.data, 'base64'));
    console.log(`Saved XAI screenshot to: ${xaiImgPath}`);

    // 4. Navigate back to GIS and open drawer for MoR-WB-2021-0001
    await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          switchMainTab('gis');
          openInspectorForProject('MoR-WB-2021-0001');
        })()
      `
    });
    await sleep(2000);

    const shotDrawer = await client.send('Page.captureScreenshot', { format: 'png' });
    const drawerImgPath = path.join(__dirname, 'dashboard_drawer_larr.png');
    fs.writeFileSync(drawerImgPath, Buffer.from(shotDrawer.data, 'base64'));
    console.log(`Saved drawer screenshot to: ${drawerImgPath}`);

    client.close();
  } finally {
    edgeProc.kill('SIGKILL');
    setTimeout(() => {
      try { fs.rmSync(USER_DATA_DIR, { recursive: true, force: true }); } catch (e) {}
    }, 1000);
  }
}

run().catch(err => {
  console.error('Error running test:', err);
  process.exit(1);
});
