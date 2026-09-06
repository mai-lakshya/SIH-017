const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const EDGE_PATH = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const USER_DATA_DIR = path.join(__dirname, '..', 'scratch_edge_user');
const PORT = 9222;

function sleep(ms) {
  return new Promise(res => setTimeout(res, ms));
}

function getJson(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => resolve(JSON.parse(data)));
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
      this.callbacks.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
  async eval(expression) {
    const res = await this.send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    return res.result ? res.result.value : undefined;
  }
}

async function zoomTest() {
  const edge = spawn(EDGE_PATH, [
    '--headless=new',
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${USER_DATA_DIR}`,
    '--disable-gpu',
    '--window-size=1440,900',
    'about:blank'
  ]);

  let wsUrl = null;
  for (let i = 0; i < 30; i++) {
    await sleep(500);
    try {
      const list = await getJson(`http://127.0.0.1:${PORT}/json`);
      if (list && list.length > 0 && list[0].webSocketDebuggerUrl) {
        wsUrl = list[0].webSocketDebuggerUrl;
        break;
      }
    } catch (e) {}
  }
  const client = new CDPClient(wsUrl);
  await client.ready();
  await client.send('Page.enable');
  await client.send('Runtime.enable');
  await client.send('Emulation.setDeviceMetricsOverride', {
    width: 1440,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false
  });

  await client.send('Page.navigate', { url: 'http://127.0.0.1:8000/map?heatmap=1' });

  for (let i = 0; i < 30; i++) {
    await sleep(500);
    const ready = await client.eval('Boolean(window.map && window.choroplethLayer && window.stateGeoJsonData)');
    if (ready) break;
  }

  // Zoom closely on J&K/Ladakh region
  console.log('Zooming into J&K/Ladakh at zoom level 7...');
  await client.eval(`
    (() => {
      let jkLayer = null;
      window.choroplethLayer.eachLayer(l => {
        const rawName = window.getFeatureStateName ? window.getFeatureStateName(l.feature) : (l.feature.properties.ST_NM || '');
        if (rawName.includes('Jammu')) jkLayer = l;
      });
      window.map.fitBounds(jkLayer.getBounds(), { padding: [20, 20], animate: false });
      jkLayer.fire('mouseover', { target: jkLayer, latlng: jkLayer.getBounds().getCenter() });
    })()
  `);

  await sleep(1500);

  const screenshot = await client.send('Page.captureScreenshot', { format: 'png' });
  const buf = Buffer.from(screenshot.data, 'base64');
  const outPath1 = path.join(__dirname, '..', 'dashboard', 'screens', 'jk_closeup_hover.png');
  const outPath2 = 'C:\\Users\\Lakshya Valecha\\.gemini\\antigravity-ide\\brain\\4393e250-8ac9-4f01-a43a-b816b1f252ef\\jk_closeup_hover.png';

  fs.writeFileSync(outPath1, buf);
  fs.writeFileSync(outPath2, buf);
  console.log('Wrote closeup screenshot to', outPath2);
  edge.kill();
}

zoomTest().catch(console.error);
