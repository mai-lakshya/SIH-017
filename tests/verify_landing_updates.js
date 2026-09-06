const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const EDGE_PATH = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const USER_DATA_DIR = path.join(__dirname, '..', 'scratch_edge_landing_' + Date.now());
const PORT = 9588;

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
  console.log('Launching headless Edge to verify Landing page updates...');
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
      height: 2200,
      deviceScaleFactor: 1,
      mobile: false
    });

    console.log('Navigating to http://127.0.0.1:8000/ ...');
    await client.send('Page.navigate', { url: 'http://127.0.0.1:8000/' });
    await sleep(2000);

    // Verify clickable capabilities cards
    const evalCaps = await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          const gis = document.getElementById('capability-gis');
          const pred = document.getElementById('capability-predictor');
          const xai = document.getElementById('capability-xai');
          const presc = document.getElementById('capability-prescriptive');
          return {
            gis: gis ? gis.getAttribute('href') : null,
            pred: pred ? pred.getAttribute('href') : null,
            xai: xai ? xai.getAttribute('href') : null,
            presc: presc ? presc.getAttribute('href') : null,
            statesCount: document.querySelectorAll('.state-shape').length,
            hasBoundaryPath: !!document.querySelector('path[stroke="#1d1d1f"]')
          };
        })()
      `,
      returnByValue: true
    });

    console.log('Capabilities and Map Verification:', evalCaps.result.value);

    // Take screenshot of Capabilities section
    const shot = await client.send('Page.captureScreenshot', { format: 'png' });
    const imgPath = path.join(__dirname, 'landing_updated_verification.png');
    fs.writeFileSync(imgPath, Buffer.from(shot.data, 'base64'));
    console.log(`Saved full landing screenshot to: ${imgPath}`);

    client.close();
  } finally {
    edgeProc.kill('SIGKILL');
    setTimeout(() => {
      try { fs.rmSync(USER_DATA_DIR, { recursive: true, force: true }); } catch (e) {}
    }, 1000);
  }
}

run().catch(err => {
  console.error('Error running verification:', err);
  process.exit(1);
});
