const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const EDGE_PATH = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const USER_DATA_DIR = path.join(__dirname, '..', 'scratch_edge_dash_' + Date.now());
const PORT = 9592;

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
  console.log('Testing dashboard banner removal and landing button text...');
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
      height: 1080,
      deviceScaleFactor: 1,
      mobile: false
    });

    // 1. Check Dashboard
    await client.send('Page.navigate', { url: 'http://127.0.0.1:8000/dashboard' });
    await sleep(2500);

    const dashCheck = await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          const heroStrip = document.querySelector('.product-hero-strip');
          const heroTitle = document.querySelector('.hero-title-group');
          const gisView = document.getElementById('view-gis');
          return {
            hasHeroStrip: !!heroStrip,
            hasHeroTitle: !!heroTitle,
            gisVisible: gisView && !gisView.classList.contains('hidden')
          };
        })()
      `,
      returnByValue: true
    });
    console.log('Dashboard Check:', dashCheck.result.value);

    const dashShot = await client.send('Page.captureScreenshot', { format: 'png' });
    const dashImgPath = path.join(__dirname, 'dashboard_banner_removed.png');
    fs.writeFileSync(dashImgPath, Buffer.from(dashShot.data, 'base64'));

    // 2. Check Landing Page
    await client.send('Page.navigate', { url: 'http://127.0.0.1:8000/' });
    await sleep(1500);

    const landingCheck = await client.send('Runtime.evaluate', {
      expression: `
        (() => {
          const links = Array.from(document.querySelectorAll('a[href="/methodology"]')).map(a => a.textContent.trim());
          return { methodologyLinks: links };
        })()
      `,
      returnByValue: true
    });
    console.log('Landing Page Check:', landingCheck.result.value);

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
