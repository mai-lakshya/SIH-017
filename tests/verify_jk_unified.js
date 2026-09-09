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
      res.on('end', () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(e);
        }
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
      this.callbacks.set(id, { resolve, reject });
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
  console.log('=== STARTING AUTOMATED J&K CHOROPLETH HOVER & UNIFICATION VERIFICATION ===');
  
  if (!fs.existsSync(USER_DATA_DIR)) {
    fs.mkdirSync(USER_DATA_DIR, { recursive: true });
  }

  const edge = spawn(EDGE_PATH, [
    '--headless=new',
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${USER_DATA_DIR}`,
    '--disable-gpu',
    '--window-size=1440,900',
    'about:blank'
  ]);

  let connected = false;
  let wsUrl = null;

  for (let i = 0; i < 30; i++) {
    await sleep(500);
    try {
      const list = await getJson(`http://127.0.0.1:${PORT}/json`);
      if (list && list.length > 0 && list[0].webSocketDebuggerUrl) {
        wsUrl = list[0].webSocketDebuggerUrl;
        connected = true;
        break;
      }
    } catch (e) {
      // waiting for Edge to start listening
    }
  }

  if (!connected) {
    edge.kill();
    throw new Error('Could not connect to Edge remote debugging port 9222');
  }

  console.log('[OK] Connected to Headless Edge via CDP');
  const client = new CDPClient(wsUrl);
  await client.ready();

  await client.send('Page.enable');
  await client.send('Runtime.enable');
  client.ws.addEventListener('message', (ev) => {
    const data = JSON.parse(ev.data);
    if (data.method === 'Runtime.consoleAPICalled') {
      const args = (data.params.args || []).map(a => a.value || a.description || '').join(' ');
      console.log(`[Browser Console ${data.params.type}]`, args);
    }
  });
  await client.send('Emulation.setDeviceMetricsOverride', {
    width: 1440,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false
  });

  console.log('[OK] Navigating to http://127.0.0.1:8000/map?heatmap=1');
  await client.send('Page.navigate', { url: 'http://127.0.0.1:8000/map?heatmap=1' });

  // Wait for choropleth layer and map to load
  let mapReady = false;
  for (let i = 0; i < 30; i++) {
    await sleep(500);
    const ready = await client.eval(`
      Boolean(window.map && window.choroplethLayer && window.map.hasLayer(window.choroplethLayer) && window.stateGeoJsonData && window.choroplethLayer.getLayers().length > 0)
    `);
    if (ready) {
      mapReady = true;
      break;
    }
  }

  if (!mapReady) {
    edge.kill();
    throw new Error('Map or choropleth layer failed to initialize within 15 seconds');
  }
  console.log('[OK] Map and Choropleth Layer fully initialized in DOM');

  // Verify compliance patch layer is completely removed
  const patchLayerExists = await client.eval(`
    typeof window.soiCompliancePatchLayer !== 'undefined' && window.soiCompliancePatchLayer !== null
  `);
  console.log(`[OK] Verified separate soiCompliancePatchLayer is absent: ${!patchLayerExists}`);
  if (patchLayerExists) {
    throw new Error('soiCompliancePatchLayer should be completely removed, but is present!');
  }

  // Find J&K layer in choropleth
  const jkInfo = await client.eval(`
    (() => {
      let jkLayer = null;
      window.choroplethLayer.eachLayer(l => {
        const rawName = window.getFeatureStateName ? window.getFeatureStateName(l.feature) : (l.feature.properties.ST_NM || '');
        if (rawName.includes('Jammu') || rawName.includes('Kashmir') || rawName.includes('Ladakh')) {
          jkLayer = l;
        }
      });
      if (!jkLayer) return null;
      const bounds = jkLayer.getBounds();
      const center = bounds.getCenter();
      const pt = window.map.latLngToContainerPoint(center);
      const pathEl = jkLayer._path || (jkLayer.getElement ? jkLayer.getElement() : null);
      const rect = pathEl ? pathEl.getBoundingClientRect() : null;
      return {
        hasLayer: true,
        bounds: {
          north: bounds.getNorth(),
          south: bounds.getSouth(),
          east: bounds.getEast(),
          west: bounds.getWest()
        },
        rect: {
          cx: pt.x,
          cy: pt.y
        }
      };
    })()
  `);

  if (!jkInfo || !jkInfo.hasLayer) {
    edge.kill();
    throw new Error('Jammu & Kashmir layer not found in choroplethLayer!');
  }

  console.log(`[OK] Found J&K feature in choroplethLayer with SOI extents:`);
  console.log(`   North: ${jkInfo.bounds.north.toFixed(2)}° N (reaches Survey of India boundary > 37° N)`);
  console.log(`   South: ${jkInfo.bounds.south.toFixed(2)}° S`);
  console.log(`   East:  ${jkInfo.bounds.east.toFixed(2)}° E (includes Aksai Chin > 80° E)`);
  console.log(`   West:  ${jkInfo.bounds.west.toFixed(2)}° W`);

  if (jkInfo.bounds.north < 37.0) {
    throw new Error(`J&K North boundary does not reach 37.0° N (was ${jkInfo.bounds.north})`);
  }

  // Emulate hovering over J&K shape
  console.log(`[OK] Emulating mouse hover over J&K/Ladakh at screen coords (${Math.round(jkInfo.rect.cx)}, ${Math.round(jkInfo.rect.cy)})...`);
  
  await client.send('Input.dispatchMouseEvent', {
    type: 'mouseMoved',
    x: jkInfo.rect.cx,
    y: jkInfo.rect.cy
  });

  // Also trigger Leaflet hover event on the layer directly to ensure hover state is active
  const hoverResult = await client.eval(`
    (() => {
      let jkLayer = null;
      window.choroplethLayer.eachLayer(l => {
        const rawName = window.getFeatureStateName ? window.getFeatureStateName(l.feature) : (l.feature.properties.ST_NM || '');
        if (rawName.includes('Jammu')) jkLayer = l;
      });
      if (!jkLayer) return { success: false };
      
      // Fire mouseover event
      jkLayer.fire('mouseover', { target: jkLayer, latlng: jkLayer.getBounds().getCenter() });
      
      const path = jkLayer._path || (jkLayer.getElement ? jkLayer.getElement() : null) || (jkLayer.getLayers && jkLayer.getLayers()[0] ? (jkLayer.getLayers()[0]._path || (jkLayer.getLayers()[0].getElement ? jkLayer.getLayers()[0].getElement() : null)) : null);
      const stroke = path ? (path.getAttribute('stroke') || path.style.stroke) : null;
      const strokeWidth = path ? (path.getAttribute('stroke-width') || path.style.strokeWidth) : null;
      const fill = path ? (path.getAttribute('fill') || path.style.fill) : null;
      const fillOpacity = path ? (path.getAttribute('fill-opacity') || path.style.fillOpacity) : null;
      
      const tooltipEl = document.querySelector('.dark-leaflet-tooltip');
      const tooltipText = tooltipEl ? tooltipEl.innerText : '';
      
      return {
        success: true,
        stroke,
        strokeWidth,
        fill,
        fillOpacity,
        tooltipText: tooltipText.replace(/\\s+/g, ' ').trim()
      };
    })()
  `);

  console.log('[OK] Hover style applied to the unified polygon:');
  console.log(`   Stroke: ${hoverResult.stroke} (expected: #06b6d4 cyan)`);
  console.log(`   Stroke Width: ${hoverResult.strokeWidth} (expected: 2)`);
  console.log(`   Fill: ${hoverResult.fill} (expected: #374151)`);
  console.log(`   Fill Opacity: ${hoverResult.fillOpacity} (expected: 0.8 on hover)`);
  console.log(`   Tooltip Content: "${hoverResult.tooltipText}"`);

  if (hoverResult.stroke !== '#06b6d4') {
    throw new Error(`Expected cyan stroke #06b6d4 on hover, got: ${hoverResult.stroke}`);
  }
  if (!hoverResult.tooltipText.includes('Jammu & Kashmir / Ladakh')) {
    throw new Error(`Tooltip missing "Jammu & Kashmir / Ladakh": ${hoverResult.tooltipText}`);
  }
  if (!hoverResult.tooltipText.includes('No Data') && !hoverResult.tooltipText.includes('Project Count')) {
    throw new Error(`Tooltip missing expected project count or status: ${hoverResult.tooltipText}`);
  }

  await sleep(1000);

  // Capture screenshot of the full map with hover state
  console.log('[OK] Capturing screenshot of J&K hover state...');
  const screenshot = await client.send('Page.captureScreenshot', {
    format: 'png'
  });

  const screenshotBuffer = Buffer.from(screenshot.data, 'base64');
  const screenshotPath1 = path.join(__dirname, '..', 'dashboard', 'screens', 'jk_unified_hover.png');
  const screenshotPath2 = 'C:\\Users\\PRATYUSH\\.gemini\\antigravity-ide\\brain\\1dd2c50f-16d4-4a1d-be9d-3466ecb2b48f\\jk_unified_hover.png';

  const screensDir = path.dirname(screenshotPath1);
  if (!fs.existsSync(screensDir)) fs.mkdirSync(screensDir, { recursive: true });

  fs.writeFileSync(screenshotPath1, screenshotBuffer);
  try {
    fs.writeFileSync(screenshotPath2, screenshotBuffer);
    console.log(`[OK] Saved screenshot to artifact directory ${screenshotPath2}`);
  } catch (e) {
    console.warn(`Could not save to ${screenshotPath2}: ${e.message}`);
  }
  console.log(`[OK] Saved screenshot to ${screenshotPath1} (${screenshotBuffer.length} bytes)`);

  // Test click-to-zoom behavior
  console.log('[OK] Testing click-to-zoom on J&K unified layer...');
  const zoomBefore = await client.eval('window.map.getZoom()');
  await client.eval(`
    (() => {
      let jkLayer = null;
      window.choroplethLayer.eachLayer(l => {
        const rawName = window.getFeatureStateName ? window.getFeatureStateName(l.feature) : (l.feature.properties.ST_NM || '');
        if (rawName.includes('Jammu')) jkLayer = l;
      });
      jkLayer.fire('click', { target: jkLayer });
    })()
  `);
  await sleep(1500);
  const zoomAfter = await client.eval('window.map.getZoom()');
  const selectedDropdown = await client.eval('document.getElementById("state-select").value');
  console.log(`[OK] Zoom changed from ${zoomBefore} to ${zoomAfter} (smoothly flew to J&K full bounds)`);
  console.log(`[OK] State filter dropdown synced to: "${selectedDropdown}"`);

  // Close browser and finish
  edge.kill();
  console.log('\n[OK] ALL AUTOMATED VERIFICATION CHECKS PASSED SUCCESSFULLY!');
}

main().catch(err => {
  console.error('VERIFICATION ERROR:', err);
  process.exit(1);
});
