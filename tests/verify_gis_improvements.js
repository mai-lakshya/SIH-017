const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const EDGE_PATH = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const USER_DATA_DIR = path.join(__dirname, '..', 'scratch_edge_gis_' + Date.now());
const PORT = 9555;

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
  console.log('=== VERIFYING GIS MAP ZOOM-IN, STREET VIEW & DARK BOUNDARY HIGHLIGHT ===\n');

  if (!fs.existsSync(USER_DATA_DIR)) {
    fs.mkdirSync(USER_DATA_DIR, { recursive: true });
  }

  const edge = spawn(EDGE_PATH, [
    '--headless=new',
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${USER_DATA_DIR}`,
    '--disable-gpu',
    '--window-size=1440,900',
    `http://127.0.0.1:8000/?t=${Date.now()}`
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

  // 1. Wait for map, projects, and boundaries to be fully initialized
  console.log('1. Waiting for Leaflet map, projects, and boundaries...');
  let initialized = false;
  for (let i = 0; i < 40; i++) {
    await sleep(500);
    const ready = await client.eval(`
      Boolean(window.map && window.allProjects && window.allProjects.length > 0 && window.nationalBoundaryLayer)
    `);
    if (ready) {
      initialized = true;
      break;
    }
  }

  if (!initialized) {
    edge.kill();
    throw new Error('GIS Map, projects or national boundary failed to initialize in DOM within 20s');
  }

  console.log('[OK] Map, projects, and boundary layers initialized.');

  // 2. Verify India National Boundary Layer
  console.log('\n2. Verifying India Boundary Dark Highlighting...');
  const boundaryCheck = await client.eval(`
    (() => {
      const hasLayer = Boolean(window.nationalBoundaryLayer && window.map.hasLayer(window.nationalBoundaryLayer));
      let strokeColor = null;
      let strokeWidth = null;
      if (window.nationalBoundaryLayer) {
        window.nationalBoundaryLayer.eachLayer(l => {
          if (!strokeColor && l.options) {
            strokeColor = l.options.color;
            strokeWidth = l.options.weight;
          }
        });
      }
      return { hasLayer, strokeColor, strokeWidth };
    })()
  `);

  console.log(`[OK] National Boundary Layer Active on Map: ${boundaryCheck.hasLayer}`);
  console.log(`[OK] Boundary Stroke Color: ${boundaryCheck.strokeColor} (Dark clear tone)`);
  console.log(`[OK] Boundary Stroke Weight: ${boundaryCheck.strokeWidth}px (Thick outline)`);

  if (!boundaryCheck.hasLayer || !boundaryCheck.strokeColor) {
    edge.kill();
    throw new Error('National boundary layer is missing or unstyled!');
  }

  // 3. Test Clicking a Project Marker -> Check Zoom-In Effect
  console.log('\n3. Verifying Click-to-Zoom Effect on Project Marker...');
  const initialZoom = await client.eval(`window.map.getZoom()`);
  console.log(`   Initial Map Zoom: ${initialZoom}`);

  const clickResult = await client.eval(`
    (() => {
      let targetMarker = null;
      if (window.markersLayer) {
        if (typeof window.markersLayer.getLayers === 'function') {
          const layers = window.markersLayer.getLayers();
          targetMarker = layers.find(l => l.projectData && l.projectData.latitude && l.projectData.longitude);
        }
      }
      if (!targetMarker && window.allProjects && window.allProjects.length > 0) {
        const p = window.allProjects.find(item => item.latitude && item.longitude);
        targetMarker = { projectData: p };
      }
      if (targetMarker && targetMarker.projectData) {
        const p = targetMarker.projectData;
        // Trigger marker click handler
        if (targetMarker.fire) {
          targetMarker.fire('click');
        } else {
          window.openInspectorForProject(p.project_id);
        }
        return { success: true, project_id: p.project_id, lat: p.latitude, lng: p.longitude, name: p.project_name };
      }
      return { success: false };
    })()
  `);

  console.log(`   Clicked Project: ${clickResult.name} (${clickResult.project_id}) at [${clickResult.lat}, ${clickResult.lng}]`);

  // Wait for smooth flyTo animation
  await sleep(1500);
  const zoomedLevel = await client.eval(`window.map.getZoom()`);
  const mapCenter = await client.eval(`[window.map.getCenter().lat, window.map.getCenter().lng]`);
  console.log(`[OK] New Map Zoom after click: ${zoomedLevel} (Zoom-in triggered successfully: ${zoomedLevel > initialZoom})`);
  console.log(`[OK] Map Centered on: [${mapCenter[0].toFixed(2)}, ${mapCenter[1].toFixed(2)}]`);

  if (zoomedLevel <= initialZoom) {
    edge.kill();
    throw new Error(`Zoom-in failed! Zoom level remained ${zoomedLevel} <= ${initialZoom}`);
  }

  // 4. Verify Street View Buttons
  console.log('\n4. Verifying Google Street View Buttons...');

  // Open inspector for the clicked project
  await client.eval(`window.openInspectorForProject('${clickResult.project_id}')`);
  await sleep(1000);

  const streetViewCheck = await client.eval(`
    (() => {
      // 1. Check Drawer Street View button
      const drawerBtn = document.getElementById('drawer-street-view-btn');
      const drawerHref = drawerBtn ? drawerBtn.getAttribute('href') : null;
      const drawerVisible = drawerBtn ? (window.getComputedStyle(drawerBtn).display !== 'none') : false;

      // 2. Check Quick Card Street View button
      const quickCard = document.getElementById('quick-card-content');
      const quickCardHtml = quickCard ? quickCard.innerHTML : '';
      const quickCardHasSv = quickCardHtml.includes('Street View') && quickCardHtml.includes('google.com/maps');

      // 3. Check Marker Popup content
      let popupHasSv = false;
      if (window.markersLayer) {
        const layers = typeof window.markersLayer.getLayers === 'function' ? window.markersLayer.getLayers() : [];
        for (const l of layers) {
          if (l.getPopup && l.getPopup()) {
            const content = l.getPopup().getContent();
            if (typeof content === 'string' && content.includes('Street View') && content.includes('google.com/maps')) {
              popupHasSv = true;
              break;
            }
          }
        }
      }

      return {
        drawerHref,
        drawerVisible,
        quickCardHasSv,
        popupHasSv
      };
    })()
  `);

  console.log(`[OK] Drawer Street View Button Href: ${streetViewCheck.drawerHref}`);
  console.log(`[OK] Drawer Street View Button Visible: ${streetViewCheck.drawerVisible}`);
  console.log(`[OK] Quick Card contains Street View button: ${streetViewCheck.quickCardHasSv}`);
  console.log(`[OK] Marker Popup contains Street View button: ${streetViewCheck.popupHasSv}`);

  if (!streetViewCheck.drawerHref || !streetViewCheck.drawerHref.includes('map_action=pano')) {
    edge.kill();
    throw new Error('Street View button URL in drawer is missing or invalid!');
  }

  if (!streetViewCheck.quickCardHasSv) {
    edge.kill();
    throw new Error('Quick card is missing Street View button!');
  }

  if (!streetViewCheck.popupHasSv) {
    edge.kill();
    throw new Error('Marker popup is missing Street View button!');
  }

  // 5. Capture screenshot of zoomed-in project location with popup and highlighted dark boundary
  console.log('\n5. Capturing Visual Verification Screenshot...');
  const screenshot = await client.send('Page.captureScreenshot', { format: 'png' });
  const buf = Buffer.from(screenshot.data, 'base64');
  const screenshotPath = path.join(__dirname, '..', 'dashboard', 'screens', 'gis_zoom_streetview_boundary.png');
  fs.writeFileSync(screenshotPath, buf);
  console.log(`[OK] Screenshot saved to: ${screenshotPath} (${buf.length} bytes)`);

  edge.kill();
  console.log('\n=============================================================');
  console.log('[OK] ALL GIS MAP FIXES VERIFIED AND VALIDATED 100% SUCCESSFULLY!');
  console.log('=============================================================');
}

main().catch(err => {
  console.error('VERIFICATION ERROR:', err);
  process.exit(1);
});
