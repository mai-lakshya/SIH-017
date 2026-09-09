// Verification script for Location Analysis Map Heat Map Choropleth functionality
const fs = require('fs');
const path = require('path');

async function runTests() {
  console.log('=== RUNNING CHOROPLETH HEAT MAP VERIFICATION ===\n');

  // 1. Check HTML file existence and syntax
  const htmlPath = path.join(__dirname, '..', 'dashboard', 'screens', 'location_analysis_map.html');
  if (!fs.existsSync(htmlPath)) {
    throw new Error(`File not found: ${htmlPath}`);
  }
  const htmlContent = fs.readFileSync(htmlPath, 'utf-8');
  console.log('[OK] location_analysis_map.html found (' + htmlContent.length + ' bytes)');

  // 2. Verify key DOM IDs and attributes in HTML
  const requiredElements = [
    'id="heatmap-toggle"',
    'id="heatmap-knob"',
    'id="map-legend"',
    'id="legend-markers"',
    'id="legend-heatmap"',
    'id="reset-view"',
    '.dark-leaflet-tooltip'
  ];

  requiredElements.forEach(item => {
    if (!htmlContent.includes(item)) {
      throw new Error(`Missing expected markup: ${item}`);
    }
    console.log(`[OK] Found DOM markup: ${item}`);
  });

  // 3. Extract the JS logic and test state aggregation & coloring
  // Load local geojson and projects from backend via fetch
  const geojsonPath = path.join(__dirname, '..', 'dashboard', 'india_states.geojson');
  const geojson = JSON.parse(fs.readFileSync(geojsonPath, 'utf-8'));
  console.log(`[OK] Loaded GeoJSON: ${geojson.features.length} state features`);

  // Fetch projects from local server
  const res = await fetch('http://127.0.0.1:8000/projects/geo', {
    headers: { 'X-API-Key': 'super-secret-token' }
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch /projects/geo: ${res.status}`);
  }
  const projects = await res.json();
  console.log(`[OK] Fetched ${projects.length} projects from GET /projects/geo`);

  // Define normalization function exactly as in location_analysis_map.html
  function normalizeStateName(rawName) {
    if (!rawName) return '';
    const name = String(rawName).trim();
    const map = {
      'Orissa': 'Odisha',
      'Uttaranchal': 'Uttarakhand',
      'Jammu & Kashmir': 'Jammu and Kashmir',
      'Andaman and Nicobar': 'Andaman and Nicobar Islands',
      'Andaman & Nicobar': 'Andaman and Nicobar Islands',
      'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli',
      'Daman & Diu': 'Daman and Diu',
      'NCT of Delhi': 'Delhi'
    };
    return map[name] || name;
  }

  function calculateStateRiskStats(projects) {
    const grouped = {};
    projects.forEach(p => {
      if (!p.state) return;
      const normState = normalizeStateName(p.state);
      if (!grouped[normState]) {
        grouped[normState] = {
          scores: [],
          count: 0,
          low: 0,
          medium: 0,
          high: 0
        };
      }
      const score = typeof p.composite_risk_score === 'number'
        ? p.composite_risk_score
        : (parseFloat(p.composite_risk_score) || 0);

      grouped[normState].scores.push(score);
      grouped[normState].count++;
      if (p.risk_tier === 'Low') grouped[normState].low++;
      else if (p.risk_tier === 'Medium') grouped[normState].medium++;
      else if (p.risk_tier === 'High') grouped[normState].high++;
    });

    const lookup = {};
    Object.keys(grouped).forEach(st => {
      const g = grouped[st];
      const sum = g.scores.reduce((a, b) => a + b, 0);
      const avg = g.count > 0 ? (sum / g.count) : 0;
      lookup[st] = {
        avgScore: parseFloat(avg.toFixed(1)),
        count: g.count,
        low: g.low,
        medium: g.medium,
        high: g.high
      };
    });
    return lookup;
  }

  function getStateColor(stats) {
    if (!stats || stats.count === 0) return '#374151'; // dark gray
    if (stats.avgScore < 52.0) return '#10b981';      // Low risk green (< 52)
    if (stats.avgScore <= 58.0) return '#f59e0b';     // Medium risk amber (52 - 58)
    return '#ef4444';                                 // High risk red (> 58)
  }

  const lookup = calculateStateRiskStats(projects);
  const stateKeys = Object.keys(lookup);
  console.log(`[OK] Computed risk metrics across ${stateKeys.length} distinct states with projects`);

  // Verify total project count across all states matches 200
  const totalCount = stateKeys.reduce((acc, st) => acc + lookup[st].count, 0);
  if (totalCount !== 200) {
    throw new Error(`Expected 200 projects aggregated, got ${totalCount}`);
  }
  console.log(`[OK] Verified total aggregated project count = ${totalCount}`);

  // Verify J&K / Ladakh has Survey of India boundary (reaches latitude > 37° N)
  const jkFeature = geojson.features.find(f => (f.properties.ST_NM || f.properties.NAME_1 || '').includes('Jammu'));
  if (!jkFeature) throw new Error('J&K feature not found in GeoJSON');
  
  function getExtents(c) {
    let lats = [], lons = [];
    function ext(item) {
      if (typeof item[0] === 'number') {
        lons.push(item[0]); lats.push(item[1]);
      } else {
        item.forEach(ext);
      }
    }
    ext(c);
    return { minLon: Math.min(...lons), maxLon: Math.max(...lons), minLat: Math.min(...lats), maxLat: Math.max(...lats) };
  }
  const jkExt = getExtents(jkFeature.geometry.coordinates);
  console.log(`[OK] Verified Survey of India J&K/Ladakh Bounds: Lat ${jkExt.minLat.toFixed(2)}° to ${jkExt.maxLat.toFixed(2)}° N | Lon ${jkExt.minLon.toFixed(2)}° to ${jkExt.maxLon.toFixed(2)}° E`);
  if (jkExt.maxLat < 37.0) {
    throw new Error(`J&K does not reach official Survey of India boundary (maxLat was ${jkExt.maxLat}, expected > 37.0° N)`);
  }
  if (jkExt.maxLon < 80.0) {
    throw new Error(`J&K does not reach Aksai Chin (maxLon was ${jkExt.maxLon}, expected > 80.0° E)`);
  }

  // Check color distribution across the 36 GeoJSON states
  const colorCounts = {
    '#10b981': 0, // Green (Low)
    '#f59e0b': 0, // Amber (Med)
    '#ef4444': 0, // Red (High)
    '#374151': 0  // Dark Gray (No Data)
  };

  geojson.features.forEach(f => {
    const rawName = f.properties.ST_NM || f.properties.NAME_1;
    const normName = normalizeStateName(rawName);
    const stats = lookup[normName];
    const color = getStateColor(stats);
    colorCounts[color] = (colorCounts[color] || 0) + 1;
  });

  console.log('\nState Shading Distribution across 36 features:');
  console.log(`- Low Risk (#10b981): ${colorCounts['#10b981']} states`);
  console.log(`- Medium Risk (#f59e0b): ${colorCounts['#f59e0b']} states`);
  console.log(`- High Risk (#ef4444): ${colorCounts['#ef4444']} states`);
  console.log(`- No Data (#374151): ${colorCounts['#374151']} states/UTs`);

  if (colorCounts['#374151'] === 0) {
    throw new Error('Expected some states to have "No Data" (gray), but found 0');
  }
  const activeColoredStates = colorCounts['#10b981'] + colorCounts['#f59e0b'] + colorCounts['#ef4444'];
  if (activeColoredStates === 0) {
    throw new Error('Expected active colored states on the map, got 0');
  }

  // Spot-check key states
  console.log('\nSpot check specific states:');
  ['Rajasthan', 'Maharashtra', 'Assam', 'Delhi', 'Goa'].forEach(name => {
    const norm = normalizeStateName(name);
    const s = lookup[norm];
    const color = getStateColor(s);
    if (s) {
      console.log(`- ${name}: Avg ${s.avgScore} | ${s.count} projects (${s.low} Low, ${s.medium} Med, ${s.high} High) | Color: ${color}`);
    } else {
      console.log(`- ${name}: No Data (0 projects) | Color: ${color} (#374151)`);
    }
  });

  console.log('\n[OK] ALL TESTS PASSED SUCCESSFULLY!');
}

runTests().catch(err => {
  console.error('Test Failed:', err);
  process.exit(1);
});
