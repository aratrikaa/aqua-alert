/**
 * app.js -- AquaAlert Frontend
 * ============================
 * Fetches data from the Flask API and renders:
 *  - Leaflet.js interactive map with GeoJSON district outlines (choropleth)
 *    + custom risk pin markers on top
 *  - Sidebar district list with filter buttons
 *  - Detail panel with SHAP explanations + Chart.js sparklines
 *  - Alerts tab with Critical districts sorted by risk score
 *
 * BUG FIXES vs v1:
 *  - Marker click no longer conflicts with popup (popup opens on click;
 *    "View Full Analysis" button inside popup opens the detail panel).
 *  - selectDistrict() is only triggered by sidebar / popup button / alert card.
 *  - GeoJSON district boundary layer provides choropleth colouring.
 *  - Removed arrow unicode that caused cp1252 crash.
 */

'use strict';

// ── State ──────────────────────────────────────────────────────────────────────
let allDistricts   = [];
let markerMap      = {};   // district_id -> Leaflet marker
let geoLayerMap    = {};   // district_id -> Leaflet GeoJSON layer
let activeDistId   = null;
let leafletMap     = null;
let depthChart     = null;
let currentFilter  = 'all';
let districtGeoJSON = null;  // raw GeoJSON FeatureCollection

// ── Colour helpers ─────────────────────────────────────────────────────────────
const RISK_COLOR = {
  Critical: '#f43f5e',
  Moderate: '#f59e0b',
  Safe:     '#22d3a0',
};

const RISK_FILL_OPACITY = {
  Critical: 0.35,
  Moderate: 0.28,
  Safe:     0.22,
};

function riskClass(label) {
  return label.toLowerCase();
}

// ── Initialisation ─────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  initMap();
  showLoadingOverlay('Warming up AI pipeline\u2026', true);
  await waitForBackend();
  hideLoadingOverlay();
  await loadDistricts();
  setStatus('online', `Live \u2014 ${allDistricts.length} districts`);
});

/**
 * Poll /api/health until the backend is ready (model loaded).
 * Shows a warm-up message with elapsed seconds so the user
 * knows the app is working, not broken.
 */
async function waitForBackend(maxWaitMs = 120_000, intervalMs = 2000) {
  const start = Date.now();
  while (Date.now() - start < maxWaitMs) {
    try {
      const res = await fetch('/api/health');
      if (res.ok) return;   // 200 = ready
      const elapsed = Math.round((Date.now() - start) / 1000);
      showLoadingOverlay(
        `Warming up AI pipeline\u2026 (${elapsed}s)\n` +
        'Model loading on Render free tier \u2014 this takes ~30\u201360s on a cold start.',
        true
      );
    } catch (_) {
      // network not ready yet
    }
    await new Promise(r => setTimeout(r, intervalMs));
  }
  // Timed out — try anyway
}

function showLoadingOverlay(message, show) {
  let overlay = document.getElementById('loading-overlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'loading-overlay';
    document.body.appendChild(overlay);
  }
  const [line1, line2] = message.split('\n');
  overlay.innerHTML = `
    <div class="loader-logo">AquaAlert</div>
    <div class="loader-sub">${line1}</div>
    ${line2 ? `<div class="loader-hint">${line2}</div>` : ''}
    <div class="spinner"></div>
  `;
  overlay.classList.toggle('hidden', !show);
}

function hideLoadingOverlay() {
  const overlay = document.getElementById('loading-overlay');
  if (overlay) {
    overlay.classList.add('hidden');
    setTimeout(() => overlay.remove(), 600);
  }
}

function setStatus(state, text) {
  const dot  = document.getElementById('status-dot');
  const span = document.getElementById('status-text');
  dot.className    = 'status-dot ' + (state === 'online' ? 'online' : '');
  span.textContent = text;
}

// ── Map initialisation ─────────────────────────────────────────────────────────

function initMap() {
  leafletMap = L.map('map', {
    center: [24.5, 77.5],   // centred over India
    zoom: 5,
    zoomControl: false,
    attributionControl: true,
  });

  // OpenStreetMap standard tiles -- completely free, no API key ever needed.
  // We apply a CSS dark-mode invert filter (class: osm-dark) so it matches
  // the dashboard's dark theme without any paid tile provider.
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19,
    className: 'osm-dark',  // applied to every tile <img>
  }).addTo(leafletMap);

  L.control.zoom({ position: 'bottomright' }).addTo(leafletMap);
}

// ── Data loading ───────────────────────────────────────────────────────────────

async function loadDistricts() {
  try {
    const res    = await fetch('/api/districts');
    allDistricts = await res.json();

    await buildDistrictLayers(allDistricts);   // GeoJSON choropleth
    buildSidebarList(allDistricts);
    buildMapMarkers(allDistricts);
    updateStats(allDistricts);
    loadAlerts();
    applyURLState();        // deep-link: auto-open ?district=ID if present
    loadFeatureImportance(); // about tab SHAP chart
  } catch (err) {
    console.error('Failed to load districts:', err);
    setStatus('error', 'API error');
  }
}

// ── GeoJSON choropleth district boundary layer ────────────────────────────────

/**
 * Load districts.geojson and draw filled polygons on the map,
 * coloured by risk level. Falls back gracefully if the file fails to load.
 */
async function buildDistrictLayers(districts) {
  // Remove old layers
  Object.values(geoLayerMap).forEach(l => l.remove());
  geoLayerMap = {};

  // Build a risk lookup
  const riskMap = {};
  districts.forEach(d => { riskMap[d.district_id] = d; });

  // Fetch GeoJSON (only once)
  if (!districtGeoJSON) {
    try {
      const res = await fetch('/districts.geojson');
      districtGeoJSON = await res.json();
    } catch (err) {
      console.warn('Could not load districts.geojson, skipping choropleth.', err);
      return;
    }
  }

  districtGeoJSON.features.forEach(feature => {
    const distId = feature.properties.district_id;
    const d      = riskMap[distId];
    if (!d) return;

    const color   = RISK_COLOR[d.risk_label]   || '#888';
    const opacity = RISK_FILL_OPACITY[d.risk_label] || 0.2;

    const layer = L.geoJSON(feature, {
      style: {
        color:       color,
        weight:      1.5,
        opacity:     0.7,
        fillColor:   color,
        fillOpacity: opacity,
        dashArray:   '4 3',
      }
    }).addTo(leafletMap);

    // Clicking the polygon opens the marker popup
    layer.on('click', (e) => {
      L.DomEvent.stopPropagation(e);
      const marker = markerMap[distId];
      if (marker) marker.openPopup();
    });

    // Hover highlight
    layer.on('mouseover', () => layer.setStyle({ fillOpacity: opacity + 0.2, weight: 2.5 }));
    layer.on('mouseout',  () => layer.setStyle({ fillOpacity: opacity,       weight: 1.5 }));

    geoLayerMap[distId] = layer;
  });
}

// ── Sidebar ────────────────────────────────────────────────────────────────────

function buildSidebarList(districts) {
  const list = document.getElementById('district-list');
  list.innerHTML = '';

  // Sort: Critical first, then Moderate, then Safe
  const order = { Critical: 0, Moderate: 1, Safe: 2 };
  const sorted = [...districts].sort((a, b) =>
    order[a.risk_label] - order[b.risk_label] ||
    b.risk_score - a.risk_score
  );

  sorted.forEach(d => {
    const item = document.createElement('div');
    item.className  = 'district-item';
    item.id         = `list-item-${d.district_id}`;
    item.dataset.risk = d.risk_label;

    item.innerHTML = `
      <div class="di-dot ${riskClass(d.risk_label)}"></div>
      <div class="di-info">
        <div class="di-name">${d.district_name}</div>
        <div class="di-state">${d.state}</div>
      </div>
      <div class="di-risk ${riskClass(d.risk_label)}">${d.risk_label}</div>
    `;

    // Sidebar click: fly to marker, open popup, load detail
    item.addEventListener('click', () => {
      const marker = markerMap[d.district_id];
      const dist   = allDistricts.find(x => x.district_id === d.district_id);
      if (dist) leafletMap.flyTo([dist.lat, dist.lon], 8, { duration: 1.0 });
      if (marker) setTimeout(() => marker.openPopup(), 1100);
      loadDetailPanel(d.district_id);
    });

    list.appendChild(item);
  });
}

// ── Map markers (pins) ─────────────────────────────────────────────────────────

function buildMapMarkers(districts) {
  Object.values(markerMap).forEach(m => m.remove());
  markerMap = {};

  districts.forEach(d => {
    const cls  = riskClass(d.risk_label);
    const icon = L.divIcon({
      className: '',
      html:      `<div class="custom-marker ${cls}"></div>`,
      iconSize:   [36, 36],
      iconAnchor: [18, 36],
      popupAnchor:[0, -38],
    });

    const marker = L.marker([d.lat, d.lon], { icon })
      .addTo(leafletMap)
      .bindPopup(buildPopupHTML(d), {
        maxWidth:  260,
        className: 'aqua-popup',
        // Keep popup open when user moves mouse into it
        closeButton: true,
        autoClose:   false,
      });

    // !! Do NOT call selectDistrict here -- it would close the popup
    // immediately. The popup's "View Analysis" button handles that.
    markerMap[d.district_id] = marker;
  });
}

function buildPopupHTML(d) {
  const cls = riskClass(d.risk_label);
  // Use HTML entities / ASCII only -- no raw Unicode arrows (cp1252 safety)
  return `
    <div class="popup-inner">
      <div class="popup-name">${d.district_name}</div>
      <div class="popup-state">${d.state}</div>
      <div class="popup-badge ${cls}">${d.risk_label}</div>
      <div class="popup-conf">Confidence: <strong>${d.confidence}%</strong></div>
      <div class="popup-conf" style="margin-top:4px;font-size:0.7rem;color:var(--text-muted)">
        Risk Score: ${Math.round(d.risk_score * 100)}%
      </div>
      <button class="popup-detail-btn"
              id="popup-btn-${d.district_id}"
              onclick="selectDistrict('${d.district_id}')">
        View Full Analysis &#8594;
      </button>
    </div>
  `;
}

// ── Stats bar ──────────────────────────────────────────────────────────────────

function updateStats(districts) {
  const counts = { Critical: 0, Moderate: 0, Safe: 0 };
  districts.forEach(d => counts[d.risk_label]++);

  document.getElementById('stat-critical').textContent = counts.Critical;
  document.getElementById('stat-moderate').textContent = counts.Moderate;
  document.getElementById('stat-safe').textContent     = counts.Safe;
  document.getElementById('stat-total').textContent    = districts.length;
  document.getElementById('alert-badge').textContent   = counts.Critical;
}

// ── Filter ─────────────────────────────────────────────────────────────────────

function filterMarkers(risk) {
  currentFilter = risk;

  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.filter === risk);
  });

  allDistricts.forEach(d => {
    const show   = (risk === 'all') || (d.risk_label === risk);
    const item   = document.getElementById(`list-item-${d.district_id}`);
    const marker = markerMap[d.district_id];
    const layer  = geoLayerMap[d.district_id];

    if (item) item.classList.toggle('hidden', !show);

    if (marker) {
      if (show) { if (!leafletMap.hasLayer(marker)) marker.addTo(leafletMap); }
      else      { marker.remove(); }
    }
    if (layer) {
      if (show) { if (!leafletMap.hasLayer(layer)) layer.addTo(leafletMap); }
      else      { layer.remove(); }
    }
  });
}

// ── District selection (called by popup button, sidebar click, alert card) ─────

/**
 * selectDistrict -- opens the right-side detail panel for a district.
 * Does NOT close any popup (that's now the user's choice via the X button).
 */
async function selectDistrict(districtId) {
  activeDistId = districtId;

  // Highlight sidebar row
  document.querySelectorAll('.district-item').forEach(el => el.classList.remove('active'));
  const li = document.getElementById(`list-item-${districtId}`);
  if (li) {
    li.classList.add('active');
    li.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  // Show detail panel with loading state
  document.getElementById('detail-placeholder').classList.add('hidden');
  const content = document.getElementById('detail-content');
  content.classList.remove('hidden');
  content.innerHTML = `
    <div style="padding:40px;text-align:center;color:var(--text-muted);">
      <div class="spinner" style="margin:0 auto 12px"></div>
      Loading analysis...
    </div>
  `;

  await loadDetailPanel(districtId);
}

async function loadDetailPanel(districtId) {
  // Highlight sidebar item
  document.querySelectorAll('.district-item').forEach(el => el.classList.remove('active'));
  const li = document.getElementById(`list-item-${districtId}`);
  if (li) { li.classList.add('active'); li.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }

  document.getElementById('detail-placeholder').classList.add('hidden');
  document.getElementById('detail-content').classList.remove('hidden');

  try {
    const [detail, chartData] = await Promise.all([
      fetch(`/api/district/${districtId}`).then(r => r.json()),
      fetch(`/api/chart/${districtId}`).then(r => r.json()),
    ]);
    renderDetail(detail);
    renderChart(chartData, detail.risk_label);
  } catch (err) {
    console.error('Failed to load district detail:', err);
    document.getElementById('detail-content').innerHTML =
      `<div style="padding:24px;color:var(--critical-color)">Failed to load data.</div>`;
  }
}

// ── Detail Panel rendering ─────────────────────────────────────────────────────

function renderDetail(d) {
  const cls  = riskClass(d.risk_label);
  const expl = d.explanation;
  const topF = expl.top_features || [];

  // SHAP bars
  const maxSHAP = Math.max(...topF.map(f => Math.abs(f.shap_value)), 0.001);
  const shapRows = topF.map(f => {
    const pct    = Math.round((Math.abs(f.shap_value) / maxSHAP) * 100);
    const barCls = f.shap_value > 0 ? 'pos' : 'neg';
    return `
      <div class="shap-feature">
        <div class="shap-name">${f.label}</div>
        <div class="shap-bar-wrap">
          <div class="shap-bar ${barCls}" style="width:${pct}%"></div>
        </div>
        <div class="shap-val">${f.shap_value > 0 ? '+' : ''}${f.shap_value.toFixed(3)}</div>
      </div>
    `;
  }).join('');

  // Feature chips
  const feats    = d.features;
  const chipData = [
    { label: 'Depth (mbgl)',      value: feats.current_depth.toFixed(1) + ' m' },
    { label: 'Annual Trend',      value: (feats.annual_trend > 0 ? '+' : '') + feats.annual_trend.toFixed(2) + ' m/yr' },
    { label: 'Extraction Rate',   value: (feats.extraction_rate * 100).toFixed(0) + '%' },
    { label: '90d Rainfall',      value: feats.rainfall_lag_90d.toFixed(1) + ' mm' },
    { label: 'Pre-Monsoon Depth', value: feats.premonsoon_depth.toFixed(1) + ' m' },
    { label: 'Monsoon Recovery',  value: feats.monsoon_recovery.toFixed(2) + ' m' },
  ];
  const chips = chipData.map(c => `
    <div class="feature-chip">
      <div class="fc-label">${c.label}</div>
      <div class="fc-value">${c.value}</div>
    </div>
  `).join('');

  document.getElementById('detail-content').innerHTML = `
    <div class="detail-header">
      <div class="detail-district">${d.district_name}</div>
      <div class="detail-state">${d.state}</div>
    </div>

    <div class="risk-badge-large ${cls}">
      <div class="pulse-ring"></div>
      ${d.risk_label} Risk
    </div>

    <div class="confidence-row">
      <span class="conf-label">Confidence</span>
      <div class="conf-bar-wrap">
        <div class="conf-bar" style="width:${d.confidence}%"></div>
      </div>
      <span class="conf-value">${d.confidence}%</span>
    </div>

    <div class="section-label">AI Explanation (SHAP)</div>
    <div class="explanation-box">
      <div class="explanation-summary">${expl.summary}</div>
      ${shapRows}
    </div>

    <div class="section-label">Key Indicators</div>
    <div class="feature-grid">${chips}</div>

    <div class="section-label">Water Table Depth (24 months)</div>
    <div class="chart-wrap">
      <canvas id="depth-chart"></canvas>
    </div>
  `;
}

// ── Chart.js sparkline ─────────────────────────────────────────────────────────

function renderChart(data, riskLabel) {
  if (depthChart) { depthChart.destroy(); depthChart = null; }

  const ctx = document.getElementById('depth-chart');
  if (!ctx) return;

  const lineColor = RISK_COLOR[riskLabel] || '#2bbfa4';

  depthChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.labels,
      datasets: [
        {
          label:           'Depth (mbgl)',
          data:            data.depth,
          borderColor:     lineColor,
          backgroundColor: lineColor + '22',
          borderWidth:     2,
          pointRadius:     0,
          fill:            true,
          tension:         0.4,
          yAxisID:         'yDepth',
        },
        {
          label:           'Rainfall (mm)',
          data:            data.rainfall,
          borderColor:     '#0891b2',
          backgroundColor: 'transparent',
          borderWidth:     1.5,
          borderDash:      [4, 3],
          pointRadius:     0,
          fill:            false,
          tension:         0.3,
          yAxisID:         'yRain',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: {
            color: '#7fa8a0',
            font:  { family: 'Inter', size: 10 },
            boxWidth: 12,
          },
        },
        tooltip: {
          backgroundColor: '#111d30',
          titleColor:      '#e8f4f2',
          bodyColor:       '#7fa8a0',
          borderColor:     'rgba(255,255,255,0.1)',
          borderWidth:     1,
        },
      },
      scales: {
        x: {
          ticks: { color: '#4a6b65', font: { size: 9 }, maxTicksLimit: 6 },
          grid:  { color: 'rgba(255,255,255,0.04)' },
        },
        yDepth: {
          type:     'linear',
          position: 'left',
          reverse:  true,   // deeper = higher on chart
          ticks:    { color: lineColor, font: { size: 9 } },
          grid:     { color: 'rgba(255,255,255,0.04)' },
        },
        yRain: {
          type:     'linear',
          position: 'right',
          ticks:    { color: '#0891b2', font: { size: 9 } },
          grid:     { drawOnChartArea: false },
        },
      },
    },
  });
}

// ── Alerts Tab ─────────────────────────────────────────────────────────────────

async function loadAlerts() {
  try {
    const res  = await fetch('/api/alerts');
    const data = await res.json();
    renderAlerts(data);
  } catch (err) {
    console.error('Failed to load alerts:', err);
  }
}

function renderAlerts(alerts) {
  const grid = document.getElementById('alerts-grid');
  const meta = document.getElementById('alerts-meta');
  meta.textContent = `${alerts.length} critical district${alerts.length !== 1 ? 's' : ''} -- updated just now`;

  if (alerts.length === 0) {
    grid.innerHTML = `
      <div style="color:var(--safe-color);font-size:1rem;padding:32px;grid-column:1/-1">
        No Critical districts detected at this time.
      </div>`;
    return;
  }

  grid.innerHTML = alerts.map((d, i) => {
    const expl    = d.explanation;
    const topF    = (expl.top_features || []).slice(0, 3);
    const pct     = Math.round(d.risk_score * 100);
    const factors = topF.map(f => `<span class="ac-factor">${f.label}</span>`).join('');

    return `
      <div class="alert-card" style="animation-delay:${i * 0.07}s"
           onclick="openFromAlert('${d.district_id}')">
        <div class="ac-header">
          <div>
            <div class="ac-name">${d.district_name}</div>
            <div class="ac-state">${d.state}</div>
          </div>
          <div>
            <div class="ac-score">${pct}%</div>
            <div class="ac-score-label">Risk Score</div>
          </div>
        </div>
        <div class="ac-summary">${expl.summary}</div>
        <div class="ac-factors">${factors}</div>
        <div class="ac-depth-row">
          <div class="ac-metric">
            <div class="ac-metric-value">${d.features.current_depth.toFixed(1)} m</div>
            <div class="ac-metric-label">Current Depth</div>
          </div>
          <div class="ac-metric">
            <div class="ac-metric-value">${(d.features.annual_trend > 0 ? '+' : '') + d.features.annual_trend.toFixed(2)} m/yr</div>
            <div class="ac-metric-label">Annual Trend</div>
          </div>
          <div class="ac-metric">
            <div class="ac-metric-value">${(d.features.extraction_rate * 100).toFixed(0)}%</div>
            <div class="ac-metric-label">Extraction</div>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

function openFromAlert(districtId) {
  switchTab('map');
  // Wait for tab animation, then fly to the marker and open popup
  setTimeout(() => {
    const d = allDistricts.find(x => x.district_id === districtId);
    if (d) {
      leafletMap.flyTo([d.lat, d.lon], 8, { duration: 1.0 });
      const marker = markerMap[districtId];
      if (marker) setTimeout(() => marker.openPopup(), 1100);
    }
    loadDetailPanel(districtId);
  }, 200);
}

// ── Tab switching ──────────────────────────────────────────────────────────────

function switchTab(name) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById(`panel-${name}`).classList.add('active');

  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`tab-${name}`).classList.add('active');

  // Invalidate map size in case it was hidden when Leaflet initialised
  if (name === 'map' && leafletMap) {
    setTimeout(() => leafletMap.invalidateSize(), 50);
  }
}

// ── Expose to global scope (used by onclick= attributes) ───────────────────────
window.switchTab        = switchTab;
window.filterMarkers    = filterMarkers;
window.selectDistrict   = selectDistrict;
window.openFromAlert    = openFromAlert;
window.searchDistricts  = searchDistricts;
window.exportCSV        = exportCSV;

// ── Search ─────────────────────────────────────────────────────────────────────

function searchDistricts(query) {
  const q = query.trim().toLowerCase();
  document.querySelectorAll('.district-item').forEach(el => {
    const name  = el.querySelector('.di-name')?.textContent.toLowerCase()  || '';
    const state = el.querySelector('.di-state')?.textContent.toLowerCase() || '';
    const show  = !q || name.includes(q) || state.includes(q);
    el.classList.toggle('hidden', !show);
  });
}

// ── URL Deep-Linking ────────────────────────────────────────────────────────────

/**
 * On load: check ?district=DIST_ID in URL and auto-open that district.
 * On select: push the district ID into the URL so the link is shareable.
 */
function applyURLState() {
  const params = new URLSearchParams(window.location.search);
  const distId = params.get('district');
  if (distId && allDistricts.find(d => d.district_id === distId)) {
    const d = allDistricts.find(x => x.district_id === distId);
    leafletMap.flyTo([d.lat, d.lon], 8, { duration: 1.0 });
    setTimeout(() => {
      const marker = markerMap[distId];
      if (marker) marker.openPopup();
      loadDetailPanel(distId);
    }, 1200);
  }
}

function pushDistrictURL(distId) {
  const url = new URL(window.location);
  url.searchParams.set('district', distId);
  window.history.pushState({}, '', url);
}

// Patch selectDistrict to also push URL state
const _origSelectDistrict = selectDistrict;
window.selectDistrict = async function(distId) {
  pushDistrictURL(distId);
  await _origSelectDistrict(distId);
};

// ── Export CSV ─────────────────────────────────────────────────────────────────

function exportCSV() {
  if (!allDistricts.length) return;

  const headers = [
    'District', 'State', 'Risk Level', 'Confidence (%)', 'Risk Score',
    'Depth (mbgl)', 'Annual Trend (m/yr)', 'Extraction Rate',
    '90d Rainfall (mm)', 'Pre-Monsoon Depth', 'Post-Monsoon Depth', 'Monsoon Recovery'
  ];

  const rows = allDistricts.map(d => [
    d.district_name,
    d.state,
    d.risk_label,
    d.confidence,
    (d.risk_score * 100).toFixed(1),
    d.features.current_depth.toFixed(2),
    d.features.annual_trend.toFixed(3),
    (d.features.extraction_rate * 100).toFixed(0) + '%',
    d.features.rainfall_lag_90d.toFixed(1),
    d.features.premonsoon_depth.toFixed(2),
    d.features.postmonsoon_depth.toFixed(2),
    d.features.monsoon_recovery.toFixed(3),
  ]);

  const csv = [headers, ...rows]
    .map(r => r.map(v => `"${v}"`).join(','))
    .join('\n');

  const blob = new Blob([csv], { type: 'text/csv' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = `aquaalert_districts_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ── Global Feature Importance Chart ────────────────────────────────────────────

async function loadFeatureImportance() {
  try {
    const res   = await fetch('/api/feature-importance');
    if (!res.ok) return;
    const items = await res.json();
    renderFeatureImportance(items);
  } catch (err) {
    console.warn('Feature importance not available:', err);
  }
}

function renderFeatureImportance(items) {
  const container = document.getElementById('fi-chart');
  if (!container) return;

  container.innerHTML = items.map((item, i) => `
    <div class="fi-row" style="animation-delay:${i * 0.06}s">
      <div class="fi-label">${item.label}</div>
      <div class="fi-bar-wrap">
        <div class="fi-bar" style="width:${item.pct}%"></div>
      </div>
      <div class="fi-pct">${item.pct}%</div>
    </div>
  `).join('');
}
