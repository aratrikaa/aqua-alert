# AquaAlert — Groundwater Depletion Early-Warning System

> An AI-powered dashboard that classifies groundwater depletion risk across Indian districts into **Safe / Moderate / Critical** tiers, explains *why* each district was classified that way using SHAP, and presents everything on an interactive Leaflet.js map.

---

## 🚀 Quick Start

```bash
# 1. Clone / enter the project directory
cd "aqua alert"

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Open **http://localhost:5000** in your browser.

The first run automatically:
1. Generates synthetic time-series data (`data/groundwater_timeseries.csv`)
2. Engineers features for each district
3. Trains Random Forest + XGBoost classifiers (picks the better one by CV)
4. Computes SHAP explanations per district
5. Serves the dashboard

Subsequent runs reuse cached files — startup takes < 3 seconds.

---

## 🗂 Project Structure

```
aqua-alert/
├── app.py                         # Main entry point  (python app.py)
├── requirements.txt
├── README.md
│
├── data/
│   ├── generate_data.py           # Synthetic CGWB/IMD data generator
│   ├── groundwater_timeseries.csv # Auto-generated on first run
│   └── districts.json             # District metadata (lat/lon, profile)
│
├── features/
│   └── engineering.py             # Feature computation from time-series
│
├── model/
│   ├── train.py                   # RF + XGBoost training & selection
│   ├── explain.py                 # SHAP value computation + plain-English text
│   ├── aqua_model.joblib          # Saved model (auto-generated)
│   └── label_encoder.joblib       # Saved encoder (auto-generated)
│
├── api/
│   └── routes.py                  # Flask Blueprint — REST endpoints
│
└── frontend/
    ├── index.html                  # Dashboard HTML
    ├── style.css                   # Teal/dark design system
    └── app.js                      # Leaflet map, SHAP panel, Chart.js
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/districts` | All districts with risk labels, coordinates, confidence |
| GET | `/api/district/<id>` | Full detail + SHAP explanation for one district |
| GET | `/api/alerts` | Critical-only list sorted by risk score |
| GET | `/api/chart/<id>` | Last 24 months of depth + rainfall for sparkline |

---

## 📊 Features Used by the Model

| Feature | Description |
|---------|-------------|
| `rainfall_lag_30d` | Cumulative rainfall in the last 1 month |
| `rainfall_lag_60d` | Cumulative rainfall in the last 2 months |
| `rainfall_lag_90d` | Cumulative rainfall in the last 3 months |
| `premonsoon_depth` | Mean water table depth Mar–May (mbgl) |
| `postmonsoon_depth` | Mean water table depth Oct–Nov (mbgl) |
| `monsoon_recovery` | Change in depth from pre→post monsoon |
| `annual_trend` | Linear slope of depth over last 3 years (m/yr) |
| `extraction_rate` | Normalised pumping proxy (0–1) |
| `current_depth` | Most recent water table depth (mbgl) |

---

## 🔌 Plugging in Real Data (CGWB / IMD)

### Replace synthetic data with real CGWB data

1. Download dynamic groundwater level data from  
   **[CGWB](https://cgwb.gov.in/GW-data-access.html)** or **[India-WRIS](https://indiawris.gov.in/)**.

2. Format your CSV to match this schema:

   ```
   date,district_id,district_name,state,water_depth_mbgl,rainfall_mm,extraction_rate
   2023-01-01,PB_LDH,Ludhiana,Punjab,21.4,2.1,0.85
   ...
   ```

3. Place the file at `data/groundwater_timeseries.csv`  
   and update `data/districts.json` with the correct `lat/lon` for each `district_id`.

4. Delete the cached model files to trigger re-training:
   ```bash
   del model\aqua_model.joblib model\label_encoder.joblib
   ```

5. Restart: `python app.py`

### Adding real IMD rainfall

Download monthly gridded rainfall from  
**[IMD Pune Data](https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html)**,  
aggregate to district level, and replace the `rainfall_mm` column.

---

## 🤖 Model Details

- **Baseline**: Random Forest (300 trees, balanced class weights)
- **Primary**: XGBoost (200 estimators, lr=0.1)
- **Selection**: 3-fold stratified cross-validation → higher accuracy wins
- **Explainability**: TreeSHAP (`shap.TreeExplainer`) — per-prediction contributions
- **Classes**: Safe (0) · Moderate (1) · Critical (2)

---

## 🎨 Dashboard Features

- **Interactive Map** — Leaflet.js with OpenStreetMap dark tiles (free, no API key); colour-coded custom markers + district boundary overlays (green / amber / red)
- **Sidebar** — filterable district list with one-click selection
- **Detail Panel** — risk badge, confidence bar, SHAP explanation, feature chips, 24-month depth/rainfall chart
- **Alerts Panel** — all Critical districts sorted by P(Critical), with key drivers shown as tags
- **About Page** — pipeline documentation

---

## 📋 Requirements

- Python ≥ 3.10
- Modern browser (Chrome, Firefox, Edge)
- Internet connection (for map tiles and CDN fonts on first load; can be made offline with local assets)

---

*Built as a college mini-project demonstrating end-to-end AI/ML pipeline: data → features → classification → explainability → interactive dashboard.*
