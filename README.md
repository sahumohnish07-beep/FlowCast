# FlowCast: AI-Powered Smart Traffic Flow Prediction Platform

[![Python Version](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**FlowCast** is an intelligent, end-to-end traffic analytics and forecasting system designed to predict traffic volume, congestion indices, estimated travel times, and accident risk. By synthesizing IoT sensor readings, real-time meteorological conditions, and temporal/calendar indicators, FlowCast empowers urban planners, commuters, and traffic management centers with predictive intelligence.

---

## 🌟 Key Features

- **Multimodal Prediction**: Forecasts traffic volume, road congestion levels, segment travel times, and localized accident probabilities.
- **Temporal & Spatial Feature Engineering**: Cyclical time encoding, multi-interval lag features, rolling statistics, and weather impact metrics.
- **Strict Time-Based Validation**: Time-series aware data splitting prevents future lookahead bias and data leakage.
- **Hybrid Modeling Pipeline**: Benchmark regressors, gradient boosted decision trees (XGBoost), and deep sequence neural networks (PyTorch).
- **High-Performance REST API**: Fast, async inference endpoints powered by FastAPI and validated with Pydantic schemas.
- **Interactive Geospatial Dashboard**: Streamlit web interface featuring dynamic Folium route maps and Plotly analytical dashboards.

---

## 🏗️ Project Architecture & Directory Layout

```text
FlowCast/
├── data/
│   ├── raw/             # Unprocessed, immutable sensor and weather datasets (.gitignored)
│   └── processed/       # Cleaned, standardized, feature-engineered tabular/tensor datasets
├── notebooks/           # Exploratory data analysis (EDA) and experimental prototypes
├── src/                 # Modular, production-grade source code
│   ├── data/            # Data ingestion, schema validation, and cleaning pipelines
│   ├── features/        # Feature extraction, cyclical encoders, and transformers
│   ├── models/          # Model architectures, training routines, and evaluation metrics
│   └── api/             # FastAPI routers, request/response models, and inference engine
├── dashboard/           # Streamlit user interface, custom styles, and Folium/Plotly views
├── models/              # Serialized model artifacts, weights, and scalers (.gitignored binaries)
├── tests/               # Pytest unit and integration test suites
├── docs/                # Architecture diagrams, API specs, and project reports
├── .env.example         # Template for environment configuration and secrets
├── .gitignore           # Git ignore rules for virtualenvs, binaries, data, and cache
├── requirements.txt     # Locked production and development dependencies
└── README.md            # Project overview, documentation, and phase progress
```

---

## 📋 Project Phases & Progress Checklist

- [x] **Phase 0: Project Setup & Repository Scaffolding**
  - Project directory structure created (`data/`, `notebooks/`, `src/`, `dashboard/`, `models/`, `tests/`, `docs/`)
  - Virtual environment initialized and verified
  - Dependency configuration (`requirements.txt`) established
  - Git repository, `.gitignore`, and `.env.example` configured
  - Initial synchronization to GitHub main branch
- [x] **Phase 1: Data Collection & Ingestion**
  - Metro Interstate Traffic Volume dataset automated downloader (`src/data/traffic_loader.py`, 48,204 rows)
  - Open-Meteo Historical Weather Archive client (`src/data/weather_client.py`, 52,584 hourly observations)
  - Statutory holiday and calendar feature synthesizer (`src/data/calendar_generator.py`, 5,205 calendar days)
  - Unified reproducible download script (`src/data/download.py`) saving to `data/raw/`
  - Data sources documentation authored in `docs/data_sources.md`
  - Automated test coverage via `tests/test_data_collection.py`
- [x] **Phase 2: Exploratory Data Analysis (EDA) & Preprocessing**
  - Interactive notebook `notebooks/01_eda.ipynb` analyzing volume dynamics, diurnal curves, weather, and holidays
  - High-resolution visualizations saved to `docs/images/` (`traffic_by_hour.png`, `traffic_by_weekday.png`, `traffic_by_weather.png`, `traffic_by_holiday.png`, `correlation_heatmap.png`)
  - Robust sensor cleaning and multimodal fusion pipeline in `src/features/preprocess.py`
  - 55 engineered features: cyclical temporal sine/cosine transforms, weather severity scores, leak-free lag features (1h, 2h, 3h, 24h, 168h), and rolling aggregations (3h, 6h, 24h)
  - Multi-task targets modeled: continuous `traffic_volume`, categorical `congestion_level`, BPR corridor `travel_time_min`, and localized `accident_risk_index`
  - Chronological time-based data split saved to `data/processed/`: Train (37,071 rows, 2012–2016), Val (8,760 rows, 2017), Test (6,552 rows, 2018)
  - Automated test coverage via `tests/test_preprocessing.py`
- [x] **Phase 3: Baseline Models (Linear Regression, Random Forest, XGBoost)**
  - End-to-end training, cross-validation, and evaluation pipeline in `src/models/baseline.py`
  - Evaluated with strict chronological metrics (MAE, RMSE, R²) on held-out 2018 test data
  - Metric comparisons saved to `docs/baseline_results.csv`
  - Forecast evaluation plot saved to `docs/images/actual_vs_predicted_baseline.png`
  - Best model artifact (`XGBoost`, Test MAE: 146.60, R²: 0.9869) serialized to `models/best_baseline_model.joblib`
  - Automated unit test coverage in `tests/test_baseline_models.py`
- [ ] **Phase 4: Deep Learning & Sequence Modeling (PyTorch)**
  - Sequence data loader construction (sliding window temporal sequences)
  - Recurrent neural networks (LSTM / GRU) or Temporal Convolutional Networks
  - Multi-horizon traffic forecasting and training checkpoints
- [ ] **Phase 5: Model Evaluation, Comparison & Serialization**
  - Cross-model performance comparison (MAE, RMSE, MAPE, R², F1-score)
  - Model serialization, metadata logging, and inference pipeline packaging
- [ ] **Phase 6: REST API Development (FastAPI)**
  - Async FastAPI endpoints for single-point and batch predictions
  - Pydantic input validation and health check endpoints
  - Swagger/OpenAPI documentation
- [ ] **Phase 7: Interactive Dashboard & Geospatial Visualization**
  - Streamlit dashboard layout with multi-page navigation
  - Folium interactive road network maps with traffic heatmaps
  - Plotly time-series visualizers for predicted vs. actual traffic flows
- [ ] **Phase 8: End-to-End System Testing & Verification**
  - Automated unit and integration tests via pytest
  - System verification, load testing, and final documentation polish

---

## 📊 Key EDA Findings & Insights

- **Bimodal Commute Curves**: Weekdays exhibit pronounced twin peaks during the morning rush (7:00–9:00 AM, $\sim 5,100\text{ veh/hr}$) and evening rush (4:00–6:00 PM, $\sim 5,600\text{ veh/hr}$). Weekends transition into a smoother unimodal profile peaking between 1:00 PM and 4:00 PM ($\sim 4,200\text{ veh/hr}$).
- **Strong Temporal Autocorrelation**: Lag variables `traffic_lag_1h` ($r = 0.88$) and `traffic_lag_24h` ($r = 0.85$) represent dominant predictive signals, validating our time-series sequence approach.
- **Statutory Holiday Drop**: National holidays produce an average traffic volume drop of $\sim 32\%$ ($\sim 2,270\text{ veh/hr}$ vs. $\sim 3,340\text{ veh/hr}$ on standard days), while holiday eves record extended afternoon outbound surges.
- **Weather Impedance**: Adverse conditions (heavy snow, freezing rain, low visibility) suppress maximum vehicular capacity while driving up modeled travel delays and localized accident risks.
- **Zero-Leakage Splitting**: Strictly chronological partitioning guarantees realistic evaluation on untouched 2018 test data:
  - **Train**: October 2012 – December 2016 (37,071 rows, $\sim 71\%$)
  - **Validation**: January 2017 – December 2017 (8,760 rows, $\sim 17\%$)
  - **Test**: January 2018 – September 2018 (6,552 rows, $\sim 12\%$)

---

## 📈 Phase 3: Baseline Model Evaluation Results

Models trained on 2012–2016 historical data and evaluated strictly on chronologically held-out 2018 test observations ($N = 6,552$):

| Model Architecture | Test MAE (veh/hr) | Test RMSE (veh/hr) | Test $R^2$ | Val MAE (veh/hr) | Val $R^2$ | Top Performer |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Regressor** | **146.60** | **226.48** | **0.9869** | **152.77** | **0.9855** | 🏆 Best Model |
| **Random Forest Regressor** | 154.04 | 242.31 | 0.9850 | 159.93 | 0.9839 | Strong Baseline |
| **Linear Regression (Ridge)** | 270.12 | 385.16 | 0.9620 | 272.56 | 0.9625 | Linear Baseline |

> [!NOTE]
> The top model artifact has been serialized to `models/best_baseline_model.joblib`. Detailed evaluation predictions and time-series overlays are preserved in `docs/images/actual_vs_predicted_baseline.png` and `docs/baseline_results.csv`.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/sahumohnish07-beep/FlowCast.git
cd FlowCast
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
```

### 5. Run Verification & Tests
```bash
pytest tests/
```

---

## 👥 Authors & College Project Attribution

- **Project**: FlowCast - AI-Powered Smart Traffic Flow Prediction Platform
- **Lead Developer**: Mohnish Sahu
