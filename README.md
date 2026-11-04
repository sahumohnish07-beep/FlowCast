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
- [ ] **Phase 1: Data Ingestion & Exploratory Data Analysis (EDA)**
  - Synthetic / real-world sensor, weather, and calendar data generation / ingestion
  - Data cleaning, missing-value imputation, and schema validation
  - Exploratory analysis of volume distributions, peak-hour behaviors, and weather correlations
- [ ] **Phase 2: Feature Engineering & Preprocessing**
  - Temporal encoding (hour-of-day, day-of-week, cyclical sine/cosine transforms)
  - Time-series lag features (t-15m, t-30m, t-1h, t-24h) and moving window aggregations
  - Weather severity indices & holiday calendar flags
  - Strict chronological train/validation/test splitting
- [ ] **Phase 3: Classical & Tree-Based Modeling (XGBoost / Scikit-Learn)**
  - Baseline linear/ridge models
  - XGBoost regressors and classifiers for traffic volume and congestion severity
  - Hyperparameter tuning and feature importance analysis
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
