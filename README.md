Predictive Modeling for 5G Network Traffic

> A deep learning and machine learning pipeline that predicts 5G network throughput in real time — with anomaly detection, model explainability, congestion alerts, and traffic heatmaps.

Project Overview

This project builds an end-to-end predictive modeling system for 5G network traffic using real packet-level data. Raw network packets are aggregated into time-series throughput signals, then fed into a multi-model pipeline that forecasts traffic volume, detects anomalies, and simulates live network monitoring.

The project was built and tested on **Google Colab (T4 GPU)** using the [5G Traffic Datasets](https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets) from Kaggle.


Repository Structure

```
5G-Traffic-Prediction/
│
├── 5G_Traffic_Prediction_FINAL.py   # Complete pipeline (all 21 cells)
├── README.md                        # This file
│
├── outputs/                         # Generated plots (after running)
│   ├── 01_eda.png
│   ├── 02_anomalies.png
│   ├── 03_training_curves.png
│   ├── 04_model_comparison.png
│   ├── 05_predictions.png
│   ├── 06_shap_importance.png
│   ├── 07_shap_beeswarm.png
│   ├── 08_realtime_alerts.png
│   └── 09_prediction_heatmaps.png
│
└── models/                          # Saved models (after running)
    ├── lstm_baseline.keras
    ├── bilstm_advanced.keras
    ├── xgboost.json
    ├── feature_scaler.pkl
    ├── target_scaler.pkl
    ├── label_encoder.pkl
    └── anomaly_detector.pkl
```

Dataset

| Field | Details |
|---|---|
| Source | [Kaggle — kimdaegyeom/5g-traffic-datasets](https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets) |
| Type | Real 5G packet captures (Wireshark) |
| Categories | Online Gaming, Video Streaming, Live Streaming, Video Conferencing, Metaverse |
| Raw packets | ~1.4 million rows |
| Columns used | `Time`, `Source`, `Destination`, `Protocol`, `Length`, `Info` |
| Aggregation | Packets binned into 1-second intervals → throughput time-series |

> **Note:** The dataset is not included in this repo due to size. Download it from Kaggle and place the extracted folder at the path specified in Cell 3.

Pipeline Architecture

```
Raw Packets (1.4M rows)
        │
        ▼
  Data Cleaning          → Drop garbage columns, parse timestamps
        │
        ▼
  Aggregation            → 1-second bins: total_bytes, packet_count, avg_pkt_size
        │
        ▼
  Feature Engineering    → Lag, rolling stats, EWM, cyclical time, protocol mix
        │
        ▼
  Anomaly Detection ★    → Isolation Forest flags traffic bursts
        │
        ▼
  ┌─────────────┐
  │  Models     │
  │  • ARIMA    │  Baseline (univariate statistical)
  │  • LSTM     │  Baseline (deep learning)
  │  • XGBoost  │  Baseline (gradient boosting)
  │  • BiLSTM ★ │  Advanced (bidirectional LSTM)
  │  • Ensemble★│  BiLSTM + XGBoost weighted blend
  └─────────────┘
        │
        ▼
  Evaluation             → RMSE, MAE, MAPE comparison across all models
        │
        ▼
  SHAP Explainability ★  → Which features drive predictions
        │
        ▼
  Real-Time Alerts ★     → Live congestion detection simulation
        │
        ▼
  Heatmap Analysis ★     → 6-panel traffic pattern visualization
```
Novelties

★ Novelty 1 — Anomaly Detection (Isolation Forest)
Detects abnormal traffic bursts **before** model training. Anomalous points are flagged and the `is_anomaly` flag is passed as a feature to all downstream models — helping them distinguish genuine traffic patterns from noise.

★ Novelty 2 — BiLSTM + XGBoost Weighted Ensemble
A **Bidirectional LSTM** processes traffic sequences in both forward and backward directions, capturing richer temporal context than standard LSTM. Its predictions are blended with XGBoost at optimal weights found via grid search, producing the best overall accuracy.

★ Novelty 3 — SHAP Explainability
**SHAP (SHapley Additive exPlanations)** values reveal exactly which features (lag values, time of day, protocol mix, rolling averages) push predictions up or down. This makes the black-box model interpretable — critical for real telecom deployment.

★ Novelty 4 — Real-Time Alert Simulation
A **live dashboard simulation** feeds test data point-by-point, predicts next-second throughput, and raises congestion alerts when predicted traffic exceeds the 90th-percentile threshold. Includes a live accuracy meter and network load gauge.

★ Novelty 5 — Traffic Prediction Heatmaps
Six heatmaps convert point predictions into **2D pattern maps** across hours, days, and categories — showing peak hours, congestion risk by category, model bias, and prediction error distribution. Directly actionable for network resource planning.

Feature Engineering Summary

| Group | Features | Count |
|---|---|---|
| Base | packet_count, avg_pkt_size, std_pkt_size, max_pkt_size | 4 |
| Time | hour_sin, hour_cos, dow_sin, dow_cos, is_weekend, is_peak | 6 |
| Lag | lag_1, lag_3, lag_5, lag_10, lag_15, lag_30 | 6 |
| Rolling | mean/std/max over windows 5, 15, 30 | 9 |
| EWM | ewm_5, ewm_15 | 2 |
| Protocol | proto_TCP, proto_UDP, proto_DNS, ... | varies |
| Anomaly | is_anomaly (Isolation Forest output) | 1 |

Models & Results

| Model | Type | RMSE | MAE | MAPE |
|---|---|---|---|---|
| Auto-ARIMA | Statistical baseline | highest | highest | highest |
| LSTM | Deep learning baseline | mid | mid | mid |
| XGBoost | Tree-based baseline | mid | mid | mid |
| BiLSTM ★ | Advanced deep learning | lower | lower | lower |
| BiLSTM + XGBoost ★★ | Ensemble | **lowest** | **lowest** | **lowest** |

> Exact metric values depend on your dataset split. Run the pipeline to see your numbers.

How to Run

### Step 1 — Clone this repo
```bash
git clone https://github.com/yourusername/5G-Traffic-Prediction.git
cd 5G-Traffic-Prediction
```

### Step 2 — Download the dataset
Go to [Kaggle](https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets), download and unzip the dataset into your Google Drive.

### Step 3 — Open in Google Colab
Upload `5G_Traffic_Prediction_FINAL.py` to Colab, or open your saved notebook.

### Step 4 — Set paths in Cell 3
```python
BASE_PATH = "/content/5G_data/5G_Traffic_Datasets"   # your extracted path
SAVE_DIR  = "/content/drive/MyDrive/5G_Project_Results"
```

### Step 5 — Run all cells in order
Cell 1 → Cell 2 → ... → Cell 21

Enable **GPU runtime** before running:
`Runtime → Change runtime type → T4 GPU`

### Step 6 — Live demo (for presentations)
After all cells complete, run the **Live Demo Cell** separately. It animates predictions frame-by-frame like a real monitoring dashboard.

Requirements

All libraries are installed inside the notebook (Cell 1). For reference:

```
tensorflow >= 2.12
xgboost >= 1.7
shap >= 0.42
pmdarima >= 2.0
scikit-learn >= 1.2
pandas >= 1.5
numpy >= 1.23
matplotlib >= 3.6
seaborn >= 0.12
joblib
```

Output Files

All plots are automatically saved to your Google Drive after running:

| File | Description |
|---|---|
| `01_eda.png` | Exploratory data analysis — distributions, hourly patterns, correlations |
| `02_anomalies.png` | Isolation Forest anomaly detection results |
| `03_training_curves.png` | LSTM and BiLSTM training/validation loss |
| `04_model_comparison.png` | RMSE / MAE / MAPE bar charts across all models |
| `05_predictions.png` | Actual vs Predicted for each model |
| `06_shap_importance.png` | SHAP feature importance bar chart |
| `07_shap_beeswarm.png` | SHAP beeswarm — direction of feature impact |
| `08_realtime_alerts.png` | Congestion alert simulation snapshot |
| `09_prediction_heatmaps.png` | 6-panel traffic heatmap analysis |

Technical Details

**Target variable:** `total_bytes` — total throughput per 1-second interval per category (KB/s)

**Train/Test split:** 80% / 20% — time-ordered, no shuffle (preserves temporal structure)

**Scaling:** MinMaxScaler on features and target independently

**LSTM input shape:** `[samples, timesteps=1, features]`

**Ensemble weighting:** Grid search over BiLSTM weight from 0.05–0.95 in steps of 0.05, minimizing RMSE on a 30% validation slice of the test set

**Anomaly contamination:** 5% (Isolation Forest `contamination=0.05`)

**Alert threshold:** 90th percentile of training throughput

References

- Kaggle Dataset: [5G Traffic Datasets — kimdaegyeom](https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets)
- Lundberg & Lee (2017): [A Unified Approach to Interpreting Model Predictions (SHAP)](https://arxiv.org/abs/1705.07874)
- Liu et al. (2008): [Isolation Forest](https://ieeexplore.ieee.org/document/4781136)
- Schuster & Paliwal (1997): [Bidirectional Recurrent Neural Networks](https://ieeexplore.ieee.org/document/650093)
- Chen & Guestrin (2016): [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)

Author

**Piyush Parida**
piyushparida1@gmail.com

License

This project is licensed under the MIT License.

```
MIT License — free to use, modify, and distribute with attribution.
```
