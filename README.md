# LPA — Load Prediction Application

Predictive autoscaler for Kubernetes. A GRU neural network forecasts CPU utilization **60 seconds ahead**, exposes the prediction as a Prometheus metric, and KEDA uses it to scale pods *before* a traffic spike arrives — instead of reacting after CPU is already saturated.

---

## How It Works

```
Prometheus (K8s)
      │  every 5s: CPU utilization [0,1], RPS
      ▼
  Collector ──────────────────────────────────────────────────────────────────┐
      │                                                                        │
      │  buffer last 9 points → POST /predict                                 │
      ▼                                                                        │
  GRU Predictor                                                               │
      │  4 features → GRU(64) → predicted_cpu_util (+60s)                     │
      ▼                                                                        │
  Prometheus Gauge                  TimescaleDB ◄──────────────────────────────┘
  gru_predicted_cpu_util            (lpa_metrics)
      │
      ▼
  KEDA ScaledObject  ──►  K8s HPA  ──►  scale cpu-service
  (proactive trigger)     (reactive fallback)
```

**Key idea:** With a 60s prediction horizon and a 15s pod startup time, the new pod is *ready* when the spike actually hits.

---

## Services

| Service | Port | Description |
|---------|------|-------------|
| `timescale_api` | **5000** | Central REST API — metric storage, model registry, settings |
| `lstm-predictor` | **6000** | GRU inference service with hot-swap model loading |
| `collector` | **8001** | Async worker — polls Prometheus every 5s, fills prediction buffer, exposes Prometheus gauge |
| `dashboard` | **8501** | Streamlit UI — metrics charts, model registry, upload, settings, logs |
| `timescaledb` | **5432** | PostgreSQL + TimescaleDB |
| `pgadmin` | **5050** | Database administration UI |
| `loki` | **3100** | Log aggregation |
| `promtail` | — | Log shipper (Docker logs → Loki) |
| `grafana` | **3000** | Observability dashboard |

---

## Quick Start

### Prerequisites

- Docker ≥ 24 and Docker Compose v2
- A running Prometheus instance accessible from the Docker host
- Kubernetes cluster with KEDA installed (for autoscaling)

```bash
git clone <repo-url>
cd diploma
docker compose up -d --build
```

Wait ~30s for health checks, then open:

| URL | What |
|-----|------|
| http://localhost:8501 | Streamlit Dashboard |
| http://localhost:5000/docs | TimescaleAPI Swagger |
| http://localhost:6000/docs | GRU Predictor Swagger |
| http://localhost:3000 | Grafana (admin / admin) |
| http://localhost:5050 | pgAdmin (admin@example.com / admin) |
| http://localhost:8001/metrics | Collector Prometheus exporter |

### First-run configuration

1. Open Dashboard → **⚙️ Settings**
2. Set **Prometheus URL** (e.g. `http://host.docker.internal:9090/api/v1/query`)
3. Set PromQL queries for CPU, RAM and RPS
4. Enable **Collector Active** → **Save**

The collector starts gathering metrics within 5 seconds. Predictions begin after 21 points accumulate (~105s).

---

## Data Flow (5-second cycle)

```
1. Fetch CPU utilization [0,1] and RPS from Prometheus
2. Save raw CPU reading (for later actual_cpu sync)
3. Sync actual_cpu for past predictions where target_ts ≈ now
4. Append {cpu_util, rps} to circular buffer (maxlen = MODEL_INPUT_STEPS + 1 = 9)
5. When buffer is full (9 points):
     a. Build 4 features: pct_change(cpu), pct_change(rps), cpu_level, log1p(rps)
     b. POST /predict → GRU → predicted_cpu_util ∈ [0, 1]
     c. Set Prometheus gauge: gru_predicted_cpu_util
     d. POST /metrics/predict → save row to DB
6. KEDA polls: max(gru_predicted_cpu_util) * 100 * replicas_ready → threshold 50
```

---

## GRU Model

| Parameter | Value | Description |
|-----------|-------|-------------|
| Architecture | GRU(64) + Dropout(0.35) + Dense(1, sigmoid) | ~13K parameters |
| Features | pct_change(cpu), pct_change(rps), cpu_level, log1p(rps) | 4 features, hardware-agnostic |
| Input | 8 steps × 4 features (40s window) | scale-invariant velocity + level |
| Output | cpu_util[t+12] point prediction ∈ [0, 1] | 60s ahead (sigmoid, no scaler_y) |
| Loss | Quantile loss (q=0.75) | bias toward over-provisioning |

### Training workflow

```
1. Run Locust scenarios against cpu-service with 1 fixed replica, NO HPA
   → collect clean RPS→CPU data via Collector → export from DB
2. Train in notebooks/deep_learning_percentage_absolute_out.ipynb
   → produces: model.keras + scaler_X.joblib (no scaler_y needed)
3. Upload via Dashboard → Upload Model (multipart: .keras + .joblib)
4. Activate via Dashboard → Model Registry → Activate
   → triggers hot-swap in GRU Predictor (zero downtime)
```

---

## KEDA Integration

```yaml
# test_deployment/predictive_hpa.yaml
triggers:
  - type: prometheus          # PROACTIVE — ML prediction
    metricType: AverageValue
    query: max(gru_predicted_cpu_util) * 100 * scalar(kube_deployment_status_replicas_ready{...})
    threshold: '50'           # stable when prediction < 50%

  - type: cpu                 # REACTIVE — fallback (vanilla HPA behaviour)
    metricType: AverageValue
    value: "500"              # 500m per pod (50% of 1000m limit)
```

The dual-trigger design ensures the system **cannot perform worse than vanilla HPA**.

---

## API Reference

### TimescaleAPI (`localhost:5000`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/metrics/history` | Fetch recent prediction records |
| `GET` | `/metrics/history/range` | Fetch records in time range (for fine-tuning) |
| `POST` | `/metrics/predict` | Save a new prediction record |
| `PUT` | `/metrics/{id}/actual` | Sync actual_cpu for a past prediction |
| `GET` | `/models` | List all registered models |
| `POST` | `/models` | Register a new model |
| `GET` | `/models/active` | Get currently active model (used by predictor on cold start) |
| `PUT` | `/models/{version}/activate` | Activate + trigger hot-swap |
| `POST` | `/models/upload` | Upload `.keras` + 2 × `.joblib` scalers (multipart) |
| `POST` | `/models/{version}/evaluate` | Calculate real MSE/MAE from DB |
| `GET` | `/settings` | Fetch system settings |
| `PUT` | `/settings` | Update Prometheus URL, PromQL queries, collector toggle |

### GRU Predictor (`localhost:6000`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Run inference; body: `{"history": [{cpu, rps} × 9]}` |
| `POST` | `/reload` | Hot-swap model (no restart); body: `{version, model_path, scaler_x_path, scaler_y_path}` |
| `POST` | `/retrain` | Start background fine-tuning |
| `GET` | `/status` | Current model version and status |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_URL` | `http://timescale_api:5000` | TimescaleAPI base URL |
| `PREDICTOR_URL` | `http://lstm-predictor:6000` | GRU Predictor base URL |
| `MODELS_DIR` | `/app/lstm_module/ml_models` | Model files directory |
| `SCALERS_DIR` | `/app/lstm_module/scalers` | Scaler files directory |
| `MODEL_INPUT_STEPS` | `8` | History window size (steps × 5s = 40s context) |
| `MODEL_FEATURES` | `4` | Number of input features (pct_cpu, pct_rps, cpu_level, rps_level) |

All variables are defined in [app/config.py](app/config.py) and can be overridden via Docker environment.

---

## Load Testing & A/B Experiment

Located in [test_deployment/locusts/](test_deployment/locusts/):

| File | Description |
|------|-------------|
| `locustfile_scenarios.py` | All 6 scenarios in sequence (~72 min total) |
| `scenario_1_linear_ramp.py` | Gradual 1→30 users over 10 min |
| `scenario_2_step_function.py` | Sudden jumps between load levels |
| `scenario_3_sine_wave.py` | Periodic sin² wave |
| `scenario_4_multi_spike.py` | Short bursts with quiet pauses — best for showing predictive advantage |
| `scenario_5_random_walk.py` | Unpredictable (seeded RNG) |
| `scenario_6_quiet_burst.py` | Quiet baseline + rare large bursts — best for showing predictive advantage |
| `run_all_scenarios.sh` | Runs all 6 with 2-min cooldown between |

### A/B comparison (Vanilla HPA vs Predictive KEDA)

```bash
# Round A — Vanilla HPA (reactive baseline)
kubectl delete scaledobject --all
kubectl apply -f test_deployment/vanilla_hpa.yaml
locust -f locusts/locustfile_scenarios.py --headless -u 30 -r 5 --run-time 40m

# Round B — Predictive KEDA
kubectl delete hpa --all
kubectl apply -f test_deployment/predictive_hpa.yaml
locust -f locusts/locustfile_scenarios.py --headless -u 30 -r 5 --run-time 40m
```

Key metrics to compare: P99 latency during spike onset, scale-up lag, CPU throttling duration, wasted pod-seconds.

**Note:** `cpu-service.yaml` uses `initialDelaySeconds: 15` to simulate a slow-starting application (e.g. Spring Boot). This makes the 30s prediction horizon meaningful — the pod is ready exactly when the spike arrives.

---

## Project Structure

```
diploma/
├── app/
│   ├── config.py                   # Global config: URLs, dirs, MODEL_INPUT_STEPS, MODEL_FEATURES
│   ├── timescale_api/              # Central REST API (FastAPI + SQLAlchemy + TimescaleDB)
│   │   └── api/
│   │       ├── routes/             # metrics.py, model.py, settings.py, logs.py
│   │       ├── models.py           # ORM: MetricEntry, ModelRegistry, SystemSettings, SystemLog
│   │       └── database.py
│   ├── lstm_module/                # GRU Predictor microservice (FastAPI, port 6000)
│   │   ├── api/routes.py           # /predict, /reload, /retrain, /status
│   │   ├── core/config.py          # PROJECT_NAME only
│   │   └── services/
│   │       ├── model_manager.py    # Thread-safe predict + hot-swap + fine-tune
│   │       └── utils.py            # Fine-tune pipeline, data preparation
│   ├── collector/                  # Metric collection worker (port 8001)
│   │   ├── worker.py               # Main async loop (5s interval)
│   │   └── services/
│   │       ├── prometheus.py       # PromQL fetch
│   │       └── api_client.py       # TimescaleAPI + Predictor client
│   ├── dashboard/
│   │   └── app.py                  # Streamlit (5 pages: Metrics, Registry, Upload, Settings, Logs)
│   ├── shared/
│   │   ├── schemas.py              # All Pydantic models
│   │   ├── utils.py
│   │   └── logger.py
│   ├── ml_models/                  # .keras model files
│   ├── scalers/                    # .joblib scaler files
│   └── Dockerfile
├── notebooks/
│   ├── deep_learning_percentage_absolute_out.ipynb  # Main training notebook (current production model)
│   ├── hyperparam_search.ipynb                      # Walk-forward CV: window/horizon/feature search
│   └── cpu_load_deployment.ipynb                    # EDA of collected dataset
├── test_deployment/
│   ├── cpu-service.yaml            # Target deployment (bcrypt CPU load, startup 15s)
│   ├── predictive_hpa.yaml         # KEDA ScaledObject (ML + CPU dual trigger)
│   ├── vanilla_hpa.yaml            # Standard HPA for A/B baseline
│   ├── podinfo.yaml                # Alternative target deployment
│   └── locusts/                    # 6 Locust scenarios + runner
├── configs/                        # Loki, Promtail, Grafana provisioning
├── docker-compose.yaml
└── pyproject.toml                  # Black, isort, mypy config
```

---

## Code Quality

```bash
black app/
isort app/
mypy
```

Configuration in [pyproject.toml](pyproject.toml): Black line length 88, mypy checks `app/` only.
