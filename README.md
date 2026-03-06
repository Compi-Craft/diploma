# LPA — Load Prediction Application

A microservice system that collects infrastructure metrics (CPU, RAM, RPS) from Prometheus, feeds them into an LSTM neural network for short-term forecasting, and exposes everything through a real-time Streamlit dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Network                           │
│                                                                 │
│  Prometheus ──► Collector ──► LSTM Predictor ──► TimescaleDB   │
│  (external)       :8001          :6000              :5432       │
│                     │                │                │         │
│                     └────────────────┴──► TimescaleAPI          │
│                                              :5000              │
│                                                │                │
│                                           Dashboard             │
│                                             :8501               │
│                                                                 │
│  Loki :3100 ◄─ Promtail ◄─ Docker logs                         │
│  Grafana :3000 ──► Loki                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Services

| Service | Image / Build | Port | Description |
|---|---|---|---|
| `timescale_api` | `./app` | **5000** | Central REST API; stores metrics & model registry in TimescaleDB |
| `lstm-predictor` | `./app` | **6000** | FastAPI service that loads LSTM model and serves predictions |
| `collector` | `./app` | **8001** | Async worker; polls Prometheus every 15 s, calls predictor, saves results |
| `dashboard` | `./app` | **8501** | Streamlit UI — metrics charts, model registry, upload, settings |
| `timescaledb` | `timescale/timescaledb:latest-pg14` | **5432** | PostgreSQL + TimescaleDB extension for time-series data |
| `pgadmin` | `dpage/pgadmin4` | **5050** | Database administration UI |
| `loki` | `grafana/loki:2.9.0` | **3100** | Log aggregation backend |
| `promtail` | `grafana/promtail:2.9.0` | — | Log shipper; tails Docker container logs → Loki |
| `grafana` | `grafana/grafana:10.2.0` | **3000** | Observability dashboard (logs from Loki) |

---

## Quick Start

### Prerequisites

- Docker ≥ 24 and Docker Compose v2
- A running Prometheus instance accessible from the Docker host

```bash
git clone <repo-url>
cd diploma
docker compose up -d --build
```

Wait for health checks to pass (≈ 30 s), then open:

| URL | What |
|---|---|
| http://localhost:8501 | Streamlit Dashboard |
| http://localhost:5000/docs | TimescaleAPI Swagger |
| http://localhost:6000/docs | LSTM Predictor Swagger |
| http://localhost:3000 | Grafana (admin / admin) |
| http://localhost:5050 | pgAdmin (admin@example.com / admin) |
| http://localhost:8001/metrics | Collector Prometheus exporter |

### First-run configuration

1. Open the Dashboard → **⚙️ Settings**.
2. Set **Prometheus URL** (e.g. `http://host.docker.internal:9090/api/v1/query`).
3. Set the three PromQL queries for CPU, RAM and RPS.
4. Enable **Collector Active** and click **Save Settings**.

The collector will start gathering metrics within 15 seconds.

---

## Data Flow

```
Every 15 s:
  Collector
    ├─ GET  /settings          → TimescaleAPI   (fetch Prometheus URL + PromQL)
    ├─ POST Prometheus         → query CPU / RAM / RPS instant values
    ├─ PUT  /metrics/sync      → TimescaleAPI   (back-fill actual_value for past forecasts)
    ├─ (buffer ≥ 10 points)
    │   ├─ POST /predict       → LSTM Predictor (10-step window → next-interval forecast)
    │   └─ POST /metrics/predict → TimescaleAPI (persist forecast record)
    └─ Prometheus Gauges       → lstm_predicted_cpu_cores / ram / rps
```

---

## API Reference

### TimescaleAPI (`localhost:5000`)

#### Metrics

| Method | Path | Description |
|---|---|---|
| `PUT` | `/metrics/sync` | Back-fill `actual_value` for predictions whose `target_ts` has passed |
| `POST` | `/metrics/predict` | Save a new prediction record |
| `GET` | `/metrics/history?resource=cpu&limit=100` | Fetch history for dashboard charts |

#### Model Registry

| Method | Path | Description |
|---|---|---|
| `GET` | `/models` | List all registered models |
| `POST` | `/models` | Register a new model (JSON body) |
| `GET` | `/models/active` | Get the currently active model (used by predictor on cold start) |
| `PUT` | `/models/{version}/activate` | Activate a model and trigger hot-swap in the predictor |
| `POST` | `/models/upload` | Upload `.keras` model + `.pkl` scaler via multipart form |

#### Settings

| Method | Path | Description |
|---|---|---|
| `GET` | `/settings` | Fetch current system settings |
| `PUT` | `/settings` | Update Prometheus URL, PromQL queries, collector on/off flag |

#### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |

---

### LSTM Predictor (`localhost:6000`)

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Run inference; body: `{ "history": [ {cpu, ram, rps} × 10 ] }` |
| `POST` | `/reload` | Hot-swap model; body: `{ "version", "model_path", "scaler_path" }` |
| `GET` | `/status` | Returns current model version and status |

---

## Model Management

### Cold Start

On startup the LSTM Predictor calls `GET /models/active` on the TimescaleAPI. If a model is found it loads the `.keras` + `.pkl` files from the shared volume. If no model is registered it falls back to a dummy (zeros) model until a real one is activated.

### Hot Swap

Activating a model via the Dashboard or `PUT /models/{version}/activate`:

1. TimescaleAPI flips `is_active` in the database.
2. A background task calls `POST /reload` on the LSTM Predictor with the new file paths.
3. The predictor loads the new model under a `threading.Lock` — zero downtime, no restart.

### Uploading a Custom Model

Via the Dashboard → **📤 Upload Model**:
- Upload a Keras `.keras` model and a scikit-learn `.pkl` scaler.
- Optionally supply a version string, MSE and MAE metrics.
- Files are saved to the shared volume; a registry record is created.
- Activate the uploaded model from the **🗂️ Model Registry** page.

### Training Workflow

```
1. Collect data from /metrics/history (or use your own dataset)
2. Train an LSTM: input shape (batch, 10, 3), output shape (batch, 3)
3. Fit a StandardScaler on the training data, save with joblib.dump()
4. Save the Keras model with model.save("model.keras")
5. Upload via Dashboard → Upload Model
6. Activate via Dashboard → Model Registry → Activate
```

---

## LSTM Model Configuration

Configured in [app/lstm_module/core/config.py](app/lstm_module/core/config.py):

| Parameter | Default | Description |
|---|---|---|
| `MODEL_INPUT_STEPS` | `10` | Number of historical points fed to the model |
| `MODEL_FEATURES` | `3` | Number of features: cpu, ram, rps |

Default prediction horizon: **60 seconds** (configurable per request via `horizon_seconds`).

---

## Configuration

### Environment Variables

| Variable | Service | Default | Description |
|---|---|---|---|
| `PORT` | timescale_api | `5000` | HTTP listen port |
| `PORT` | lstm-predictor | `6000` | HTTP listen port |
| `API_URL` | dashboard | `http://timescale_api:5000` | TimescaleAPI base URL |
| `POSTGRES_PASSWORD` | timescaledb | `lpa_password` | DB password |
| `POSTGRES_DB` | timescaledb | `lpa_database` | DB name |

TimescaleAPI connects to `postgresql+asyncpg://postgres:lpa_password@timescaledb/lpa_database` by default (hardcoded in [app/timescale_api/api/database.py](app/timescale_api/api/database.py)).

### Monitoring Stack Config Files

| File | Purpose |
|---|---|
| `configs/loki-config.yml` | Loki storage and ingestion config |
| `configs/promtail-config.yml` | Promtail Docker log scraping targets |
| `configs/grafana-datasources.yml` | Auto-provision Loki as Grafana datasource |

---

## Dashboard

The Streamlit dashboard ([app/dashboard/app.py](app/dashboard/app.py)) has four pages:

**📈 Metrics** — Auto-refreshing Plotly charts (one per resource: CPU / RAM / RPS) showing:
- Actual measured values
- LSTM predicted values (dashed, for the next interval)
- Retrospective actuals plotted at the predicted target time

Configurable refresh interval (5–60 s) and history window (20–300 points) from the sidebar.

**🗂️ Model Registry** — Table of all registered models with version, status, MSE, MAE, created date. Supports one-click model activation.

**📤 Upload Model** — Form to upload a `.keras` Keras model and `.pkl` scaler with optional version string and metrics.

**⚙️ Settings** — Edit Prometheus URL, PromQL queries for each resource, and toggle the collector on/off.

---

## Load Testing

Located in [test_deployment/](test_deployment/):

```bash
# Basic load test
locust -f test_deployment/locustfile.py --host http://localhost:5000

# Extended test (all endpoints)
locust -f test_deployment/locustfile_extended.py --host http://localhost:5000
```

---

## Development

### Running Services Locally

Each service sets its own module path. From `app/`:

```bash
# TimescaleAPI
python timescale_api/run.py

# LSTM Predictor
python lstm_module/main.py

# Collector
python -m collector.worker
```

### Code Quality

```bash
# Formatting
black app/
isort app/

# Type checking
mypy .
```

Configuration in [pyproject.toml](pyproject.toml): Black line length 88, mypy strict mode (`disallow_untyped_defs = true`).

---

## Project Structure

```
diploma/
├── app/
│   ├── timescale_api/          # Central REST API (FastAPI + SQLAlchemy + TimescaleDB)
│   │   └── api/
│   │       ├── routes/
│   │       │   ├── metrics.py  # /metrics endpoints
│   │       │   ├── model.py    # /models endpoints
│   │       │   └── settings.py # /settings endpoints
│   │       ├── models.py       # SQLAlchemy ORM models
│   │       ├── schemas.py      # Pydantic schemas
│   │       ├── database.py     # Async DB session
│   │       └── utils.py        # Version generator, predictor notifier
│   ├── lstm_module/            # LSTM Predictor microservice (FastAPI)
│   │   ├── api/routes.py       # /predict, /reload, /status
│   │   ├── services/
│   │   │   └── model_manager.py # Thread-safe model hot-swap
│   │   ├── models/schemas.py   # Pydantic schemas
│   │   └── core/config.py      # MODEL_INPUT_STEPS, MODEL_FEATURES
│   ├── collector/              # Metric collection worker
│   │   ├── worker.py           # Main async loop (15 s interval)
│   │   └── services/
│   │       ├── prometheus.py   # PromQL fetch
│   │       ├── predictor.py    # LSTM Predictor client
│   │       └── api_client.py   # TimescaleAPI client
│   ├── dashboard/
│   │   └── app.py              # Streamlit dashboard (4 pages)
│   ├── ml_models/              # Default LSTM model (.keras)
│   ├── scalers/                # Default scaler (.pkl)
│   ├── requirements.txt
│   └── Dockerfile
├── test_deployment/
│   ├── locustfile.py           # Basic load test
│   └── locustfile_extended.py  # Extended load test
├── configs/
│   ├── loki-config.yml
│   ├── promtail-config.yml
│   └── grafana-datasources.yml
├── tools/
│   └── format_data.py          # Data preparation utility
├── docker-compose.yaml
└── pyproject.toml              # Black, isort, mypy config
```
