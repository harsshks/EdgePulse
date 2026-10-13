# EdgePulse — Vehicle Telemetry & Fleet Health Platform

Real-time vehicle telemetry ingestion, edge processing, persistence, and REST-based fleet data access.

EdgePulse is a real-time vehicle telemetry platform designed to simulate how connected vehicles can transmit operational data to an edge processing system, persist that data, and expose it through a backend API for fleet monitoring and analysis.

The project models a simplified real-world scenario:

> A fleet operator needs a reliable way to collect vehicle telemetry, validate incoming data, store historical readings, and make vehicle health information available to monitoring systems.

---

## 1. Overview

EdgePulse demonstrates a complete, locally running telemetry pipeline. The system uses a simulated vehicle that continuously generates randomized sensor data, sending it via MQTT to a central broker. An edge consumer script subscribes to this data flow, validates it, adds timestamps, and persists it to a local SQLite database (and a JSONL audit log). Finally, a FastAPI backend exposes this stored data via a set of clean REST API endpoints for external clients to consume. 

Currently, EdgePulse successfully simulates data ingestion, storage, and access on a single local machine. The long-term objective of this project is to evolve this pipeline into a cloud-native, containerized architecture that includes CI/CD, metric monitoring, and eventual time-series machine learning to identify abnormal vehicle behaviors.

---

## 2. Real-World Problem

In the real world, a fleet of delivery vehicles continuously generates telemetry such as:
- Speed
- Engine RPM
- Engine Temperature
- Vehicle IDs
- Timestamps

The challenge is not merely collecting this data. The system needs to ingest telemetry continuously at scale, process the incoming messages, validate the sensor readings, persist historical data safely, and expose that data to other services in a structured way (e.g., dashboards or alert systems).

EdgePulse models this problem exactly. While the vehicle is currently simulated in software, the architecture leverages the same fundamental communication concepts (like MQTT for IoT messaging and REST for API access) used in actual connected-device systems.

---

## 3. Project Objective

EdgePulse is being developed incrementally. The project is designed to grow from a simple local script into a fully-fledged cloud architecture.

**The evolution path:**
Local Scripts & MQTT (Current) → Edge Processing & Database (Current) → REST API (Current) → Docker → CI/CD → Cloud Deployment → System Monitoring → Time-Series ML → Anomaly Detection (Predictive Maintenance).

**Current Capabilities (Implemented):**
- ✅ Software-based vehicle telemetry generation
- ✅ Asynchronous MQTT publish/subscribe messaging
- ✅ Real-time data validation and processing
- ✅ Relational data persistence (SQLite) and audit logging (JSONL)
- ✅ REST API with Swagger documentation for querying fleet telemetry

**Future Capabilities (Not Yet Implemented):**
- ❌ Docker containerization
- ❌ CI/CD pipelines
- ❌ Cloud infrastructure deployment
- ❌ Advanced observability / Grafana dashboards
- ❌ Machine learning or predictive maintenance models

---

## 4. Current Architecture

```text
                         EdgePulse
              Vehicle Telemetry Pipeline

┌─────────────────────┐
│  Vehicle Simulator  │
│  simulator/         │
│  publisher.py       │
└──────────┬──────────┘
           │
           │ MQTT Publish
           │ telemetry/car1
           ▼
┌─────────────────────┐
│   MQTT Broker       │
│   Eclipse Mosquitto │
│   localhost:1883    │
└──────────┬──────────┘
           │
           │ MQTT Subscribe
           ▼
┌─────────────────────┐
│   Edge Consumer     │
│   edge/consumer.py  │
├─────────────────────┤
│ JSON Parsing        │
│ Validation          │
│ Timestamping        │
│ Persistence         │
└──────────┬──────────┘
           │
           ├──────────────────┐
           │                  │
           ▼                  ▼
┌─────────────────┐   ┌────────────────────┐
│     SQLite      │   │   JSONL Audit Log  │
│ edgepulse.db    │   │ telemetry.jsonl    │
└────────┬────────┘   └────────────────────┘
         │
         │ SQL Queries
         ▼
┌─────────────────────┐
│    FastAPI Backend  │
│    api/main.py      │
├─────────────────────┤
│ REST API            │
│ Request Validation  │
│ Response Models     │
└──────────┬──────────┘
           │
           ▼
┌────────────────────────────┐
│ REST Clients               │
│ Browser / curl / Postman   │
│ Other services             │
└────────────────────────────┘
```

---

## 5. End-to-End Data Flow

Here is the lifecycle of a single telemetry message in EdgePulse:

1. **Generation:** `simulator/publisher.py` generates a telemetry reading (e.g. speed, RPM, temp).
2. **Publishing:** The simulator publishes this payload as a JSON string to the MQTT broker on topic `telemetry/car1`.
3. **Broker Transport:** Eclipse Mosquitto receives the message.
4. **Subscription:** `edge/consumer.py`, which is listening to `telemetry/car1`, receives the message.
5. **Parsing & Validation:** The consumer parses the JSON and passes it to `edge/validation.py`. It checks for required fields, numeric formats, and acceptable data ranges (e.g. speed cannot be negative).
6. **Enrichment:** The consumer adds a server-side ISO-8601 timestamp to the reading.
7. **Persistence:** The processed record is inserted into the `telemetry` table in the local `edgepulse.db` SQLite database.
8. **Audit Logging:** A copy of the record is appended to the `logs/telemetry.jsonl` file.
9. **API Request:** A client makes an HTTP GET request to `http://127.0.0.1:8000/telemetry/latest`.
10. **Query & Response:** `api/main.py` queries the SQLite database, formats the row using a Pydantic model, and returns a JSON response to the client.

**Realistic Payload Example (after processing):**
```json
{
  "id": 142,
  "timestamp": "2026-09-17T03:10:24.482012",
  "vehicle_id": "car_01",
  "speed": 82.0,
  "rpm": 3400.0,
  "engine_temp": 95.0
}
```

---

## 6. Communication Architecture

The system purposely splits concerns across different protocols:

- **Vehicle → Broker (MQTT):** MQTT is lightweight, handles poor connectivity gracefully, and uses a Publish/Subscribe model. This is perfect for IoT devices pushing high-frequency data asynchronously without needing to wait for a response.
- **Broker → Edge Consumer (MQTT):** The consumer subscribes to the broker and handles the data stream as it arrives. 
- **Edge → Database (SQLite/SQL):** The consumer uses SQL to insert data into a local relational database for structured storage.
- **Client → API (HTTP/REST):** Clients query the data on-demand. HTTP is the universal standard for client-server communication.
- **API → Database (SQL):** The API uses SQL to fetch the requested records.

**Why separate MQTT and HTTP?**
The vehicle should not be tightly coupled to the API, nor should it wait for an HTTP response for every reading. Using MQTT as an ingestion buffer means the API can safely go down, or scale up independently, without affecting the car's ability to report data.

---

## 7. Core Components

### Vehicle Simulator
**Source:** `simulator/publisher.py`
Simulates a vehicle (`car_01`) continuously generating data. Every 1 second, it randomizes speed (40–140 km/h), RPM (2000–6000), and engine temperature (80–110°C). It serializes this data into JSON and publishes it to the `telemetry/car1` topic via the MQTT broker configured in `config.py`.

### MQTT Broker
An instance of **Eclipse Mosquitto** running locally on port 1883. It acts as the routing hub connecting the simulator and the edge consumer.

### Edge Consumer
**Source:** `edge/consumer.py`
Runs continuously, waiting for messages on `telemetry/car1`. Upon receiving a payload, it:
1. Catches JSON decoding errors safely.
2. Validates the data using the validation module.
3. Injects a server timestamp.
4. Saves to SQLite and appends to a JSONL audit log.
It is robust enough to log warnings for malformed data without crashing.

### Validation Layer
**Source:** `edge/validation.py`
Extracted to keep business rules centralized. It ensures:
- All required keys exist (`vehicle_id`, `speed`, `rpm`, `engine_temp`).
- All sensor values are numeric.
- `speed` and `rpm` are non-negative.
- `engine_temp` is within a realistic range (-50°C to 300°C).

### Database Layer
**Source:** `database/db.py`
A local SQLite database (`edgepulse.db`) mapping to a single table (`telemetry`). It provides efficient insertion and retrieval mechanisms (e.g., finding the latest record or filtering by `vehicle_id`). Using a relational database over flat files allows for performant SQL querying by the FastAPI application.

### FastAPI Backend
**Source:** `api/main.py`
A modern, asynchronous REST API. It handles incoming HTTP requests, queries the SQLite database, validates data shapes with Pydantic response models, and serves data to end-clients. It automatically serves interactive OpenAPI (Swagger) docs.

---

## 8. Telemetry Model

**Raw Payload (From Simulator):**
```json
{
  "vehicle_id": "car_01",
  "speed": 115,
  "rpm": 4500,
  "engine_temp": 92
}
```

**Processed Payload (Stored & Served):**
```json
{
  "id": 1,
  "timestamp": "2026-09-17T03:10:24.482012",
  "vehicle_id": "car_01",
  "speed": 115.0,
  "rpm": 4500.0,
  "engine_temp": 92.0
}
```

---

## 9. Repository Contents

- `/simulator` - Vehicle simulation script (`publisher.py`).
- `/edge` - MQTT consumer (`consumer.py`) and data validation logic (`validation.py`).
- `/api` - FastAPI web application (`main.py`).
- `/database` - SQLite connection and query layer (`db.py`).
- `/tests` - Pytest suite covering the consumer, database, and API layers.
- `/logs` - Output directory for the append-only `telemetry.jsonl` audit log.
- `config.py` - Centralized configuration variables (host, port, topic, paths).
- `requirements.txt` - Python package dependencies.
- `README.md` - This documentation.

---

## 10. Installation & Setup

**Prerequisites:**
- Python 3.12+
- Eclipse Mosquitto installed and running as a service locally.

**Environment Setup:**
1. Clone the repository and navigate to the project root.
2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 11. Running the System

You will need multiple terminal windows (with the `.venv` activated in each) to run the full pipeline locally.

**1. MQTT Broker**
Ensure Mosquitto is running on your machine (port 1883).

**2. Edge Consumer**
Start the consumer to listen for telemetry and initialize the database.
```powershell
python edge/consumer.py
```

**3. Vehicle Simulator**
Start publishing data.
```powershell
python simulator/publisher.py
```

**4. FastAPI Backend**
Start the REST API server.
```powershell
uvicorn api.main:app --reload
```

---

## 12. API Documentation (Endpoints)

Once FastAPI is running, you can interactively test all endpoints by visiting the Swagger UI at `http://127.0.0.1:8000/docs`.

### Available Endpoints:
- **`GET /`** - Root status check.
- **`GET /health`** - System health check. Returns `200 OK` if the database is reachable, `503 Service Unavailable` if it is not.
- **`GET /telemetry/latest`** - Returns the single most recent telemetry record. Returns `404 Not Found` if the database is empty.
- **`GET /telemetry`** - Returns a list of recent telemetry records. Accepts a `limit` query parameter (default 20, max 100). Returns `422 Unprocessable Entity` for invalid limits.
- **`GET /telemetry/{vehicle_id}`** - Returns recent telemetry filtered by vehicle ID. Accepts a `limit` query parameter. Returns `404 Not Found` if the vehicle has no records.

---

## 13. Testing

EdgePulse includes a full test suite built with `pytest` covering validation logic, database queries, and API endpoints. The tests use temporary in-memory databases and do not pollute your main `edgepulse.db`.

Run the entire test suite:
```powershell
pytest tests/ -v
```

---

## Phase 1 — Stateful Vehicle Simulation + Fault Injection

Phase 1 replaces the simple random simulator with a realistic, stateful multi-vehicle simulation capable of producing temporally correlated telemetry and controlled fault scenarios. The output is designed to become the foundation for future anomaly-detection and predictive-maintenance experiments.

Phase 1 does NOT implement any ML or anomaly detection. It only creates the high-quality synthetic dataset that future phases will consume.

---

### Simulator Architecture

```
simulator/
├── vehicle.py      — VehicleProfile, VehicleStateMachine, VehicleState
├── faults.py       — FaultScenario, FaultController, FaultType, FaultState
└── publisher.py    — CLI entry point, MQTT publisher, run_simulation()
```

The three layers are independent:
- `vehicle.py` — pure state machine, no I/O
- `faults.py` — pure fault logic, no I/O
- `publisher.py` — wires them together and handles MQTT + CLI

---

### Vehicle States

```
STOPPED → IDLE → ACCELERATING → CRUISING → DECELERATING → IDLE → ...
```

| State | Speed | RPM | Temperature |
|---|---|---|---|
| STOPPED | 0 | 0 | ambient |
| IDLE | 0 | ~850 | gradually falls to idle baseline |
| ACCELERATING | ↑ rising | ↑ rising + boost | ↑ rising under load |
| CRUISING | ≈ stable ±noise | ≈ stable ±noise | ≈ stable at baseline |
| DECELERATING | ↓ falling | ↓ falling | ↓ slowly falls |

### Sensor Relationships

- **RPM** = `idle_rpm + speed × rpm_per_kmh + accel_boost` (during acceleration)
- **engine_temp** — converges toward a state-dependent target at a vehicle-specific rate
- **battery_voltage** — slow mean-reversion to nominal with small Gaussian noise
- **fuel_level** — decreases proportionally to load; never refuels in simulation

### Vehicle Fleet

| Vehicle | Baseline Temp | Idle RPM | Cruise Speed | Notes |
|---|---|---|---|---|
| CAR-01 | 88°C | 820 | 100 km/h | Reference vehicle |
| CAR-02 | 93°C | 900 | 95 km/h | Runs hotter, heats faster |
| CAR-03 | 86°C | 800 | 110 km/h | Cooler, faster, lower fuel use |

---

### Fault Scenarios

| Fault | What it does | Telemetry signature |
|---|---|---|
| COOLING_DEGRADATION | engine_temp rises above normal | Gradual temp increase; slow recovery |
| RPM_INSTABILITY | RPM oscillates around expected value | RPM std dev rises while speed is stable |
| BATTERY_DEGRADATION | battery_voltage drops and becomes noisy | Voltage mean decreases; variance increases |
| SENSOR_DRIFT | Temperature sensor bias accumulates | Reported temp drifts above actual over time |
| EXCESSIVE_IDLING | Vehicle stays in IDLE far longer than normal | Long stretches of speed=0 and low RPM |

All faults produce **valid telemetry** — values stay within Phase 0 validation ranges.

### Fault Lifecycle

```
NORMAL → STARTING → ACTIVE → RECOVERING → RESOLVED → NORMAL
```

Each fault has configurable `ramp_time` (STARTING), `duration` (ACTIVE), and `recovery_time` (RECOVERING).

---

### Ground-Truth Labeling

Fault metadata is recorded separately from the production telemetry so it never contaminates the MQTT pipeline or SQLite schema. Every simulated step produces a `GroundTruthRecord`:

```
sim_time | vehicle_id | message_id | event_timestamp | fault_type | fault_state | severity
```

This allows future ML models to be evaluated precisely: "at timestamp T, was vehicle V experiencing a fault, and if so, which one?"

---

### Running the Simulator

Activate the virtual environment first:
```powershell
.venv\Scripts\Activate.ps1
```

**Normal mode** (healthy vehicles only):
```powershell
python simulator/publisher.py --mode normal --seed 42 --steps 600
```

**Fault mode** (one fault per vehicle):
```powershell
python simulator/publisher.py --mode fault --seed 42 --steps 900
```

**Mixed mode** (multiple faults per vehicle with recovery):
```powershell
python simulator/publisher.py --mode mixed --seed 7 --steps 1800
```

**Export ground-truth CSV** for ML experiments:
```powershell
python simulator/publisher.py --mode mixed --seed 42 --steps 1800 --export data/ground_truth.csv
```

**Options:**
```
--mode     normal | fault | mixed   (default: normal)
--seed     integer                  (default: random — not reproducible)
--steps    integer                  (default: 600)
--dt       float seconds            (default: 1.0)
--export   path/to/gt.csv           (optional)
--vehicles CAR-01 CAR-02 ...        (default: all three)
```

### Reproducibility

Running with the same `--seed` produces substantially identical telemetry. Use seeds in experiments to ensure results are comparable across algorithm versions.

---

### Current Milestone

**Phase 1 — Stateful Simulation + Fault Injection** ✅

Realistic multi-vehicle simulation with five fault types, full lifecycle tracking, ground-truth labeling, reproducible seeds, and CSV export for ML experiments.

---

## Future Roadmap

| Phase | Focus |
|---|---|
| ✅ 0 | Foundation hardening — MQTT pipeline, SQLite, FastAPI |
| ✅ 1 | Stateful simulation, fault injection, ground-truth labeling |
| ✅ 2 | Edge feature engineering + vehicle-specific baselines |
| ✅ 3 | LSTM-based temporal anomaly detection |
| ⬜ 4 | Docker containerisation |
| ⬜ 5 | CI/CD — GitHub Actions |
| ⬜ 6 | Cloud deployment (AWS) |
| ⬜ 7 | Monitoring — Prometheus + Grafana |
| ⬜ 8 | Production hardening + predictive maintenance |

---

## Phase 2 — Feature Engineering and Statistical Baselines

Phase 2 adds temporal feature extraction and vehicle-specific statistical anomaly detection, building on Phase 1 telemetry.

**New modules:**
- `edge/features.py` — rolling features: temp mean/std/slope, RPM variability, speed change rate, voltage trend, fuel rate, idle duration
- `edge/baseline.py` — per-vehicle Welford running baseline; configurable readiness threshold; outlier and fault-period exclusion
- `edge/scoring.py` — z-score deviation scoring with configurable ELEVATED/SEVERE thresholds
- `edge/pipeline.py` — wires feature extraction, baseline updates, and scoring; called after raw telemetry is safely stored

**API endpoints added:** `/features/{id}`, `/baseline/{id}`, `/scores/{id}`

**Phase 2 is complete.** ✅

---

## Phase 3 — LSTM Autoencoder Anomaly Detection

Phase 3 adds an LSTM autoencoder that learns normal temporal patterns and detects anomalous sequences through reconstruction error. The existing Phase 2 statistical detector remains unchanged.

### What Phase 3 adds

- **LSTM autoencoder** (`lstm/model.py`): encoder-decoder architecture, input `(batch, seq_len=30, 12 features)`, hidden dim 32, MSE reconstruction loss
- **Data pipeline** (`lstm/dataset.py`): leakage-safe splits, median imputation, MinMax scaling, boundary-safe sequence building
- **Training** (`lstm/train.py`): offline training with early stopping, best-checkpoint restoration, explicit CLI
- **Artifacts** (`lstm/artifacts.py`): atomic save/load of model weights, scaler/imputer parameters, threshold, and metadata
- **Inference** (`lstm/inference.py`): threshold calibration (95th-percentile of healthy val errors), per-vehicle bounded sequence buffers, `AnomalyScorer`
- **Evaluation** (`lstm/evaluate.py`): comparative evaluation vs Phase 2 with sequence-level and event-level metrics
- **API endpoints**: `GET /lstm/status`, `GET /lstm/anomaly/{vehicle_id}`

### Phase 2 vs Phase 3 comparison

| Aspect | Phase 2 (Statistical) | Phase 3 (LSTM) |
|---|---|---|
| Method | Z-score per feature vs running baseline | MSE reconstruction error on temporal sequences |
| What it detects | Per-feature deviation from per-vehicle norm | Unusual temporal pattern across all 12 features |
| Training required | No (baselines update online) | Yes (offline, explicit `python -m lstm.train`) |
| Warm-up | ~30 samples per feature | `seq_len` (default 30) observations |
| Result | Per-feature severity level | Single reconstruction error + threshold |
| Interpretability | "feature X is N std devs above baseline" | "sequence is temporally unusual" |
| Online retraining | Automatic (online Welford update) | Never (offline only) |

### Dependencies

PyTorch and NumPy are required. Both are listed in `requirements.txt` and installed in the `.venv`.

```powershell
pip install -r requirements.txt
```

### Step 1: Prepare training data and train the model

Training uses the existing Phase 1 simulator internally. No manual data export is needed.

```powershell
# Train with default settings (seed=42, 600 steps/vehicle, seq_len=30, hidden=32)
python -m lstm.train

# Train with explicit options
python -m lstm.train --seq-len 30 --hidden-dim 32 --epochs 100 --seed 42 --model-dir models/
```

Configurable parameters:

| Parameter | Default | Description |
|---|---|---|
| `--seq-len` | 30 | Observations per sequence |
| `--hidden-dim` | 32 | LSTM hidden state size |
| `--epochs` | 100 | Maximum training epochs |
| `--batch-size` | 32 | Mini-batch size |
| `--lr` | 1e-3 | Adam learning rate |
| `--patience` | 10 | Early stopping patience |
| `--seed` | 42 | Random seed |
| `--model-dir` | models/ | Artifact output directory |
| `--train-steps` | 600 | Simulation steps for training split |
| `--val-steps` | 600 | Simulation steps for validation split |

The trained model is saved to `models/` (ignored by git).

### Step 2: Calibrate the anomaly threshold

Threshold calibration uses healthy validation sequences only and is run automatically during evaluation, or can be triggered explicitly in Python:

```python
from lstm.inference import calibrate_threshold
from lstm.artifacts import load_model
from lstm.dataset import prepare_standard_splits

train_split, val_split, _ = prepare_standard_splits(seq_len=30)
artifact = load_model("models/")
threshold = calibrate_threshold(
    val_sequences=val_split.sequences,
    model=artifact["model"],
    imputer=artifact["imputer"],
    scaler=artifact["scaler"],
    percentile=95.0,
    model_dir="models/",
)
```

### Step 3: Run evaluation

```powershell
# Evaluate using an existing trained model
python -m lstm.evaluate

# Retrain and evaluate in one step
python -m lstm.evaluate --retrain --epochs 100

# Full options
python -m lstm.evaluate --model-dir models/ --out-dir evaluation-results/ --test-steps 900
```

Results are written to `evaluation-results/` (ignored by git):
- `results.json` — machine-readable metrics
- `report.md` — human-readable Markdown report

### Step 4: Enable LSTM inference in the consumer

The LSTM inference hook is disabled by default. To enable it, initialise `FeaturePipeline` with:

```python
from edge.pipeline import FeaturePipeline
pipeline = FeaturePipeline(lstm_enabled=True, lstm_model_dir="models/")
```

Or set environment variables and modify `edge/consumer.py` to pass `lstm_enabled=True`.

### Interpreting reconstruction error and threshold

- **Reconstruction error** is the MSE between the model's reconstructed sequence and the original input.
- A **high reconstruction error** means the sequence is temporally unusual relative to the healthy patterns learned during training.
- The **threshold** is the 95th percentile of reconstruction errors on healthy validation sequences.
- A result above the threshold is flagged as `is_anomalous=True`.
- Reconstruction error does **not** identify a specific fault type. It only indicates that the observed temporal pattern is unusual.
- A false alarm can occur when a vehicle operates in a mode not represented in training (e.g. prolonged idling, unusual acceleration).

### Current limitations and known failure modes

- All training, validation, and test data are generated from a software simulator. Real-world performance may differ significantly.
- Phase 2 baseline may not converge well with short training runs (~600 steps). Longer runs improve baseline quality.
- Overlapping sliding windows inflate sequence-level TP/FP counts. Use event-level metrics for a fair comparison.
- The LSTM is trained on normal driving across three vehicles. Faults that affect all vehicles simultaneously (e.g. shared sensor error) may not be detected if they also shift the baseline.
- Wall-clock `event_timestamp` from the simulator reflects execution speed, not simulation time. The dataset pipeline uses `sim_time` to guarantee monotonic timestamps. This is documented in `lstm/dataset.py`.
- Do not interpret simulator results as a guarantee of real-world anomaly detection accuracy.

### Phase 3 is complete. ✅
