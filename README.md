# EdgePulse — Cloud-Native Vehicle Telemetry Platform

## Project Overview
EdgePulse is a practical learning project that simulates a real vehicle telemetry pipeline. A vehicle simulator publishes sensor data over MQTT. An edge consumer receives, validates, and persists that data. A FastAPI backend exposes the data via a REST API.

The goal is to build this step-by-step, understanding every layer from the network protocol to the HTTP endpoint.

---

## Current Architecture (Week 3)

```
🚗 Vehicle Simulator (simulator/publisher.py)
        │
        │  MQTT publish → topic: telemetry/car1
        ▼
📡 Mosquitto MQTT Broker (localhost:1883)
        │
        │  MQTT subscribe
        ▼
🖥️  Edge Consumer (edge/consumer.py)
        │
        ├── JSON parsing
        ├── Validation (edge/validation.py)
        ├── Timestamping
        ├── SQLite persistence (database/db.py → edgepulse.db)
        └── JSONL audit log (logs/telemetry.jsonl)
        
🌐 FastAPI REST API (api/main.py)
        │
        │  Reads from edgepulse.db
        ▼
REST Clients (browser, Postman, curl, other services)
```

### Two separate communication channels
| Channel | Protocol | Direction | Purpose |
|---|---|---|---|
| Vehicle → Edge | MQTT | push (async) | Real-time telemetry ingestion |
| Client → API | HTTP (REST) | request/response | Query stored telemetry |

MQTT and HTTP are independent. The consumer runs 24/7 writing to SQLite. The API reads from SQLite whenever a client asks.

---

## Technologies

| Technology | Purpose |
|---|---|
| Python 3.12 | Core language |
| paho-mqtt | MQTT client library |
| Eclipse Mosquitto | MQTT broker |
| SQLite | Embedded persistent database |
| FastAPI | HTTP REST API framework |
| Pydantic | Response data validation and serialisation |
| Uvicorn | ASGI server that runs FastAPI |
| pytest | Test runner |

---

## Project Structure

```
EdgePulse/
├── simulator/
│   └── publisher.py        # Vehicle simulator — generates and publishes telemetry
├── edge/
│   ├── consumer.py         # MQTT subscriber — validates, timestamps, persists
│   └── validation.py       # Telemetry validation rules (shared module)
├── api/
│   └── main.py             # FastAPI application and route definitions
├── database/
│   └── db.py               # SQLite connection and query functions
├── tests/
│   ├── test_consumer.py    # Validation and consumer tests
│   ├── test_database.py    # SQLite layer tests
│   └── test_api.py         # API endpoint tests
├── logs/
│   └── telemetry.jsonl     # Append-only audit log (runtime, gitignored)
├── config.py               # Central configuration (broker host, port, paths)
├── requirements.txt        # Python dependencies
├── edgepulse.db            # SQLite database (runtime, gitignored)
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.12+
- Eclipse Mosquitto installed and running

### 1. Mosquitto (Windows)
Download from [mosquitto.org/download](https://mosquitto.org/download/), install, then start the service:
```powershell
Start-Service -Name "mosquitto"
```

### 2. Python Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Full Pipeline

You need four separate terminals with the virtual environment active in each.

**Terminal 1 — MQTT Broker** (if not already running as a service):
```powershell
mosquitto
```

**Terminal 2 — Edge Consumer:**
```powershell
python edge/consumer.py
```
Expected output:
```
🖥️  Starting Edge Consumer...
🗄️  Database ready.
✅ Connected to broker at localhost:1883
📡 Subscribed to topic: telemetry/car1
```

**Terminal 3 — Vehicle Simulator:**
```powershell
python simulator/publisher.py
```
Expected output:
```
🚗 Starting Vehicle Simulator (car_01)...
✅ Connected to broker at localhost:1883
📤 Published to telemetry/car1: {"vehicle_id": "car_01", "speed": 87, ...}
```

**Terminal 4 — FastAPI:**
```powershell
uvicorn api.main:app --reload
```
Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```
The `--reload` flag restarts the server automatically when you edit `api/main.py`. Remove it in production.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service identity and status |
| GET | `/health` | Health check — verifies DB is reachable |
| GET | `/telemetry/latest` | The most recently received telemetry record |
| GET | `/telemetry` | List of recent records (default 20, max 100) |
| GET | `/telemetry/{vehicle_id}` | Records for a specific vehicle |

### Query Parameters
`GET /telemetry?limit=5` — returns the 5 most recent records.
`GET /telemetry/car_01?limit=10` — returns 10 records for car_01.

`limit` must be an integer between 1 and 100. FastAPI returns 422 automatically if it isn't.

### Example Response — `/telemetry/latest`
```json
{
  "id": 42,
  "timestamp": "2026-08-27T10:35:01.123456",
  "vehicle_id": "car_01",
  "speed": 97.0,
  "rpm": 4200.0,
  "engine_temp": 91.0
}
```

### HTTP Status Codes Used
| Code | Meaning |
|---|---|
| 200 | Success |
| 404 | No records found for that vehicle or the table is empty |
| 422 | Query parameter failed validation (e.g. limit=0) |
| 503 | Database is unreachable |

---

## Swagger / OpenAPI Docs

With the API running, open your browser at:
```
http://127.0.0.1:8000/docs
```
FastAPI automatically generates interactive documentation from the route definitions and Pydantic models. You can execute API calls directly from the browser — no Postman required. This is one of FastAPI's most practical features.

---

## SQLite Database

The database file `edgepulse.db` is created automatically when the consumer or API starts. You can inspect it directly:
```powershell
# Open the SQLite shell (if sqlite3.exe is on your PATH)
sqlite3 edgepulse.db
.tables
SELECT * FROM telemetry ORDER BY id DESC LIMIT 5;
.quit
```

### Why SQLite alongside JSONL?
- The JSONL file is a raw audit trail — human-readable, easy to inspect with a text editor.
- SQLite lets you query: "give me the last 20 records for car_01 ordered by time." Doing that from JSONL requires reading the whole file in Python.
- Both are retained in Week 3 for learning value. In a production system you'd pick one.

---

## Configuration

All tunable values live in `config.py`:
```python
MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "telemetry/car1"
LOG_FILE = "logs/telemetry.jsonl"
DATABASE_PATH = "edgepulse.db"
```
Change values here once instead of hunting through multiple files.

---

## Testing

```powershell
# Run all tests
.venv\Scripts\pytest tests/ -v

# Run a specific file
.venv\Scripts\pytest tests/test_api.py -v
```

24 tests across three files:
- `test_consumer.py` — validation rules, process_telemetry function
- `test_database.py` — insert, retrieve, latest, limit, vehicle filter
- `test_api.py` — all endpoints including 404 and 422 error cases

Tests use temporary SQLite databases so they never touch `edgepulse.db`.

---

## Example Telemetry Payload

Published by the simulator (raw, before processing):
```json
{"vehicle_id": "car_01", "speed": 120, "rpm": 5200, "engine_temp": 94}
```

After the consumer processes it (stored in SQLite and JSONL):
```json
{
  "id": 1,
  "timestamp": "2026-08-27T10:35:01.123456",
  "vehicle_id": "car_01",
  "speed": 120,
  "rpm": 5200,
  "engine_temp": 94
}
```

---

## Week 2 MQTT Pipeline

Week 2 established the core telemetry pipeline:
- Python virtual environment and paho-mqtt
- MQTT publisher (vehicle simulator)
- MQTT broker (Mosquitto)
- MQTT subscriber (edge consumer)
- JSON parsing, field validation, timestamping
- Local JSONL logging
- Git/GitHub, README, basic tests

This pipeline is fully preserved in Week 3.

---

## Current Milestone

**Week 3 — Local Backend** ✅

MQTT pipeline + SQLite persistence + FastAPI REST API running locally.

---

## Future Roadmap

| Week | Focus |
|---|---|
| Week 4 | Docker — containerise consumer, API, broker |
| Week 5 | CI/CD — GitHub Actions, automated tests on push |
| Week 6 | Cloud — deploy to AWS (ECS or EC2) |
| Week 7 | Monitoring — Prometheus metrics, Grafana dashboards |
| Week 8 | Streaming — Kafka or MQTT → event stream |
| Week 9 | AI/MLOps — anomaly detection, predictive maintenance |
| Future | Kubernetes, managed databases (RDS/TimescaleDB) |

Not implemented yet. Not added yet. Each has a reason for its week.
