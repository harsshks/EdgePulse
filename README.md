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