# EdgePulse

## What is EdgePulse?
EdgePulse is a Cloud-Native Vehicle Telemetry Platform built as a practical learning project. It simulates vehicle telemetry data, transmits it via MQTT, and processes it at the edge.

## Current Architecture (Week 2)
```
🚗 Vehicle Simulator (simulator/)
        │
        │  MQTT publish (Topic: telemetry/car1)
        ▼
📡 MQTT Broker (Mosquitto)
        │
        │  MQTT subscribe
        ▼
🖥️ Edge Consumer (edge/)
        │
        │  Parse → Validate → Timestamp
        ▼
📁 Local Telemetry Log (logs/telemetry.jsonl)
```

## Technologies
- **Python 3.12**
- **paho-mqtt** (MQTT Client Library)
- **Eclipse Mosquitto** (MQTT Broker)

## Project Structure
- `simulator/`: Contains the Python script that generates fake vehicle telemetry.
- `edge/`: Contains the Python script that receives and processes the telemetry.
- `logs/`: Directory where the processed JSONL telemetry logs are stored.
- `tests/`: Directory reserved for future unit testing.

## Prerequisites
1. **Python 3.12+** installed.
2. **Mosquitto** installed and running locally on port `1883`.

## Setup Instructions

### 1. MQTT Broker Setup (Windows)
1. Download Mosquitto from [mosquitto.org](https://mosquitto.org/download/).
2. Install with default settings.
3. Open an Administrator PowerShell and run:
   ```powershell
   Start-Service -Name "mosquitto"
   ```

### 2. Python Environment Setup
Open a standard terminal in the project root and run:
```powershell
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Running the Pipeline

You need two separate terminals. Ensure your virtual environment is active in both (or use the full `.venv\Scripts\python.exe` path).

**Terminal 1 (Start the Consumer):**
```powershell
python edge/consumer.py
```

**Terminal 2 (Start the Simulator):**
```powershell
python simulator/publisher.py
```

## Example Telemetry Payload
The simulator sends raw JSON to the broker:
```json
{"vehicle_id": "car_01", "speed": 120, "rpm": 5200, "engine_temp": 94}
```

The consumer processes it and logs it with a timestamp to `logs/telemetry.jsonl`:
```json
{"vehicle_id": "car_01", "speed": 120, "rpm": 5200, "engine_temp": 94, "timestamp": "2026-08-16T12:00:00.123456"}
```

## Future Roadmap (Not yet implemented)
- **FastAPI**: REST API to serve telemetry data.
- **Database**: Migrate from local JSONL logging to PostgreSQL/TimescaleDB.
- **Docker**: Containerize the consumer, publisher, and broker.
- **Cloud/Kubernetes**: Deploy to AWS/GCP using K8s.
- **CI/CD**: Automated testing and deployment pipelines.
- **Monitoring**: Prometheus and Grafana dashboards.
- **AI/MLOps**: Predictive maintenance modeling.
