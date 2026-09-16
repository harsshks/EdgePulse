# EdgePulse — Vehicle Telemetry & Fleet Health Platform

EdgePulse is a real-time vehicle telemetry platform designed to simulate how connected vehicles can transmit operational data to an edge processing system, persist that data, and expose it through a backend API for fleet monitoring and analysis.

The project models a simplified real-world scenario:

> A fleet operator needs a reliable way to collect vehicle telemetry, validate incoming data, store historical readings, and make vehicle health information available to monitoring systems.

EdgePulse is being built incrementally, starting with local telemetry ingestion and backend services and progressing toward containerized deployment, cloud infrastructure, observability, and time-series anomaly detection.

---

## 1. The Problem

Modern connected vehicles continuously generate telemetry such as:

- Vehicle speed
- Engine RPM
- Engine temperature
- Vibration and other sensor readings
- Vehicle identifiers
- Timestamps

Collecting this data is only the first step.

A useful fleet monitoring system must also:

1. Receive telemetry continuously.
2. Handle asynchronous sensor data.
3. Validate incoming data.
4. Process data before persistence.
5. Store historical telemetry.
6. Provide an interface for querying vehicle data.
7. Detect abnormal vehicle behavior.
8. Eventually support cloud deployment and monitoring.

EdgePulse is designed as a small-scale implementation of this pipeline.

The vehicle is simulated in software, but the architecture is designed around the same fundamental communication and backend concepts used in connected-device systems.

---

# 2. What is EdgePulse?

EdgePulse consists of four primary components:

### Vehicle Simulator

Simulates a connected vehicle and periodically publishes telemetry data.

### MQTT Broker

Acts as the messaging layer between vehicles and the edge processing system.

### Edge Consumer

Receives telemetry, validates and processes the data, adds timestamps, and persists it.

### FastAPI Backend

Provides REST APIs through which clients can retrieve the stored telemetry.

At the current stage, the system runs locally using:

- Python
- MQTT
- Mosquitto
- SQLite
- FastAPI
- Pydantic
- pytest

---

# 3. Current Architecture

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