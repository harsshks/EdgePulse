# database/db.py
# SQLite persistence layer for EdgePulse telemetry.
#
# WHY SQLite instead of just JSONL?
# - JSONL is a flat append-only log. To answer "what is the latest reading for
#   car_01?" you have to read the entire file and filter in Python.
# - SQLite stores data in a structured table. You can ask it questions like
#   SELECT * WHERE vehicle_id='car_01' ORDER BY id DESC LIMIT 1
#   and the database engine handles it efficiently.
# - SQLite requires zero external setup (no server, no credentials). The whole
#   database is a single file (edgepulse.db). Perfect for a learning project.
#
# We KEEP the JSONL log as well:
# - It acts as a raw audit trail / backup.
# - It's human-readable with a text editor.
# - In production you'd pick one, but for learning it's valuable to see both.

import sqlite3
from config import DATABASE_PATH


def get_connection() -> sqlite3.Connection:
    """
    Opens (or creates) the SQLite database file and returns a connection.
    Setting row_factory means rows come back as dict-like objects,
    so we can write row["speed"] instead of row[2].

    check_same_thread=False is required when the same database is accessed
    from multiple threads (e.g. FastAPI's thread pool, the MQTT consumer,
    and tests). SQLite handles this safely as long as each function opens
    its own connection and closes it when done — which is exactly what we do.
    """
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Creates the telemetry table if it doesn't already exist.
    Safe to call every time the application starts — CREATE TABLE IF NOT EXISTS
    is idempotent (running it twice has no side effect).

    Table design:
      id           — INTEGER PRIMARY KEY auto-incremented by SQLite.
                     A primary key uniquely identifies every row. No two rows
                     can share the same id. It's the 'address' of a record.
      timestamp    — ISO-8601 string from the edge consumer.
      vehicle_id   — Which vehicle sent this reading.
      speed        — km/h
      rpm          — Engine revolutions per minute.
      engine_temp  — Degrees Celsius.
    """
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                vehicle_id  TEXT    NOT NULL,
                speed       REAL    NOT NULL,
                rpm         REAL    NOT NULL,
                engine_temp REAL    NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


def insert_telemetry(record: dict) -> None:
    """
    Inserts one validated telemetry record into the database.
    Uses parameterized values (the ? placeholders) to prevent SQL injection —
    a good habit even in local projects.
    """
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO telemetry (timestamp, vehicle_id, speed, rpm, engine_temp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record["timestamp"],
                record["vehicle_id"],
                record["speed"],
                record["rpm"],
                record["engine_temp"],
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_latest_telemetry() -> dict | None:
    """
    Returns the single most recently inserted telemetry row, or None if the
    table is empty.
    ORDER BY id DESC picks the highest (most recently inserted) id.
    LIMIT 1 tells SQLite to stop after one row — efficient.
    """
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM telemetry ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_telemetry(limit: int = 20) -> list[dict]:
    """
    Returns the most recent `limit` telemetry rows, newest first.
    The caller (FastAPI endpoint) validates the limit before passing it here.
    """
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM telemetry ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_telemetry_by_vehicle(vehicle_id: str, limit: int = 20) -> list[dict]:
    """
    Returns recent telemetry for a specific vehicle.
    Useful when the fleet grows beyond a single car.
    """
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM telemetry WHERE vehicle_id = ? ORDER BY id DESC LIMIT ?",
            (vehicle_id, limit),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
