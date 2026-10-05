# database/db.py
# SQLite persistence layer for EdgePulse telemetry.
#
# Schema design (Phase 0):
#
#   id                 INTEGER PK AUTOINCREMENT — internal row identifier
#   message_id         TEXT UNIQUE              — deduplication key from the vehicle/consumer
#   event_timestamp    TEXT NOT NULL            — when the vehicle generated the reading (UTC ISO-8601)
#   ingestion_timestamp TEXT NOT NULL           — when EdgePulse received and stored it (UTC ISO-8601)
#   vehicle_id         TEXT NOT NULL            — which vehicle sent this
#   speed              REAL NOT NULL            — km/h
#   rpm                REAL NOT NULL            — engine RPM
#   engine_temp        REAL NOT NULL            — °C
#
# WHY two timestamps?
#   event_timestamp    = vehicle clock time. Needed for offline/delayed telemetry
#                        where the vehicle buffers data and sends it later.
#   ingestion_timestamp = server time when EdgePulse stored it.
#                        Always reliable because it comes from the server clock.
#   Without both, you cannot tell whether a reading is delayed or real-time.
#
# WHY message_id?
#   MQTT at QoS 1 may deliver the same message more than once on reconnect.
#   Without a unique message_id, the same telemetry event gets stored twice.
#   INSERT OR IGNORE with a UNIQUE constraint on message_id silently discards duplicates.
#
# WHY indexes?
#   Queries filter by vehicle_id and sort by event_timestamp.
#   Without indexes, SQLite scans every row. With indexes, lookups are O(log n).

import logging
import sqlite3

from config import DATABASE_PATH

logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    """
    Opens (or creates) the SQLite database and returns a connection.
    row_factory=sqlite3.Row lets callers use row["column"] notation.
    check_same_thread=False: each function opens and closes its own connection,
    so cross-thread use is safe here.
    """
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrent read/write performance
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """
    Creates the telemetry table and indexes if they do not already exist.
    Safe to call every startup — CREATE IF NOT EXISTS is idempotent.
    """
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id          TEXT    UNIQUE,
                event_timestamp     TEXT    NOT NULL,
                ingestion_timestamp TEXT    NOT NULL,
                vehicle_id          TEXT    NOT NULL,
                speed               REAL    NOT NULL,
                rpm                 REAL    NOT NULL,
                engine_temp         REAL    NOT NULL
            )
        """)
        # Index for filtering by vehicle
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vehicle_id
            ON telemetry (vehicle_id)
        """)
        # Index for time-ordered queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_timestamp
            ON telemetry (event_timestamp)
        """)
        conn.commit()
        logger.debug("Database initialised: table and indexes ready.")
    except sqlite3.Error as e:
        logger.error("Failed to initialise database: %s", e)
        raise
    finally:
        conn.close()


def insert_telemetry(record: dict) -> bool:
    """
    Inserts one validated telemetry record.

    Uses INSERT OR IGNORE so duplicate message_ids are silently skipped
    rather than raising an error. Returns True if inserted, False if duplicate.

    Raises sqlite3.Error for genuine database failures (caller should log and
    decide whether to continue).
    """
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO telemetry
                (message_id, event_timestamp, ingestion_timestamp,
                 vehicle_id, speed, rpm, engine_temp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("message_id"),           # None is fine — UNIQUE allows multiple NULLs
                record["event_timestamp"],
                record["ingestion_timestamp"],
                record["vehicle_id"],
                record["speed"],
                record["rpm"],
                record["engine_temp"],
            ),
        )
        conn.commit()
        inserted = cursor.rowcount > 0
        if not inserted:
            logger.warning(
                "Duplicate message_id '%s' for vehicle '%s' — skipped.",
                record.get("message_id"), record.get("vehicle_id"),
            )
        return inserted
    except sqlite3.Error as e:
        conn.rollback()
        logger.error("Database insert failed for vehicle '%s': %s", record.get("vehicle_id"), e)
        raise
    finally:
        conn.close()


def get_latest_telemetry() -> dict | None:
    """Returns the most recently inserted row, or None if the table is empty."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM telemetry ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error as e:
        logger.error("Failed to query latest telemetry: %s", e)
        raise
    finally:
        conn.close()


def get_telemetry(limit: int = 20) -> list[dict]:
    """Returns the most recent `limit` rows, newest first."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM telemetry ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error("Failed to query telemetry (limit=%d): %s", limit, e)
        raise
    finally:
        conn.close()


def get_telemetry_by_vehicle(vehicle_id: str, limit: int = 20) -> list[dict]:
    """Returns recent telemetry for a specific vehicle, newest first."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT * FROM telemetry
            WHERE vehicle_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (vehicle_id, limit),
        )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error(
            "Failed to query telemetry for vehicle '%s': %s", vehicle_id, e
        )
        raise
    finally:
        conn.close()
