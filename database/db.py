# database/db.py
# SQLite persistence layer for EdgePulse telemetry.
#
# Schema (Phase 2 — extended from Phase 0):
#
#   id                  INTEGER PK AUTOINCREMENT
#   message_id          TEXT UNIQUE              — deduplication key
#   event_timestamp     TEXT NOT NULL            — vehicle clock time (UTC ISO-8601)
#   ingestion_timestamp TEXT NOT NULL            — server receipt time (UTC ISO-8601)
#   vehicle_id          TEXT NOT NULL            — indexed
#   speed               REAL NOT NULL            — km/h
#   rpm                 REAL NOT NULL            — revolutions per minute
#   engine_temp         REAL NOT NULL            — °C
#   battery_voltage     REAL                     — V  (nullable: Phase 0 records lack this)
#   fuel_level          REAL                     — %  (nullable: Phase 0 records lack this)
#
# Phase 2 also adds:
#   Table: vehicle_features   — persisted feature vectors per vehicle step
#   Table: vehicle_baselines  — running baseline statistics per vehicle/feature
#
# Migration strategy:
#   ALTER TABLE ADD COLUMN is safe and idempotent (SQLite ignores if column exists via
#   the try/except pattern in _migrate_telemetry_schema).

import logging
import sqlite3

from config import DATABASE_PATH

logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    """
    Opens (or creates) the SQLite database and returns a connection.
    row_factory=sqlite3.Row lets callers use row["column"] notation.
    check_same_thread=False: each function opens and closes its own connection.
    """
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """
    Creates all required tables and indexes if they do not already exist.
    Migrates the telemetry table to Phase 2 schema if necessary.
    Safe to call every startup — idempotent.
    """
    conn = get_connection()
    try:
        # --- telemetry table (Phase 0 core) ---
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
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vehicle_id
            ON telemetry (vehicle_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_timestamp
            ON telemetry (event_timestamp)
        """)

        # --- Phase 2 migration: add battery_voltage and fuel_level columns ---
        _add_column_if_missing(conn, "telemetry", "battery_voltage", "REAL")
        _add_column_if_missing(conn, "telemetry", "fuel_level", "REAL")

        # --- vehicle_features table (Phase 2) ---
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_features (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id          TEXT    NOT NULL,
                event_timestamp     TEXT    NOT NULL,
                window_size         INTEGER NOT NULL,
                feature_json        TEXT    NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vf_vehicle_id
            ON vehicle_features (vehicle_id)
        """)

        # --- vehicle_baselines table (Phase 2) ---
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_baselines (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id      TEXT    NOT NULL,
                feature_name    TEXT    NOT NULL,
                sample_count    INTEGER NOT NULL DEFAULT 0,
                mean            REAL,
                variance        REAL,
                updated_at      TEXT    NOT NULL,
                UNIQUE (vehicle_id, feature_name)
            )
        """)

        conn.commit()
        logger.debug("Database initialised: all tables and indexes ready.")
    except sqlite3.Error as e:
        logger.error("Failed to initialise database: %s", e)
        raise
    finally:
        conn.close()


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
    """Adds a column to a table if it does not already exist. Idempotent."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        logger.info("Migrated %s: added column '%s %s'", table, column, col_type)


def insert_telemetry(record: dict) -> bool:
    """
    Inserts one validated telemetry record.
    Returns True if inserted, False if duplicate message_id.
    Raises sqlite3.Error on genuine database failures.
    """
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO telemetry
                (message_id, event_timestamp, ingestion_timestamp,
                 vehicle_id, speed, rpm, engine_temp, battery_voltage, fuel_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("message_id"),
                record["event_timestamp"],
                record["ingestion_timestamp"],
                record["vehicle_id"],
                record["speed"],
                record["rpm"],
                record["engine_temp"],
                record.get("battery_voltage"),   # nullable
                record.get("fuel_level"),         # nullable
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
        cursor = conn.execute("SELECT * FROM telemetry ORDER BY id DESC LIMIT 1")
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
            ORDER BY id DESC LIMIT ?
            """,
            (vehicle_id, limit),
        )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error("Failed to query telemetry for vehicle '%s': %s", vehicle_id, e)
        raise
    finally:
        conn.close()


def get_telemetry_by_vehicle_ordered(vehicle_id: str, limit: int = 200) -> list[dict]:
    """Returns telemetry for a vehicle ordered oldest-first (for feature windows)."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT * FROM telemetry
            WHERE vehicle_id = ?
            ORDER BY event_timestamp ASC LIMIT ?
            """,
            (vehicle_id, limit),
        )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error("Failed to query ordered telemetry for vehicle '%s': %s", vehicle_id, e)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Baseline persistence
# ---------------------------------------------------------------------------

def upsert_baseline(vehicle_id: str, feature_name: str,
                    sample_count: int, mean: float, variance: float,
                    updated_at: str) -> None:
    """Insert or update a baseline row for a given vehicle/feature."""
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO vehicle_baselines
                (vehicle_id, feature_name, sample_count, mean, variance, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(vehicle_id, feature_name) DO UPDATE SET
                sample_count = excluded.sample_count,
                mean         = excluded.mean,
                variance     = excluded.variance,
                updated_at   = excluded.updated_at
            """,
            (vehicle_id, feature_name, sample_count, mean, variance, updated_at),
        )
        conn.commit()
    except sqlite3.Error as e:
        logger.error("Failed to upsert baseline %s/%s: %s", vehicle_id, feature_name, e)
        raise
    finally:
        conn.close()


def get_baselines_for_vehicle(vehicle_id: str) -> list[dict]:
    """Returns all stored baseline rows for a vehicle."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM vehicle_baselines WHERE vehicle_id = ?",
            (vehicle_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error("Failed to get baselines for vehicle '%s': %s", vehicle_id, e)
        raise
    finally:
        conn.close()
