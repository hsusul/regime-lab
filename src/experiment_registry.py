"""Lightweight SQLite experiment registry for RegimeLab.

The registry is additive to the MVP file-based `reports/experiments.json`
metadata. JSON remains supported as a fallback while SQLite provides an optional
active-model pointer and structured query surface.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence


REPORTS_DIR = Path("reports")
DEFAULT_REGISTRY_PATH = REPORTS_DIR / "regimelab.db"
SCHEMA_VERSION = 1
JSON_FIELDS = {
    "tickers",
    "training_date_range",
    "test_date_range",
    "metrics",
    "environment",
}


class ExperimentRegistryError(Exception):
    """Base exception for SQLite registry failures."""


class RegistryExperimentNotFoundError(ExperimentRegistryError):
    """Raised when an experiment does not exist in the registry."""


def connect(db_path: Path = DEFAULT_REGISTRY_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with row dictionaries enabled."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_registry(db_path: Path = DEFAULT_REGISTRY_PATH) -> None:
    """Create registry schema if it does not already exist."""
    with connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS registry_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                model_type TEXT NOT NULL,
                tickers TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                feature_version TEXT,
                labeling_version TEXT,
                training_date_range TEXT,
                test_date_range TEXT,
                metrics TEXT,
                environment TEXT,
                active INTEGER NOT NULL DEFAULT 0,
                metrics_path TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO registry_metadata (key, value)
            VALUES ('schema_version', ?)
            """,
            (str(SCHEMA_VERSION),),
        )


def registry_exists(db_path: Path = DEFAULT_REGISTRY_PATH) -> bool:
    """Return whether a registry database exists."""
    return db_path.exists()


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True)


def _json_loads(value: str | None) -> Any:
    if not value:
        return {}
    return json.loads(value)


def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    for field in JSON_FIELDS:
        record[field] = _json_loads(record.get(field))
    record["active"] = bool(record["active"])
    return record


def _metrics_from_record(record: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(record.get("metrics") or {})
    for key in ["accuracy", "macro_f1"]:
        if key in record:
            metrics[key] = record[key]
    return metrics


def normalize_experiment_record(
    record: dict[str, Any],
    *,
    active: bool | None = None,
) -> dict[str, Any]:
    """Normalize file-based or metrics-enriched metadata for SQLite storage."""
    return {
        "experiment_id": str(record["experiment_id"]),
        "created_at": str(record.get("created_at", "")),
        "status": str(record.get("status", "completed")),
        "model_type": str(record.get("model_type", "")),
        "tickers": list(record.get("tickers", [])),
        "artifact_path": str(record.get("artifact_path", "")),
        "feature_version": record.get("feature_version"),
        "labeling_version": record.get("labeling_version"),
        "training_date_range": record.get("training_date_range", {}),
        "test_date_range": record.get("test_date_range", {}),
        "metrics": _metrics_from_record(record),
        "environment": record.get("environment", {}),
        "active": bool(record.get("active", False) if active is None else active),
        "metrics_path": record.get("metrics_path"),
    }


def upsert_experiment(
    record: dict[str, Any],
    *,
    db_path: Path = DEFAULT_REGISTRY_PATH,
    active: bool | None = None,
) -> dict[str, Any]:
    """Insert or update one experiment in the SQLite registry."""
    initialize_registry(db_path)
    normalized = normalize_experiment_record(record, active=active)
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO experiments (
                experiment_id,
                created_at,
                status,
                model_type,
                tickers,
                artifact_path,
                feature_version,
                labeling_version,
                training_date_range,
                test_date_range,
                metrics,
                environment,
                active,
                metrics_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(experiment_id) DO UPDATE SET
                created_at = excluded.created_at,
                status = excluded.status,
                model_type = excluded.model_type,
                tickers = excluded.tickers,
                artifact_path = excluded.artifact_path,
                feature_version = excluded.feature_version,
                labeling_version = excluded.labeling_version,
                training_date_range = excluded.training_date_range,
                test_date_range = excluded.test_date_range,
                metrics = excluded.metrics,
                environment = excluded.environment,
                active = CASE
                    WHEN excluded.active = 1 THEN 1
                    ELSE experiments.active
                END,
                metrics_path = excluded.metrics_path
            """,
            (
                normalized["experiment_id"],
                normalized["created_at"],
                normalized["status"],
                normalized["model_type"],
                _json_dumps(normalized["tickers"]),
                normalized["artifact_path"],
                normalized["feature_version"],
                normalized["labeling_version"],
                _json_dumps(normalized["training_date_range"]),
                _json_dumps(normalized["test_date_range"]),
                _json_dumps(normalized["metrics"]),
                _json_dumps(normalized["environment"]),
                int(normalized["active"]),
                normalized["metrics_path"],
            ),
        )
    if normalized["active"]:
        activate_experiment(normalized["experiment_id"], db_path=db_path)
    return normalized


def list_experiments(
    *,
    db_path: Path = DEFAULT_REGISTRY_PATH,
    limit: int | None = None,
    model_type: str | None = None,
    ticker: str | None = None,
) -> list[dict[str, Any]]:
    """List registry experiments newest first with optional filters."""
    if not registry_exists(db_path):
        return []
    initialize_registry(db_path)
    query = "SELECT * FROM experiments"
    conditions: list[str] = []
    params: list[Any] = []
    if model_type:
        conditions.append("model_type = ?")
        params.append(model_type)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    with connect(db_path) as connection:
        rows = [_row_to_record(row) for row in connection.execute(query, params)]

    if ticker:
        normalized_ticker = ticker.upper()
        rows = [
            row
            for row in rows
            if normalized_ticker in {str(item).upper() for item in row.get("tickers", [])}
        ]
    return rows


def get_experiment(
    experiment_id: str,
    *,
    db_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Return one registry experiment by ID."""
    if not registry_exists(db_path):
        raise RegistryExperimentNotFoundError(f"Registry does not exist: {db_path}")
    initialize_registry(db_path)
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
    if row is None:
        raise RegistryExperimentNotFoundError(
            f"Experiment {experiment_id!r} not found in {db_path}."
        )
    return _row_to_record(row)


def activate_experiment(
    experiment_id: str,
    *,
    db_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Mark one experiment as active and clear all other active flags."""
    existing = get_experiment(experiment_id, db_path=db_path)
    with connect(db_path) as connection:
        connection.execute("UPDATE experiments SET active = 0")
        connection.execute(
            "UPDATE experiments SET active = 1 WHERE experiment_id = ?",
            (experiment_id,),
        )
    existing["active"] = True
    return existing


def active_experiment(
    *,
    db_path: Path = DEFAULT_REGISTRY_PATH,
    require_valid_artifact: bool = True,
) -> dict[str, Any] | None:
    """Return the active experiment, optionally requiring an existing artifact."""
    if not registry_exists(db_path):
        return None
    initialize_registry(db_path)
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM experiments WHERE active = 1 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    record = _row_to_record(row)
    if require_valid_artifact and not Path(str(record.get("artifact_path", ""))).exists():
        return None
    return record


def update_experiment_metrics(
    experiment_id: str,
    metrics: dict[str, Any],
    *,
    metrics_path: Path | None = None,
    db_path: Path = DEFAULT_REGISTRY_PATH,
) -> None:
    """Update metrics JSON for an experiment if the registry exists."""
    if not registry_exists(db_path):
        return
    existing = get_experiment(experiment_id, db_path=db_path)
    merged_metrics = dict(existing.get("metrics") or {})
    merged_metrics.update(metrics)
    with connect(db_path) as connection:
        connection.execute(
            """
            UPDATE experiments
            SET metrics = ?, metrics_path = ?
            WHERE experiment_id = ?
            """,
            (
                _json_dumps(merged_metrics),
                None if metrics_path is None else str(metrics_path),
                experiment_id,
            ),
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the experiment registry CLI parser."""
    parser = argparse.ArgumentParser(description="Manage the RegimeLab SQLite registry.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Path to SQLite registry database.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list", help="List registry experiments.")
    list_parser.add_argument("--limit", type=int)
    list_parser.add_argument("--model-type")
    list_parser.add_argument("--ticker")

    show_parser = subparsers.add_parser("show", help="Show one experiment.")
    show_parser.add_argument("experiment_id")

    activate_parser = subparsers.add_parser("activate", help="Mark an experiment active.")
    activate_parser.add_argument("experiment_id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for SQLite registry management."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "list":
            records = list_experiments(
                db_path=args.db_path,
                limit=args.limit,
                model_type=args.model_type,
                ticker=args.ticker,
            )
            print(json.dumps(records, indent=2))
            return 0
        if args.command == "show":
            print(json.dumps(get_experiment(args.experiment_id, db_path=args.db_path), indent=2))
            return 0
        if args.command == "activate":
            record = activate_experiment(args.experiment_id, db_path=args.db_path)
            print(f"activated: {record['experiment_id']}")
            return 0
    except ExperimentRegistryError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
