"""SQLite-backed inspection audit trail and dashboard queries."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


class InspectionDatabase:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS inspections (
                    inspection_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    product TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    detections_json TEXT NOT NULL,
                    ai_observations_json TEXT NOT NULL,
                    quality_json TEXT NOT NULL,
                    model_version TEXT,
                    processing_ms REAL,
                    original_image_path TEXT,
                    annotated_image_path TEXT
                )"""
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_inspections_timestamp ON inspections(timestamp)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_inspections_product ON inspections(product)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_inspections_outcome ON inspections(outcome)")

    def add_inspection(self, record: dict[str, Any]) -> None:
        payload = (
            record["inspection_id"], record.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            record["product"], record["mode"], record["provider"], record["outcome"], record["explanation"],
            json.dumps(record.get("detections", []), ensure_ascii=True),
            json.dumps(record.get("ai_observations", []), ensure_ascii=True),
            json.dumps(record.get("quality", {}), ensure_ascii=True),
            record.get("model_version"), record.get("processing_ms"),
            record.get("original_image_path"), record.get("annotated_image_path"),
        )
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO inspections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                payload,
            )

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        result["detections"] = json.loads(result.pop("detections_json"))
        result["ai_observations"] = json.loads(result.pop("ai_observations_json"))
        result["quality"] = json.loads(result.pop("quality_json"))
        return result

    def get_inspection(self, inspection_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM inspections WHERE inspection_id = ?", (inspection_id,)).fetchone()
        return self._decode(row)

    def list_inspections(
        self,
        *,
        search: str = "",
        product: str = "All",
        outcome: str = "All",
        method: str = "All",
        date_from: str = "",
        date_to: str = "",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if search.strip():
            clauses.append("inspection_id LIKE ?")
            values.append(f"%{search.strip()}%")
        if product != "All":
            clauses.append("product = ?")
            values.append(product)
        if outcome != "All":
            clauses.append("outcome = ?")
            values.append(outcome)
        if method != "All":
            clauses.append("provider = ?")
            values.append(method)
        if date_from:
            clauses.append("date(timestamp) >= date(?)")
            values.append(date_from)
        if date_to:
            clauses.append("date(timestamp) <= date(?)")
            values.append(date_to)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM inspections {where} ORDER BY timestamp DESC LIMIT ?",
                (*values, max(1, min(int(limit), 5000))),
            ).fetchall()
        return [self._decode(row) for row in rows if row is not None]

    def dashboard_data(
        self,
        *,
        product: str = "All",
        outcome: str = "All",
        date_from: str = "",
        date_to: str = "",
    ) -> dict[str, Any]:
        outcomes = ["PASS / ACCEPTED", "FAIL / REJECTED", "REVIEW REQUIRED", "RECAPTURE REQUIRED"]
        clauses: list[str] = []
        values: list[Any] = []
        if product != "All":
            clauses.append("product = ?")
            values.append(product)
        if outcome != "All":
            clauses.append("outcome = ?")
            values.append(outcome)
        if date_from:
            clauses.append("date(timestamp) >= date(?)")
            values.append(date_from)
        if date_to:
            clauses.append("date(timestamp) <= date(?)")
            values.append(date_to)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        trend_clauses = [*clauses, "timestamp >= datetime('now', '-30 days')"]
        trend_where = f"WHERE {' AND '.join(trend_clauses)}"
        with self._connection() as connection:
            total = connection.execute(f"SELECT COUNT(*) FROM inspections {where}", values).fetchone()[0]
            outcome_rows = connection.execute(
                f"SELECT outcome, COUNT(*) AS count FROM inspections {where} GROUP BY outcome", values
            ).fetchall()
            category_rows = connection.execute(
                f"SELECT product, outcome, COUNT(*) AS count FROM inspections {where} GROUP BY product, outcome ORDER BY product",
                values,
            ).fetchall()
            trend_rows = connection.execute(
                f"""SELECT date(timestamp) AS day, outcome, COUNT(*) AS count
                   FROM inspections {trend_where}
                   GROUP BY date(timestamp), outcome ORDER BY day""",
                values,
            ).fetchall()
            recent_rows = connection.execute(
                f"SELECT * FROM inspections {where} ORDER BY timestamp DESC LIMIT 10", values
            ).fetchall()
            observation_rows = connection.execute(
                f"SELECT detections_json, ai_observations_json FROM inspections {where}", values
            ).fetchall()
        counts = dict.fromkeys(outcomes, 0)
        counts.update({row["outcome"]: row["count"] for row in outcome_rows})
        defects: dict[str, int] = {}
        for row in observation_rows:
            for item in json.loads(row["detections_json"]):
                label = str(item.get("label", ""))
                if item.get("kind") == "defined defect" and label:
                    defects[label] = defects.get(label, 0) + 1
            for item in json.loads(row["ai_observations_json"]):
                label = str(item.get("defect", item.get("potential_defect", "")))
                if label:
                    defects[label] = defects.get(label, 0) + 1
        return {
            "total": total,
            "counts": counts,
            "accepted_rate": counts[outcomes[0]] / total if total else 0.0,
            "rejected_rate": counts[outcomes[1]] / total if total else 0.0,
            "by_category": [dict(row) for row in category_rows],
            "common_observations": sorted(defects.items(), key=lambda item: (-item[1], item[0]))[:10],
            "trend": [dict(row) for row in trend_rows],
            "recent": [self._decode(row) for row in recent_rows],
        }