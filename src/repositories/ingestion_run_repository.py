"""
Repository для ingestion_runs — checkpoint/idempotency (Phase 5, раздел 17).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Connection, desc, select

from src.db.schema import ingestion_runs


def start_run(conn: Connection, source: str) -> int:
    result = conn.execute(
        ingestion_runs.insert().values(
            source=source, started_at=datetime.now(timezone.utc), status="running"
        )
    )
    return result.inserted_primary_key[0]


def finish_run(
    conn: Connection,
    run_id: int,
    status: str,
    records_fetched: int,
    checkpoint: Optional[dict],
    error: Optional[str] = None,
) -> None:
    conn.execute(
        ingestion_runs.update()
        .where(ingestion_runs.c.id == run_id)
        .values(
            finished_at=datetime.now(timezone.utc),
            status=status,
            records_fetched=records_fetched,
            checkpoint=checkpoint,
            error=error,
        )
    )


def get_last_successful_checkpoint(conn: Connection, source: str) -> Optional[dict]:
    """
    "Какие данные уже загружены?" — Phase 5, раздел 17. Отвечает одним
    запросом, не требует пересчёта по всей таблице matches
    (docs/data-pipeline.md, раздел Checkpoints).
    """
    row = conn.execute(
        select(ingestion_runs.c.checkpoint)
        .where(ingestion_runs.c.source == source, ingestion_runs.c.status == "succeeded")
        .order_by(desc(ingestion_runs.c.finished_at))
        .limit(1)
    ).first()
    return row[0] if row else None
