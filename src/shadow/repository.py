"""
PHASE 15 — хранение снимков и разрешений.

Слой намеренно узкий: вставка снимка, перевод состояния вперёд, вставка
разрешения, чтение. Метода «обновить снимок» здесь нет и быть не должно —
неизменяемость дополнительно закреплена триггером в СУБД (миграция
a1b2c3d4e5f6), так что даже обход этого модуля не поможет.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List, Optional

from sqlalchemy import Engine, and_, select, text

from src.db.schema import prediction_resolutions, prediction_snapshots
from src.shadow import states
from src.shadow.snapshot import PredictionSnapshot, ResolutionRecord


class DuplicatePrediction(Exception):
    """Прогноз с таким id уже существует. Возбуждается явно, чтобы дубликат
    нельзя было создать молча (PART U-8)."""


class DuplicateResolution(Exception):
    """Разрешение для этого прогноза уже записано (PART U-9)."""


def insert_snapshot(conn, snap: PredictionSnapshot) -> None:
    exists = conn.execute(
        select(prediction_snapshots.c.prediction_id)
        .where(prediction_snapshots.c.prediction_id == snap.prediction_id)
    ).first()
    if exists:
        raise DuplicatePrediction(
            f"снимок {snap.prediction_id} уже существует "
            f"(match_key={snap.match_key}, ts={snap.prediction_timestamp})")
    row = snap.to_row()
    row["created_at"] = row.get("created_at") or datetime.now(timezone.utc)
    conn.execute(prediction_snapshots.insert().values(**row))


def advance_state(conn, prediction_id: str, new_state: str,
                  invalid_reason: Optional[str] = None) -> None:
    cur = conn.execute(
        select(prediction_snapshots.c.state)
        .where(prediction_snapshots.c.prediction_id == prediction_id)
    ).scalar_one_or_none()
    if cur is None:
        raise KeyError(f"снимка {prediction_id} нет")
    if not states.can_transition(cur, new_state):
        raise ValueError(f"недопустимый переход {cur} -> {new_state}")
    vals = {"state": new_state}
    # invalid_reason разрешено писать только вместе с переходом в INVALID
    # и только до публикации — после публикации триггер СУБД отклонит.
    if new_state == states.INVALID and invalid_reason and not states.is_published(cur):
        vals["invalid_reason"] = invalid_reason
    conn.execute(prediction_snapshots.update()
                 .where(prediction_snapshots.c.prediction_id == prediction_id)
                 .values(**vals))


def insert_resolution(conn, rec: ResolutionRecord) -> None:
    exists = conn.execute(
        select(prediction_resolutions.c.prediction_id)
        .where(prediction_resolutions.c.prediction_id == rec.prediction_id)
    ).first()
    if exists:
        raise DuplicateResolution(
            f"разрешение для {rec.prediction_id} уже записано")
    conn.execute(prediction_resolutions.insert().values(**rec.to_row()))


def get_snapshot(conn, prediction_id: str):
    return conn.execute(
        select(prediction_snapshots)
        .where(prediction_snapshots.c.prediction_id == prediction_id)).first()


def list_snapshots(conn, *, source: Optional[str] = None,
                   state: Optional[str] = None, limit: int = 10000) -> List:
    q = select(prediction_snapshots)
    conds = []
    if source:
        conds.append(prediction_snapshots.c.source == source)
    if state:
        conds.append(prediction_snapshots.c.state == state)
    if conds:
        q = q.where(and_(*conds))
    q = q.order_by(prediction_snapshots.c.prediction_timestamp.asc()).limit(limit)
    return conn.execute(q).fetchall()


def list_unresolved(conn, *, source: Optional[str] = None, limit: int = 5000) -> List:
    """Снимки, у которых ещё нет записи разрешения."""
    q = (select(prediction_snapshots)
         .outerjoin(prediction_resolutions,
                    prediction_resolutions.c.prediction_id
                    == prediction_snapshots.c.prediction_id)
         .where(prediction_resolutions.c.prediction_id.is_(None)))
    if source:
        q = q.where(prediction_snapshots.c.source == source)
    q = q.order_by(prediction_snapshots.c.prediction_timestamp.asc()).limit(limit)
    return conn.execute(q).fetchall()


def list_resolved(conn, *, source: Optional[str] = None, limit: int = 100000) -> List:
    """Пары (снимок, разрешение) в хронологическом порядке прогноза."""
    q = (select(prediction_snapshots, prediction_resolutions)
         .join(prediction_resolutions,
               prediction_resolutions.c.prediction_id
               == prediction_snapshots.c.prediction_id))
    if source:
        q = q.where(prediction_snapshots.c.source == source)
    q = q.order_by(prediction_snapshots.c.prediction_timestamp.asc()).limit(limit)
    return conn.execute(q).fetchall()


def counts_by_state(conn) -> dict:
    rows = conn.execute(text(
        "SELECT source, state, count(*) n FROM prediction_snapshots "
        "GROUP BY source, state ORDER BY source, state")).fetchall()
    return {f"{r.source}/{r.state}": r.n for r in rows}
