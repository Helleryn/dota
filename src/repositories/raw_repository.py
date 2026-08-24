"""
Repository для raw-слоя (Phase 5, раздел 6-7; docs/database-design.md,
ADR-002). Единственное место в проекте, которое пишет в raw_responses —
адаптеры (OpenDotaSource) сообщают о сыром ответе через callback, но сами
не знают о существовании БД (docs/architecture.md, разделение слоёв).
"""

from __future__ import annotations

from sqlalchemy import Connection, select

from src.datasources.base import RawResponseRecord
from src.db.schema import raw_responses


def save_raw_response(conn: Connection, record: RawResponseRecord) -> bool:
    """
    Возвращает True, если запись реально вставлена, False — если это
    байт-в-байт повтор уже сохранённого ответа (Phase 5, раздел 6:
    "content hash / request identity для предотвращения бессмысленных
    дублей"). Разные content_hash для того же (source, endpoint, params) —
    ЛЕГИТИМНАЯ новая строка (источник изменил ответ со временем — ценная
    для аудита информация, docs/database-design.md: raw_responses append-only).
    """
    existing = conn.execute(
        select(raw_responses.c.id).where(
            raw_responses.c.source == record.source,
            raw_responses.c.endpoint == record.endpoint,
            raw_responses.c.content_hash == record.content_hash,
        )
    ).first()
    if existing is not None:
        return False

    conn.execute(
        raw_responses.insert().values(
            source=record.source,
            endpoint=record.endpoint,
            request_params=record.request_params,
            fetched_at=record.fetched_at,
            http_status=record.http_status,
            response_body=record.response_body,
            content_hash=record.content_hash,
        )
    )
    return True
