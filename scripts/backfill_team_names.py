#!/usr/bin/env python3
"""
PHASE 10 — заполнение `teams.name`/`teams.tag` из реестра команд OpenDota.

## Зачем

Аудит Phase 10.0 обнаружил, что только 93 из 7 933 team_id в нашей БД имеют
имя. Причина: bulk-backfill (`OpenDotaExplorerSource`, Phase 6-7) берёт имя
из `matches.radiant_team_name`/`dire_team_name`, а эти поля в БД OpenDota
**100% NULL** на всём диапазоне 2021-2026 (проверено запросом). Имена
получили только те команды, которые попали в ingestion через `/proMatches`
(Phase 5), где поля `radiant_name`/`dire_name` заполнены.

Отдельная таблица `teams` в OpenDota содержит 22 397 команд, у ВСЕХ есть
name и tag, и она покрывает 8 814 наших team_id.

## КРИТИЧЕСКАЯ ОГОВОРКА О ВРЕМЕНИ

Реестр `teams` — это СНИМОК НА СЕГОДНЯ. В нём нет ни `valid_from`, ни
истории переименований. Поэтому загруженное имя:

    * НЕ является именем команды на момент исторического матча;
    * для переименовавшейся команды это её ТЕКУЩЕЕ имя.

Из этого следует прямое ограничение для Phase 10 (см.
`reports/phase10-identity-plan.md`): имя нельзя использовать как
point-in-time свидетельство идентичности. Оно пригодно только как слабое
подтверждение поверх ростерных доказательств, которые point-in-time по
своей природе (`match_players` привязаны к конкретному матчу).

Запуск:
    python3 scripts/backfill_team_names.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text

from src.config import load_settings
from src.datasources.http_client import HttpClientConfig, RateLimitedHttpClient
from src.db.engine import make_engine


def main() -> int:
    settings = load_settings()
    engine = make_engine(settings)

    with engine.connect() as conn:
        local_ids = [r[0] for r in conn.execute(text("SELECT team_id FROM teams")).fetchall()]
    print(f"team_id в локальной БД: {len(local_ids)}")

    client = RateLimitedHttpClient(HttpClientConfig(
        base_url=settings.opendota_base_url,
        timeout_seconds=settings.request_timeout_seconds,
        max_retries=settings.max_retries,
        requests_per_minute=settings.opendota_rate_limit_per_min,
    ))
    try:
        payload = client.get_json("/explorer", params={
            "sql": "SELECT team_id, name, tag FROM teams WHERE name IS NOT NULL"
        })
    finally:
        client.close()

    if payload.get("err"):
        raise ValueError(f"/explorer error: {payload['err']}")

    registry = {r["team_id"]: (r.get("name"), r.get("tag")) for r in payload.get("rows", [])}
    print(f"Записей в реестре OpenDota: {len(registry)}")

    updates = [
        {"tid": tid, "name": registry[tid][0], "tag": registry[tid][1]}
        for tid in local_ids if tid in registry
    ]
    print(f"Совпало с нашими team_id: {len(updates)}")

    with engine.begin() as conn:
        for batch_start in range(0, len(updates), 1000):
            batch = updates[batch_start:batch_start + 1000]
            conn.execute(
                text("UPDATE teams SET name = :name, tag = :tag WHERE team_id = :tid"),
                batch,
            )

    with engine.connect() as conn:
        n_named = conn.execute(text("SELECT count(*) FROM teams WHERE name IS NOT NULL")).scalar()
    print(f"teams с именем после заполнения: {n_named} из {len(local_ids)}")
    print("ВНИМАНИЕ: это ТЕКУЩИЕ имена (снимок реестра), НЕ имена на момент матча.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
