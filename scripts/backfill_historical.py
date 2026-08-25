#!/usr/bin/env python3
"""
Исторический backfill 2021-01-01 → сегодня через OpenDotaExplorerSource
(Phase 6 — baseline ML требует многолетний датасет, см.
reports/phase5-live-summary.md, раздел Recommendation).

Стратегия (см. src/datasources/opendota_explorer.py за обоснование):
для каждого календарного квартала запрашивается ОДИН bulk /explorer-запрос
(professional+premium tier, известные team_id/winner/duration), из которого
оставляются MATCHES_PER_QUARTER самых ранних матчей квартала — та же логика
truncation, что run_ingestion() уже применяет к limit (Phase 5,
src/ingestion/opendota_pipeline.py). Это даёт РЕАЛЬНУЮ многолетнюю выборку
без пагинации через /proMatches (физически неосуществимой на таком
диапазоне за разумное время, см. docstring адаптера).

Ограничение (документируется, не скрывается): сэмплирование "первые N
матчей квартала" даёт непрерывность Elo/recent-form ВНУТРИ квартала, но
разрывы МЕЖДУ кварталами (RatingEngine не видит матчи, не попавшие в
выборку) — это не полная история команды, а репрезентативная выборка.
Обсуждается в reports/phase6-summary.md.

Запуск:
    python3 scripts/backfill_historical.py --since 2021-01-01 --matches-per-quarter 120
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_settings
from src.datasources.opendota_explorer import OpenDotaExplorerSource
from src.db.engine import make_engine
from src.ingestion.opendota_pipeline import run_ingestion, sync_reference_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("backfill_historical")


def quarter_bounds(start: date, end: date):
    """Возвращает список (quarter_start, quarter_end) [since, until) datetime UTC."""
    bounds = []
    y, q = start.year, (start.month - 1) // 3 + 1
    while True:
        q_start_month = (q - 1) * 3 + 1
        q_start = date(y, q_start_month, 1)
        if q_start >= end:
            break
        if q == 4:
            q_end = date(y + 1, 1, 1)
        else:
            q_end = date(y, q_start_month + 3, 1)
        bounds.append((
            datetime.combine(max(q_start, start), datetime.min.time(), tzinfo=timezone.utc),
            datetime.combine(min(q_end, end), datetime.min.time(), tzinfo=timezone.utc),
        ))
        if q == 4:
            y, q = y + 1, 1
        else:
            q += 1
    return bounds


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", type=str, default="2021-01-01")
    parser.add_argument("--until", type=str, default=None, help="по умолчанию — сегодня")
    parser.add_argument("--matches-per-quarter", type=int, default=120)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    engine = make_engine(settings)

    since_date = date.fromisoformat(args.since)
    until_date = date.fromisoformat(args.until) if args.until else datetime.now(timezone.utc).date()

    sync_reference_data(engine)

    quarters = quarter_bounds(since_date, until_date)
    logger.info("backfill_plan", extra={"quarters": len(quarters), "matches_per_quarter": args.matches_per_quarter})

    totals = {"fetched": 0, "upserted": 0, "skipped": 0}
    for q_since, q_until in quarters:
        source = OpenDotaExplorerSource.from_config(
            base_url=settings.opendota_base_url,
            timeout_seconds=settings.request_timeout_seconds,
            max_retries=settings.max_retries,
            rate_limit_per_min=settings.opendota_rate_limit_per_min,
        )
        try:
            result = run_ingestion(engine, source, since=q_since, until=q_until, limit=args.matches_per_quarter)
        finally:
            source.close()

        totals["fetched"] += result.matches_fetched
        totals["upserted"] += result.matches_upserted
        totals["skipped"] += result.matches_skipped_blocking_error
        print(
            f"{q_since.date()} — {q_until.date()}: "
            f"получено={result.matches_fetched} загружено={result.matches_upserted} "
            f"пропущено={result.matches_skipped_blocking_error}"
        )

    print(f"\nИТОГО: получено={totals['fetched']} загружено={totals['upserted']} пропущено={totals['skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
