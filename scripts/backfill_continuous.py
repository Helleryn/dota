#!/usr/bin/env python3
"""
PHASE 6.5 — непрерывный (без квартальных разрывов) исторический backfill,
2021-01-01 → сегодня, через OpenDotaExplorerSource (Phase 6).

В отличие от scripts/backfill_historical.py (Phase 6, 120 самых ранних
матчей на квартал — сознательный sampling ради экономии времени), здесь
запрашивается ВЕСЬ pro/premium tier по КАЖДОМУ КАЛЕНДАРНОМУ ГОДУ одним
`/explorer`-запросом, без cap — реальная скорость ingestion (~166
матчей/сек, измерено на январе 2024 в этой сессии) делает это дешевле, чем
предполагалось в Phase 6: полный диапазон 2021-2026 (~117k матчей) — это
~12 минут, а не часы.

Идемпотентно (upsert по match_id) — не создаёт дублей и не удаляет уже
загруженные Phase 5/6 данные (в том числе квартальный sample) — просто
заполняет пропуски между ними.

Запуск:
    python3 scripts/backfill_continuous.py --since 2021-01-01
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
logger = logging.getLogger("backfill_continuous")


def year_bounds(start: date, end: date):
    bounds = []
    y = start.year
    while True:
        y_start = date(y, 1, 1)
        if y_start >= end:
            break
        y_end = date(y + 1, 1, 1)
        bounds.append((
            datetime.combine(max(y_start, start), datetime.min.time(), tzinfo=timezone.utc),
            datetime.combine(min(y_end, end), datetime.min.time(), tzinfo=timezone.utc),
        ))
        y += 1
    return bounds


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", type=str, default="2021-01-01")
    parser.add_argument("--until", type=str, default=None)
    parser.add_argument("--cap-per-year", type=int, default=50000, help="safety cap, не должен реально сработать")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    engine = make_engine(settings)

    since_date = date.fromisoformat(args.since)
    until_date = date.fromisoformat(args.until) if args.until else datetime.now(timezone.utc).date()

    sync_reference_data(engine)

    years = year_bounds(since_date, until_date)
    logger.info("continuous_backfill_plan", extra={"years": len(years)})

    totals = {"fetched": 0, "upserted": 0, "skipped": 0}
    for y_since, y_until in years:
        source = OpenDotaExplorerSource.from_config(
            base_url=settings.opendota_base_url,
            timeout_seconds=settings.request_timeout_seconds,
            max_retries=settings.max_retries,
            rate_limit_per_min=settings.opendota_rate_limit_per_min,
        )
        try:
            result = run_ingestion(engine, source, since=y_since, until=y_until, limit=args.cap_per_year)
        finally:
            source.close()

        totals["fetched"] += result.matches_fetched
        totals["upserted"] += result.matches_upserted
        totals["skipped"] += result.matches_skipped_blocking_error
        print(
            f"{y_since.date()} — {y_until.date()}: "
            f"получено={result.matches_fetched} загружено={result.matches_upserted} "
            f"пропущено={result.matches_skipped_blocking_error}"
        )
        if result.matches_fetched >= args.cap_per_year:
            print(f"  ВНИМАНИЕ: достигнут cap-per-year={args.cap_per_year} — год мог быть обрезан!")

    print(f"\nИТОГО: получено={totals['fetched']} загружено={totals['upserted']} пропущено={totals['skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
