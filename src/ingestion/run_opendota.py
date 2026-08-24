#!/usr/bin/env python3
"""
CLI ingestion pipeline (Phase 5, раздел 8, 16-18).

Запуск (offline fixture mode — по умолчанию, т.к. сеть к api.opendota.com
недоступна из этой среды, см. docs/environment-constraints.md):

    python3 -m src.ingestion.run_opendota --source fixtures --limit 100

Запуск против реального OpenDota (в среде с доступом в интернет):

    python3 -m src.ingestion.run_opendota --source live --limit 100

Идемпотентно: повторный запуск с тем же --since не создаёт дублей
(upsert по natural key, Phase 5 раздел 16) и продолжает с последнего
checkpoint, если --since не передан явно (Phase 5 раздел 17).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

from src.config import load_settings
from src.datasources.opendota import OpenDotaSource
from src.datasources.opendota_fixtures import build_fixture_opendota_source
from src.db.engine import make_engine
from src.ingestion.opendota_pipeline import run_ingestion, sync_reference_data
from src.repositories import ingestion_run_repository as run_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("ingestion.cli")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=["fixtures", "live"], default="fixtures",
        help="fixtures — offline mode (по умолчанию, см. docs/environment-constraints.md); "
             "live — реальный OpenDota API (требует сетевого доступа)",
    )
    parser.add_argument("--limit", type=int, default=100, help="макс. число матчей за прогон (Phase 5, раздел 8)")
    parser.add_argument("--since", type=str, default=None, help="ISO-дата, по умолчанию — последний checkpoint или DATA_START_DATE")
    parser.add_argument("--until", type=str, default=None, help="ISO-дата, по умолчанию — сейчас")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    engine = make_engine(settings)

    logger.info("syncing_reference_data")
    sync_reference_data(engine)

    with engine.begin() as conn:
        checkpoint = run_repo.get_last_successful_checkpoint(conn, "opendota")

    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    elif checkpoint and checkpoint.get("since"):
        since = datetime.fromisoformat(checkpoint["since"])
        logger.info("resuming_from_checkpoint", extra={"since": since.isoformat()})
    elif settings.data_start_date:
        since = datetime.combine(settings.data_start_date, datetime.min.time(), tzinfo=timezone.utc)
    else:
        since = datetime.now(timezone.utc) - timedelta(days=365)

    until = datetime.fromisoformat(args.until).replace(tzinfo=timezone.utc) if args.until else datetime.now(timezone.utc)

    # on_raw_response=None здесь намеренно — run_ingestion сам подключит
    # реальный callback, пишущий в raw_responses (см. src/ingestion/opendota_pipeline.py).
    if args.source == "fixtures":
        source = build_fixture_opendota_source(on_raw_response=None)
        logger.warning(
            "OFFLINE FIXTURE MODE — данные синтетические, НЕ реальные матчи. "
            "Для реального ingestion нужна среда с доступом к api.opendota.com "
            "(docs/environment-constraints.md)."
        )
    else:
        source = OpenDotaSource.from_config(
            base_url=settings.opendota_base_url,
            api_key=settings.opendota_api_key,
            timeout_seconds=settings.request_timeout_seconds,
            max_retries=settings.max_retries,
            rate_limit_per_min=settings.opendota_rate_limit_per_min,
            on_raw_response=None,
        )

    logger.info("ingestion_started", extra={"since": since.isoformat(), "until": until.isoformat(), "limit": args.limit})

    try:
        result = run_ingestion(engine, source, since=since, until=until, limit=args.limit)
    finally:
        source.close()

    logger.info(
        "ingestion_finished",
        extra={
            "matches_fetched": result.matches_fetched,
            "matches_upserted": result.matches_upserted,
            "matches_skipped_blocking_error": result.matches_skipped_blocking_error,
            "duplicate_match_ids_in_batch": result.duplicate_match_ids_in_batch,
            "raw_responses_written": result.raw_responses_written,
            "raw_responses_deduped": result.raw_responses_deduped,
            "checkpoint": result.checkpoint,
        },
    )
    print(f"Матчей получено: {result.matches_fetched}")
    print(f"Матчей загружено: {result.matches_upserted}")
    print(f"Пропущено (блокирующие ошибки валидации): {result.matches_skipped_blocking_error}")
    print(f"Дублей match_id внутри батча: {len(result.duplicate_match_ids_in_batch)}")
    print(f"Raw-ответов записано: {result.raw_responses_written}, дедуплицировано: {result.raw_responses_deduped}")
    print(f"Предупреждений валидации: {len(result.validation_issues)}")
    print(f"Новый checkpoint: {result.checkpoint}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
