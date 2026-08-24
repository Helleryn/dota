"""
Ingestion pipeline для OpenDota (Phase 5, раздел 8, 11-14, 16-17):

    Extract -> Raw -> Validate -> Normalize -> Deduplicate -> Enrich -> DB

Идемпотентно (повторный запуск не создаёт дублей, upsert по natural key),
с checkpoint-based incremental sync, с накоплением статистики для
data quality report (Phase 5, раздел 13).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import Engine
from sqlalchemy.exc import IntegrityError

from src.datasources.base import DataSource
from src.datasources.http_client import PermanentHttpError, TransientHttpError
from src.normalization.enrich import load_heroes_reference, load_patches_reference
from src.normalization.normalize import normalize_match, normalize_player, normalize_team
from src.normalization.validate import ValidationIssue, find_duplicate_match_ids, has_blocking_errors, validate_match
from src.repositories import ingestion_run_repository as run_repo
from src.repositories import match_repository as repo
from src.repositories import raw_repository as raw_repo

logger = logging.getLogger("ingestion.opendota")


@dataclass
class IngestionResult:
    run_id: int
    matches_fetched: int = 0
    matches_upserted: int = 0
    matches_skipped_blocking_error: int = 0
    duplicate_match_ids_in_batch: List[int] = field(default_factory=list)
    validation_issues: List[ValidationIssue] = field(default_factory=list)
    raw_responses_written: int = 0
    raw_responses_deduped: int = 0
    checkpoint: Optional[dict] = None
    status: str = "running"
    error: Optional[str] = None


def sync_reference_data(engine: Engine) -> None:
    """Синхронизация справочников heroes/patches из вендоренных
    dotaconstants-файлов (Phase 5, не зависит от сети — статические
    файлы в src/normalization/data/, см. README там же)."""
    heroes_data = load_heroes_reference()
    patches_data = load_patches_reference()
    with engine.begin() as conn:
        n_heroes = repo.sync_heroes(conn, heroes_data)
        n_patches = repo.sync_patches(conn, patches_data)
    logger.info("reference_data_synced", extra={"heroes": n_heroes, "patches": n_patches})


def run_ingestion(
    engine: Engine,
    source: DataSource,
    since: datetime,
    until: datetime,
    limit: Optional[int] = None,
) -> IngestionResult:
    """
    Extract -> Raw -> Validate -> Normalize -> Enrich -> upsert в БД.

    limit — Phase 5, раздел 8: "начни с ~100 матчей, не с полного диапазона" —
    жёсткий потолок на число матчей за один прогон, не на число HTTP-запросов
    (детальный запрос на матч — это ещё +1-2 вызова сверх /proMatches).
    """
    with engine.begin() as conn:
        run_id = run_repo.start_run(conn, source.name)

    result = IngestionResult(run_id=run_id)

    def on_raw(record):
        with engine.begin() as conn:
            inserted = raw_repo.save_raw_response(conn, record)
        if inserted:
            result.raw_responses_written += 1
        else:
            result.raw_responses_deduped += 1

    # on_raw_response — публичный изменяемый атрибут DataSource-адаптеров
    # (см. src/datasources/opendota.py). Подключаем здесь, а не в момент
    # конструирования source, потому что колбэк должен писать статистику
    # именно в ЭТОТ IngestionResult (создаётся только что, вместе с run_id).
    if getattr(source, "on_raw_response", "unset") is None:
        source.on_raw_response = on_raw

    raw_matches = list(source.fetch_matches(since, until))
    result.matches_fetched = len(raw_matches)

    if limit is not None and len(raw_matches) > limit:
        # Оставляем САМЫЕ РАННИЕ в диапазоне (устойчивое к повторным
        # прогонам поведение) — не случайную подвыборку.
        raw_matches = sorted(raw_matches, key=lambda m: m.start_time)[:limit]

    normalized = [normalize_match(m) for m in raw_matches]
    result.duplicate_match_ids_in_batch = find_duplicate_match_ids(normalized)

    max_start_time: Optional[datetime] = None

    for nm in normalized:
        issues = validate_match(nm)
        result.validation_issues.extend(issues)

        if has_blocking_errors(issues):
            result.matches_skipped_blocking_error += 1
            logger.warning("match_skipped_blocking_error", extra={"match_id": nm.match_id, "issues": str(issues)})
            continue

        # Детальный ответ (/matches/{id}) не гарантирован для каждого матча
        # (не распарсен, удалён, временно недоступен у источника) — не повод
        # ронять весь batch. Извлекаем ДО транзакции, чтобы сетевая ошибка
        # не держала открытым соединение с БД.
        try:
            picks_bans = list(source.fetch_picks_bans(nm.match_id))
            players = list(source.fetch_player_matches(nm.match_id))
        except (PermanentHttpError, TransientHttpError) as e:
            picks_bans, players = [], []
            result.validation_issues.append(
                ValidationIssue(nm.match_id, "match_detail", f"недоступен детальный ответ: {e}", "warning")
            )
            logger.warning("match_detail_unavailable", extra={"match_id": nm.match_id, "error": str(e)})

        try:
            with engine.begin() as conn:
                radiant = normalize_team(nm.radiant_team_id, nm.radiant_team_name)
                dire = normalize_team(nm.dire_team_id, nm.dire_team_name)
                now = datetime.now(timezone.utc)
                if radiant:
                    repo.upsert_team(conn, radiant.team_id, radiant.name, radiant.tag, now)
                if dire:
                    repo.upsert_team(conn, dire.team_id, dire.name, dire.tag, now)
                if nm.league_id is not None:
                    repo.upsert_league(conn, nm.league_id, name=None, tier=nm.league_tier)

                repo.upsert_match(
                    conn,
                    match_id=nm.match_id,
                    start_time=nm.start_time,
                    duration_seconds=nm.duration_seconds,
                    radiant_team_id=nm.radiant_team_id,
                    dire_team_id=nm.dire_team_id,
                    radiant_win=nm.radiant_win,
                    league_id=nm.league_id,
                    patch_id=nm.patch_id,
                    series_id=nm.series_id,
                    series_type=nm.series_type,
                    source=nm.source,
                    ingested_at=nm.ingested_at,
                )

                if picks_bans:
                    repo.upsert_picks_bans(conn, nm.match_id, picks_bans)

                for p in players:
                    normalized_player = normalize_player(p.account_id)
                    if normalized_player:
                        repo.upsert_player(conn, normalized_player.account_id, normalized_player.name, now)
                if players:
                    repo.upsert_match_players(conn, nm.match_id, players)
        except IntegrityError as e:
            # Реальный сценарий, не гипотетический: hero_id из ответа
            # источника может отсутствовать в вендоренном справочнике heroes
            # (пространство ID у Dota 2 имеет разрывы — упразднённые ID
            # старых героев). Один такой матч не должен ронять весь batch —
            # ядро матча просто не сохраняется в этом прогоне, ошибка
            # фиксируется для data quality report, откатывается только его
            # транзакция (остальные матчи уже закоммичены отдельно).
            result.matches_skipped_blocking_error += 1
            result.validation_issues.append(
                ValidationIssue(nm.match_id, "db_write", f"IntegrityError при записи: {e.orig}", "error")
            )
            logger.error("match_db_write_failed", extra={"match_id": nm.match_id, "error": str(e.orig)})
            continue

        result.matches_upserted += 1
        if max_start_time is None or nm.start_time > max_start_time:
            max_start_time = nm.start_time

    # +1 микросекунда делает границу для следующего прогона исключающей:
    # fetch_matches() пропускает строки с start_time < since (строго), так что
    # без сдвига последний обработанный матч (start_time == since) на
    # следующем запуске снова прошёл бы фильтр (idempotent, но лишний вызов).
    result.checkpoint = (
        {"since": (max_start_time + timedelta(microseconds=1)).isoformat()} if max_start_time else None
    )
    result.status = "succeeded"

    with engine.begin() as conn:
        run_repo.finish_run(
            conn, run_id, status="succeeded", records_fetched=result.matches_upserted, checkpoint=result.checkpoint
        )

    return result
