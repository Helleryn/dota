"""
Normalization layer (Phase 5, раздел 11): normalize_match/normalize_team/
normalize_player/normalize_patch. Приводит типы, нормализует timestamps,
триммит строки, применяет enrichment (patch_id). НЕ занимается валидацией
бизнес-правил (src/normalization/validate.py) и НЕ пишет в БД
(src/repositories/) — чистые функции.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.datasources.base import RawMatch
from src.normalization.enrich import resolve_patch_id


@dataclass(frozen=True)
class NormalizedMatch:
    match_id: int
    start_time: datetime
    duration_seconds: int
    radiant_team_id: Optional[int]
    dire_team_id: Optional[int]
    radiant_team_name: Optional[str]
    dire_team_name: Optional[str]
    radiant_win: bool
    league_id: Optional[int]
    league_tier: Optional[str]
    patch_id: Optional[int]
    series_id: Optional[int]
    series_type: Optional[int]
    source: str
    ingested_at: datetime


@dataclass(frozen=True)
class NormalizedTeam:
    team_id: int
    name: Optional[str]
    tag: Optional[str]


@dataclass(frozen=True)
class NormalizedPlayer:
    account_id: int
    name: Optional[str]


def _clean_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def normalize_match(raw: RawMatch, *, ingested_at: Optional[datetime] = None) -> NormalizedMatch:
    start_time = raw.start_time
    if start_time.tzinfo is None:
        # Источник ОБЯЗАН отдавать aware-datetime (RawMatch это гарантирует
        # в OpenDotaSource), но normalize защищается от naive datetime на
        # случай другого будущего источника (Liquipedia/STRATZ), который
        # может отдать его иначе.
        start_time = start_time.replace(tzinfo=timezone.utc)

    return NormalizedMatch(
        match_id=raw.match_id,
        start_time=start_time,
        duration_seconds=raw.duration_seconds,
        radiant_team_id=raw.radiant_team_id,
        dire_team_id=raw.dire_team_id,
        radiant_team_name=_clean_str(raw.radiant_team_name),
        dire_team_name=_clean_str(raw.dire_team_name),
        radiant_win=raw.radiant_win,
        league_id=raw.league_id,
        league_tier=raw.league_tier,
        patch_id=resolve_patch_id(start_time),
        series_id=raw.series_id,
        series_type=raw.series_type,
        source=raw.source,
        ingested_at=ingested_at or datetime.now(timezone.utc),
    )


def normalize_team(team_id: Optional[int], name: Optional[str] = None, tag: Optional[str] = None) -> Optional[NormalizedTeam]:
    if team_id is None:
        return None  # матч без привязанной команды — валидный, но не создаём "команду None"
    return NormalizedTeam(team_id=team_id, name=_clean_str(name), tag=_clean_str(tag))


def normalize_player(account_id: Optional[int], name: Optional[str] = None) -> Optional[NormalizedPlayer]:
    if account_id is None:
        return None  # анонимный игрок (Steam privacy) — валидный случай, docs/data-feasibility.md
    return NormalizedPlayer(account_id=account_id, name=_clean_str(name))


def normalize_patch(start_time: datetime) -> Optional[int]:
    """Отдельная точка входа, если нужно определить патч без полного normalize_match."""
    return resolve_patch_id(start_time)
