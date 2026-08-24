"""
Validation layer (Phase 5, раздел 12). Проверяет бизнес-правила поверх уже
нормализованных данных — не HTTP/сеть, не запись в БД, чистые функции.

Два уровня серьёзности:
  "error"   — строка не должна попадать в normalized-слой как есть
              (например, невозможная дата) — либо не грузится, либо
              грузится с явным флагом проблемы, решает вызывающий pipeline.
  "warning" — грузится, но фиксируется в data quality report (Phase 5,
              раздел 13) — например, матч без league_tier из premium/professional.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List

from src.normalization.normalize import NormalizedMatch

# Про-матч по нашему quality-фильтру (ADR-001, VERIFIED из исходников
# odota/core: `tier IN ('premium', 'professional')`). Единственное место в
# проекте, где определён этот список — используется и validate_match, и
# ingestion pipeline при построении отчётов.
PRO_LEAGUE_TIERS = {"premium", "professional"}

# Dota 2 публично вышла в июле 2013 (релиз), но матч-история в Steam Web
# API технически доступна с более раннего периода бета-тестирования —
# используем консервативную границу 2011-01-01 (начало закрытой беты),
# см. docs/data-sources.md. Более точная нижняя граница для PRO-матчей
# специфично определяется через REQUIRES LIVE VERIFICATION
# (scripts/verify_data_source.py), это лишь защита от заведомо невозможных дат.
MIN_VALID_START_TIME = datetime(2011, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class ValidationIssue:
    match_id: int
    field: str
    problem: str
    severity: str  # "error" | "warning"


def validate_match(match: NormalizedMatch, *, now: datetime | None = None) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    now = now or datetime.now(timezone.utc)

    if match.radiant_win is None:
        issues.append(ValidationIssue(match.match_id, "radiant_win", "не должен быть None", "error"))

    if (
        match.radiant_team_id is not None
        and match.dire_team_id is not None
        and match.radiant_team_id == match.dire_team_id
    ):
        issues.append(
            ValidationIssue(match.match_id, "team_ids", "radiant_team_id == dire_team_id — невозможная ситуация", "error")
        )

    if match.start_time < MIN_VALID_START_TIME or match.start_time > now:
        issues.append(
            ValidationIssue(match.match_id, "start_time", f"невозможная дата: {match.start_time.isoformat()}", "error")
        )

    if match.duration_seconds <= 0:
        issues.append(
            ValidationIssue(match.match_id, "duration_seconds", f"недопустимая длительность: {match.duration_seconds}", "error")
        )

    if match.radiant_team_id is None or match.dire_team_id is None:
        issues.append(ValidationIssue(match.match_id, "team_ids", "отсутствует team_id одной из сторон", "warning"))

    if match.league_tier not in PRO_LEAGUE_TIERS:
        issues.append(
            ValidationIssue(match.match_id, "league_tier", f"не входит в PRO_LEAGUE_TIERS: {match.league_tier!r}", "warning")
        )

    if match.patch_id is None:
        issues.append(ValidationIssue(match.match_id, "patch_id", "патч не определён (start_time вне справочника)", "warning"))

    return issues


def has_blocking_errors(issues: Iterable[ValidationIssue]) -> bool:
    return any(i.severity == "error" for i in issues)


def find_duplicate_match_ids(matches: Iterable[NormalizedMatch]) -> List[int]:
    """
    Дубли ВНУТРИ одного батча, полученного от источника (не путать с
    идемпотентностью upsert на уровне БД, которая решает повторные ЗАПУСКИ
    ingestion, а не дубли внутри одного ответа API) — Phase 5, раздел 12/14.
    """
    seen: set[int] = set()
    duplicates: List[int] = []
    for m in matches:
        if m.match_id in seen:
            duplicates.append(m.match_id)
        seen.add(m.match_id)
    return duplicates


def is_pro_match(match: NormalizedMatch) -> bool:
    return match.league_tier in PRO_LEAGUE_TIERS
