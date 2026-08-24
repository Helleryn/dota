"""
Абстрактный интерфейс источников данных (Phase 2.2 — Source Abstraction).

Это НЕ production-клиент. Это фиксация контракта: любой источник данных
(OpenDota, Liquipedia, STRATZ, ...) должен уметь отдавать данные в ОДНОМ
и том же внутреннем формате, чтобы downstream-код (normalized-слой,
feature engineering, ML dataset) не зависел от конкретного API.

Реальные сетевые клиенты (HTTP-запросы, retry, rate limiting) появятся
в Phase 5. Здесь — только контракт и доменные структуры данных.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional


@dataclass(frozen=True)
class RawMatch:
    """Матч в унифицированном виде, независимо от источника."""

    match_id: int
    start_time: datetime
    duration_seconds: int
    radiant_team_id: Optional[int]
    dire_team_id: Optional[int]
    radiant_team_name: Optional[str]
    dire_team_name: Optional[str]
    radiant_win: bool
    league_id: Optional[int]
    league_tier: Optional[str]  # 'premium' | 'professional' | ...
    series_id: Optional[int]
    series_type: Optional[int]  # 0/1/2 -> предположительно Bo1/Bo3/Bo5, см. docs/data-feasibility.md
    patch: Optional[str]  # проставляется по start_time через odota/dotaconstants, не приходит от источника

    # источник и происхождение записи — обязательны для Raw-слоя (см. Phase 2.3)
    source: str = ""
    fetched_at: Optional[datetime] = None


@dataclass(frozen=True)
class RawPickBan:
    match_id: int
    is_pick: bool
    hero_id: int
    team: int  # 0 = radiant, 1 = dire (согласно реальному примеру ответа Steam API)
    order: int


@dataclass(frozen=True)
class RawPlayerMatch:
    match_id: int
    account_id: Optional[int]
    team_id: Optional[int]
    hero_id: int
    is_radiant: bool
    kills: int
    deaths: int
    assists: int
    gold_per_min: int
    xp_per_min: int


@dataclass(frozen=True)
class RawResponseRecord:
    """
    Один сырой HTTP-ответ источника + метаданные запроса — то, что
    записывается в raw_responses (docs/database-design.md, ADR-002).
    Адаптеры (OpenDotaSource и т.д.) сообщают об этом через callback
    on_raw_response, сами не пишут в БД — сохраняет принцип "domain-логика
    не зависит от конкретного хранилища" (docs/architecture.md).
    """

    source: str
    endpoint: str
    request_params: dict
    fetched_at: datetime
    http_status: int
    response_body: object  # уже распарсенный JSON (dict/list), не строка
    content_hash: str  # sha256 от нормализованного тела ответа — для дедупликации


@dataclass(frozen=True)
class RosterChange:
    """Событие смены состава команды. Источник — как правило Liquipedia."""

    team_id: int
    account_id: int
    changed_at: datetime
    change_type: str  # 'joined' | 'left' | 'stand_in'
    source: str = ""


class DataSource(ABC):
    """
    Контракт источника данных. Каждая реализация (OpenDotaSource,
    LiquipediaSource, StratzSource, ...) должна привести ответ своего
    API к этим унифицированным структурам.

    Важно: методы принимают временные границы (since/until), а не
    "последние N матчей" — это обязательное требование для
    воспроизводимого, детерминированного построения датасета.
    """

    name: str

    @abstractmethod
    def fetch_matches(
        self, since: datetime, until: datetime
    ) -> Iterable[RawMatch]:
        """Все матчи в полуоткрытом интервале [since, until)."""
        raise NotImplementedError

    @abstractmethod
    def fetch_picks_bans(self, match_id: int) -> Iterable[RawPickBan]:
        raise NotImplementedError

    @abstractmethod
    def fetch_player_matches(self, match_id: int) -> Iterable[RawPlayerMatch]:
        raise NotImplementedError

    def fetch_roster_changes(self, team_id: int) -> Iterable[RosterChange]:
        """
        Не все источники это поддерживают (например, OpenDota — нет).
        Источники без этих данных просто не переопределяют метод —
        пустой iterator по умолчанию, а не исключение, чтобы вызывающий
        код мог агрегировать несколько источников без try/except на
        каждый вызов.
        """
        return []
