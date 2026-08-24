"""
Feature Set 0 (Phase 5, раздел 25; docs/features.md, Feature Set 0):

    team_elo, elo_difference, recent_winrate_5, recent_winrate_difference,
    days_since_last_match

Чистая логика, без БД — принимает поток матчей в хронологическом порядке,
использует RatingEngine (src/ratings/engine.py) для Elo и собственный
in-memory tracker для recent form/last-match-date. Та же гарантия
leakage-safety, что у RatingEngine: признаки для матча читаются ДО
обновления состояния этим же матчем (docs/data-leakage.md).

Признаки, которые НЕ вошли (см. docs/features.md, но недостаточно надёжны
для Feature Set 0 без polноценного ростер-реконструктора или Liquipedia —
Phase 5 прямо требует не добавлять признак "просто ради количества"):
roster stability, patch winrate (мало матчей на патч в MVP-выборке —
статистически шумно), head-to-head (то же самое).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, Iterable, List, Optional, Protocol

from src.ratings.engine import RatingEngine

RECENT_FORM_WINDOW = 5


class MatchForFeatures(Protocol):
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_win: bool


@dataclass(frozen=True)
class FeatureRow:
    match_id: int
    as_of_timestamp: datetime  # момент, на который признаки актуальны (= start_time матча)

    radiant_team_id: int
    dire_team_id: int

    radiant_elo: float
    dire_elo: float
    elo_difference: float  # radiant - dire

    radiant_recent_winrate: Optional[float]  # None = недостаточно истории (Phase 5, раздел 9)
    dire_recent_winrate: Optional[float]
    recent_winrate_difference: Optional[float]

    radiant_days_since_last_match: Optional[float]
    dire_days_since_last_match: Optional[float]

    radiant_matches_played_before: int  # для missing-value стратегии/аудита
    dire_matches_played_before: int

    radiant_win: bool  # целевая переменная — ЕДИНСТВЕННОЕ поле, известное только post-hoc


class _TeamFormTracker:
    def __init__(self, window: int = RECENT_FORM_WINDOW):
        self._window = window
        self._results: Dict[int, Deque[bool]] = defaultdict(lambda: deque(maxlen=window))
        self._last_match_time: Dict[int, datetime] = {}
        self._matches_played: Dict[int, int] = defaultdict(int)

    def recent_winrate(self, team_id: int) -> Optional[float]:
        results = self._results.get(team_id)
        if not results:
            return None
        return sum(results) / len(results)

    def days_since_last_match(self, team_id: int, as_of: datetime) -> Optional[float]:
        last = self._last_match_time.get(team_id)
        if last is None:
            return None
        return (as_of - last).total_seconds() / 86400.0

    def matches_played(self, team_id: int) -> int:
        return self._matches_played.get(team_id, 0)

    def record(self, team_id: int, won: bool, at: datetime) -> None:
        self._results[team_id].append(won)
        self._last_match_time[team_id] = at
        self._matches_played[team_id] += 1


def build_feature_set_0(matches: Iterable[MatchForFeatures]) -> List[FeatureRow]:
    """
    matches ОБЯЗАН быть отсортирован по start_time по возрастанию —
    функция это не проверяет сама (ответственность вызывающего кода,
    как и у RatingEngine.process_match), но КАЖДАЯ строка использует
    только состояние, накопленное по предыдущим элементам итератора.
    """
    rating_engine = RatingEngine()
    form_tracker = _TeamFormTracker()
    rows: List[FeatureRow] = []

    for match in matches:
        # 1. ЧТЕНИЕ состояния ДО этого матча (pre-match, leakage-safe).
        radiant_winrate = form_tracker.recent_winrate(match.radiant_team_id)
        dire_winrate = form_tracker.recent_winrate(match.dire_team_id)
        winrate_diff = (
            radiant_winrate - dire_winrate if radiant_winrate is not None and dire_winrate is not None else None
        )

        radiant_days = form_tracker.days_since_last_match(match.radiant_team_id, match.start_time)
        dire_days = form_tracker.days_since_last_match(match.dire_team_id, match.start_time)

        radiant_matches_before = form_tracker.matches_played(match.radiant_team_id)
        dire_matches_before = form_tracker.matches_played(match.dire_team_id)

        # RatingEngine.process_match() уже возвращает pre-match рейтинги в
        # snapshot.*_pre — читаем их и обновляем состояние ОДНИМ вызовом
        # (сам RatingEngine это гарантирует, см. src/ratings/engine.py).
        snapshot = rating_engine.process_match(match)  # noqa: этот вызов ОБНОВЛЯЕТ состояние — см. ниже

        rows.append(
            FeatureRow(
                match_id=match.match_id,
                as_of_timestamp=match.start_time,
                radiant_team_id=match.radiant_team_id,
                dire_team_id=match.dire_team_id,
                radiant_elo=snapshot.radiant_pre,
                dire_elo=snapshot.dire_pre,
                elo_difference=snapshot.radiant_pre - snapshot.dire_pre,
                radiant_recent_winrate=radiant_winrate,
                dire_recent_winrate=dire_winrate,
                recent_winrate_difference=winrate_diff,
                radiant_days_since_last_match=radiant_days,
                dire_days_since_last_match=dire_days,
                radiant_matches_played_before=radiant_matches_before,
                dire_matches_played_before=dire_matches_before,
                radiant_win=match.radiant_win,
            )
        )

        # 2. ОБНОВЛЕНИЕ состояния — строго ПОСЛЕ того, как признаки для
        # этого матча уже зафиксированы (form_tracker; RatingEngine уже
        # обновлён вызовом process_match выше).
        form_tracker.record(match.radiant_team_id, match.radiant_win, match.start_time)
        form_tracker.record(match.dire_team_id, not match.radiant_win, match.start_time)

    return rows
