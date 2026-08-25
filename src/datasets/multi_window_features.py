"""
Phase 6.5 (раздел 11-12) — расширение Feature Set 0 несколькими окнами
recent form (3/5/10/20) и rest-признаками (matches_last_N_days), для
ablation "какое окно реально несёт сигнал". НЕ заменяет
`src/datasets/feature_set_0.py` (Phase 5, используется PredictionService/
DatasetBuilder как есть) — отдельный, аддитивный расчёт для экспериментов
этой фазы, использующий тот же `RatingEngine` (Elo не дублируется) и тот же
принцип leakage-safety (читаем состояние ДО матча, обновляем строго после).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, Iterable, List, Optional, Protocol

from src.ratings.engine import RatingEngine

FORM_WINDOWS = (3, 5, 10, 20)
REST_WINDOWS_DAYS = (7, 14, 30)


class MatchForFeatures(Protocol):
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_win: bool


@dataclass(frozen=True)
class MultiWindowFeatureRow:
    match_id: int
    as_of_timestamp: datetime
    radiant_team_id: int
    dire_team_id: int

    elo_difference: float

    # {window_size: difference}, None если у ХОТЯ БЫ ОДНОЙ из команд нет истории
    recent_winrate_difference: Dict[int, Optional[float]]

    days_since_last_match_difference: Optional[float]
    matches_last_n_days_difference: Dict[int, int]  # 0 по умолчанию — отсутствие матчей, не пропуск

    radiant_matches_played_before: int
    dire_matches_played_before: int

    radiant_win: bool


class _MultiWindowTeamTracker:
    """Хранит СТРОГО ПРОШЛУЮ историю результатов и дат матчей на команду —
    windows считаются из одного и того же deque(maxlen=max(FORM_WINDOWS)),
    меньшие окна — просто срез последних K элементов (не отдельные
    структуры) — гарантирует консистентность между окнами по построению
    (не может быть, что form_3 и form_5 посчитаны по разным подвыборкам)."""

    def __init__(self):
        max_window = max(FORM_WINDOWS)
        self._results: Dict[int, Deque[bool]] = defaultdict(lambda: deque(maxlen=max_window))
        self._match_times: Dict[int, Deque[datetime]] = defaultdict(lambda: deque(maxlen=max_window))
        self._last_match_time: Dict[int, datetime] = {}
        self._matches_played: Dict[int, int] = defaultdict(int)

    def recent_winrate(self, team_id: int, window: int) -> Optional[float]:
        results = self._results.get(team_id)
        if not results:
            return None
        recent = list(results)[-window:]
        return sum(recent) / len(recent) if recent else None

    def days_since_last_match(self, team_id: int, as_of: datetime) -> Optional[float]:
        last = self._last_match_time.get(team_id)
        return (as_of - last).total_seconds() / 86400.0 if last is not None else None

    def matches_last_n_days(self, team_id: int, as_of: datetime, n_days: int) -> int:
        times = self._match_times.get(team_id)
        if not times:
            return 0
        cutoff = as_of.timestamp() - n_days * 86400.0
        return sum(1 for t in times if t.timestamp() >= cutoff)

    def matches_played(self, team_id: int) -> int:
        return self._matches_played.get(team_id, 0)

    def record(self, team_id: int, won: bool, at: datetime) -> None:
        self._results[team_id].append(won)
        self._match_times[team_id].append(at)
        self._last_match_time[team_id] = at
        self._matches_played[team_id] += 1


def build_multi_window_features(matches: Iterable[MatchForFeatures], k_factor: float = 32.0) -> List[MultiWindowFeatureRow]:
    """
    matches ОБЯЗАН быть отсортирован по (start_time, match_id) — тот же
    tie-breaker, что и `_load_pro_matches` (src/datasets/builder.py, Phase
    6.5 раздел 10) — не проверяется здесь (ответственность вызывающего
    кода, как и у build_feature_set_0).

    k_factor — параметризован ради Elo K-анализа (Phase 6.5, раздел 20):
    K подбирается ТОЛЬКО на validation, не на test (см. scripts/phase6_5_pipeline.py).
    """
    rating_engine = RatingEngine(k_factor=k_factor)
    tracker = _MultiWindowTeamTracker()
    rows: List[MultiWindowFeatureRow] = []

    for match in matches:
        radiant_winrates = {w: tracker.recent_winrate(match.radiant_team_id, w) for w in FORM_WINDOWS}
        dire_winrates = {w: tracker.recent_winrate(match.dire_team_id, w) for w in FORM_WINDOWS}
        winrate_diff = {
            w: (radiant_winrates[w] - dire_winrates[w])
            if radiant_winrates[w] is not None and dire_winrates[w] is not None
            else None
            for w in FORM_WINDOWS
        }

        radiant_days = tracker.days_since_last_match(match.radiant_team_id, match.start_time)
        dire_days = tracker.days_since_last_match(match.dire_team_id, match.start_time)
        days_diff = radiant_days - dire_days if radiant_days is not None and dire_days is not None else None

        matches_n_days_diff = {
            n: tracker.matches_last_n_days(match.radiant_team_id, match.start_time, n)
            - tracker.matches_last_n_days(match.dire_team_id, match.start_time, n)
            for n in REST_WINDOWS_DAYS
        }

        radiant_matches_before = tracker.matches_played(match.radiant_team_id)
        dire_matches_before = tracker.matches_played(match.dire_team_id)

        snapshot = rating_engine.process_match(match)  # обновляет Elo — вызывается ПОСЛЕ чтения form/rest выше

        rows.append(
            MultiWindowFeatureRow(
                match_id=match.match_id,
                as_of_timestamp=match.start_time,
                radiant_team_id=match.radiant_team_id,
                dire_team_id=match.dire_team_id,
                elo_difference=snapshot.radiant_pre - snapshot.dire_pre,
                recent_winrate_difference=winrate_diff,
                days_since_last_match_difference=days_diff,
                matches_last_n_days_difference=matches_n_days_diff,
                radiant_matches_played_before=radiant_matches_before,
                dire_matches_played_before=dire_matches_before,
                radiant_win=match.radiant_win,
            )
        )

        tracker.record(match.radiant_team_id, match.radiant_win, match.start_time)
        tracker.record(match.dire_team_id, not match.radiant_win, match.start_time)

    return rows
