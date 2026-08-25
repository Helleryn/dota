"""
Phase 9, PART A/B/C — как ПРАВИЛЬНО агрегировать силу пятёрки.

Phase 8 показала, что player-Elo несёт сигнал, но использовала одну грубую
агрегацию (`mean`). Здесь считаются ВСЕ кандидатные представления в ОДНОМ
walk-forward проходе, чтобы их можно было сравнить на идентичной выборке:

PART A — представление силы пятёрки:
    mean, sum, median, min, max, std, spread (max-min)
    weakest_link  = mean - std   (штраф за слабое звено)
    star_power    = max - mean   (насколько пятёрка «держится» на лидере)

PART B — roster delta (изменение состава):
    roster_strength_delta       — сила текущей пятёрки минус сила ПРЕДЫДУЩЕЙ
                                  пятёрки этой же команды, обе оценённые по
                                  ТЕКУЩИМ (на момент t) рейтингам. Это
                                  изолирует «кто в составе» от «как менялись
                                  рейтинги»: если состав не менялся, ровно 0.
    player_replacement_delta    — Elo пришедших минус Elo ушедших
    n_replacements_{7,14,30}d   — сколько замен за окно
    cum_roster_delta_{7,30}d    — накопленная сумма delta за окно

PART C — текущая пятёрка против исторической силы клуба:
    current_five_vs_team_elo    — mean(player Elo) - team Elo

Гипотеза Part B/C: team Elo реагирует на усиление состава ТОЛЬКО через
последующие результаты (медленно, ~K за матч). Разница «сила пятёрки минус
сила клуба» видна СРАЗУ в первом же матче нового состава.

## Leakage

Все рейтинги — из `_PlayerRatingEngine` (Phase 8), обновляются строго ПОСЛЕ
фиксации признаков. Предыдущий состав — из прошлых матчей команды.
Состав текущего матча берётся из `match_players` этого матча — то же
соглашение Phase 7/8 (в проде требует анонса лайнапа), НЕ игровая
статистика.
"""

from __future__ import annotations

import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, FrozenSet, Iterable, List, Optional, Protocol, Tuple

from src.datasets.player_features import DEFAULT_K, _PlayerRatingEngine

REPLACEMENT_WINDOWS_DAYS = (7, 14, 30)


class MatchWithRoster(Protocol):
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_roster: FrozenSet[int]
    dire_roster: FrozenSet[int]
    radiant_win: bool


@dataclass(frozen=True)
class TeamSideFeatures:
    # PART A
    elo_mean: float
    elo_sum: float
    elo_median: float
    elo_min: float
    elo_max: float
    elo_std: float
    elo_spread: float
    weakest_link: float
    star_power: float
    # PART B
    roster_strength_delta: float
    player_replacement_delta: float
    n_replacements: Dict[int, int]
    cum_roster_delta: Dict[int, float]
    # PART C
    current_five_vs_team_elo: float


@dataclass(frozen=True)
class RosterRepresentationRow:
    match_id: int
    as_of_timestamp: datetime
    radiant: TeamSideFeatures
    dire: TeamSideFeatures
    radiant_win: bool


class _RosterHistory:
    """Предыдущий состав команды и история событий замены."""

    def __init__(self):
        self._prev_roster: Dict[int, FrozenSet[int]] = {}
        # (время, delta силы) для окон
        self._events: Dict[int, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=500))

    def previous(self, team_id: int) -> Optional[FrozenSet[int]]:
        return self._prev_roster.get(team_id)

    def count_in_window(self, team_id: int, now_ts: float, days: int) -> int:
        ev = self._events.get(team_id)
        if not ev:
            return 0
        cutoff = now_ts - days * 86400.0
        return sum(1 for t, _ in ev if t >= cutoff)

    def sum_delta_in_window(self, team_id: int, now_ts: float, days: int) -> float:
        ev = self._events.get(team_id)
        if not ev:
            return 0.0
        cutoff = now_ts - days * 86400.0
        return sum(d for t, d in ev if t >= cutoff)

    def observe(self, team_id: int, roster: FrozenSet[int], now_ts: float, strength_delta: float) -> None:
        prev = self._prev_roster.get(team_id)
        if prev is not None and roster and roster != prev:
            self._events[team_id].append((now_ts, strength_delta))
        if roster:
            self._prev_roster[team_id] = roster


def _side_features(
    engine: _PlayerRatingEngine,
    history: _RosterHistory,
    team_id: int,
    roster: FrozenSet[int],
    now_ts: float,
) -> Tuple[TeamSideFeatures, float]:
    ratings = [engine.rating(p) for p in roster] or [engine.base_rating]
    mean = sum(ratings) / len(ratings)
    median = statistics.median(ratings)
    lo, hi = min(ratings), max(ratings)
    std = statistics.pstdev(ratings) if len(ratings) > 1 else 0.0

    prev = history.previous(team_id)
    if prev:
        prev_ratings = [engine.rating(p) for p in prev]
        prev_mean = sum(prev_ratings) / len(prev_ratings)
        # ОБЕ пятёрки оценены ТЕКУЩИМИ рейтингами -> при неизменном составе ровно 0
        roster_strength_delta = mean - prev_mean
        incoming = roster - prev
        outgoing = prev - roster
        repl_delta = (
            sum(engine.rating(p) for p in incoming) / len(incoming) if incoming else 0.0
        ) - (
            sum(engine.rating(p) for p in outgoing) / len(outgoing) if outgoing else 0.0
        )
    else:
        roster_strength_delta = 0.0
        repl_delta = 0.0

    feats = TeamSideFeatures(
        elo_mean=mean,
        elo_sum=sum(ratings),
        elo_median=median,
        elo_min=lo,
        elo_max=hi,
        elo_std=std,
        elo_spread=hi - lo,
        weakest_link=mean - std,
        star_power=hi - mean,
        roster_strength_delta=roster_strength_delta,
        player_replacement_delta=repl_delta,
        n_replacements={w: history.count_in_window(team_id, now_ts, w) for w in REPLACEMENT_WINDOWS_DAYS},
        cum_roster_delta={w: history.sum_delta_in_window(team_id, now_ts, w) for w in (7, 30)},
        current_five_vs_team_elo=mean - engine.team_rating(team_id),
    )
    return feats, roster_strength_delta


def build_roster_representation(
    matches: Iterable[MatchWithRoster],
    k_factor: float = DEFAULT_K,
) -> List[RosterRepresentationRow]:
    """matches ОБЯЗАН быть отсортирован по (start_time, match_id)."""
    engine = _PlayerRatingEngine(k_factor=k_factor)
    history = _RosterHistory()
    rows: List[RosterRepresentationRow] = []

    for m in matches:
        ts = m.start_time.timestamp()
        r_feats, r_delta = _side_features(engine, history, m.radiant_team_id, m.radiant_roster, ts)
        d_feats, d_delta = _side_features(engine, history, m.dire_team_id, m.dire_roster, ts)

        rows.append(RosterRepresentationRow(
            match_id=m.match_id,
            as_of_timestamp=m.start_time,
            radiant=r_feats,
            dire=d_feats,
            radiant_win=m.radiant_win,
        ))

        # состояние обновляется строго ПОСЛЕ фиксации признаков
        history.observe(m.radiant_team_id, m.radiant_roster, ts, r_delta)
        history.observe(m.dire_team_id, m.dire_roster, ts, d_delta)
        engine.process(m.radiant_roster, m.dire_roster, m.radiant_team_id, m.dire_team_id, m.radiant_win)

    return rows


def to_feature_dict(row: RosterRepresentationRow) -> dict:
    """Разности radiant-dire — формат, ожидаемый моделями проекта."""
    r, d = row.radiant, row.dire
    out = {
        "match_id": row.match_id,
        # PART A — каждая агрегация отдельно
        "elo_mean_diff": r.elo_mean - d.elo_mean,
        "elo_sum_diff": r.elo_sum - d.elo_sum,
        "elo_median_diff": r.elo_median - d.elo_median,
        "elo_min_diff": r.elo_min - d.elo_min,
        "elo_max_diff": r.elo_max - d.elo_max,
        "elo_std_diff": r.elo_std - d.elo_std,
        "elo_spread_diff": r.elo_spread - d.elo_spread,
        "weakest_link_diff": r.weakest_link - d.weakest_link,
        "star_power_diff": r.star_power - d.star_power,
        # PART B
        "roster_strength_delta_diff": r.roster_strength_delta - d.roster_strength_delta,
        "player_replacement_delta_diff": r.player_replacement_delta - d.player_replacement_delta,
        # PART C
        "five_vs_team_elo_diff": r.current_five_vs_team_elo - d.current_five_vs_team_elo,
    }
    for w in REPLACEMENT_WINDOWS_DAYS:
        out[f"n_replacements_{w}d_diff"] = r.n_replacements[w] - d.n_replacements[w]
    for w in (7, 30):
        out[f"cum_roster_delta_{w}d_diff"] = r.cum_roster_delta[w] - d.cum_roster_delta[w]
    return out
