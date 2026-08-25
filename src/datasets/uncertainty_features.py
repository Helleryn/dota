"""
PHASE 13 — ковариаты неопределённости (PART C/E/J/L).

Это **не признаки для предсказания исхода**. Их задача другая: описать,
насколько мало модель знает о конкретном матче. Отсюда два следствия,
определяющих устройство модуля:

1. величины намеренно симметричны или односторонни (min / max по сторонам),
   а не «radiant минус dire»: неопределённость не имеет знака — матч
   одинаково труден, если мало данных о любой из команд;
2. ни одна величина не должна коррелировать с ИСХОДОМ по построению —
   иначе это скрытый признак силы, а не меры незнания.

Все значения читаются ДО матча и обновляются строго ПОСЛЕ фиксации
(контракт всех Feature Set модулей проекта). Единственное исключение по
смыслу — `days_since_patch`: дата релиза патча известна заранее, это
календарный факт, а не статистика будущего.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

HERO_HALF_LIFE_DAYS = 90.0
NEW_HERO_THRESHOLD = 20.0     # меньше стольких затухающих игр — герой «редкий»


class PlayerInMatch(Protocol):
    account_id: int
    hero_id: int
    is_radiant: bool


class MatchForUncertainty(Protocol):
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_win: bool
    players: Sequence[PlayerInMatch]


@dataclass(frozen=True)
class UncertaintyRow:
    match_id: int
    as_of_timestamp: datetime
    # --- объём истории ---
    team_matches_min: int          # у худше изученной из двух команд
    team_matches_max: int
    player_matches_min: int        # у худше изученного из десяти игроков
    player_matches_mean: float
    # --- новизна состава (PART L) ---
    new_players_max: int           # сколько игроков не играли в прошлом матче команды
    new_players_total: int
    roster_matches_together_min: int
    roster_age_days_min: Optional[float]
    # --- новизна героев (PART E) ---
    hero_games_min: float          # затухающие игры самого редкого из 10 героев
    rare_heroes_count: int
    # --- переходы игроков (PART M) ---
    # ВНИМАНИЕ: две следующие величины опираются на `team_id` и потому
    # ЗАГРЯЗНЕНЫ фрагментацией идентичности команд (Phase 10: 7838 team_id,
    # медиана 3 матча). Замер Phase 13: пики ровно на 5 и 10 «переходах»
    # (3471 и 1224 матча) — это смена team_id целиком, а не переход игроков.
    # Оставлены для сопоставимости и как документированный дефект.
    transferred_players_total: int
    days_since_transfer_min: Optional[float]
    # --- переход БЕЗ опоры на team_id (устойчиво к фрагментации) ---
    # «Переход» определяется через смену ПАРТНЁРОВ: если у игрока сменилось
    # больше половины четвёрки, это смена окружения независимо от того, под
    # каким team_id команда записана.
    teammate_churn_max: float        # макс. по сторонам доля сменившихся партнёров
    teammate_churn_mean: float
    players_with_new_teammates: int  # сколько игроков сменили >половины партнёров
    radiant_win: bool


class _HeroDecay:
    def __init__(self, hl: float):
        self.hl = hl
        self._g: Dict[int, float] = defaultdict(float)
        self._t: Dict[int, float] = {}

    def games(self, h: int, ts: float) -> float:
        last = self._t.get(h)
        if last is None:
            return 0.0
        dt = (ts - last) / 86400.0
        return self._g[h] * (0.5 ** (dt / self.hl)) if dt > 0 else self._g[h]

    def observe(self, h: int, ts: float) -> None:
        self._g[h] = self.games(h, ts) + 1.0
        self._t[h] = ts


def build_uncertainty_features(
    matches: Iterable[MatchForUncertainty],
    hero_half_life_days: float = HERO_HALF_LIFE_DAYS,
) -> List[UncertaintyRow]:
    """`matches` ОБЯЗАН быть отсортирован по (start_time, match_id)."""
    team_games: Dict[int, int] = defaultdict(int)
    player_games: Dict[int, int] = defaultdict(int)
    last_roster: Dict[int, frozenset] = {}          # team_id -> состав прошлого матча
    roster_streak: Dict[int, int] = defaultdict(int)  # сколько матчей подряд тот же состав
    roster_since: Dict[int, float] = {}             # ts начала текущего состава
    player_team: Dict[int, int] = {}                # последняя команда игрока
    player_team_since: Dict[int, float] = {}        # с какого момента он в ней
    player_mates: Dict[int, frozenset] = {}         # партнёры игрока в прошлом матче
    heroes = _HeroDecay(hero_half_life_days)

    rows: List[UncertaintyRow] = []

    for m in matches:
        ts = m.start_time.timestamp()
        sides: Dict[bool, List[PlayerInMatch]] = {True: [], False: []}
        for p in m.players:
            sides[bool(p.is_radiant)].append(p)
        team_of = {True: m.radiant_team_id, False: m.dire_team_id}

        tg = [team_games[team_of[s]] for s in (True, False)]
        pg = [player_games[p.account_id] for p in m.players]

        new_players, together, ages = [], [], []
        for s in (True, False):
            tid = team_of[s]
            roster = frozenset(p.account_id for p in sides[s])
            prev = last_roster.get(tid)
            new_players.append(len(roster - prev) if prev is not None else len(roster))
            together.append(roster_streak[tid] if prev == roster else 0)
            since = roster_since.get(tid)
            ages.append((ts - since) / 86400.0 if since is not None and prev == roster else 0.0)

        hg = [heroes.games(p.hero_id, ts) for p in m.players]

        transferred, since_transfer = 0, []
        for p in m.players:
            prev_team = player_team.get(p.account_id)
            cur_team = team_of[bool(p.is_radiant)]
            if prev_team is not None and prev_team != cur_team:
                transferred += 1
            st = player_team_since.get(p.account_id)
            if st is not None:
                since_transfer.append((ts - st) / 86400.0)

        churn_by_side, new_mates = [], 0
        for s_ in (True, False):
            vals = []
            for p in sides[s_]:
                mates = frozenset(q.account_id for q in sides[s_] if q.account_id != p.account_id)
                prev = player_mates.get(p.account_id)
                if prev is None or not prev:
                    vals.append(0.0)      # нет истории — не выдумываем смену
                    continue
                ch = 1.0 - len(mates & prev) / len(prev)
                vals.append(ch)
                if ch > 0.5:
                    new_mates += 1
            churn_by_side.append(sum(vals) / len(vals) if vals else 0.0)

        rows.append(UncertaintyRow(
            match_id=m.match_id,
            as_of_timestamp=m.start_time,
            team_matches_min=int(min(tg)),
            team_matches_max=int(max(tg)),
            player_matches_min=int(min(pg)) if pg else 0,
            player_matches_mean=float(sum(pg) / len(pg)) if pg else 0.0,
            new_players_max=int(max(new_players)) if new_players else 0,
            new_players_total=int(sum(new_players)),
            roster_matches_together_min=int(min(together)) if together else 0,
            roster_age_days_min=float(min(ages)) if ages else None,
            hero_games_min=float(min(hg)) if hg else 0.0,
            rare_heroes_count=int(sum(1 for g in hg if g < NEW_HERO_THRESHOLD)),
            transferred_players_total=transferred,
            days_since_transfer_min=float(min(since_transfer)) if since_transfer else None,
            teammate_churn_max=float(max(churn_by_side)) if churn_by_side else 0.0,
            teammate_churn_mean=float(sum(churn_by_side) / len(churn_by_side)) if churn_by_side else 0.0,
            players_with_new_teammates=int(new_mates),
            radiant_win=m.radiant_win,
        ))

        # ---------- обновление строго ПОСЛЕ фиксации признаков ----------
        for s in (True, False):
            tid = team_of[s]
            roster = frozenset(p.account_id for p in sides[s])
            team_games[tid] += 1
            if last_roster.get(tid) == roster:
                roster_streak[tid] += 1
            else:
                roster_streak[tid] = 1
                roster_since[tid] = ts
            last_roster[tid] = roster
        for p in m.players:
            player_games[p.account_id] += 1
            cur_team = team_of[bool(p.is_radiant)]
            if player_team.get(p.account_id) != cur_team:
                player_team_since[p.account_id] = ts
            player_team[p.account_id] = cur_team
            player_mates[p.account_id] = frozenset(
                q.account_id for q in sides[bool(p.is_radiant)] if q.account_id != p.account_id)
            heroes.observe(p.hero_id, ts)

    return rows


UNCERTAINTY_COLUMNS: Tuple[str, ...] = (
    "team_matches_min", "team_matches_max",
    "player_matches_min", "player_matches_mean",
    "new_players_max", "new_players_total",
    "roster_matches_together_min", "roster_age_days_min",
    "hero_games_min", "rare_heroes_count",
    "transferred_players_total", "days_since_transfer_min",
    "teammate_churn_max", "teammate_churn_mean", "players_with_new_teammates",
)


def to_feature_dict(row: UncertaintyRow) -> Dict[str, object]:
    d: Dict[str, object] = {"match_id": row.match_id}
    for c in UNCERTAINTY_COLUMNS:
        v = getattr(row, c)
        d[c] = float(v) if v is not None else float("nan")
    return d
