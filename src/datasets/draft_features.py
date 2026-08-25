"""
Phase 7, раздел 10-14 — draft features. Walk-forward: ВСЕ статистики
(hero win rate, team-hero history, pick popularity, hero matchup win rate)
считаются ТОЛЬКО по матчам со start_time СТРОГО раньше текущего (раздел
12: "hero_winrate = all historical matches" для 2023-прогноза запрещено).

Semantics picks_bans.team проверена ЭМПИРИЧЕСКИ на реальных данных (не
предположена из названия поля, раздел 10 задания): team=0 <-> is_radiant
(2000/2000 совпадений с match_players.is_radiant по тому же герою, см.
reports/phase7-summary.md).

Уровни (раздел 15 задания), собраны в ОДНОМ walk-forward проходе ради
эффективности (не 4 отдельных прохода):
  D1 — hero strength (overall + patch-scoped, раздел 13)
  D2 — team-hero history
  D3 — pick popularity (глобальная частота пика героя, прокси "pick statistics")
  D4/D14 — draft interaction (hero-vs-hero historical matchup win rate)

НЕ реализовано в этой фазе (см. reports/phase7-summary.md, Limitations):
ban-specific frequency features (раздел 10 задания предлагает pick+ban
statistics как ОДИН уровень DRAFT-3 — здесь взята pick-часть как
представительная, ban-статистика — кандидат для отдельного эксперимента).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

from sqlalchemy import Engine, text


class MatchWithDraft(Protocol):
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    patch_id: Optional[int]
    radiant_picks: Tuple[int, ...]
    dire_picks: Tuple[int, ...]
    radiant_win: bool


@dataclass(frozen=True)
class _MatchDraftRow:
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    patch_id: Optional[int]
    radiant_picks: Tuple[int, ...]
    dire_picks: Tuple[int, ...]
    radiant_win: bool


def load_matches_with_draft(engine: Engine) -> List[_MatchDraftRow]:
    """pro/premium матчи с ПОЛНЫМ draft (5 пиков на сторону) — матчи с
    частичным/отсутствующим draft исключены (см. DRAFT_COMPLETE_SET,
    раздел 2 задания: основной continuous dataset НЕ удаляется, это
    отдельная выборка только для draft-экспериментов)."""
    sql = text(
        """
        SELECT m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id, m.radiant_win, m.patch_id,
               array_agg(pb.hero_id ORDER BY pb.ord) FILTER (WHERE pb.is_pick AND pb.team = 0) AS radiant_picks,
               array_agg(pb.hero_id ORDER BY pb.ord) FILTER (WHERE pb.is_pick AND pb.team = 1) AS dire_picks
        FROM matches m
        JOIN leagues l ON l.league_id = m.league_id
        JOIN picks_bans pb ON pb.match_id = m.match_id
        WHERE l.tier IN ('professional', 'premium')
          AND m.radiant_team_id IS NOT NULL AND m.dire_team_id IS NOT NULL
          AND m.radiant_win IS NOT NULL
        GROUP BY m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id, m.radiant_win, m.patch_id
        HAVING array_length(array_agg(pb.hero_id) FILTER (WHERE pb.is_pick AND pb.team = 0), 1) = 5
           AND array_length(array_agg(pb.hero_id) FILTER (WHERE pb.is_pick AND pb.team = 1), 1) = 5
        ORDER BY m.start_time ASC, m.match_id ASC
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    return [
        _MatchDraftRow(
            match_id=r.match_id, start_time=r.start_time, radiant_team_id=r.radiant_team_id,
            dire_team_id=r.dire_team_id, patch_id=r.patch_id,
            radiant_picks=tuple(r.radiant_picks), dire_picks=tuple(r.dire_picks), radiant_win=r.radiant_win,
        )
        for r in rows
    ]


@dataclass(frozen=True)
class DraftFeatureRow:
    match_id: int
    as_of_timestamp: datetime
    radiant_team_id: int
    dire_team_id: int

    radiant_hero_strength: Optional[float]  # D1: mean overall win rate своих 5 героев
    dire_hero_strength: Optional[float]
    radiant_hero_strength_patch: Optional[float]  # D1 + раздел 13: patch-scoped
    dire_hero_strength_patch: Optional[float]

    radiant_team_hero_experience: float  # D2: mean игр команды с этими героями (0 = валидно, не пропуск)
    dire_team_hero_experience: float
    radiant_team_hero_winrate: Optional[float]  # D2
    dire_team_hero_winrate: Optional[float]

    radiant_pick_popularity: Optional[float]  # D3
    dire_pick_popularity: Optional[float]

    matchup_advantage: Optional[float]  # D4/D14: mean win rate radiant-героев против dire-героев (5x5), с точки зрения radiant

    radiant_win: bool


class _DraftStatsTracker:
    def __init__(self):
        self._hero_games: Dict[int, int] = defaultdict(int)
        self._hero_wins: Dict[int, int] = defaultdict(int)
        self._hero_games_patch: Dict[Tuple[int, int], int] = defaultdict(int)  # (patch_id, hero_id)
        self._hero_wins_patch: Dict[Tuple[int, int], int] = defaultdict(int)
        self._team_hero_games: Dict[Tuple[int, int], int] = defaultdict(int)  # (team_id, hero_id)
        self._team_hero_wins: Dict[Tuple[int, int], int] = defaultdict(int)
        self._total_drafts: int = 0
        self._matchup_games: Dict[Tuple[int, int], int] = defaultdict(int)  # (hero_a, hero_b) -> games where a's side played vs b
        self._matchup_wins: Dict[Tuple[int, int], int] = defaultdict(int)  # a's side won

    def hero_winrate(self, hero_id: int) -> Optional[float]:
        g = self._hero_games.get(hero_id, 0)
        return self._hero_wins.get(hero_id, 0) / g if g > 0 else None

    def hero_winrate_patch(self, patch_id: Optional[int], hero_id: int) -> Optional[float]:
        if patch_id is None:
            return None
        g = self._hero_games_patch.get((patch_id, hero_id), 0)
        return self._hero_wins_patch.get((patch_id, hero_id), 0) / g if g > 0 else None

    def team_hero_games(self, team_id: int, hero_id: int) -> int:
        return self._team_hero_games.get((team_id, hero_id), 0)

    def team_hero_winrate(self, team_id: int, hero_id: int) -> Optional[float]:
        g = self._team_hero_games.get((team_id, hero_id), 0)
        return self._team_hero_wins.get((team_id, hero_id), 0) / g if g > 0 else None

    def pick_popularity(self, hero_id: int) -> Optional[float]:
        if self._total_drafts == 0:
            return None
        return self._hero_games.get(hero_id, 0) / self._total_drafts

    def matchup_winrate(self, hero_a: int, hero_b: int) -> Optional[float]:
        g = self._matchup_games.get((hero_a, hero_b), 0)
        return self._matchup_wins.get((hero_a, hero_b), 0) / g if g > 0 else None

    def observe(self, radiant_team_id: int, dire_team_id: int, radiant_picks, dire_picks, radiant_win: bool, patch_id: Optional[int]) -> None:
        self._total_drafts += 1
        for hero_id, team_id, won in (
            *[(h, radiant_team_id, radiant_win) for h in radiant_picks],
            *[(h, dire_team_id, not radiant_win) for h in dire_picks],
        ):
            self._hero_games[hero_id] += 1
            self._hero_wins[hero_id] += int(won)
            if patch_id is not None:
                self._hero_games_patch[(patch_id, hero_id)] += 1
                self._hero_wins_patch[(patch_id, hero_id)] += int(won)
            self._team_hero_games[(team_id, hero_id)] += 1
            self._team_hero_wins[(team_id, hero_id)] += int(won)

        for ha in radiant_picks:
            for hb in dire_picks:
                self._matchup_games[(ha, hb)] += 1
                self._matchup_wins[(ha, hb)] += int(radiant_win)
                self._matchup_games[(hb, ha)] += 1
                self._matchup_wins[(hb, ha)] += int(not radiant_win)


def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


def build_draft_features(matches: Iterable[MatchWithDraft]) -> List[DraftFeatureRow]:
    """matches ОБЯЗАН быть отсортирован по (start_time, match_id)."""
    tracker = _DraftStatsTracker()
    rows: List[DraftFeatureRow] = []

    for match in matches:
        r_strength = _mean_or_none([tracker.hero_winrate(h) for h in match.radiant_picks])
        d_strength = _mean_or_none([tracker.hero_winrate(h) for h in match.dire_picks])
        r_strength_patch = _mean_or_none([tracker.hero_winrate_patch(match.patch_id, h) for h in match.radiant_picks])
        d_strength_patch = _mean_or_none([tracker.hero_winrate_patch(match.patch_id, h) for h in match.dire_picks])

        r_team_hero_exp = sum(tracker.team_hero_games(match.radiant_team_id, h) for h in match.radiant_picks) / 5.0
        d_team_hero_exp = sum(tracker.team_hero_games(match.dire_team_id, h) for h in match.dire_picks) / 5.0
        r_team_hero_wr = _mean_or_none([tracker.team_hero_winrate(match.radiant_team_id, h) for h in match.radiant_picks])
        d_team_hero_wr = _mean_or_none([tracker.team_hero_winrate(match.dire_team_id, h) for h in match.dire_picks])

        r_pick_pop = _mean_or_none([tracker.pick_popularity(h) for h in match.radiant_picks])
        d_pick_pop = _mean_or_none([tracker.pick_popularity(h) for h in match.dire_picks])

        matchup_values = [
            tracker.matchup_winrate(ha, hb) for ha in match.radiant_picks for hb in match.dire_picks
        ]
        matchup_advantage = _mean_or_none(matchup_values)

        rows.append(
            DraftFeatureRow(
                match_id=match.match_id,
                as_of_timestamp=match.start_time,
                radiant_team_id=match.radiant_team_id,
                dire_team_id=match.dire_team_id,
                radiant_hero_strength=r_strength,
                dire_hero_strength=d_strength,
                radiant_hero_strength_patch=r_strength_patch,
                dire_hero_strength_patch=d_strength_patch,
                radiant_team_hero_experience=r_team_hero_exp,
                dire_team_hero_experience=d_team_hero_exp,
                radiant_team_hero_winrate=r_team_hero_wr,
                dire_team_hero_winrate=d_team_hero_wr,
                radiant_pick_popularity=r_pick_pop,
                dire_pick_popularity=d_pick_pop,
                matchup_advantage=matchup_advantage,
                radiant_win=match.radiant_win,
            )
        )

        tracker.observe(match.radiant_team_id, match.dire_team_id, match.radiant_picks, match.dire_picks, match.radiant_win, match.patch_id)

    return rows
