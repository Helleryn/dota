"""
Phase 7, раздел 5-9 — historical roster reconstruction + roster
features. НЕ использует `/teams/{id}/players` (all-time агрегат, Phase 5
explicitly established это ненадёжно для point-in-time) — реконструирует
состав walk-forward из `match_players` (кто реально играл за команду в
каждом матче, известно до начала матча в реальности — публикуется до
драфта, не в процессе игры).

Walk-forward safety: то же правило, что `feature_set_0.py`/
`multi_window_features.py` — признаки читаются ДО обновления состояния
текущим матчем.

Определения (см. reports/phase7-summary.md за обоснование):
  - "активный состав" команды = набор account_id, сыгравших её ПОСЛЕДНИЙ
    матч (до текущего).
  - "текущий матч состав" читается из match_players ЭТОГО ЖЕ матча —
    считается pre-draft информацией (составы объявляются до матча в
    реальности), НЕ in-match статистикой (kills/gpm и т.п. НЕ используются
    здесь).
  - roster period = непрерывная последовательность матчей с ОДИНАКОВЫМ
    набором account_id.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Dict, FrozenSet, Iterable, List, Optional, Protocol

from sqlalchemy import Engine, text

REST_WINDOWS_DAYS = (7, 30, 90)


@dataclass(frozen=True)
class _MatchWithRosterRow:
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_roster: FrozenSet[int]
    dire_roster: FrozenSet[int]
    radiant_win: bool


def load_matches_with_rosters(engine: Engine) -> List[_MatchWithRosterRow]:
    """
    pro/premium tier матчи, для которых есть хотя бы частичные
    `match_players` — состав каждой команды собирается из account_id
    ИМЕННО ЭТОГО матча (group by is_radiant), НЕ из `/teams/{id}/players`
    (Phase 5: all-time агрегат, ненадёжен как point-in-time roster).
    Матчи без единой строки в match_players ни для одной из команд —
    не включаются (нет данных для roster-признаков), см.
    `reports/phase7-summary.md`, раздел Data coverage.
    """
    sql = text(
        """
        SELECT m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id, m.radiant_win,
               array_remove(array_agg(mp.account_id) FILTER (WHERE mp.is_radiant), NULL) AS radiant_roster,
               array_remove(array_agg(mp.account_id) FILTER (WHERE NOT mp.is_radiant), NULL) AS dire_roster
        FROM matches m
        JOIN leagues l ON l.league_id = m.league_id
        JOIN match_players mp ON mp.match_id = m.match_id
        WHERE l.tier IN ('professional', 'premium')
          AND m.radiant_team_id IS NOT NULL AND m.dire_team_id IS NOT NULL
          AND m.radiant_win IS NOT NULL
        GROUP BY m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id, m.radiant_win
        ORDER BY m.start_time ASC, m.match_id ASC
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()

    out = []
    for r in rows:
        start_time = r.start_time if r.start_time.tzinfo else r.start_time.replace(tzinfo=timezone.utc)
        out.append(
            _MatchWithRosterRow(
                match_id=r.match_id,
                start_time=start_time,
                radiant_team_id=r.radiant_team_id,
                dire_team_id=r.dire_team_id,
                radiant_roster=frozenset(r.radiant_roster or []),
                dire_roster=frozenset(r.dire_roster or []),
                radiant_win=r.radiant_win,
            )
        )
    return out


class MatchWithRosters(Protocol):
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_roster: FrozenSet[int]  # account_id этого матча (может быть неполным)
    dire_roster: FrozenSet[int]
    radiant_win: bool


@dataclass(frozen=True)
class RosterFeatureRow:
    match_id: int
    as_of_timestamp: datetime
    radiant_team_id: int
    dire_team_id: int

    radiant_roster_size: int
    dire_roster_size: int

    # PRE-MATCH (до этого матча, НЕ включая его собственный состав)
    radiant_roster_matches_together: int  # R2: сколько подряд матчей сыграл ТЕКУЩИЙ (пре-матчевый) состав
    dire_roster_matches_together: int
    radiant_roster_age_days: Optional[float]  # R3: с какого момента этот состав существует
    dire_roster_age_days: Optional[float]

    # Сравнение состава ЭТОГО матча с пре-матчевым "активным" составом
    radiant_player_continuity: Optional[float]  # R4
    dire_player_continuity: Optional[float]

    radiant_roster_changes: Dict[int, int]  # R5: {window_days: n_change_events}
    dire_roster_changes: Dict[int, int]

    radiant_win: bool


class _TeamRosterTracker:
    def __init__(self):
        self._active_roster: Dict[int, FrozenSet[int]] = {}
        self._roster_since: Dict[int, datetime] = {}
        self._roster_match_count: Dict[int, int] = defaultdict(int)
        self._change_events: Dict[int, Deque[datetime]] = defaultdict(lambda: deque(maxlen=200))

    def pre_match_state(self, team_id: int):
        roster = self._active_roster.get(team_id)
        since = self._roster_since.get(team_id)
        count = self._roster_match_count.get(team_id, 0)
        return roster, since, count

    def roster_age_days(self, team_id: int, as_of: datetime) -> Optional[float]:
        since = self._roster_since.get(team_id)
        if since is None:
            return None
        return (as_of - since).total_seconds() / 86400.0

    def changes_in_window(self, team_id: int, as_of: datetime, window_days: int) -> int:
        events = self._change_events.get(team_id)
        if not events:
            return 0
        cutoff = as_of.timestamp() - window_days * 86400.0
        return sum(1 for t in events if t.timestamp() >= cutoff)

    def observe(self, team_id: int, this_match_roster: FrozenSet[int], at: datetime) -> None:
        """Обновляет состояние ПОСЛЕ того, как признаки для текущего матча
        уже прочитаны (вызывающий код гарантирует порядок, как и в
        multi_window_features.build_multi_window_features)."""
        prev_roster = self._active_roster.get(team_id)

        if prev_roster is None or (this_match_roster and this_match_roster != prev_roster):
            # Новый период состава: либо первый матч команды, либо состав изменился.
            if prev_roster is not None and this_match_roster:
                self._change_events[team_id].append(at)
            self._active_roster[team_id] = this_match_roster
            self._roster_since[team_id] = at
            self._roster_match_count[team_id] = 1 if this_match_roster else 0
        else:
            self._roster_match_count[team_id] += 1


def _continuity(current: FrozenSet[int], previous: Optional[FrozenSet[int]]) -> Optional[float]:
    if not previous:
        return None
    if not current:
        return 0.0
    return len(current & previous) / len(previous)


def build_roster_features(matches: Iterable[MatchWithRosters]) -> List[RosterFeatureRow]:
    """matches ОБЯЗАН быть отсортирован по (start_time, match_id) —
    тот же tie-breaker, что и остальные Feature Set модули (Phase 6.5,
    раздел 10)."""
    tracker = _TeamRosterTracker()
    rows: List[RosterFeatureRow] = []

    for match in matches:
        radiant_prev_roster, radiant_since, radiant_count = tracker.pre_match_state(match.radiant_team_id)
        dire_prev_roster, dire_since, dire_count = tracker.pre_match_state(match.dire_team_id)

        radiant_age = tracker.roster_age_days(match.radiant_team_id, match.start_time)
        dire_age = tracker.roster_age_days(match.dire_team_id, match.start_time)

        radiant_continuity = _continuity(match.radiant_roster, radiant_prev_roster)
        dire_continuity = _continuity(match.dire_roster, dire_prev_roster)

        radiant_changes = {w: tracker.changes_in_window(match.radiant_team_id, match.start_time, w) for w in REST_WINDOWS_DAYS}
        dire_changes = {w: tracker.changes_in_window(match.dire_team_id, match.start_time, w) for w in REST_WINDOWS_DAYS}

        rows.append(
            RosterFeatureRow(
                match_id=match.match_id,
                as_of_timestamp=match.start_time,
                radiant_team_id=match.radiant_team_id,
                dire_team_id=match.dire_team_id,
                radiant_roster_size=len(match.radiant_roster),
                dire_roster_size=len(match.dire_roster),
                radiant_roster_matches_together=radiant_count,
                dire_roster_matches_together=dire_count,
                radiant_roster_age_days=radiant_age,
                dire_roster_age_days=dire_age,
                radiant_player_continuity=radiant_continuity,
                dire_player_continuity=dire_continuity,
                radiant_roster_changes=radiant_changes,
                dire_roster_changes=dire_changes,
                radiant_win=match.radiant_win,
            )
        )

        tracker.observe(match.radiant_team_id, match.radiant_roster, match.start_time)
        tracker.observe(match.dire_team_id, match.dire_roster, match.start_time)

    return rows
