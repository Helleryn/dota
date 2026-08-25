"""
Phase 7 — leakage-safety для src/datasets/roster_features.py. Тот же
принцип, что tests/leakage/test_multi_window_features_leakage.py (Phase
6.5), для НОВОГО модуля.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import FrozenSet

from src.datasets.roster_features import build_roster_features


@dataclass(frozen=True)
class M:
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_roster: FrozenSet[int]
    dire_roster: FrozenSet[int]
    radiant_win: bool


def _make_matches():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    roster_a1 = frozenset({1, 2, 3, 4, 5})
    roster_a2 = frozenset({1, 2, 3, 4, 6})  # один игрок заменён
    roster_b = frozenset({10, 11, 12, 13, 14})
    matches = [
        M(1, base, 100, 200, roster_a1, roster_b, True),
        M(2, base + timedelta(days=1), 100, 200, roster_a1, roster_b, False),
        M(3, base + timedelta(days=2), 100, 200, roster_a2, roster_b, True),  # смена состава radiant
        M(4, base + timedelta(days=3), 100, 200, roster_a2, roster_b, False),
        M(5, base + timedelta(days=10), 100, 200, roster_a2, roster_b, True),
    ]
    return matches


def _snapshot(rows):
    return {
        r.match_id: (
            r.radiant_roster_size, r.dire_roster_size,
            r.radiant_roster_matches_together, r.dire_roster_matches_together,
            r.radiant_roster_age_days, r.dire_roster_age_days,
            r.radiant_player_continuity, r.dire_player_continuity,
            tuple(r.radiant_roster_changes.items()), tuple(r.dire_roster_changes.items()),
        )
        for r in rows
    }


def test_roster_change_detected_correctly():
    rows = build_roster_features(_make_matches())
    # Матч 1: первый матч, нет истории -> matches_together=0, age=None, continuity=None
    assert rows[0].radiant_roster_matches_together == 0
    assert rows[0].radiant_roster_age_days is None
    assert rows[0].radiant_player_continuity is None

    # Матч 2: тот же состав -> matches_together читает состояние ДО матча (=1, после матча 1)
    assert rows[1].radiant_roster_matches_together == 1
    assert rows[1].radiant_player_continuity == 1.0

    # Матч 3: состав сменился (a2 вместо a1) -> continuity < 1.0 (4/5 игроков совпадают)
    assert rows[2].radiant_player_continuity == 4 / 5

    # Матч 4: новый состав (a2) второй раз подряд -> matches_together для НОВОГО периода = 1
    assert rows[3].radiant_roster_matches_together == 1
    assert rows[3].radiant_player_continuity == 1.0


def test_flipping_past_outcome_does_not_change_earlier_rows():
    matches = _make_matches()
    baseline = _snapshot(build_roster_features(matches))

    mutated = list(matches)
    mutated[2] = M(matches[2].match_id, matches[2].start_time, matches[2].radiant_team_id,
                    matches[2].dire_team_id, matches[2].radiant_roster, matches[2].dire_roster,
                    not matches[2].radiant_win)
    mutated_snap = _snapshot(build_roster_features(mutated))

    for m in matches[:2]:
        assert baseline[m.match_id] == mutated_snap[m.match_id], f"УТЕЧКА на match_id={m.match_id}"


def test_appending_future_match_does_not_change_existing_rows():
    matches = _make_matches()
    baseline = _snapshot(build_roster_features(matches))

    future = M(999, matches[-1].start_time + timedelta(days=1), matches[0].radiant_team_id,
               matches[0].dire_team_id, frozenset({1, 2, 3, 4, 6}), frozenset({10, 11, 12, 13, 14}), True)
    with_future_snap = _snapshot(build_roster_features(matches + [future]))

    for m in matches:
        assert baseline[m.match_id] == with_future_snap[m.match_id], f"УТЕЧКА на match_id={m.match_id}"


def test_roster_change_uses_this_matchs_own_roster_not_future():
    """Смена состава в match_id=3 (текущий, roster_a2) не должна влиять на
    matches_together, вычисленный ДЛЯ match_id=3 (это pre-match состояние —
    читается ДО того, как match_id=3 обновит трекер своим новым составом)."""
    rows = build_roster_features(_make_matches())
    row3 = next(r for r in rows if r.match_id == 3)
    # pre-match state для match 3 всё ещё отражает СТАРЫЙ (a1) состав, накопленный за matches 1-2
    assert row3.radiant_roster_matches_together == 2  # 2 предыдущих матча со старым составом
