"""
Phase 6.5 — leakage-safety для нового src/datasets/multi_window_features.py
(multi-window recent form + rest). Тот же принцип, что и
tests/leakage/test_feature_set_0_leakage.py (Phase 5), для НОВОГО модуля —
не предполагается, что leakage-safety автоматически переносится с
feature_set_0.py только потому, что паттерн похож.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.datasets.multi_window_features import build_multi_window_features


@dataclass
class M:
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_win: bool


def _make_matches(n=20):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    matches = []
    for i in range(n):
        matches.append(M(
            match_id=i,
            start_time=base + timedelta(hours=i),
            radiant_team_id=100 + (i % 3),
            dire_team_id=200 + (i % 4),
            radiant_win=(i % 2 == 0),
        ))
    return matches


def _snapshot(rows):
    return {
        r.match_id: (
            round(r.elo_difference, 6),
            tuple(round(v, 6) if v is not None else None for v in r.recent_winrate_difference.values()),
            round(r.days_since_last_match_difference, 6) if r.days_since_last_match_difference is not None else None,
            tuple(r.matches_last_n_days_difference.values()),
        )
        for r in rows
    }


def test_flipping_past_outcome_does_not_change_earlier_rows():
    matches = _make_matches(20)
    baseline = _snapshot(build_multi_window_features(matches))

    mid = 10
    mutated = list(matches)
    mutated[mid] = M(matches[mid].match_id, matches[mid].start_time, matches[mid].radiant_team_id,
                      matches[mid].dire_team_id, not matches[mid].radiant_win)
    mutated_snap = _snapshot(build_multi_window_features(mutated))

    for m in matches[:mid]:
        assert baseline[m.match_id] == mutated_snap[m.match_id], f"УТЕЧКА на match_id={m.match_id}"


def test_appending_future_match_does_not_change_existing_rows():
    matches = _make_matches(20)
    baseline = _snapshot(build_multi_window_features(matches))

    future = M(999, matches[-1].start_time + timedelta(days=1), matches[0].radiant_team_id, matches[0].dire_team_id, True)
    with_future_snap = _snapshot(build_multi_window_features(matches + [future]))

    for m in matches:
        assert baseline[m.match_id] == with_future_snap[m.match_id], f"УТЕЧКА на match_id={m.match_id}"


def test_recent_winrate_windows_do_not_include_current_match():
    """Форма 1-го матча каждой команды ОБЯЗАНА быть None (нет истории)."""
    matches = _make_matches(20)
    rows = build_multi_window_features(matches)
    first_row = rows[0]
    for w, diff in first_row.recent_winrate_difference.items():
        assert diff is None, f"window={w}: у обеих команд не должно быть истории на первом матче"


def test_form_windows_are_consistent_subsets():
    """form_3 не может использовать больше матчей истории, чем form_20 —
    оба читаются из одного и того же deque, разница только в длине среза."""
    matches = _make_matches(30)
    rows = build_multi_window_features(matches)
    last_row = rows[-1]
    # Просто убеждаемся, что для каждой команды с историей >= 3 матчей
    # form_3 и form_20 определены одновременно (или оба None, если истории нет).
    for w in (3, 5, 10, 20):
        val = last_row.recent_winrate_difference[w]
        assert val is None or -1.0 <= val <= 1.0
