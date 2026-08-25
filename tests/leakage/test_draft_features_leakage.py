"""
Phase 7 — leakage-safety для src/datasets/draft_features.py.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from src.datasets.draft_features import build_draft_features


@dataclass(frozen=True)
class M:
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    patch_id: Optional[int]
    radiant_picks: Tuple[int, ...]
    dire_picks: Tuple[int, ...]
    radiant_win: bool


def _make_matches(n=15):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    matches = []
    for i in range(n):
        matches.append(M(
            match_id=i,
            start_time=base + timedelta(hours=i),
            radiant_team_id=100 + (i % 3),
            dire_team_id=200 + (i % 4),
            patch_id=54,
            radiant_picks=(1, 2, 3, 4, 5),
            dire_picks=(6, 7, 8, 9, 10),
            radiant_win=(i % 2 == 0),
        ))
    return matches


def _snapshot(rows):
    return {
        r.match_id: (
            r.radiant_hero_strength, r.dire_hero_strength,
            r.radiant_hero_strength_patch, r.dire_hero_strength_patch,
            r.radiant_team_hero_experience, r.dire_team_hero_experience,
            r.radiant_team_hero_winrate, r.dire_team_hero_winrate,
            r.radiant_pick_popularity, r.dire_pick_popularity,
            r.matchup_advantage,
        )
        for r in rows
    }


def test_first_draft_has_no_history():
    rows = build_draft_features(_make_matches(5))
    assert rows[0].radiant_hero_strength is None
    assert rows[0].radiant_team_hero_winrate is None
    assert rows[0].matchup_advantage is None
    assert rows[0].radiant_team_hero_experience == 0.0  # 0 матчей — валидное значение, не None


def test_flipping_past_outcome_does_not_change_earlier_rows():
    matches = _make_matches(15)
    baseline = _snapshot(build_draft_features(matches))

    mid = 8
    mutated = list(matches)
    m = matches[mid]
    mutated[mid] = M(m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id, m.patch_id,
                      m.radiant_picks, m.dire_picks, not m.radiant_win)
    mutated_snap = _snapshot(build_draft_features(mutated))

    for m in matches[:mid]:
        assert baseline[m.match_id] == mutated_snap[m.match_id], f"УТЕЧКА на match_id={m.match_id}"


def test_appending_future_match_does_not_change_existing_rows():
    matches = _make_matches(15)
    baseline = _snapshot(build_draft_features(matches))

    future = M(999, matches[-1].start_time + timedelta(days=1), matches[0].radiant_team_id,
               matches[0].dire_team_id, 54, (11, 12, 13, 14, 15), (16, 17, 18, 19, 20), True)
    with_future_snap = _snapshot(build_draft_features(matches + [future]))

    for m in matches:
        assert baseline[m.match_id] == with_future_snap[m.match_id], f"УТЕЧКА на match_id={m.match_id}"


def test_hero_winrate_excludes_current_match_result():
    """hero_winrate для матча i не должен учитывать результат САМОГО
    матча i, даже если тот же герой встречается в этом же матче."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    matches = [
        M(1, base, 100, 200, 54, (1, 2, 3, 4, 5), (6, 7, 8, 9, 10), True),   # herod 1 wins
        M(2, base + timedelta(hours=1), 100, 200, 54, (1, 2, 3, 4, 5), (6, 7, 8, 9, 10), True),  # 2й матч с героем 1
    ]
    rows = build_draft_features(matches)
    # Признаки матча 2 читают состояние ПОСЛЕ матча 1 (герой 1: 1 игра, 1 победа -> winrate=1.0)
    assert rows[1].radiant_hero_strength == 1.0
    # Признаки матча 1 не имеют истории вообще (герой 1 впервые встречается)
    assert rows[0].radiant_hero_strength is None
