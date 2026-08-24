"""
test_duplicate_match, test_invalid_winner, test_invalid_duration,
test_same_teams (Phase 5, раздел 20).
"""

from datetime import datetime, timezone

from src.normalization.normalize import NormalizedMatch
from src.normalization.validate import (
    find_duplicate_match_ids,
    has_blocking_errors,
    is_pro_match,
    validate_match,
)


def _match(**overrides) -> NormalizedMatch:
    defaults = dict(
        match_id=1,
        start_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        duration_seconds=1800,
        radiant_team_id=100,
        dire_team_id=200,
        radiant_team_name="Team Alpha",
        dire_team_name="Team Beta",
        radiant_win=True,
        league_id=5000,
        league_tier="premium",
        patch_id=55,
        series_id=1,
        series_type=1,
        source="opendota",
        ingested_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return NormalizedMatch(**defaults)


def test_valid_match_has_no_blocking_errors():
    issues = validate_match(_match())
    assert not has_blocking_errors(issues)


def test_invalid_winner_none_is_blocking_error():
    issues = validate_match(_match(radiant_win=None))
    assert has_blocking_errors(issues)
    assert any(i.field == "radiant_win" for i in issues)


def test_same_teams_is_blocking_error():
    issues = validate_match(_match(radiant_team_id=100, dire_team_id=100))
    assert has_blocking_errors(issues)
    assert any(i.field == "team_ids" and i.severity == "error" for i in issues)


def test_invalid_duration_zero_is_blocking_error():
    issues = validate_match(_match(duration_seconds=0))
    assert has_blocking_errors(issues)


def test_invalid_duration_negative_is_blocking_error():
    issues = validate_match(_match(duration_seconds=-100))
    assert has_blocking_errors(issues)


def test_impossible_future_start_time_is_blocking_error():
    far_future = datetime(2099, 1, 1, tzinfo=timezone.utc)
    issues = validate_match(_match(start_time=far_future), now=datetime(2026, 8, 24, tzinfo=timezone.utc))
    assert has_blocking_errors(issues)
    assert any(i.field == "start_time" for i in issues)


def test_missing_team_id_is_warning_not_error():
    issues = validate_match(_match(dire_team_id=None))
    assert not has_blocking_errors(issues), "отсутствующий team_id — warning, не error (валидный edge case)"
    assert any(i.field == "team_ids" and i.severity == "warning" for i in issues)


def test_amateur_tier_is_warning_not_error():
    issues = validate_match(_match(league_tier="amateur"))
    assert not has_blocking_errors(issues)
    assert any(i.field == "league_tier" and i.severity == "warning" for i in issues)
    assert not is_pro_match(_match(league_tier="amateur"))
    assert is_pro_match(_match(league_tier="premium"))


def test_missing_patch_is_warning_not_error():
    issues = validate_match(_match(patch_id=None))
    assert not has_blocking_errors(issues)
    assert any(i.field == "patch_id" for i in issues)


def test_duplicate_match_ids_detected_within_batch():
    matches = [_match(match_id=1), _match(match_id=2), _match(match_id=1), _match(match_id=3), _match(match_id=2)]
    dups = find_duplicate_match_ids(matches)
    assert sorted(dups) == [1, 2]


def test_no_duplicates_in_clean_batch():
    matches = [_match(match_id=i) for i in range(1, 6)]
    assert find_duplicate_match_ids(matches) == []
