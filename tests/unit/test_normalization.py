"""test_match_normalization, test_team_normalization (Phase 5, раздел 20)."""

from datetime import datetime, timezone

from src.datasources.base import RawMatch
from src.normalization.normalize import normalize_match, normalize_patch, normalize_player, normalize_team


def _raw_match(**overrides) -> RawMatch:
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
        series_id=1,
        series_type=1,
        patch=None,
        source="opendota",
        fetched_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return RawMatch(**defaults)


def test_match_normalization_resolves_known_patch():
    nm = normalize_match(_raw_match())
    assert nm.match_id == 1
    assert nm.patch_id is not None
    assert nm.radiant_team_id == 100
    assert nm.dire_team_id == 200


def test_match_normalization_unknown_patch_stays_none_not_fake():
    """Phase 5, раздел 9: если патч неизвестен — NULL, не выдуманное значение."""
    ancient = _raw_match(match_id=2, start_time=datetime(2005, 1, 1, tzinfo=timezone.utc))
    nm = normalize_match(ancient)
    assert nm.patch_id is None


def test_match_normalization_adds_tzinfo_to_naive_datetime():
    naive_raw = _raw_match(match_id=3, start_time=datetime(2024, 6, 1))  # без tzinfo
    nm = normalize_match(naive_raw)
    assert nm.start_time.tzinfo is not None


def test_team_normalization_trims_whitespace():
    t = normalize_team(100, "  Team Alpha  ", " ALP ")
    assert t.name == "Team Alpha"
    assert t.tag == "ALP"


def test_team_normalization_none_team_id_returns_none():
    assert normalize_team(None, "irrelevant") is None


def test_team_normalization_empty_string_becomes_none():
    t = normalize_team(100, "   ", "")
    assert t.name is None
    assert t.tag is None


def test_player_normalization_anonymous_account_is_valid_none():
    assert normalize_player(None) is None


def test_player_normalization_preserves_account_id():
    p = normalize_player(111111, "SomePlayer")
    assert p.account_id == 111111
    assert p.name == "SomePlayer"


def test_normalize_patch_matches_normalize_match_patch_id():
    ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
    assert normalize_patch(ts) == normalize_match(_raw_match(start_time=ts)).patch_id
