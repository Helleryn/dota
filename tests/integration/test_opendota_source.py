"""
Интеграционные тесты OpenDotaSource ПРОТИВ FIXTURES (tests/fixtures/opendota/),
не против живого API — сеть к api.opendota.com недоступна в этой среде
(docs/environment-constraints.md). httpx.MockTransport подменяет транспорт,
сам код HTTP-клиента (retry/rate-limit/error handling) отрабатывает по-настоящему.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from src.datasources.base import RawResponseRecord
from src.datasources.http_client import HttpClientConfig, RateLimitedHttpClient
from src.datasources.opendota import OpenDotaSource
from src.datasources.opendota_fixtures import build_fixture_opendota_source

FIXTURES = Path(__file__).parent.parent / "fixtures" / "opendota"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


def make_source(raw_sink=None) -> OpenDotaSource:
    """Обёртка над общим fixture-билдером (src/datasources/opendota_fixtures.py)
    — переиспользуется и CLI ingestion pipeline (offline mode), не дублируется."""
    return build_fixture_opendota_source(FIXTURES, on_raw_response=raw_sink)


def test_fetch_matches_filters_by_tier_and_time_range():
    """
    Ключевая проверка: только premium/professional лиги (ADR-001), только
    диапазон [since, until), amateur-матчи (league 5001) должны быть
    ИСКЛЮЧЕНЫ (league_tier не в PRO_LEAGUE_TIERS — фильтрация происходит на
    уровне normalization/validation, а не здесь; сам адаптер лишь
    ПРОСТАВЛЯЕТ league_tier честно, включая 'amateur').
    """
    source = make_source()
    since = datetime(2024, 1, 1, tzinfo=timezone.utc)
    until = datetime(2025, 1, 1, tzinfo=timezone.utc)

    matches = list(source.fetch_matches(since, until))

    assert len(matches) == 14, f"ожидалось 14 матчей (12 premium + 2 amateur) в диапазоне, получено {len(matches)}"
    tiers_seen = {m.league_tier for m in matches}
    assert tiers_seen == {"premium", "amateur"}, "адаптер обязан честно проставлять tier, включая amateur"

    match_ids = {m.match_id for m in matches}
    assert match_ids == set(range(7001, 7015))


def test_fetch_matches_respects_since_boundary():
    """Диапазон since/until должен реально ограничивать выдачу, не просто игнорироваться."""
    source = make_source()
    # Матчи 7008-7014 находятся на первой странице; отсекаем всё до 7010.
    since = datetime.fromtimestamp(1704067200 + 9 * 86400 * 3, tz=timezone.utc)  # match_id=7010 start_time
    until = datetime(2025, 1, 1, tzinfo=timezone.utc)

    matches = list(source.fetch_matches(since, until))
    match_ids = {m.match_id for m in matches}
    assert min(match_ids) >= 7010, f"since не соблюдён: {sorted(match_ids)}"


def test_fetch_matches_team_mapping_is_radiant_dire_not_winner_loser():
    """
    Phase 5, раздел 10 — критический тест: radiant_team_id/dire_team_id
    определяются игровой ролью, НЕ исходом. Проверяем на fixture, где
    известны и radiant_win, и radiant_team_id, что порядок полей не
    "подстроен" под победителя.
    """
    source = make_source()
    matches = {m.match_id: m for m in source.fetch_matches(
        datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2025, 1, 1, tzinfo=timezone.utc)
    )}

    m7001 = matches[7001]
    # radiant_team_id/dire_team_id закодированы игровой ролью независимо от
    # того, кто победил (сверяем со значением из самого fixture, не
    # хардкодим ожидаемый исход — цель теста в стабильности МАППИНГА ролей,
    # не в конкретном результате конкретного матча).
    assert m7001.radiant_team_id == 100
    assert m7001.dire_team_id == 200
    assert m7001.radiant_win is False  # см. tests/fixtures/opendota/pro_matches_page2.json

    # Найдём матч, где radiant ПРОИГРАЛ (i=1 в генераторе, amateur-матчи radiant_win=False)
    losing_radiant_matches = [m for m in matches.values() if m.radiant_win is False]
    assert losing_radiant_matches, "в fixture должен быть хотя бы один матч, где radiant проиграл"
    m = losing_radiant_matches[0]
    # radiant_team_id всё равно указывает на игровую роль, не на "проигравшего постфактум"
    assert m.radiant_team_id is not None and m.dire_team_id is not None
    assert m.radiant_team_id != m.dire_team_id


def test_fetch_picks_bans_real_structure():
    source = make_source()
    picks_bans = list(source.fetch_picks_bans(7001))
    assert len(picks_bans) == 6
    assert picks_bans[0].is_pick is False
    assert picks_bans[0].hero_id == 99
    assert picks_bans[0].team == 1
    assert picks_bans[0].order == 0
    # Порядок сохранён как в источнике
    assert [pb.order for pb in picks_bans] == list(range(6))


def test_fetch_picks_bans_missing_is_not_an_error():
    """Матч без записанного драфта (picks_bans=None) — валидный кейс, не исключение."""
    source = make_source()
    picks_bans = list(source.fetch_picks_bans(7002))
    assert picks_bans == []


def test_fetch_player_matches_radiant_dire_split_and_anonymous_account():
    source = make_source()
    players = list(source.fetch_player_matches(7001))
    assert len(players) == 10

    radiant_players = [p for p in players if p.is_radiant]
    dire_players = [p for p in players if not p.is_radiant]
    assert len(radiant_players) == 5
    assert len(dire_players) == 5
    assert all(p.team_id == 100 for p in radiant_players)
    assert all(p.team_id == 200 for p in dire_players)

    anonymous = [p for p in players if p.account_id is None]
    assert len(anonymous) == 1, "fixture намеренно содержит одного анонимного игрока (player_slot=4)"


def test_match_detail_is_fetched_once_and_cached_for_both_methods():
    """
    fetch_picks_bans и fetch_player_matches для ОДНОГО match_id должны
    переиспользовать один HTTP-ответ (docs/data-pipeline.md — экономия квоты),
    не делать два отдельных запроса.
    """
    call_count = {"matches_7001": 0}
    page1 = _load("pro_matches_page1.json")
    leagues = _load("leagues.json")
    detail = _load("match_detail_7001.json")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/leagues":
            return httpx.Response(200, json=leagues)
        if request.url.path == "/proMatches":
            return httpx.Response(200, json=page1)
        if request.url.path == "/matches/7001":
            call_count["matches_7001"] += 1
            return httpx.Response(200, json=detail)
        return httpx.Response(404)

    config = HttpClientConfig(base_url="https://fake.test", max_retries=1,
                               backoff_base_seconds=0.01, requests_per_minute=6000)
    client = RateLimitedHttpClient(config, transport=httpx.MockTransport(handler))
    source = OpenDotaSource(client=client)

    list(source.fetch_picks_bans(7001))
    list(source.fetch_player_matches(7001))

    assert call_count["matches_7001"] == 1, "детальный ответ должен быть закэширован между вызовами"


def test_raw_response_callback_invoked_without_api_key_leak():
    """Phase 5, раздел 6-7: каждый успешный HTTP-ответ должен попадать в raw-слой
    через callback, БЕЗ api_key в сохранённых request_params (это секрет)."""
    records: list[RawResponseRecord] = []
    source = make_source(raw_sink=records.append)
    source._api_key = "SECRET_KEY_MUST_NOT_LEAK"  # noqa: SLF001 — тестируем именно эту защиту

    list(source.fetch_picks_bans(7001))

    assert records, "callback должен был сработать хотя бы раз"
    for r in records:
        assert "api_key" not in r.request_params, "api_key не должен попадать в raw_responses"
        assert r.source == "opendota"
        assert r.content_hash and len(r.content_hash) == 64  # sha256 hex


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
