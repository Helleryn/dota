"""
PHASE 16 — adversarial tests источников pre-match данных (PART V, 18 сценариев).

Все тесты работают на подставных ответах: сетевой источник для проверки
инвариантов непригоден — он не воспроизводим и его нельзя заставить
вернуть нужный отказ.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.sources.base import (
    Provenance,
    SourceConfidence,
    SourceResult,
    SourceStatus,
)
from src.sources.bo3gg import Bo3ggAdapter
from src.sources.http import HostPolicy, PolitClient
from src.sources.liquipedia import LiquipediaAdapter, parse_roster_wikitext
from src.sources.models import (
    ExternalTeamRef,
    PatchInfo,
    RoleConfidence,
    RosterMembership,
    UpcomingMatch,
)
from src.sources.resolution import resolve_roster
from src.sources.temporal import (
    TemporalRosterStore,
    patch_as_of,
    roster_data_confidence,
)
from src.sources.valve_dpc import ValveDpcAdapter

T0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
TEAM = ExternalTeamRef("test", "1", name="Team A")
OTHER = ExternalTeamRef("test", "2", name="Team B")


def prov(source="test", observed=T0, conf=SourceConfidence.MEDIUM):
    return Provenance(source, observed, conf)


def mem(pid, valid_from=None, valid_to=None, observed=T0,
        conf=SourceConfidence.MEDIUM, ref=TEAM, pos=None,
        role=RoleConfidence.UNKNOWN):
    return RosterMembership(team_ref=ref, player_id=pid, player_name=pid,
                            valid_from=valid_from, valid_to=valid_to,
                            provenance=prov(observed=observed, conf=conf),
                            position=pos, role_confidence=role)


class FakeClient(PolitClient):
    """Клиент с заранее заданными ответами. Наследуется от настоящего,
    чтобы политика интервалов проверялась той же, что в бою."""

    def __init__(self, responses):
        super().__init__("test-agent", policies={}, sleeper=lambda s: None,
                         clock=lambda: 0.0)
        self.responses = responses
        self.calls = []

    def get_json(self, url):
        self.calls.append(url)
        for frag, resp in self.responses.items():
            if frag in url:
                return resp
        return (404, None, {})


# ---------- 1/2/3. будущее не попадает в прошлый прогноз ----------

def test_future_roster_cannot_enter_old_prediction():
    """КЛЮЧЕВОЙ ТЕСТ. Замена, наблюдённая ПОСЛЕ прогноза, не должна
    попадать в состав на момент прогноза — даже если она действует с
    более раннего момента."""
    store = TemporalRosterStore()
    store.add(mem("old", valid_from=T0 - timedelta(days=30), valid_to=T0 + timedelta(hours=1)))
    # замена действует с T0+1ч, но НАБЛЮДЕНА только в T0+2ч
    store.add(mem("new", valid_from=T0 + timedelta(hours=1),
                  observed=T0 + timedelta(hours=2)))

    at_prediction = store.roster_as_of(TEAM, T0)
    assert [m.player_id for m in at_prediction] == ["old"]

    later = store.roster_as_of(TEAM, T0 + timedelta(hours=3))
    assert [m.player_id for m in later] == ["new"]


def test_roster_announced_after_prediction_is_invisible():
    """Отдельно от предыдущего: факт действует ЗАДНИМ ЧИСЛОМ, но
    объявлен позже. Прогноз, сделанный до объявления, знать о нём не может."""
    store = TemporalRosterStore()
    store.add(mem("retro", valid_from=T0 - timedelta(days=10),
                  observed=T0 + timedelta(days=1)))
    assert store.roster_as_of(TEAM, T0) == []
    assert len(store.roster_as_of(TEAM, T0 + timedelta(days=2))) == 1


def test_future_patch_cannot_enter_prediction():
    p_now = PatchInfo("7.40", T0 - timedelta(days=40), prov(observed=T0 - timedelta(days=40)))
    p_future = PatchInfo("7.41", T0 + timedelta(days=5), prov(observed=T0 - timedelta(days=1)))
    got = patch_as_of([p_now, p_future], T0)
    assert got is not None and got.name == "7.40", "будущий патч попал в прогноз"


def test_patch_known_only_later_is_not_used():
    """Патч вышел до T, но наши данные о нём получены позже."""
    p = PatchInfo("7.41", T0 - timedelta(days=1), prov(observed=T0 + timedelta(days=1)))
    assert patch_as_of([p], T0) is None


def test_historical_query_can_opt_out_of_known_at():
    """Для исторического анализа вопрос иной: «кто фактически числился».
    Отключение должно быть ЯВНЫМ, а не поведением по умолчанию."""
    store = TemporalRosterStore()
    store.add(mem("x", valid_from=T0 - timedelta(days=5), observed=T0 + timedelta(days=5)))
    assert store.roster_as_of(TEAM, T0) == []
    assert len(store.roster_as_of(TEAM, T0, require_known=False)) == 1


# ---------- 4/5/6. идентичность команд ----------

def test_team_key_is_not_the_name():
    """Phase 10: 461 пара team_id с одинаковым именем существовала
    параллельно. Одноимённые команды разных источников не должны
    сливаться."""
    store = TemporalRosterStore()
    a = ExternalTeamRef("valve_dpc", "111", name="Spirit")
    b = ExternalTeamRef("bo3gg", "222", name="Spirit")
    store.add(mem("p1", ref=a, valid_from=T0 - timedelta(days=1)))
    store.add(mem("p2", ref=b, valid_from=T0 - timedelta(days=1)))
    assert len(store.roster_as_of(a, T0)) == 1
    assert len(store.roster_as_of(b, T0)) == 1
    assert store.roster_as_of(a, T0)[0].player_id == "p1"


def test_same_name_within_one_source_still_separate_ids():
    store = TemporalRosterStore()
    a = ExternalTeamRef("valve_dpc", "111", name="Nemiga")
    b = ExternalTeamRef("valve_dpc", "999", name="Nemiga")
    store.add(mem("p1", ref=a, valid_from=T0 - timedelta(days=1)))
    store.add(mem("p2", ref=b, valid_from=T0 - timedelta(days=1)))
    assert len(store.teams()) == 2


def test_valve_team_id_is_not_auto_assigned_to_other_sources():
    r = ExternalTeamRef("bo3gg", "8221", name="Team A")
    assert r.valve_team_id is None, "соответствие идентичностей не выводится автоматически"


# ---------- 7/8. расписание: переносы и отмены ----------

def test_postponed_match_is_a_new_fact_not_an_edit():
    m1 = UpcomingMatch("bo3gg", "42", T0 + timedelta(hours=2), TEAM, OTHER, prov())
    m2 = UpcomingMatch("bo3gg", "42", T0 + timedelta(hours=8), TEAM, OTHER,
                       prov(observed=T0 + timedelta(hours=1)))
    assert m1.match_key == m2.match_key
    assert m1.scheduled_start != m2.scheduled_start
    assert m1.provenance.observed_at < m2.provenance.observed_at


def test_cancelled_match_keeps_explicit_status():
    m = UpcomingMatch("bo3gg", "42", T0 + timedelta(hours=2), TEAM, OTHER, prov(),
                      status="CANCELLED")
    assert m.status == "CANCELLED"


def test_started_match_is_excluded_by_valve_adapter():
    node = {"node_id": 1, "scheduled_time": int((T0 - timedelta(hours=1)).timestamp()),
            "team_id_1": 1, "team_id_2": 2, "has_started": True}
    body = {"node_groups": [{"nodes": [node]}]}
    a = ValveDpcAdapter(FakeClient({"GetLeagueData": (200, body, {})}))
    r = a.upcoming_for_league(1, now=T0)
    assert r.status == SourceStatus.EMPTY


def test_valve_adapter_skips_nodes_without_teams_or_time():
    body = {"node_groups": [{"nodes": [
        {"node_id": 1, "scheduled_time": 0, "team_id_1": 1, "team_id_2": 2},
        {"node_id": 2, "scheduled_time": int((T0 + timedelta(hours=3)).timestamp()),
         "team_id_1": None, "team_id_2": 2},
        {"node_id": 3, "scheduled_time": int((T0 + timedelta(hours=3)).timestamp()),
         "team_id_1": 7, "team_id_2": 8},
    ]}]}
    a = ValveDpcAdapter(FakeClient({"GetLeagueData": (200, body, {})}))
    r = a.upcoming_for_league(5, now=T0)
    assert len(r.items) == 1 and r.items[0].external_id == "5:3"


def test_duplicate_schedule_entries_share_a_key():
    m1 = UpcomingMatch("bo3gg", "42", T0 + timedelta(hours=2), TEAM, OTHER, prov())
    m2 = UpcomingMatch("bo3gg", "42", T0 + timedelta(hours=2), TEAM, OTHER, prov())
    assert m1.match_key == m2.match_key
    assert len({m1.match_key, m2.match_key}) == 1


# ---------- 9. расхождение источников ----------

def test_source_disagreement_is_not_resolved_silently():
    a = [mem(f"a{i}", valid_from=T0 - timedelta(days=1)) for i in range(5)]
    b = [mem(f"b{i}", valid_from=T0 - timedelta(days=1)) for i in range(5)]
    r = resolve_roster({"src_a": a, "src_b": b})
    assert r.is_unknown, "конфликт разрешён молча"
    assert r.conflict and "расходятся" in r.conflict


def test_official_source_wins_over_community():
    off = [mem(f"o{i}", valid_from=T0 - timedelta(days=1), conf=SourceConfidence.HIGH)
           for i in range(5)]
    com = [mem(f"c{i}", valid_from=T0 - timedelta(days=1), conf=SourceConfidence.MEDIUM)
           for i in range(5)]
    r = resolve_roster({"valve": off, "community": com})
    assert not r.is_unknown
    assert [m.player_id for m in r.resolved] == [f"o{i}" for i in range(5)]


def test_corroboration_beats_a_single_dissenter():
    same = [mem(f"s{i}", valid_from=T0 - timedelta(days=1)) for i in range(5)]
    other = [mem(f"x{i}", valid_from=T0 - timedelta(days=1)) for i in range(5)]
    r = resolve_roster({"a": same, "b": list(same), "c": other})
    assert not r.is_unknown and "Подтверждено" in r.rule.capitalize()


def test_agreement_needs_no_rule():
    same = [mem(f"s{i}", valid_from=T0 - timedelta(days=1)) for i in range(5)]
    r = resolve_roster({"a": same, "b": list(same)})
    assert not r.is_unknown and "соглас" in r.rule


# ---------- 10/11/12. отсутствующий и устаревший состав ----------

def test_missing_roster_is_unknown_not_empty_success():
    r = resolve_roster({"a": [], "b": []})
    assert r.is_unknown and "ни один источник" in (r.conflict or "")


def test_stale_roster_is_excluded_by_valid_to():
    store = TemporalRosterStore()
    store.add(mem("left", valid_from=T0 - timedelta(days=100),
                  valid_to=T0 - timedelta(days=1)))
    assert store.roster_as_of(TEAM, T0) == []


def test_data_confidence_is_the_weakest_fact_not_the_average():
    ms = [mem(f"p{i}", valid_from=T0 - timedelta(days=1), conf=SourceConfidence.HIGH)
          for i in range(4)]
    ms.append(mem("p4", valid_from=T0 - timedelta(days=1), conf=SourceConfidence.LOW))
    assert roster_data_confidence(ms) == SourceConfidence.LOW


# ---------- 13. роли ----------

def test_unknown_role_is_not_silently_upgraded():
    m = mem("p", valid_from=T0 - timedelta(days=1))
    assert m.role_confidence == RoleConfidence.UNKNOWN
    assert m.position is None


def test_liquipedia_role_is_predicted_not_confirmed():
    """Позиция из вики — исторически заявленная роль, а не подтверждение
    на конкретный матч. Повышать её до CONFIRMED нельзя (Phase 11)."""
    wt = "{{Person|flag=ru|id=Larl|name=Denis|position=2|joindate=2022-12-08}}"
    a = LiquipediaAdapter(FakeClient({"action=parse": (200,
        {"parse": {"wikitext": wt}}, {})}))
    r = a.roster("Team_X", now=T0)
    assert r.ok and len(r.items) == 1
    assert r.items[0].position == 2
    assert r.items[0].role_confidence == RoleConfidence.PREDICTED


def test_wikitext_parser_extracts_dates_and_positions():
    wt = ("{{Person|id=A|position=1|joindate=2025-01-08}}"
          "{{Person|id=B|position=5|joindate=2024-10-18|leavedate=2026-02-01}}"
          "{{Person|id=C}}")
    got = parse_roster_wikitext(wt)
    assert [g["id"] for g in got] == ["A", "B", "C"]
    assert got[1]["leavedate"] == "2026-02-01"
    assert got[2]["position"] is None


# ---------- 14/15/16. отметки времени, таймауты, лимиты ----------

def test_missing_source_timestamp_cannot_be_constructed():
    with pytest.raises(TypeError):
        Provenance("src", confidence=SourceConfidence.HIGH)   # нет observed_at


def test_source_timeout_is_unavailable_not_empty():
    a = ValveDpcAdapter(FakeClient({"GetLeagueData": (0, None, {})}))
    r = a.upcoming_for_league(1, now=T0)
    assert r.status == SourceStatus.UNAVAILABLE
    assert r.status != SourceStatus.EMPTY, "молчание источника не равно отсутствию матчей"


def test_rate_limit_is_reported_not_bypassed():
    a = LiquipediaAdapter(FakeClient({"action=parse": (429, None, {"retry-after": "60"})}))
    r = a.roster("Team_X", now=T0)
    assert r.status == SourceStatus.RATE_LIMITED
    assert r.retry_after_seconds == 60.0


def test_rate_limit_increases_interval_never_decreases_it():
    """Проверка отсутствия обхода лимита: после 429 интервал для хоста
    обязан вырасти."""
    c = PolitClient("ua", policies={"example.com": HostPolicy(min_interval=1.0)},
                    sleeper=lambda s: None, clock=lambda: 0.0)
    before = c.policies["example.com"].min_interval
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.HTTPError("http://example.com/x", 429, "Too Many Requests",
                                     {"Retry-After": "120"}, None)
    import urllib.request
    orig, urllib.request.urlopen = urllib.request.urlopen, boom
    try:
        st, _, _ = c.get_json("http://example.com/x")
    finally:
        urllib.request.urlopen = orig
    assert st == 429
    assert c.policies["example.com"].min_interval >= max(before, 120.0)


def test_liquipedia_requires_gzip_is_surfaced():
    a = LiquipediaAdapter(FakeClient({"action=parse": (406, None, {})}))
    r = a.roster("Team_X", now=T0)
    assert r.status == SourceStatus.UNAVAILABLE and "gzip" in (r.error or "")


def test_cache_prevents_repeated_requests():
    fc = FakeClient({"action=parse": (200, {"parse": {"wikitext": "{{Person|id=A}}"}}, {})})
    a = LiquipediaAdapter(fc)
    a.roster("Team_X", now=T0)
    a.roster("Team_X", now=T0)
    assert len(fc.calls) == 1, "кэш обязателен: источник упирается в 429 со второго запроса"


# ---------- 17/18. bo3.gg ----------

def test_bo3gg_missing_lineup_is_empty_not_error():
    body = {"results": [{"id": 1, "start_date": (T0 + timedelta(hours=5)).isoformat(),
                         "team1_id": 10, "team2_id": 20, "players": []}]}
    a = Bo3ggAdapter(FakeClient({"/matches": (200, body, {})}))
    up = a.upcoming(now=T0)
    assert up.ok and len(up.items) == 1
    r = a.roster_for_match(up.items[0], now=T0)
    assert r.status == SourceStatus.EMPTY


def test_bo3gg_changed_start_time_yields_new_observation():
    def body(hours):
        return {"results": [{"id": 7, "start_date": (T0 + timedelta(hours=hours)).isoformat(),
                             "team1_id": 10, "team2_id": 20}]}
    a1 = Bo3ggAdapter(FakeClient({"/matches": (200, body(3), {})}))
    a2 = Bo3ggAdapter(FakeClient({"/matches": (200, body(9), {})}))
    m1 = a1.upcoming(now=T0).items[0]
    m2 = a2.upcoming(now=T0 + timedelta(hours=1)).items[0]
    assert m1.match_key == m2.match_key
    assert m2.scheduled_start > m1.scheduled_start
    assert m2.provenance.observed_at > m1.provenance.observed_at


def test_bo3gg_skips_matches_already_started():
    body = {"results": [{"id": 1, "start_date": (T0 - timedelta(hours=1)).isoformat(),
                         "team1_id": 10, "team2_id": 20}]}
    a = Bo3ggAdapter(FakeClient({"/matches": (200, body, {})}))
    assert a.upcoming(now=T0).status == SourceStatus.EMPTY
