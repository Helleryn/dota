"""
PHASE 10 — тесты identity resolution.

Две группы:
  1. GOLD SET — вручную сконструированные кейсы (PART F): очевидные merge,
     очевидные non-merge, сложные случаи. Проверяют КОРРЕКТНОСТЬ правил.
  2. LEAKAGE — adversarial (PART P): будущие ростеры/имена/переходы не
     должны влиять на идентичность, установленную в прошлом.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import FrozenSet

import pytest

from src.identity.resolver import jaccard, normalize_name, resolve_identities

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class M:
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_roster: FrozenSet[int]
    dire_roster: FrozenSet[int]


FIVE_A = frozenset({1, 2, 3, 4, 5})
FIVE_B = frozenset({10, 11, 12, 13, 14})
OPP = frozenset({90, 91, 92, 93, 94})


def _series(team_id, roster, start_day, n, opp_id=999, opp=OPP):
    return [M(1000 * team_id + i, BASE + timedelta(days=start_day + i),
              team_id, opp_id, roster, opp) for i in range(n)]


# ============================ GOLD SET (PART F) ============================

def test_gold_obvious_merge_rebrand_same_roster():
    """ОЧЕВИДНЫЙ MERGE: команда закончила, через 10 дней тот же состав
    появился под новым team_id."""
    ms = _series(100, FIVE_A, 0, 5) + _series(200, FIVE_A, 15, 5)
    resolver, res = resolve_identities(sorted(ms, key=lambda m: (m.start_time, m.match_id)))
    assert resolver.canonical(200) == resolver.canonical(100), "очевидный ребрендинг не распознан"
    assert len(res.links) == 1
    assert res.links[0].roster_overlap == 1.0


def test_gold_obvious_non_merge_parallel_same_name():
    """ОЧЕВИДНЫЙ NON-MERGE: два team_id с ОДИНАКОВЫМ именем и одинаковым
    составом, но играющие ОДНОВРЕМЕННО. Аудит нашёл 461 такую пару в
    реальных данных — слияние по имени было бы ошибкой во всех."""
    ms = _series(100, FIVE_A, 0, 10) + _series(200, FIVE_A, 3, 10)
    names = {100: "Geek Fam", 200: "Geek Fam"}
    resolver, res = resolve_identities(sorted(ms, key=lambda m: (m.start_time, m.match_id)), names=names)
    assert resolver.canonical(200) != resolver.canonical(100), \
        "ПАРАЛЛЕЛЬНЫЕ команды слиты — критическая ошибка"
    assert len(res.links) == 0


def test_gold_obvious_non_merge_name_only():
    """NON-MERGE: одинаковое имя, но составы не пересекаются вовсе."""
    ms = _series(100, FIVE_A, 0, 5) + _series(200, FIVE_B, 20, 5)
    names = {100: "Dominion", 200: "Dominion"}
    resolver, res = resolve_identities(sorted(ms, key=lambda m: (m.start_time, m.match_id)), names=names)
    assert resolver.canonical(200) != resolver.canonical(100), "слияние только по имени запрещено"


def test_gold_non_merge_too_long_gap():
    """NON-MERGE: тот же состав, но через 2 года — это не преемственность,
    а возрождение/случайная пересборка."""
    ms = _series(100, FIVE_A, 0, 5) + _series(200, FIVE_A, 730, 5)
    resolver, _ = resolve_identities(sorted(ms, key=lambda m: (m.start_time, m.match_id)))
    assert resolver.canonical(200) != resolver.canonical(100)


def test_gold_merge_without_name_match_is_allowed():
    """СЛОЖНЫЙ КЕЙС: ребрендинг со сменой имени. Ростер — доказательство,
    имя — нет. Должно слиться с MEDIUM."""
    ms = _series(100, FIVE_A, 0, 5) + _series(200, FIVE_A, 12, 5)
    names = {100: "Kylin Esports Club", 200: "LBZS"}
    resolver, res = resolve_identities(sorted(ms, key=lambda m: (m.start_time, m.match_id)), names=names)
    assert resolver.canonical(200) == resolver.canonical(100)
    assert res.links[0].confidence == "MEDIUM"
    assert res.links[0].name_matched is False


def test_gold_name_match_raises_confidence_to_high():
    ms = _series(100, FIVE_A, 0, 5) + _series(200, FIVE_A, 12, 5)
    names = {100: "The MongolZ", 200: "The Mongolz"}   # регистр -> нормализация
    _, res = resolve_identities(sorted(ms, key=lambda m: (m.start_time, m.match_id)), names=names)
    assert res.links[0].confidence == "HIGH"
    assert res.links[0].name_matched is True


def test_gold_partial_overlap_below_threshold_not_merged():
    """2 из 5 общих игроков — обычная ротация рынка, не преемственность."""
    partial = frozenset({1, 2, 50, 51, 52})
    ms = _series(100, FIVE_A, 0, 5) + _series(200, partial, 12, 5)
    resolver, _ = resolve_identities(sorted(ms, key=lambda m: (m.start_time, m.match_id)))
    assert resolver.canonical(200) != resolver.canonical(100)


def test_gold_chain_merges_collapse():
    """Цепочка A -> B -> C должна схлопнуться в ОДИН canonical id."""
    ms = _series(100, FIVE_A, 0, 5) + _series(200, FIVE_A, 12, 5) + _series(300, FIVE_A, 24, 5)
    resolver, _ = resolve_identities(sorted(ms, key=lambda m: (m.start_time, m.match_id)))
    assert resolver.canonical(300) == resolver.canonical(200) == resolver.canonical(100)


def test_source_id_never_lost():
    """ПРАВИЛО: исходный team_id всегда восстановим; несвязанный id
    отображается сам в себя."""
    ms = _series(100, FIVE_A, 0, 5)
    resolver, _ = resolve_identities(ms)
    assert resolver.canonical(100) == 100
    assert resolver.canonical(555555) == 555555   # неизвестный id


# ============================ LEAKAGE (PART P) ============================

def test_future_matches_do_not_change_past_identity():
    """Ключевой тест: добавление будущих матчей не меняет ни одну связь,
    установленную ранее."""
    ms = sorted(_series(100, FIVE_A, 0, 5) + _series(200, FIVE_A, 12, 5),
                key=lambda m: (m.start_time, m.match_id))
    _, res_before = resolve_identities(ms)
    snapshot = [(l.source_team_id, l.canonical_team_id, l.valid_from, l.confidence) for l in res_before.links]

    future = _series(300, FIVE_B, 400, 5)
    _, res_after = resolve_identities(sorted(ms + future, key=lambda m: (m.start_time, m.match_id)))
    after = [(l.source_team_id, l.canonical_team_id, l.valid_from, l.confidence)
             for l in res_after.links if l.source_team_id in (100, 200)]
    assert snapshot == after, "будущие матчи изменили прошлую идентичность"


def test_future_rebrand_does_not_retroactively_merge():
    """Если A ребрендится в B в 2025, идентичность B активируется ТОЛЬКО
    с момента первого матча B — прошлые матчи A не переписываются."""
    ms = sorted(_series(100, FIVE_A, 0, 5) + _series(200, FIVE_A, 12, 5),
                key=lambda m: (m.start_time, m.match_id))
    _, res = resolve_identities(ms)
    link = res.links[0]
    first_b_match = min(m.start_time for m in ms if m.radiant_team_id == 200)
    assert link.valid_from == first_b_match, "valid_from не совпал с моментом появления доказательства"
    # ни одна связь не может быть активна раньше первого матча B
    assert all(l.valid_from >= first_b_match for l in res.links)


def test_identity_is_prefix_stable():
    """Идентичность, построенная на префиксе истории, обязана совпадать с
    соответствующей частью идентичности, построенной на полной истории.
    Это и есть формальное определение отсутствия утечки."""
    full = sorted(_series(100, FIVE_A, 0, 5) + _series(200, FIVE_A, 12, 5) +
                  _series(300, FIVE_B, 30, 5) + _series(400, FIVE_B, 45, 5),
                  key=lambda m: (m.start_time, m.match_id))
    _, res_full = resolve_identities(full)

    cutoff = BASE + timedelta(days=20)
    prefix = [m for m in full if m.start_time <= cutoff]
    _, res_prefix = resolve_identities(prefix)

    links_full = {l.source_team_id: (l.canonical_team_id, l.valid_from)
                  for l in res_full.links if l.valid_from <= cutoff}
    links_prefix = {l.source_team_id: (l.canonical_team_id, l.valid_from) for l in res_prefix.links}
    assert links_full == links_prefix, "идентичность зависит от будущих данных — УТЕЧКА"


def test_future_name_change_cannot_create_past_link():
    """Имя — снимок «на сегодня». Оно не должно позволять слить команды,
    которые по ростеру/времени не связаны."""
    ms = sorted(_series(100, FIVE_A, 0, 5) + _series(200, FIVE_B, 12, 5),
                key=lambda m: (m.start_time, m.match_id))
    names = {100: "Same Name", 200: "Same Name"}
    resolver, res = resolve_identities(ms, names=names)
    assert len(res.links) == 0
    assert resolver.canonical(200) != resolver.canonical(100)


# ============================ утилиты ============================

def test_name_normalization_is_conservative():
    assert normalize_name("The MongolZ") == normalize_name("The Mongolz")
    assert normalize_name("Team  KEV") == "team kev"
    assert normalize_name("PSG.LGD") == "psg lgd"
    # НЕ должно склеивать разные сущности
    assert normalize_name("Entity") != normalize_name("Entity Academy")


def test_jaccard():
    assert jaccard(FIVE_A, FIVE_A) == 1.0
    assert jaccard(FIVE_A, FIVE_B) == 0.0
    assert jaccard(frozenset({1, 2, 3}), frozenset({1, 2, 4})) == pytest.approx(0.5)
