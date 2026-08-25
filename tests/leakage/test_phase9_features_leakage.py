"""
Phase 9 — adversarial leakage tests для roster_representation.py и
hero_strength_schemes.py.

Особое внимание к time decay (план Phase 9.0 пометил его как рискованный):
затухание не должно позволять будущим матчам влиять на прошлые прогнозы.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import FrozenSet, Optional, Tuple

import pytest

from src.datasets.hero_strength_schemes import SCHEMES, build_hero_scheme_features
from src.datasets.roster_representation import build_roster_representation, to_feature_dict

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class MR:
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_roster: FrozenSet[int]
    dire_roster: FrozenSet[int]
    radiant_win: bool


@dataclass(frozen=True)
class MD:
    match_id: int
    start_time: datetime
    patch_id: Optional[int]
    radiant_picks: Tuple[int, ...]
    dire_picks: Tuple[int, ...]
    radiant_bans: Tuple[int, ...]
    dire_bans: Tuple[int, ...]
    radiant_win: bool


def _rmatches(n=12):
    a = frozenset({1, 2, 3, 4, 5})
    b = frozenset({10, 11, 12, 13, 14})
    return [MR(i, BASE + timedelta(days=i), 100, 200, a, b, i % 2 == 0) for i in range(n)]


def _dmatches(n=12, patch=54):
    return [MD(i, BASE + timedelta(days=i), patch, (1, 2, 3, 4, 5), (6, 7, 8, 9, 10),
               (11, 12), (13, 14), i % 2 == 0) for i in range(n)]


def _rsnap(rows):
    return {r.match_id: tuple(sorted(to_feature_dict(r).items())) for r in rows}


def _dsnap(rows):
    return {r.match_id: (tuple(round(r.strength_diff[s], 9) for s in SCHEMES),
                         round(r.pick_rate_diff, 9), round(r.ban_rate_diff, 9),
                         round(r.contest_rate_diff, 9)) for r in rows}


# --- будущий результат не меняет прошлое ---

def test_roster_future_result_does_not_affect_past():
    ms = _rmatches()
    base = _rsnap(build_roster_representation(ms))
    mid = 6
    mut = list(ms)
    m = ms[mid]
    mut[mid] = MR(m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id,
                  m.radiant_roster, m.dire_roster, not m.radiant_win)
    after = _rsnap(build_roster_representation(mut))
    for x in ms[:mid]:
        assert base[x.match_id] == after[x.match_id]


def test_hero_schemes_future_result_does_not_affect_past():
    ms = _dmatches()
    base = _dsnap(build_hero_scheme_features(ms))
    mid = 6
    mut = list(ms)
    m = ms[mid]
    mut[mid] = MD(m.match_id, m.start_time, m.patch_id, m.radiant_picks, m.dire_picks,
                  m.radiant_bans, m.dire_bans, not m.radiant_win)
    after = _dsnap(build_hero_scheme_features(mut))
    for x in ms[:mid]:
        assert base[x.match_id] == after[x.match_id]


# --- добавление будущих матчей не меняет прошлое (ключевой тест для time decay) ---

def test_time_decay_future_matches_do_not_affect_past():
    """Ленивое затухание применяется от ПРОШЛОГО наблюдения к текущему
    моменту. Матч, добавленный ПОЗЖЕ, не должен изменить ни одного
    прошлого значения ни в одной из 5 схем."""
    ms = _dmatches()
    base = _dsnap(build_hero_scheme_features(ms))
    future = MD(999, ms[-1].start_time + timedelta(days=200), 54,
                (1, 2, 3, 4, 5), (6, 7, 8, 9, 10), (11, 12), (13, 14), True)
    after = _dsnap(build_hero_scheme_features(ms + [future]))
    for x in ms:
        assert base[x.match_id] == after[x.match_id], f"time-decay утечка на {x.match_id}"


def test_roster_future_matches_do_not_affect_past():
    ms = _rmatches()
    base = _rsnap(build_roster_representation(ms))
    future = MR(999, ms[-1].start_time + timedelta(days=90), 100, 200,
                frozenset({1, 2, 3, 4, 77}), frozenset({10, 11, 12, 13, 14}), True)
    after = _rsnap(build_roster_representation(ms + [future]))
    for x in ms:
        assert base[x.match_id] == after[x.match_id]


# --- будущая смена состава не меняет прошлое ---

def test_future_roster_change_does_not_affect_past():
    ms = _rmatches()
    base = _rsnap(build_roster_representation(ms))
    mid = 8
    mut = list(ms)
    m = ms[mid]
    mut[mid] = MR(m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id,
                  frozenset({1, 2, 3, 4, 55}), m.dire_roster, m.radiant_win)
    after = _rsnap(build_roster_representation(mut))
    for x in ms[:mid]:
        assert base[x.match_id] == after[x.match_id]


# --- отсутствие истории даёт нейтральные значения ---

def test_first_match_is_neutral():
    rows = build_roster_representation(_rmatches(3))
    f = to_feature_dict(rows[0])
    assert f["elo_mean_diff"] == pytest.approx(0.0)
    assert f["roster_strength_delta_diff"] == pytest.approx(0.0)
    assert f["player_replacement_delta_diff"] == pytest.approx(0.0)
    assert f["five_vs_team_elo_diff"] == pytest.approx(0.0)

    drows = build_hero_scheme_features(_dmatches(3))
    for s in SCHEMES:
        assert drows[0].strength_diff[s] == pytest.approx(0.0)
    assert drows[0].pick_rate_diff == pytest.approx(0.0)
    assert drows[0].ban_rate_diff == pytest.approx(0.0)


# --- поведенческие проверки (PART B/C) ---

def test_roster_delta_is_zero_when_roster_unchanged():
    """Ключевое свойство конструкции: обе пятёрки оценены ТЕКУЩИМИ
    рейтингами, поэтому при неизменном составе delta ровно 0 — признак
    реагирует на СМЕНУ СОСТАВА, а не на дрейф рейтингов."""
    rows = build_roster_representation(_rmatches(10))
    for r in rows[1:]:
        assert r.radiant.roster_strength_delta == pytest.approx(0.0)
        assert r.dire.roster_strength_delta == pytest.approx(0.0)


def test_roster_delta_reacts_to_replacement():
    strong = frozenset({1, 2, 3, 4, 5})
    weak = frozenset({10, 11, 12, 13, 14})
    ms = [MR(i, BASE + timedelta(days=i), 100, 200, strong, weak, True) for i in range(20)]
    # игрок 5 (накопил высокий рейтинг) заменён новичком 90 (рейтинг 1000)
    ms.append(MR(50, BASE + timedelta(days=30), 100, 200, frozenset({1, 2, 3, 4, 90}), weak, True))
    rows = build_roster_representation(ms)
    last = rows[-1]
    assert last.radiant.roster_strength_delta < 0, "ослабление состава не отражено"
    assert last.radiant.player_replacement_delta < 0, "replacement delta не отражает замену"


def test_five_vs_team_elo_detects_stronger_five_than_club_history():
    """PART C: команда без истории (team Elo=1000), но с сильными игроками
    -> current_five_vs_team_elo > 0 сразу, в первом же матче."""
    strong = frozenset({1, 2, 3, 4, 5})
    weak = frozenset({10, 11, 12, 13, 14})
    ms = [MR(i, BASE + timedelta(days=i), 100, 200, strong, weak, True) for i in range(20)]
    # те же сильные игроки под НОВЫМ team_id
    ms.append(MR(50, BASE + timedelta(days=30), 777, 888, strong, frozenset({20, 21, 22, 23, 24}), True))
    last = build_roster_representation(ms)[-1]
    assert last.radiant.current_five_vs_team_elo > 0, "новый team_id с сильной пятёркой не распознан"


def test_patch_scoped_scheme_resets_on_patch_change():
    """S2/S5 обязаны обнуляться на смене патча, S1/S4 — нет.

    Исходы намеренно АСИММЕТРИЧНЫ (radiant всегда выигрывает на патче 54):
    при симметричных исходах strength_diff = 0 у ЛЮБОЙ схемы просто по
    симметрии, и тест ничего бы не проверял.
    """
    ms = [MD(i, BASE + timedelta(days=i), 54, (1, 2, 3, 4, 5), (6, 7, 8, 9, 10),
             (11, 12), (13, 14), True) for i in range(8)]
    ms += [MD(100 + i, BASE + timedelta(days=20 + i), 55, (1, 2, 3, 4, 5), (6, 7, 8, 9, 10),
              (11, 12), (13, 14), True) for i in range(3)]
    rows = build_hero_scheme_features(ms)

    # последний матч ДО смены патча: patch-scoped схемы накопили сигнал
    assert rows[7].strength_diff["same_patch"] > 0.0

    first_after_patch = rows[8]
    assert first_after_patch.strength_diff["same_patch"] == pytest.approx(0.0), "same_patch не обнулился"
    assert first_after_patch.strength_diff["patch_decay"] == pytest.approx(0.0), "patch_decay не обнулился"
    # глобальная схема, наоборот, обязана помнить историю через границу патча
    assert first_after_patch.strength_diff["global"] > 0.0, "global потерял историю на смене патча"
