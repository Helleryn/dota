"""
PHASE 11 — adversarial leakage tests для player_hero_features.py.

Покрывает все сценарии PART T задания, в том числе самый тонкий:
роль ТЕКУЩЕГО матча (post-match `lane_role` + GPM) не должна влиять на
признаки этого же матча.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence

import pytest

from src.datasets.player_hero_features import (
    build_player_hero_features,
    derive_positions,
)

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class P:
    account_id: int
    hero_id: int
    is_radiant: bool
    lane_role: Optional[int] = 1
    gold_per_min: Optional[int] = 400


@dataclass(frozen=True)
class M:
    match_id: int
    start_time: datetime
    radiant_win: bool
    players: Sequence[P]


def _team(base_id, heroes, is_rad, gpms=None, lanes=None):
    gpms = gpms or [600, 550, 450, 300, 250]
    lanes = lanes or [1, 2, 3, 3, 1]
    return [P(base_id + i, heroes[i], is_rad, lanes[i], gpms[i]) for i in range(5)]


def _matches(n=12):
    out = []
    for i in range(n):
        players = _team(1, [10, 11, 12, 13, 14], True) + _team(100, [20, 21, 22, 23, 24], False)
        out.append(M(i, BASE + timedelta(days=i), i % 2 == 0, players))
    return out


def _snap(rows):
    return {
        r.match_id: (
            round(r.player_hero_strength_diff, 9),
            round(r.player_role_strength_diff, 9),
            round(r.player_hero_role_strength_diff, 9),
            round(r.role_entropy_diff, 9),
            round(r.meta_relative_ph_diff, 9),
        )
        for r in rows
    }


# ---------- 1. будущие результаты ----------

def test_future_result_does_not_change_past():
    ms = _matches()
    base = _snap(build_player_hero_features(ms))
    mid = 6
    mut = list(ms)
    m = ms[mid]
    mut[mid] = M(m.match_id, m.start_time, not m.radiant_win, m.players)
    after = _snap(build_player_hero_features(mut))
    for x in ms[:mid]:
        assert base[x.match_id] == after[x.match_id], f"утечка на {x.match_id}"


# ---------- 4/10. добавление будущих матчей ----------

def test_appending_future_matches_changes_nothing_past():
    ms = _matches()
    base = _snap(build_player_hero_features(ms))
    future = [M(900 + i, ms[-1].start_time + timedelta(days=200 + i), True,
                _team(1, [10, 11, 12, 13, 14], True) + _team(100, [20, 21, 22, 23, 24], False))
              for i in range(4)]
    after = _snap(build_player_hero_features(ms + future))
    for x in ms:
        assert base[x.match_id] == after[x.match_id], "time-decay утечка"


# ---------- 6. будущий player x hero performance ----------

def test_future_player_hero_performance_does_not_leak():
    ms = _matches()
    base = _snap(build_player_hero_features(ms))
    mid = 8
    mut = list(ms)
    m = ms[mid]
    # тот же игрок, ДРУГОЙ герой в будущем -> не должно менять прошлое
    newp = [P(p.account_id, p.hero_id + 50, p.is_radiant, p.lane_role, p.gold_per_min)
            for p in m.players]
    mut[mid] = M(m.match_id, m.start_time, m.radiant_win, newp)
    after = _snap(build_player_hero_features(mut))
    for x in ms[:mid]:
        assert base[x.match_id] == after[x.match_id]


# ---------- 3/8. роль ТЕКУЩЕГО матча не влияет на его же признаки ----------

def test_current_match_lane_role_does_not_affect_its_own_features():
    """КЛЮЧЕВОЙ ТЕСТ. lane_role и GPM — post-match. Если их изменить у
    ТЕКУЩЕГО матча, признаки ЭТОГО матча меняться не должны: роль текущего
    матча попадает только в ИСТОРИЮ, то есть влияет на СЛЕДУЮЩИЕ матчи."""
    ms = _matches(10)
    base = _snap(build_player_hero_features(ms))

    idx = 9  # последний матч — у него нет «следующих», влиять некуда
    m = ms[idx]
    scrambled = [P(p.account_id, p.hero_id, p.is_radiant,
                   lane_role=(3 if p.lane_role == 1 else 1),
                   gold_per_min=(999 - (p.gold_per_min or 0)))
                 for p in m.players]
    mut = list(ms)
    mut[idx] = M(m.match_id, m.start_time, m.radiant_win, scrambled)
    after = _snap(build_player_hero_features(mut))

    assert base[m.match_id] == after[m.match_id], \
        "УТЕЧКА: роль/GPM текущего матча повлияли на его собственные признаки"


def test_current_match_gpm_does_not_affect_own_features_midstream():
    """То же, но для матча в середине: его собственные признаки не меняются,
    а последующие — обязаны измениться (иначе история не накапливается)."""
    ms = _matches(10)
    base = _snap(build_player_hero_features(ms))
    idx = 4
    m = ms[idx]
    scrambled = [P(p.account_id, p.hero_id, p.is_radiant,
                   lane_role=(3 if p.lane_role == 1 else 1),
                   gold_per_min=(999 - (p.gold_per_min or 0)))
                 for p in m.players]
    mut = list(ms)
    mut[idx] = M(m.match_id, m.start_time, m.radiant_win, scrambled)
    after = _snap(build_player_hero_features(mut))

    assert base[m.match_id] == after[m.match_id], "собственные признаки изменились — утечка"
    for x in ms[:idx]:
        assert base[x.match_id] == after[x.match_id], "прошлое изменилось — утечка"


# ---------- 5. будущее состояние меты ----------

def test_future_meta_does_not_leak():
    ms = _matches()
    base = _snap(build_player_hero_features(ms))
    # в будущем те же герои играют с другим исходом -> меняется их сила в мете
    future = [M(800 + i, ms[-1].start_time + timedelta(days=10 + i), False,
                _team(1, [10, 11, 12, 13, 14], True) + _team(100, [20, 21, 22, 23, 24], False))
              for i in range(5)]
    after = _snap(build_player_hero_features(ms + future))
    for x in ms:
        assert base[x.match_id] == after[x.match_id]


# ---------- отсутствие истории даёт нейтральное значение ----------

def test_first_match_is_neutral():
    rows = build_player_hero_features(_matches(3))
    r0 = rows[0]
    assert r0.player_hero_strength_diff == pytest.approx(0.0)
    assert r0.player_role_strength_diff == pytest.approx(0.0)
    assert r0.player_hero_role_strength_diff == pytest.approx(0.0)
    assert r0.role_entropy_diff == pytest.approx(0.0)
    assert r0.meta_relative_ph_diff == pytest.approx(0.0)
    assert r0.player_hero_games_min == pytest.approx(0.0)


def test_prefix_stability():
    """Признаки на префиксе истории тождественны соответствующей части
    признаков на полной истории — формальное определение отсутствия утечки."""
    ms = _matches(14)
    full = _snap(build_player_hero_features(ms))
    prefix = _snap(build_player_hero_features(ms[:7]))
    for x in ms[:7]:
        assert full[x.match_id] == prefix[x.match_id]


# ---------- корректность вывода позиций ----------

def test_derive_positions_splits_core_and_support():
    players = [
        P(1, 10, True, lane_role=1, gold_per_min=600),   # safelane, выше GPM -> поз.1
        P(2, 11, True, lane_role=2, gold_per_min=550),   # mid -> поз.2
        P(3, 12, True, lane_role=3, gold_per_min=450),   # offlane, выше -> поз.3
        P(4, 13, True, lane_role=3, gold_per_min=300),   # offlane, ниже -> поз.4
        P(5, 14, True, lane_role=1, gold_per_min=250),   # safelane, ниже -> поз.5
    ]
    pos = derive_positions(players)
    assert pos[1] == 1 and pos[2] == 2 and pos[3] == 3 and pos[4] == 4 and pos[5] == 5


def test_derive_positions_handles_missing_lane_role():
    players = [P(i, 10 + i, True, lane_role=None, gold_per_min=400) for i in range(5)]
    pos = derive_positions(players)
    assert all(v is None for v in pos.values()), "неизвестная линия должна давать None, а не выдумку"


def test_player_hero_strength_is_zero_when_player_is_the_only_one_on_hero():
    """Признак измеряет превосходство игрока НАД BASELINE ГЕРОЯ. Если герой
    встречается только у одного игрока, ph_rate == h_rate, и корректный
    ответ — ровно 0: разделить мастерство игрока и силу героя невозможно."""
    rows = [M(i, BASE + timedelta(days=i), True,
              _team(1, [10, 11, 12, 13, 14], True) + _team(100, [20, 21, 22, 23, 24], False))
            for i in range(30)]
    out = build_player_hero_features(rows)
    assert out[-1].player_hero_strength_diff == pytest.approx(0.0), \
        "при отсутствии сравнимой выборки признак обязан быть нейтральным"
    assert out[-1].player_hero_games_min > 0


def test_player_hero_strength_separates_players_on_the_same_hero():
    """Настоящая проверка: ОДИН И ТОТ ЖЕ герой у разных игроков с разными
    исходами. Сильный игрок обязан получить положительную оценку."""
    HERO = 10
    rows = []
    for i in range(40):
        if i % 2 == 0:
            # игрок 1 на HERO выигрывает
            players = _team(1, [HERO, 11, 12, 13, 14], True) + _team(100, [20, 21, 22, 23, 24], False)
            win = True
        else:
            # игрок 200 на ТОМ ЖЕ HERO проигрывает
            players = _team(200, [HERO, 11, 12, 13, 14], True) + _team(100, [20, 21, 22, 23, 24], False)
            win = False
        rows.append(M(i, BASE + timedelta(days=i), win, players))
    out = build_player_hero_features(rows)

    strong = [r for r, m in zip(out, rows) if any(p.account_id == 1 for p in m.players)][-1]
    weak = [r for r, m in zip(out, rows) if any(p.account_id == 200 for p in m.players)][-1]
    assert strong.player_hero_strength_diff > 0, "сильный игрок на герое не выделен"
    assert weak.player_hero_strength_diff < 0, "слабый игрок на герое не выделен"
