"""
PHASE 13 — adversarial leakage tests для ковариат неопределённости.

Ковариаты неопределённости опаснее обычных признаков: соблазн подсмотреть
будущее здесь сильнее (например, «сколько всего матчей сыграла команда» —
величина, которую естественно считать по всей истории, и это была бы
утечка). Плюс требование PART C: ковариата не должна коррелировать с
исходом по построению, иначе это скрытый признак силы.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

import pytest

from src.datasets.uncertainty_features import (
    UNCERTAINTY_COLUMNS,
    build_uncertainty_features,
    to_feature_dict,
)

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class P:
    account_id: int
    hero_id: int
    is_radiant: bool


@dataclass(frozen=True)
class M:
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_win: bool
    players: Sequence[P]


def _players(rad_accs=(1, 2, 3, 4, 5), dire_accs=(101, 102, 103, 104, 105),
             rad_heroes=(10, 11, 12, 13, 14), dire_heroes=(20, 21, 22, 23, 24)):
    return ([P(rad_accs[i], rad_heroes[i], True) for i in range(5)]
            + [P(dire_accs[i], dire_heroes[i], False) for i in range(5)])


def _matches(n=12, rad_team=1, dire_team=2):
    return [M(i, BASE + timedelta(days=i), rad_team, dire_team, i % 2 == 0, _players())
            for i in range(n)]


def _snap(rows):
    """NaN не равен сам себе, поэтому в снимке он заменяется маркером —
    иначе сравнение снимков падало бы на пропусках, а не на утечках."""
    def norm(v):
        return "NaN" if isinstance(v, float) and v != v else v
    return {r.match_id: tuple(sorted((k, norm(v)) for k, v in to_feature_dict(r).items()))
            for r in rows}


# ---------- Q1/Q10: будущий исход не влияет на прошлое ----------

def test_future_result_does_not_change_past():
    ms = _matches()
    base = _snap(build_uncertainty_features(ms))
    idx = 6
    m = ms[idx]
    mut = list(ms)
    mut[idx] = M(m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id,
                 not m.radiant_win, m.players)
    after = _snap(build_uncertainty_features(mut))
    for x in ms[:idx]:
        assert base[x.match_id] == after[x.match_id]


def test_result_does_not_enter_covariates_at_all():
    """Сильнее предыдущего: переворачиваем исход у ВСЕХ матчей. Ковариаты
    неопределённости обязаны не измениться НИГДЕ — они меры незнания, а не
    силы, и результат в них входить не должен вообще."""
    ms = _matches(10)
    base = _snap(build_uncertainty_features(ms))
    flipped = [M(m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id,
                 not m.radiant_win, m.players) for m in ms]
    after = _snap(build_uncertainty_features(flipped))
    assert base == after, "исход просочился в ковариаты неопределённости"


# ---------- Q3/Q4/Q9: будущие матчи не меняют прошлое ----------

def test_appending_future_matches_changes_nothing_past():
    ms = _matches(10)
    base = _snap(build_uncertainty_features(ms))
    future = [M(900 + i, ms[-1].start_time + timedelta(days=100 + i), 1, 2, True, _players())
              for i in range(5)]
    after = _snap(build_uncertainty_features(ms + future))
    for x in ms:
        assert base[x.match_id] == after[x.match_id], "time-decay/накопление утекли назад"


def test_prefix_stability():
    ms = _matches(14)
    full = _snap(build_uncertainty_features(ms))
    prefix = _snap(build_uncertainty_features(ms[:7]))
    for x in ms[:7]:
        assert full[x.match_id] == prefix[x.match_id]


# ---------- Q7: состав текущего матча не считается «известным заранее» ----------

def test_first_match_has_no_history():
    r = build_uncertainty_features(_matches(1))[0]
    assert r.team_matches_min == 0 and r.team_matches_max == 0
    assert r.player_matches_min == 0
    assert r.hero_games_min == 0.0
    assert r.rare_heroes_count == 10
    assert r.transferred_players_total == 0


def test_counts_accumulate_forward_only():
    rows = build_uncertainty_features(_matches(5))
    assert [r.team_matches_min for r in rows] == [0, 1, 2, 3, 4]
    assert [r.player_matches_min for r in rows] == [0, 1, 2, 3, 4]


# ---------- PART L: новизна состава ----------

def test_roster_change_is_detected_on_the_match_it_happens():
    ms = _matches(6)
    changed = M(6, BASE + timedelta(days=6), 1, 2, True,
                _players(rad_accs=(1, 2, 3, 4, 99)))
    rows = build_uncertainty_features(ms + [changed])
    assert rows[-1].new_players_max == 1, "замена не замечена в матче, где произошла"
    assert rows[-1].roster_matches_together_min == 0, "новый состав не может иметь стажа"
    assert rows[-2].new_players_max == 0


def test_stable_roster_accumulates_matches_together():
    rows = build_uncertainty_features(_matches(6))
    assert rows[-1].roster_matches_together_min >= 4
    assert rows[-1].roster_age_days_min is not None and rows[-1].roster_age_days_min > 0


# ---------- PART M: переходы игроков ----------

def test_transfer_detected_when_player_switches_team():
    ms = _matches(5, rad_team=1, dire_team=2)
    # игрок 1 переходит в команду 3
    moved = M(5, BASE + timedelta(days=5), 3, 2, True,
              _players(rad_accs=(1, 201, 202, 203, 204)))
    rows = build_uncertainty_features(ms + [moved])
    assert rows[-1].transferred_players_total >= 1, "переход игрока не замечен"


def test_no_transfer_flag_when_everyone_stays():
    rows = build_uncertainty_features(_matches(6))
    assert all(r.transferred_players_total == 0 for r in rows)


# ---------- PART E: новизна героев ----------

def test_rare_heroes_count_drops_as_heroes_accumulate_history():
    rows = build_uncertainty_features(_matches(30))
    assert rows[0].rare_heroes_count == 10
    assert rows[-1].rare_heroes_count < 10, "герои так и не перестали считаться редкими"


def test_new_hero_is_flagged_rare_even_late_in_history():
    ms = _matches(30)
    late = M(30, BASE + timedelta(days=30), 1, 2, True,
             _players(rad_heroes=(200, 11, 12, 13, 14)))
    rows = build_uncertainty_features(ms + [late])
    assert rows[-1].hero_games_min == 0.0, "невиданный герой обязан иметь нулевую историю"


def test_all_columns_present_in_feature_dict():
    d = to_feature_dict(build_uncertainty_features(_matches(3))[-1])
    for c in UNCERTAINTY_COLUMNS:
        assert c in d
        assert isinstance(d[c], float)


# ---------- PART M: переход БЕЗ опоры на team_id ----------

def test_teammate_churn_is_zero_for_stable_roster():
    rows = build_uncertainty_features(_matches(6))
    assert rows[-1].teammate_churn_max == pytest.approx(0.0)
    assert rows[-1].players_with_new_teammates == 0


def test_teammate_churn_detects_player_moving_to_new_environment():
    """Игрок 1 уходит к совершенно новым партнёрам. У него самого смена
    окружения полная, у четверых новичков истории нет вовсе, поэтому
    средняя по пятёрке доля = 1/5. Счётчик игроков со сменившимся
    окружением обязан показать ровно одного."""
    ms = _matches(6, rad_team=1, dire_team=2)
    moved = M(6, BASE + timedelta(days=6), 1, 2, True,
              _players(rad_accs=(1, 301, 302, 303, 304)))
    rows = build_uncertainty_features(ms + [moved])
    assert rows[-1].players_with_new_teammates == 1
    assert rows[-1].teammate_churn_max == pytest.approx(0.2)


def test_teammate_churn_is_high_when_whole_five_regroups():
    """Две устоявшиеся пятёрки меняются половинами: у КАЖДОГО игрока
    сменилось больше половины партнёров. Здесь величина обязана быть большой."""
    a = [M(i, BASE + timedelta(days=i), 1, 2, True,
           _players(rad_accs=(1, 2, 3, 4, 5), dire_accs=(11, 12, 13, 14, 15)))
         for i in range(8)]
    mixed = M(8, BASE + timedelta(days=8), 1, 2, True,
              _players(rad_accs=(1, 2, 13, 14, 15), dire_accs=(11, 12, 3, 4, 5)))
    rows = build_uncertainty_features(a + [mixed])
    assert rows[-1].teammate_churn_max > 0.5
    # У двоих оставшихся на стороне сменилось 3 партнёра из 4 (0.75 > 0.5),
    # у троих пришедших — ровно 2 из 4 (0.5, порог строгий). Итого по 2 на
    # сторону. Число проверяется точно, чтобы тест ловил смену семантики.
    assert rows[-1].players_with_new_teammates == 4


def test_teammate_churn_ignores_pure_team_id_change():
    """КЛЮЧЕВОЙ ТЕСТ. Та же пятёрка играет под ДРУГИМ team_id —
    ровно та фрагментация идентичности, что нашла Phase 10. Величина,
    опирающаяся на team_id, объявит пять переходов; устойчивая обязана
    показать ноль."""
    ms = _matches(6, rad_team=1, dire_team=2)
    renamed = M(6, BASE + timedelta(days=6), 777, 2, True, _players())
    rows = build_uncertainty_features(ms + [renamed])
    last = rows[-1]
    assert last.transferred_players_total == 5, "контроль: team_id-версия обязана ошибиться"
    assert last.teammate_churn_max == pytest.approx(0.0), \
        "устойчивая величина не должна реагировать на переименование команды"
    assert last.players_with_new_teammates == 0


def test_teammate_churn_result_independent():
    ms = _matches(8)
    base = _snap(build_uncertainty_features(ms))
    flipped = [M(m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id,
                 not m.radiant_win, m.players) for m in ms]
    assert base == _snap(build_uncertainty_features(flipped))
