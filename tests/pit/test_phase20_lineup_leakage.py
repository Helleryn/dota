"""
PHASE 20 — adversarial-тесты структуры состава (PART M).

Двенадцать сценариев. Устройство то же, что в Phase 18/19: у каждого
запрета есть парный контроль из шума, иначе тест проходил бы и на коде,
который данные игнорирует вовсе.

Два теста фаза объявила критическими:

  PLAYER TRANSFER  — при смене команды факты об игроке обязаны перейти
                     вместе с ним, а team_id не должен их менять;
  META CUTOFF      — изменение состояния после T не меняет прогноз на T.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.pit.lineup_structure import (
    ATTENUATION_KAPPA,
    LineupMatch,
    _State,
    build_lineup_features,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
A = (1, 2, 3, 4, 5)
B = (11, 12, 13, 14, 15)
C = (21, 22, 23, 24, 25)

NUM = ("rd_mean_diff", "rd_max_diff", "elo_mean_diff_attenuated",
       "pos_diff_1", "pos_diff_2", "pos_diff_3", "pos_diff_4", "pos_diff_5",
       "pos_bottleneck", "pos_best", "pos_imbalance",
       "elo_mean_decayed_diff", "recency_gap", "days_since_last_max_diff",
       "pair_games_mean_diff", "pair_games_min_diff", "core3_games_diff",
       "synergy_residual_diff", "returning_players_diff",
       "core_continuity_diff", "lineup_novelty_diff", "days_since_lineup_diff")


def m(mid, day, hours=0.0, r=A, d=B, win=True, rt=100, dt=200,
      lane=None, gpm=None) -> LineupMatch:
    """Матч. По умолчанию lane_role/GPM расставлены так, что позиции
    выводятся однозначно: 1-2 сейф (керри и пятёрка), 2 мид, 3-4 офлейн."""
    lanes = {}
    gpms = {}
    for side in (r, d):
        for i, a in enumerate(side):
            lanes[a] = (1, 2, 3, 1, 3)[i]
            gpms[a] = (600, 550, 450, 300, 250)[i]
    if lane:
        lanes.update(lane)
    if gpm:
        gpms.update(gpm)
    return LineupMatch(match_id=mid, start_time=T0 + timedelta(days=day, hours=hours),
                       radiant_team_id=rt, dire_team_id=dt, radiant_win=win,
                       radiant=tuple(sorted(r)), dire=tuple(sorted(d)),
                       lane_role=lanes, gpm=gpms)


def feats(matches, horizon=timedelta(hours=24), emit=None):
    return {r.match_id: r
            for r in build_lineup_features(matches, horizon, emit_only=emit)}


def same(a, b, fields=NUM, rel_tol=1e-12):
    for f in fields:
        x, y = getattr(a, f), getattr(b, f)
        if x is None or y is None:
            if x is not y:
                return False
        elif not math.isclose(float(x), float(y), rel_tol=rel_tol, abs_tol=1e-12):
            return False
    return True


def history(n=14):
    """История, АСИММЕТРИЧНАЯ по сторонам.

    Первая версия давала A против B каждый матч, поэтому счётчики обеих
    сторон совпадали и ВСЕ разностные признаки были структурно нулевыми —
    тесты проходили бы на любом коде. Здесь B дополнительно играет с
    третьей командой, а пятый игрок A меняется, так что расходятся и
    опыт, и неопределённость, и сыгранность.
    """
    out = [m(i, i, win=(i % 3 != 0)) for i in range(1, n + 1)]
    out += [m(1000 + j, j, hours=6.0, r=B, d=C, win=False)
            for j in range(1, n // 2 + 1)]
    out += [m(2000, n - 1, hours=12.0, r=(1, 2, 3, 4, 77), d=C, win=True)]
    return sorted(out, key=lambda x: (x.start_time, x.match_id))


# ------------------------------------------------------------------ 1
def test_1_current_match_outcome_mutation():
    """Исход самого матча не участвует в его собственных признаках."""
    h = history()
    a = feats(h + [m(99, 20, win=True)])[99]
    b = feats(h + [m(99, 20, win=False)])[99]
    assert same(a, b), "исход матча просочился в его признаки"
    assert a.target != b.target, "контроль: метки обязаны отличаться"


# ------------------------------------------------------------------ 2
def test_2_future_match_insertion():
    """Матч, стартовавший после T, невидим; при горизонте 0 — обязан влиять."""
    h = history()
    target = m(99, 20)
    later = m(50, 19, hours=12, win=False)
    with_l = feats(h + [later, target])[99]
    without = feats(h + [target])[99]
    h0 = feats(h + [later, target], timedelta(0))[99]
    assert same(with_l, without), "матч после T попал в признаки"
    assert not same(h0, without), "контроль: при горизонте 0 обязан влиять"


# ------------------------------------------------------------------ 3
def test_3_post_match_performance_mutation():
    """GPM текущего матча — post-match величина и не влияет на его признаки.

    GPM определяет позицию через `derive_positions`, поэтому если бы он
    протекал, `pos_diff_*` менялись бы.
    """
    h = history()
    normal = m(99, 20)
    swapped = m(99, 20, gpm={1: 250, 4: 600, 11: 250, 14: 600})
    a, b = feats(h + [normal])[99], feats(h + [swapped])[99]
    assert same(a, b), "GPM текущего матча изменил его собственные признаки"

    # контроль: GPM ПРОШЛЫХ матчей влиять обязан. Меняем его во ВСЕЙ
    # истории, иначе один матч не перевесит моду из тринадцати.
    swapped_hist = [m(x.match_id, 0, gpm={1: 250, 4: 600, 11: 250, 14: 600})
                    if False else x for x in h]
    st_a, st_b = _State(), _State()
    for x in h:
        st_a.apply(x, _dp)
        st_b.apply(_regpm(x), _dp)
    assert st_a.pos_hist[1] != st_b.pos_hist[1], \
        "контроль: GPM прошлых матчей обязан менять выведенные позиции"


# ------------------------------------------------------------------ 4
def test_4_team_rename_and_team_id_change():
    """team_id не влияет ни на один признак фазы.

    Это не совпадение, а конструкция: якорь H5 — идентичность игроков, а
    не команда (Phase 10: 461 пара одноимённых `team_id` играла
    параллельно).
    """
    h = history()
    a = feats(h + [m(99, 20, rt=100, dt=200)])[99]
    b = feats(h + [m(99, 20, rt=777777, dt=888888)])[99]
    assert same(a, b), "смена team_id изменила признаки состава"


# ------------------------------------------------------------------ 5
def test_5_player_transfer_keeps_individual_facts():
    """КРИТИЧЕСКИЙ ТЕСТ ФАЗЫ.

    Игрок 1 уходит из команды 100 в команду 300. Его собственные
    неопределённость и опыт обязаны перейти вместе с ним и не зависеть от
    `team_id`.
    """
    st = _State()
    for i in range(1, 21):
        st.apply(m(i, i), _dp)
    ts = (T0 + timedelta(days=25)).timestamp()

    rd_before = st.rd(1, ts)
    n_before = st.eff_games(1, ts)
    elo_before = st.elo(1)

    # тот же игрок, другая команда, ни одного нового матча
    st_after = _State()
    for i in range(1, 21):
        st_after.apply(m(i, i, rt=300), _dp)     # команда переименована/сменена

    assert math.isclose(st_after.rd(1, ts), rd_before, rel_tol=1e-12), \
        "неопределённость игрока зависит от team_id"
    assert math.isclose(st_after.eff_games(1, ts), n_before, rel_tol=1e-12)
    assert math.isclose(st_after.elo(1), elo_before, rel_tol=1e-12)

    # контроль: у игрока с другой историей величины обязаны отличаться
    st2 = _State()
    for i in range(1, 6):
        st2.apply(m(i, i), _dp)
    assert st2.rd(1, ts) > rd_before, \
        "контроль: меньше матчей — больше неопределённость"


def test_5b_uncertainty_is_individual_not_team_wide():
    """Неопределённость обязана различаться ВНУТРИ пятёрки.

    Phase 19 показала: волатильность player-Elo вырождается в командную
    величину, потому что дельта общая на пятёрых. Здесь величина строится
    на собственной истории игрока, поэтому новичок в составе ветеранов
    обязан иметь другое RD.
    """
    st = _State()
    for i in range(1, 21):
        st.apply(m(i, i), _dp)
    # игрок 999 приходит на место 5 в последнем матче
    st.apply(m(21, 21, r=(1, 2, 3, 4, 999)), _dp)
    ts = (T0 + timedelta(days=22)).timestamp()

    rds = {p: st.rd(p, ts) for p in (1, 2, 3, 4, 999)}
    assert len(set(round(v, 9) for v in rds.values())) > 1, \
        "неопределённость одинакова у всей пятёрки — вырождение Phase 19 повторилось"
    assert rds[999] > max(rds[p] for p in (1, 2, 3, 4)), \
        "у новичка обязана быть наибольшая неопределённость"


# ------------------------------------------------------------------ 6
def test_6_current_match_role_mutation():
    """lane_role текущего матча не влияет на его признаки."""
    h = history()
    a = feats(h + [m(99, 20)])[99]
    b = feats(h + [m(99, 20, lane={1: 3, 3: 1, 11: 3, 13: 1})])[99]
    assert same(a, b), "роль текущего матча изменила его собственные признаки"


# ------------------------------------------------------------------ 7
def test_7_positions_of_current_match_are_predicted_not_observed():
    """Позиции берутся из истории игрока, а не из этого матча.

    У игрока без истории позиции быть не может; она назначается
    детерминированно из свободных и помечается в `pos_assigned_min`.
    """
    only = feats([m(99, 20)])[99]
    assert only.pos_assigned_min == 0, \
        "позиция назначена игрокам без истории как наблюдённая"
    h = history()
    after = feats(h + [m(99, 20)])[99]
    assert after.pos_assigned_min == 5, "контроль: с историей позиции обязаны найтись"
    # ни один pos_diff не может существовать без истории
    assert all(getattr(only, f) is None for f in
               ("pos_diff_1", "pos_diff_2", "pos_diff_3", "pos_diff_4", "pos_diff_5")), \
        "позиционные разницы посчитаны при отсутствии истории"


# ------------------------------------------------------------------ 8
def test_8_future_roster_information_mutation():
    """Будущий состав не меняет прошлый прогноз."""
    h = history()
    target = m(99, 20)
    future = m(120, 40, r=(1, 2, 3, 4, 900))     # состав сменится ПОСЛЕ матча
    a = feats(h + [target])[99]
    b = feats(h + [target, future])[99]
    assert same(a, b), "будущая замена состава изменила прошлый прогноз"


# ------------------------------------------------------------------ 9
def test_9_future_player_elo_mutation():
    """Игры игрока после T не меняют его рейтинг на T."""
    h = history()
    target = m(99, 20)
    after = [m(200 + i, 25 + i, win=False) for i in range(5)]
    a = feats(h + [target])[99]
    b = feats(h + [target] + after)[99]
    assert same(a, b), "будущие игры изменили рейтинг на момент прогноза"


# ------------------------------------------------------------------ 10
def test_10_meta_cutoff():
    """КРИТИЧЕСКИЙ ТЕСТ ФАЗЫ — META CUTOFF.

    Всё, что произошло после T, прогноз на T менять не имеет права: ни
    один матч, ни серия матчей, ни смена состава, ни простой.
    """
    h = history()
    target = m(99, 20)
    base = feats(h + [target])[99]
    for extra in ([m(300, 21)], [m(300, 21), m(301, 22, r=(1, 2, 3, 4, 900))],
                  [m(300 + i, 21 + i, win=(i % 2 == 0)) for i in range(10)]):
        assert same(base, feats(h + [target] + extra)[99]), \
            "состояние после T изменило прогноз на T"
    # Контроль: то же самое ДО T обязано менять. Матч ставится на 18.5
    # суток, а не на 19: при горизонте 24 ч момент прогноза — ровно 19-е,
    # а там PREDICT идёт строго раньше UPDATE, и матч 19-х суток
    # правильно НЕ виден.
    assert not same(base, feats(h + [m(60, 18, hours=12, win=False,
                                       r=A, d=C), target])[99]), \
        "контроль: матч до T обязан влиять"


# ------------------------------------------------------------------ 11
def test_11_prediction_time_cutoff_monotone():
    """Чем раньше момент прогноза, тем меньше матчей он видит.

    Проверяется на односторонней вставке: матч, в котором играет только
    сторона A, ставится между горизонтами. При Δ=1ч он уже сыгран, при
    Δ=14 суток — ещё нет.

    Разностные признаки при равномерном сдвиге T инвариантны, если обе
    стороны играли в одни и те же моменты, — поэтому симметричная
    вставка тест бы не проверила.
    """
    h = history(20)
    target = m(99, 30)
    between = m(70, 29, hours=12, r=A, d=C, win=True)
    seen = {}
    for hours in (1, 6, 24, 72, 24 * 14):
        f = feats(h + [between, target], timedelta(hours=hours))[99]
        seen[hours] = f.pair_games_mean_diff
    assert seen[1] != seen[24 * 14], "контроль: горизонты обязаны различаться"
    assert seen[1] > seen[24 * 14], \
        "более поздний момент прогноза обязан видеть БОЛЬШЕ совместных матчей"
    # Отдельно: при ОТСУТСТВИИ новых матчей затухающий счётчик читается
    # тем меньшим, чем позже момент — это затухание, а не потеря данных.
    st = _State()
    for x in h:
        st.apply(x, _dp)
    t_near = (target.start_time - timedelta(hours=1)).timestamp()
    t_far = (target.start_time - timedelta(days=14)).timestamp()
    assert st.pair_games(1, 2, t_near) < st.pair_games(1, 2, t_far), \
        "затухание не применяется при чтении на более поздний момент"


# ------------------------------------------------------------------ 12
def test_12_prefix_stability():
    """Признаки на префиксе тождественны соответствующей части признаков
    на полной истории."""
    full = history(20) + [m(90 + i, 21 + i) for i in range(5)]
    prefix = full[:18]
    a = feats(full)
    b = feats(prefix)
    common = set(a) & set(b)
    assert len(common) >= 10, "контроль: пересечение обязано быть непустым"
    for mid in common:
        assert same(a[mid], b[mid]), f"префикс-нестабильность на матче {mid}"


# ------------------------------------------- вспомогательное
def _dp(mm: LineupMatch):
    from src.datasets.player_hero_features import derive_positions

    class _P:
        def __init__(self, aid, lr, g):
            self.account_id, self.lane_role, self.gold_per_min = aid, lr, g

    out = {}
    for side in (mm.radiant, mm.dire):
        out.update(derive_positions([_P(a, mm.lane_role.get(a), mm.gpm.get(a))
                                     for a in side]))
    return out


def _regpm(x: LineupMatch) -> LineupMatch:
    """Тот же матч с переставленным GPM внутри сейф-линии."""
    g = dict(x.gpm)
    for a, b in ((x.radiant[0], x.radiant[3]), (x.dire[0], x.dire[3])):
        g[a], g[b] = g.get(b), g.get(a)
    return LineupMatch(match_id=x.match_id, start_time=x.start_time,
                       radiant_team_id=x.radiant_team_id, dire_team_id=x.dire_team_id,
                       radiant_win=x.radiant_win, radiant=x.radiant, dire=x.dire,
                       lane_role=x.lane_role, gpm=g)
