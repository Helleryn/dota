"""
PHASE 17 — adversarial tests point-in-time движка (PART R).

Центральный инвариант фазы: признаки на момент T зависят ТОЛЬКО от
матчей, начавшихся строго раньше T. Матч, стартовавший между T и началом
прогнозируемого матча, попадать в признаки не должен — именно этим
point-in-time отличается от обычного walk-forward.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.pit.engine import (
    PitMatch,
    PointInTimeState,
    RosterProvider,
    build_point_in_time_features,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
RA = frozenset({1, 2, 3, 4, 5})
DA = frozenset({11, 12, 13, 14, 15})


def m(mid, day_offset, r_team=100, d_team=200, win=True, hours=0,
      rr=RA, dd=DA, picks=((1, 2, 3, 4, 5), (6, 7, 8, 9, 10))):
    return PitMatch(match_id=mid,
                    start_time=T0 + timedelta(days=day_offset, hours=hours),
                    radiant_team_id=r_team, dire_team_id=d_team, radiant_win=win,
                    radiant_roster=rr, dire_roster=dd,
                    radiant_picks=picks[0], dire_picks=picks[1])


def feats(matches, horizon, rp=None, **kw):
    return {f.match_id: f for f in
            build_point_in_time_features(matches, horizon, roster_provider=rp, **kw)}


# ---------- 1. главный инвариант: матч между T и стартом не виден ----------

def test_match_between_prediction_time_and_start_is_invisible():
    """КЛЮЧЕВОЙ ТЕСТ ФАЗЫ.

    Команда 100 играет матч за 12 часов до прогнозируемого. При горизонте
    24 часа этот матч ещё не состоялся к моменту прогноза и в признаки
    попасть не может; при горизонте 0 — обязан попасть.
    """
    target = m(99, 10)
    interm = m(50, 9, hours=12, win=False)          # старт за 12 ч до target
    seq = [interm, target]

    f24 = feats(seq, timedelta(hours=24))[99]
    f0 = feats(seq, timedelta(0))[99]
    only_target = feats([target], timedelta(hours=24))[99]

    assert f24.elo_difference == only_target.elo_difference, \
        "матч, стартовавший ПОСЛЕ момента прогноза, попал в признаки"
    assert f0.elo_difference != f24.elo_difference, \
        "контроль: при горизонте 0 промежуточный матч обязан влиять"


def test_prediction_at_equals_start_minus_horizon():
    f = feats([m(1, 5)], timedelta(hours=3))[1]
    assert f.prediction_at == f.start_time - timedelta(hours=3)
    assert f.horizon_hours == pytest.approx(3.0)


def test_match_starting_exactly_at_prediction_time_is_not_counted():
    """Матч, стартующий ровно в момент прогноза, исхода к этому моменту
    ещё не имеет."""
    target = m(2, 10)
    same = m(3, 9, hours=0, win=False)      # ровно T = start(target) − 24ч
    f = feats([same, target], timedelta(hours=24))[2]
    alone = feats([target], timedelta(hours=24))[2]
    assert f.elo_difference == alone.elo_difference


# ---------- 2/3. будущие результаты и рейтинги ----------

def test_future_result_does_not_change_earlier_features():
    seq = [m(i, i) for i in range(1, 8)]
    base = feats(seq, timedelta(hours=1))
    flipped = list(seq)
    flipped[5] = m(6, 6, win=False)
    after = feats(flipped, timedelta(hours=1))
    for i in range(1, 6):
        assert base[i].elo_difference == after[i].elo_difference
        assert base[i].elo_mean_diff == after[i].elo_mean_diff


def test_appending_future_matches_changes_nothing():
    seq = [m(i, i) for i in range(1, 6)]
    base = feats(seq, timedelta(hours=1))
    extra = seq + [m(90 + i, 200 + i) for i in range(3)]
    after = feats(extra, timedelta(hours=1))
    for i in range(1, 6):
        assert base[i] == after[i]


def test_longer_horizon_never_sees_more_than_shorter():
    """Признаки на большем горизонте строятся по подмножеству истории."""
    seq = [m(i, i // 2, hours=(i % 2) * 6) for i in range(1, 20)]
    f1 = feats(seq, timedelta(hours=1))
    f7 = feats(seq, timedelta(days=7))
    for k in f1:
        assert f7[k].radiant_matches_before <= f1[k].radiant_matches_before


# ---------- 4/5. player-Elo привязан к игроку, а не к team_id ----------

def test_player_elo_follows_the_player_not_the_team():
    """PART D. Игрок X уходит из команды 100 в команду 300. Его сила
    обязана уйти вместе с ним, а не остаться у team_id."""
    X = 1
    hist = [m(i, i, rr=RA, dd=DA, win=True) for i in range(1, 21)]
    # X переходит в новую команду 300 с новыми партнёрами
    moved = m(50, 30, r_team=300, rr=frozenset({X, 21, 22, 23, 24}), dd=DA)
    # состав 100 без X
    replaced = m(51, 30, hours=1, r_team=100,
                 rr=frozenset({2, 3, 4, 5, 25}), dd=DA)
    out = feats(hist + [moved, replaced], timedelta(hours=1),
                rp=RosterProvider({50: (frozenset({X, 21, 22, 23, 24}), DA),
                                   51: (frozenset({2, 3, 4, 5, 25}), DA)}))
    st = PointInTimeState()
    for x in hist:
        st.apply(x)
    assert st.p_elo(X) > st.p_elo(21), "рейтинг игрока не перенёсся вместе с ним"
    assert out[50].elo_mean_diff is not None and out[51].elo_mean_diff is not None


def test_team_identity_does_not_transfer_rating_between_people():
    st = PointInTimeState()
    for i in range(1, 16):
        st.apply(m(i, i))
    known = st.p_elo(1)
    newcomer = st.p_elo(999)
    assert newcomer == pytest.approx(1000.0), "новый игрок получил чужую историю"
    assert known != newcomer


# ---------- 6/7. состав ----------

def test_unknown_roster_yields_none_not_defaults():
    """Неизвестный состав обязан давать None, а не значения по умолчанию:
    подстановка «сегодняшнего» состава прямо запрещена фазой."""
    f = feats([m(1, 5)], timedelta(hours=1), rp=None)[1]
    assert f.elo_mean_diff is None and f.five_vs_team_elo_diff is None
    assert f.roster_known_radiant == 0 and f.roster_known_dire == 0
    assert f.elo_difference is not None, "признаки без состава обязаны считаться"


def test_partial_roster_is_not_silently_completed():
    rp = RosterProvider({1: (frozenset({1, 2, 3}), DA)})     # только трое
    f = feats([m(1, 5)], timedelta(hours=1), rp=rp)[1]
    assert f.roster_known_radiant == 3
    assert f.elo_mean_diff is not None      # считается по тому, что известно
    assert f.roster_known_radiant != 5, "неполный состав не должен «дополняться»"


def test_todays_roster_is_not_used_for_yesterday():
    """Провайдер отдаёт состав по match_id. Матч, которого в провайдере
    нет, не получает состав от соседнего матча."""
    rp = RosterProvider({2: (RA, DA)})
    out = feats([m(1, 5), m(2, 6)], timedelta(hours=1), rp=rp)
    assert out[1].elo_mean_diff is None
    assert out[2].elo_mean_diff is not None


# ---------- 8. драфт ----------

def test_pre_draft_mode_does_not_compute_hero_feature():
    f = feats([m(1, 5)], timedelta(hours=1), include_draft=False)[1]
    assert f.hero_exp_decay_diff is None, \
        "до драфта выбранных героев не существует — признак не должен вычисляться"


def test_draft_aware_mode_computes_hero_feature():
    seq = [m(i, i) for i in range(1, 30)]
    f = feats(seq, timedelta(0), include_draft=True)[29]
    assert f.hero_exp_decay_diff is not None


def test_future_draft_actions_do_not_change_features():
    """PART F: действие драфта позже prediction_at не должно влиять."""
    seq = [m(i, i) for i in range(1, 15)]
    base = feats(seq, timedelta(hours=1), include_draft=True)
    changed = list(seq)
    changed[13] = m(14, 14, picks=((90, 91, 92, 93, 94), (95, 96, 97, 98, 99)))
    after = feats(changed, timedelta(hours=1), include_draft=True)
    for i in range(1, 14):
        assert base[i].hero_exp_decay_diff == after[i].hero_exp_decay_diff


# ---------- 9. пул героев в мете ----------

def test_pool_meta_uses_past_heroes_not_this_match_draft():
    """Признак пула обязан меняться при смене ПРОШЛЫХ героев и НЕ
    меняться при смене героев самого прогнозируемого матча.

    Фикстура намеренно АСИММЕТРИЧНА: игроки 1-5 играли на героях 1-5 и
    побеждали, игроки 21-25 — на героях 6-10 и проигрывали. На
    симметричной фикстуре (обе стороны всегда одни и те же герои) обе
    половины теста прошли бы тождественно и ничего не проверяли.
    """
    STRONG = frozenset({1, 2, 3, 4, 5})
    WEAK = frozenset({21, 22, 23, 24, 25})
    hist = [m(i, i, rr=STRONG, dd=WEAK, win=True,
              picks=((1, 2, 3, 4, 5), (6, 7, 8, 9, 10))) for i in range(1, 30)]

    target = m(99, 40, rr=STRONG, dd=WEAK)
    rp = RosterProvider({99: (STRONG, WEAK)})
    base = feats(hist + [target], timedelta(hours=1), rp=rp)[99]
    assert base.pool_meta_diff is not None and base.pool_size_min > 0
    assert base.pool_meta_diff > 0, "пул победителей обязан быть сильнее"

    # 1. смена героев ЭТОГО матча не должна влиять
    other_now = m(99, 40, rr=STRONG, dd=WEAK,
                  picks=((80, 81, 82, 83, 84), (85, 86, 87, 88, 89)))
    f_now = feats(hist + [other_now], timedelta(hours=1), rp=rp)[99]
    assert base.pool_meta_diff == f_now.pool_meta_diff, \
        "признак пула использовал драфт прогнозируемого матча"

    # 2. зеркальный состав обязан дать зеркальный знак
    mirrored = m(98, 40, rr=WEAK, dd=STRONG)
    rp2 = RosterProvider({98: (WEAK, STRONG)})
    f_mirror = feats(hist + [mirrored], timedelta(hours=1), rp=rp2)[98]
    assert f_mirror.pool_meta_diff == pytest.approx(-base.pool_meta_diff), \
        "контроль: обмен составов обязан менять знак признака"


def test_pool_meta_is_none_without_roster():
    f = feats([m(1, 5)], timedelta(hours=1), rp=None)[1]
    assert f.pool_meta_diff is None and f.pool_size_min == 0


# ---------- 10. эмиссия и популяция ----------

def test_emit_only_restricts_output_not_state_updates():
    """Матч вне emit-множества обязан обновлять состояние, но не выдавать
    строку. Это ровно тот дефект, который разошёлся с замороженным
    конвейером на 2.1 Elo, пока популяции не разделили."""
    seq = [m(i, i) for i in range(1, 6)]
    full = feats(seq, timedelta(hours=1))
    partial = feats(seq, timedelta(hours=1), emit_only={5})
    assert set(partial) == {5}
    assert partial[5].elo_difference == full[5].elo_difference
    assert partial[5].radiant_matches_before == 4


def test_deterministic_across_runs():
    seq = [m(i, i) for i in range(1, 12)]
    a = feats(seq, timedelta(hours=2))
    b = feats(list(reversed(seq)), timedelta(hours=2))
    assert a == b, "результат зависит от порядка подачи — нарушена сортировка"


def test_empty_history_gives_neutral_elo():
    f = feats([m(1, 0)], timedelta(hours=1))[1]
    assert f.elo_difference == pytest.approx(0.0)
    assert f.form_3_difference is None
    assert f.radiant_matches_before == 0
