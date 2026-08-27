"""
PHASE 21 — unit и adversarial тесты контекста соперников (§17).

Восемнадцать сценариев. Устройство то же, что в Phase 18–20: у каждого
запрета парный контроль из шума, иначе тест проходил бы и на коде,
который данные игнорирует вовсе.

Три теста фаза считает критическими:

  №4/№5  историческая сила соперника — снимок НА ТОТ МОМЕНТ, а не
         сегодняшний Elo; это единственная утечка фазы, которую не ловит
         проверка порядка событий;
  №7     «нет истории» ≠ «нулевой перформанс»;
  №14    сегодняшний состав не приписывается старым встречам.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.pit.opponent_context import (
    HALF_LIFE_DAYS,
    CtxMatch,
    _Ctx,
    _expected,
    build_opponent_context,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
A, B, C, D, E = 100, 200, 300, 400, 500
RA = frozenset({1, 2, 3, 4, 5})
RB = frozenset({11, 12, 13, 14, 15})

NUM = ("oas_diff_7", "oas_diff_14", "oas_diff_30", "recent_winrate_diff_30",
       "common_opponent_delta", "h2h_residual_decayed", "h2h_winrate_decayed",
       "h2h_matches_decayed", "h2h_days_since_last", "h2h_roster_overlap",
       "h2h_residual_roster")
COUNTS = ("recent_n_min_7", "recent_n_min_14", "recent_n_min_30",
          "common_opponent_count")


def m(mid, day, hours=0.0, r=A, d=B, win=True, rr=RA, dd=RB) -> CtxMatch:
    return CtxMatch(match_id=mid, start_time=T0 + timedelta(days=day, hours=hours),
                    radiant_team_id=r, dire_team_id=d, radiant_win=win,
                    radiant_roster=rr, dire_roster=dd)


def feats(matches, horizon=timedelta(hours=24), emit=None):
    return {r.match_id: r
            for r in build_opponent_context(matches, horizon, emit_only=emit)}


def same(a, b, fields=NUM + COUNTS, rel_tol=1e-12):
    for f in fields:
        x, y = getattr(a, f), getattr(b, f)
        if x is None or y is None:
            if x is not y:
                return False
        elif not math.isclose(float(x), float(y), rel_tol=rel_tol, abs_tol=1e-12):
            return False
    return True


def history():
    """Асимметричная история: A и B играют с разными соперниками и с
    разным успехом, C — общий соперник обеих."""
    return [
        m(1, 1, r=A, d=C, win=True),
        m(2, 3, r=B, d=C, win=False),
        m(3, 5, r=A, d=D, win=True),
        m(4, 7, r=B, d=E, win=True),
        m(5, 9, r=A, d=C, win=False),
        m(6, 11, r=B, d=C, win=True),
        m(7, 13, r=A, d=B, win=True),
        m(8, 15, r=A, d=E, win=False),
        m(9, 17, r=B, d=D, win=True),
    ]


# ------------------------------------------------------------------ 1
def test_1_future_opponent_match_cannot_affect_feature():
    h = history()
    target = m(99, 30)
    later = m(50, 40, r=A, d=C, win=False)      # уже ПОСЛЕ целевого
    assert same(feats(h + [target])[99], feats(h + [target, later])[99]), \
        "матч соперника из будущего повлиял на признаки"


# ------------------------------------------------------------------ 2
def test_2_current_match_cannot_affect_feature():
    """Целевой матч не входит ни в свой recent-контекст, ни в свой H2H."""
    h = history()
    f = feats(h + [m(99, 30)])[99]
    solo = feats([m(99, 30)])[99]
    assert solo.recent_n_min_30 == 0, "матч посчитал сам себя в недавних"
    assert solo.h2h_matches_decayed == 0.0, "матч попал в собственный H2H"
    assert f.recent_n_min_30 > 0, "контроль: с историей контекст обязан быть"


# ------------------------------------------------------------------ 3
def test_3_changing_current_outcome_cannot_affect_feature():
    h = history()
    a = feats(h + [m(99, 30, win=True)])[99]
    b = feats(h + [m(99, 30, win=False)])[99]
    assert same(a, b), "исход целевого матча просочился в его признаки"
    assert a.target != b.target, "контроль: метки обязаны отличаться"


# ------------------------------------------------------------------ 4
def test_4_historical_opponent_elo_is_a_snapshot():
    """КРИТИЧЕСКИЙ ТЕСТ ФАЗЫ.

    Сила соперника фиксируется в момент сыгранного матча. Матчи
    соперника ПОСЛЕ этого меняют его текущий Elo, но не имеют права
    менять снимок, по которому считался residual.
    """
    st = _Ctx()
    st.apply(m(1, 1, r=A, d=C, win=True))
    snap = st.played[A][0].opp_elo
    elo_c_then = st.team_elo(C)

    # C играет ещё пять матчей и проваливается
    for i in range(5):
        st.apply(m(10 + i, 2 + i, r=C, d=D, win=False))

    assert st.team_elo(C) < elo_c_then - 10, "контроль: Elo соперника обязан упасть"
    assert st.played[A][0].opp_elo == snap, \
        "снимок силы соперника переписан его будущими матчами"


def test_4b_snapshot_is_taken_before_the_match_updates_elo():
    """Снимок берётся ДО применения матча, а не после."""
    st = _Ctx()
    before = st.team_elo(A)
    st.apply(m(1, 1, r=A, d=C, win=True))
    assert st.played[A][0].own_elo == before, \
        "снимок собственной силы взят ПОСЛЕ обновления рейтинга"
    assert st.team_elo(A) > before, "контроль: победа обязана поднять Elo"


# ------------------------------------------------------------------ 5
def test_5_todays_elo_cannot_leak_into_historical_residual():
    """Residual, посчитанный на T, не меняется от матчей соперника
    между тем матчем и T — потому что использует снимок, а не текущий Elo."""
    h = [m(1, 1, r=A, d=C, win=True), m(2, 2, r=B, d=D, win=True)]
    target = m(99, 20)
    # C рушится между историей и целевым матчем
    collapse = [m(30 + i, 5 + i, r=C, d=E, win=False) for i in range(5)]

    a = feats(h + [target])[99]
    b = feats(h + collapse + [target])[99]
    assert math.isclose(a.oas_diff_30 or 0.0, b.oas_diff_30 or 0.0, abs_tol=1e-12), \
        "сегодняшняя сила соперника протекла в исторический residual"


# ------------------------------------------------------------------ 6
def test_6_team_rename_does_not_manufacture_results():
    """Смена team_id не создаёт истории: у новой команды её нет."""
    h = history()
    a = feats(h + [m(99, 30, r=A, d=B)])[99]
    renamed = feats(h + [m(99, 30, r=777, d=B)])[99]
    assert renamed.recent_n_min_30 == 0, "переименованная команда унаследовала историю"
    assert renamed.h2h_matches_decayed == 0.0, "переименованная команда унаследовала H2H"
    assert a.recent_n_min_30 > 0, "контроль: у исходной команды история есть"


# ------------------------------------------------------------------ 7
def test_7_zero_history_is_not_zero_performance():
    """КРИТИЧЕСКИЙ ТЕСТ ФАЗЫ.

    «Не сыграла ни одного матча» обязано отличаться от «сыграла ровно по
    ожиданию». Первое — None, второе — 0.0.
    """
    solo = feats([m(99, 30)])[99]
    assert solo.oas_diff_30 is None, "отсутствие истории выдано как нулевой перформанс"
    assert solo.recent_n_min_30 == 0
    assert solo.common_opponent_delta is None
    assert solo.h2h_residual_decayed is None

    # контроль: команда сыграла ровно по ожиданию -> ноль, а не None
    st = _Ctx()
    st.apply(m(1, 1, r=A, d=B, win=True))
    st.apply(m(2, 2, r=A, d=B, win=False))
    val, n = st.oas(A, (T0 + timedelta(days=3)).timestamp(), 30)
    assert n == 2 and val is not None, "сыгранные матчи выданы как отсутствие данных"
    assert abs(val) < 0.2, "контроль: около нуля при размене побед"


# ------------------------------------------------------------------ 8
def test_8_common_opponent_count_is_explicit():
    """Один общий соперник и пять различаются только через явный счётчик."""
    st = _Ctx()
    st.apply(m(1, 1, r=A, d=C, win=True))
    st.apply(m(2, 2, r=B, d=C, win=False))
    now = (T0 + timedelta(days=5)).timestamp()
    d1, n1 = st.common_opponents(A, B, now)
    assert n1 == 1 and d1 is not None

    for i, opp in enumerate((D, E, 600, 700)):
        st.apply(m(10 + i * 2, 3 + i, r=A, d=opp, win=True))
        st.apply(m(11 + i * 2, 3 + i, hours=1.0, r=B, d=opp, win=False))
    now2 = (T0 + timedelta(days=10)).timestamp()
    d5, n5 = st.common_opponents(A, B, now2)
    assert n5 == 5, f"счётчик общих соперников неверен: {n5}"
    assert d5 is not None and d5 > 0, "A выигрывала у всех общих — дельта обязана быть > 0"


def test_8b_common_opponents_excludes_the_two_teams_themselves():
    """A и B не считаются общими соперниками друг друга."""
    st = _Ctx()
    st.apply(m(1, 1, r=A, d=B, win=True))
    d, n = st.common_opponents(A, B, (T0 + timedelta(days=3)).timestamp())
    assert n == 0 and d is None, "встреча пары засчитана как общий соперник"


# ------------------------------------------------------------------ 9
def test_9_h2h_cannot_use_current_match():
    h = [m(1, 1, r=A, d=B, win=True), m(2, 5, r=A, d=B, win=False)]
    f = feats(h + [m(99, 20)])[99]
    solo = feats([m(99, 20)])[99]
    assert solo.h2h_matches_decayed == 0.0, "текущий матч попал в собственный H2H"
    assert f.h2h_matches_decayed > 0.0, "контроль: прошлые встречи обязаны учитываться"


# ------------------------------------------------------------------ 10
def test_10_h2h_cannot_use_future_rematch():
    h = [m(1, 1, r=A, d=B, win=True)]
    target = m(99, 20)
    rematch = m(50, 40, r=A, d=B, win=False)
    assert same(feats(h + [target])[99], feats(h + [target, rematch])[99]), \
        "будущая переигровка изменила H2H прошлого прогноза"


# ------------------------------------------------------------------ 11
def test_11_old_h2h_decay_is_monotonic():
    """Чем старее встреча, тем меньше её вес."""
    st = _Ctx()
    st.apply(m(1, 1, r=A, d=B, win=True))
    prev = None
    for days in (2, 30, 90, 180, 365):
        now = (T0 + timedelta(days=days)).timestamp()
        n = st.head_to_head(A, B, now, RA)["n"]
        if prev is not None:
            assert n < prev, f"вес встречи не убывает на {days} днях"
        prev = n
    # Период полураспада соблюдён. Отсчёт с дня 2, а не с дня 1: матч
    # стартует ровно в день 1, а `head_to_head` берёт строго `ts < now` —
    # то самое правило «PREDICT раньше UPDATE», из-за которого в момент
    # старта встреча ещё не существует.
    n0 = st.head_to_head(A, B, (T0 + timedelta(days=2)).timestamp(), RA)["n"]
    n_hl = st.head_to_head(A, B, (T0 + timedelta(days=2 + HALF_LIFE_DAYS)).timestamp(),
                           RA)["n"]
    assert math.isclose(n_hl, n0 / 2.0, rel_tol=1e-9), "полураспад не соблюдён"


# ------------------------------------------------------------------ 12
def test_12_context_is_point_in_time():
    """Всё, что произошло после T, прогноз на T менять не имеет права."""
    h = history()
    target = m(99, 30)
    base = feats(h + [target])[99]
    for extra in ([m(300, 31, r=A, d=C)],
                  [m(300 + i, 31 + i, r=A, d=B, win=(i % 2 == 0)) for i in range(6)]):
        assert same(base, feats(h + [target] + extra)[99]), \
            "состояние после T изменило прогноз на T"
    # контроль: то же ДО T обязано менять
    assert not same(base, feats(h + [m(60, 28, r=A, d=C, win=False), target])[99]), \
        "контроль: матч до T обязан влиять"


# ------------------------------------------------------------------ 13
def test_13_roster_overlap_uses_only_prediction_time_information():
    """Пересечение считается с составом, поданным на момент T."""
    h = [m(1, 1, r=A, d=B, win=True, rr=RA)]
    full = feats(h + [m(99, 20, rr=RA)])[99]
    half = feats(h + [m(99, 20, rr=frozenset({1, 2, 3, 90, 91}))])[99]
    none = feats(h + [m(99, 20, rr=frozenset({90, 91, 92, 93, 94}))])[99]
    assert full.h2h_roster_overlap == pytest.approx(1.0)
    assert half.h2h_roster_overlap == pytest.approx(0.6)
    assert none.h2h_roster_overlap == pytest.approx(0.0)
    assert none.h2h_residual_roster is None, \
        "residual с нулевым пересечением обязан быть None, а не нулём"


# ------------------------------------------------------------------ 14
def test_14_current_roster_not_assigned_retroactively_to_old_h2h():
    """КРИТИЧЕСКИЙ ТЕСТ ФАЗЫ.

    Состав старой встречи — тот, что играл ТОГДА. Сменившийся сегодня
    состав не переписывает её задним числом.
    """
    old = frozenset({1, 2, 3, 4, 5})
    new = frozenset({1, 2, 90, 91, 92})
    h = [m(1, 1, r=A, d=B, win=True, rr=old)]
    f = feats(h + [m(99, 20, rr=new)])[99]
    assert f.h2h_roster_overlap == pytest.approx(0.4), \
        f"пересечение посчитано не по составу той встречи: {f.h2h_roster_overlap}"

    # состав старой встречи в состоянии не мутирует
    st = _Ctx()
    st.apply(m(1, 1, r=A, d=B, win=True, rr=old))
    st.apply(m(2, 10, r=A, d=C, win=True, rr=new))
    assert st.h2h[(A, B)][0].roster == old, "состав старой встречи переписан"


# ------------------------------------------------------------------ 15
def test_15_repeated_execution_is_deterministic():
    h = history()
    seq = h + [m(99, 30)]
    a, b = feats(seq)[99], feats(seq)[99]
    assert same(a, b, rel_tol=0.0), "повторный прогон дал другой ответ"


# ------------------------------------------------------------------ 16
def test_16_predict_happens_before_update():
    """Матч, стартующий РОВНО в момент прогноза, ещё не имеет исхода."""
    h = history()
    target = m(99, 30)                       # T при горизонте 24ч = день 29
    exactly_at_T = m(70, 29, r=A, d=C, win=False)
    at_T = feats(h + [exactly_at_T, target])[99]
    without = feats(h + [target])[99]
    assert same(at_T, without), "матч, стартующий в момент прогноза, попал в признаки"
    # контроль: чуть раньше — обязан попасть
    earlier = feats(h + [m(70, 28, hours=23, r=A, d=C, win=False), target])[99]
    assert not same(earlier, without), "контроль: матч до T обязан влиять"


# ------------------------------------------------------------------ 17
def test_17_no_post_draft_data_in_any_feature():
    """LEVEL 1: в структуре матча нет ни героев, ни ролей, ни драфта."""
    fields = set(CtxMatch.__dataclass_fields__)
    forbidden = {"radiant_picks", "dire_picks", "picks", "bans", "lane_role",
                 "hero_id", "gold_per_min", "duration_seconds"}
    assert not (fields & forbidden), f"post-draft поля в CtxMatch: {fields & forbidden}"
    row_fields = set(__import__("src.pit.opponent_context", fromlist=["x"])
                     .OpponentContextRow.__dataclass_fields__)
    assert not any("hero" in f or "pick" in f or "ban" in f or "role" in f
                   for f in row_fields), "post-draft признак в выдаче LEVEL 1"


# ------------------------------------------------------------------ 18
def test_18_prefix_stability():
    """Признаки на префиксе тождественны части признаков полной истории.

    Это и есть формальное «в отбор не попадает информация из будущего»:
    ни одна строка не зависит от того, что идёт дальше по времени.
    """
    full = history() + [m(90 + i, 20 + i * 2, r=A, d=B, win=(i % 2 == 0))
                        for i in range(6)]
    prefix = full[:11]
    a, b = feats(full), feats(prefix)
    common = set(a) & set(b)
    assert len(common) >= 8, "контроль: пересечение обязано быть непустым"
    for mid in common:
        assert same(a[mid], b[mid]), f"префикс-нестабильность на матче {mid}"


# ---------------------------- unit: сама формула -------------------
def test_unit_expected_is_symmetric_and_monotone():
    assert _expected(1000, 1000) == pytest.approx(0.5)
    assert _expected(1400, 1000) > 0.9
    assert _expected(1000, 1400) < 0.1
    assert _expected(1200, 1000) + _expected(1000, 1200) == pytest.approx(1.0)


def test_unit_oas_sign_matches_over_and_under_performance():
    """Победа над более сильным даёт положительный residual, поражение
    от более слабого — отрицательный."""
    st = _Ctx()
    st.apply(m(1, 1, r=C, d=D, win=True))
    st.apply(m(2, 2, r=C, d=D, win=True))
    st.apply(m(3, 3, r=C, d=D, win=True))     # C стала сильнее базы
    now = (T0 + timedelta(days=4)).timestamp()
    assert st.team_elo(C) > st.team_elo(A), "контроль фикстуры: C обязана быть сильнее"

    st.apply(m(4, 4, r=A, d=C, win=True))     # A обыграла более сильную
    up, _ = st.oas(A, (T0 + timedelta(days=5)).timestamp(), 30)
    assert up is not None and up > 0, f"победа над фаворитом дала residual {up}"

    st2 = _Ctx()
    st2.apply(m(1, 1, r=D, d=C, win=False))
    st2.apply(m(2, 2, r=D, d=C, win=False))
    st2.apply(m(3, 3, r=D, d=C, win=False))   # D слабее базы
    st2.apply(m(4, 4, r=A, d=D, win=False))   # A проиграла более слабой
    down, _ = st2.oas(A, (T0 + timedelta(days=5)).timestamp(), 30)
    assert down is not None and down < 0, f"поражение от аутсайдера дало residual {down}"


def test_unit_windows_are_nested():
    """Окно 30 дней обязано включать всё, что видит окно 7."""
    st = _Ctx()
    st.apply(m(1, 1, r=A, d=B, win=True))     # 20 дней назад
    st.apply(m(2, 18, r=A, d=C, win=True))    # 3 дня назад
    now = (T0 + timedelta(days=21)).timestamp()
    _, n7 = st.oas(A, now, 7)
    _, n14 = st.oas(A, now, 14)
    _, n30 = st.oas(A, now, 30)
    assert (n7, n14, n30) == (1, 1, 2), f"вложенность окон нарушена: {n7} {n14} {n30}"
