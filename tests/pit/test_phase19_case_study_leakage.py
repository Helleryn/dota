"""
PHASE 19 — adversarial-тесты исторического case study (PART R).

Двенадцать сценариев из задачи фазы. Устройство то же, что в Phase 18:
у каждого запрета есть парный контроль из шума, иначе тест проходил бы и
на коде, который данные игнорирует вовсе.

Отдельно проверяется структура серии Bo5: при большом Δ момент прогноза
для поздней игры лежит раньше окончания ранних, при малом — позже.
Именно поэтому «прогноз на игру 5 за 30 минут» и «прогноз на серию»
разные вещи, и смешивать их нельзя.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.pit.engine import (
    PitMatch,
    RosterProvider,
    build_point_in_time_features,
)
from src.sources.base import Provenance, SourceConfidence
from src.sources.models import ExternalTeamRef, PatchInfo, RoleConfidence, RosterMembership
from src.sources.temporal import TemporalRosterStore, patch_as_of

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
SPIRIT = frozenset({321580662, 106305042, 302214028, 218231587, 847565596})
VISION = frozenset({1044002267, 106573901, 195108598, 164199202, 73401082})
REF = ExternalTeamRef(source="test", external_id="spirit", name="Team Spirit")


def m(mid, day, hours=0.0, r_team=7119388, d_team=9572001, win=True,
      rr=SPIRIT, dd=VISION, picks=((1, 2, 3, 4, 5), (6, 7, 8, 9, 10))) -> PitMatch:
    return PitMatch(match_id=mid, start_time=T0 + timedelta(days=day, hours=hours),
                    radiant_team_id=r_team, dire_team_id=d_team, radiant_win=win,
                    radiant_roster=rr, dire_roster=dd,
                    radiant_picks=picks[0], dire_picks=picks[1])


def feats(matches, horizon, rp=None, **kw):
    return {f.match_id: f
            for f in build_point_in_time_features(matches, horizon,
                                                  roster_provider=rp, **kw)}


FULL_RP = RosterProvider({})


def rp_for(*mids) -> RosterProvider:
    return RosterProvider({mid: (SPIRIT, VISION) for mid in mids})


def history(n=12):
    """Асимметричная история: Spirit чаще выигрывает, рейтинги расходятся."""
    return [m(i, i, win=(i % 3 != 0)) for i in range(1, n + 1)]


ALL_FIELDS = ("elo_difference", "form_3_difference", "elo_mean_diff",
              "five_vs_team_elo_diff", "radiant_matches_before",
              "dire_matches_before", "pool_meta_diff")


def same_exact(a, b, fields=ALL_FIELDS):
    return all(getattr(a, f) == getattr(b, f) for f in fields)


def same(a, b, fields=ALL_FIELDS, rel_tol=1e-12):
    """Равенство признаков с допуском на последние биты.

    Допуск не косметический, и вот почему. `_DecayHeroStrength.strength`
    вызывает `_decay`, который ПИШЕТ в состояние: продвигает часы
    затухания героя. Значит чтение признака одного матча меняет
    округление при чтении признака другого — величины остаются
    математически теми же (затухание мультипликативно: f1·f2 ≡ f1 затем
    f2), но не побитово теми же.

    Утечкой это быть не может: часы двигаются только вперёд и только до
    момента прогноза, а PREDICT идут в хронологическом порядке. Но
    контракт класса «читается на PREDICT, меняется на UPDATE» нарушен, и
    тест ниже это фиксирует явно, а не прячет за допуском.
    """
    for f in fields:
        x, y = getattr(a, f), getattr(b, f)
        if x is None or y is None:
            if x is not y:
                return False
        elif isinstance(x, float) or isinstance(y, float):
            if not math.isclose(float(x), float(y), rel_tol=rel_tol, abs_tol=1e-15):
                return False
        elif x != y:
            return False
    return True


# ------------------------------------------------------------------ 1
def test_1_target_result_removed_prediction_unchanged():
    """Результат самого матча не участвует в его собственных признаках."""
    hist = history()
    win = m(99, 20, win=True)
    lose = m(99, 20, win=False)
    rp = rp_for(99)

    a = feats(hist + [win], timedelta(hours=24), rp)[99]
    b = feats(hist + [lose], timedelta(hours=24), rp)[99]

    assert same(a, b), "исход матча просочился в его собственные признаки"
    assert a.target != b.target, "контроль: сама метка обязана отличаться"


# ------------------------------------------------------------------ 2
def test_2_future_matches_removed_prediction_unchanged():
    """Матчи после T не влияют; при горизонте 0 — обязаны влиять."""
    hist = history()
    target = m(99, 20)
    later = m(50, 19, hours=12, win=False)     # старт за 12 ч до целевого
    rp = rp_for(99)

    with_later = feats(hist + [later, target], timedelta(hours=24), rp)[99]
    without = feats(hist + [target], timedelta(hours=24), rp)[99]
    h0 = feats(hist + [later, target], timedelta(0), rp)[99]

    assert same(with_later, without), "матч, стартовавший после T, попал в признаки"
    assert h0.elo_difference != without.elo_difference, \
        "контроль: при горизонте 0 он обязан влиять"


# ------------------------------------------------------------------ 3
def test_3_post_match_player_elo_forbidden():
    """player-Elo на момент T не учитывает игры, начавшиеся позже T."""
    hist = history()
    target = m(99, 20)
    between = m(50, 19, hours=12, win=False, r_team=7119388, d_team=555)
    rp = rp_for(99)

    a = feats(hist + [between, target], timedelta(hours=24), rp)[99]
    b = feats(hist + [target], timedelta(hours=24), rp)[99]
    c = feats(hist + [between, target], timedelta(0), rp)[99]

    assert a.elo_mean_diff == b.elo_mean_diff, "player-Elo обновлён будущей игрой"
    assert a.five_vs_team_elo_diff == b.five_vs_team_elo_diff
    assert c.elo_mean_diff != b.elo_mean_diff, "контроль: при горизонте 0 обязан отличаться"


# ------------------------------------------------------------------ 4
def test_4_post_match_hero_stats_forbidden():
    """Сила героев на T не знает исходов матчей, начавшихся позже T."""
    STRONG, WEAK = (1, 2, 3, 4, 5), (21, 22, 23, 24, 25)
    hist = [m(i, i, picks=(STRONG, WEAK), win=True) for i in range(1, 9)]
    target = m(99, 20, picks=(STRONG, WEAK))
    between = m(50, 19, hours=12, win=False, picks=(STRONG, WEAK), d_team=555,
                dd=frozenset({901, 902, 903, 904, 905}))
    rp = rp_for(99)

    a = feats(hist + [between, target], timedelta(hours=24), rp, include_draft=True)[99]
    b = feats(hist + [target], timedelta(hours=24), rp, include_draft=True)[99]
    c = feats(hist + [between, target], timedelta(0), rp, include_draft=True)[99]

    assert b.hero_exp_decay_diff is not None, "контроль фикстуры: значение обязано существовать"
    assert a.hero_exp_decay_diff == b.hero_exp_decay_diff, \
        "исход будущего матча изменил силу героев"
    assert c.hero_exp_decay_diff != b.hero_exp_decay_diff, \
        "контроль: при горизонте 0 обязан отличаться"


# ------------------------------------------------------------------ 5
def test_5_post_match_meta_forbidden():
    """pool_meta_diff — снимок меты на T, а не «на сегодня»."""
    STRONG, WEAK = (1, 2, 3, 4, 5), (21, 22, 23, 24, 25)
    hist = [m(i, i, picks=(STRONG, WEAK), win=True) for i in range(1, 9)]
    target = m(99, 20)
    between = m(50, 19, hours=12, win=False, picks=(STRONG, WEAK), d_team=555,
                dd=frozenset({901, 902, 903, 904, 905}))
    rp = rp_for(99)

    a = feats(hist + [between, target], timedelta(hours=24), rp)[99]
    b = feats(hist + [target], timedelta(hours=24), rp)[99]
    c = feats(hist + [between, target], timedelta(0), rp)[99]

    assert b.pool_meta_diff is not None, "контроль фикстуры: значение обязано существовать"
    assert a.pool_meta_diff == b.pool_meta_diff, "мета обновлена матчем из будущего"
    assert c.pool_meta_diff != b.pool_meta_diff, "контроль: при горизонте 0 обязан отличаться"


# ------------------------------------------------------------------ 6
def test_6_todays_roster_cannot_rewrite_historical_lineup():
    """Состав фиксируется на T и не дополняется сегодняшним."""
    hist = history()
    target = m(99, 20)
    known_then = RosterProvider({99: (frozenset({321580662, 106305042, 302214028}),
                                      VISION)})
    known_today = rp_for(99)

    a = feats(hist + [target], timedelta(hours=24), known_then)[99]
    b = feats(hist + [target], timedelta(hours=24), known_today)[99]

    assert (a.roster_known_radiant, a.roster_known_dire) == (3, 5), \
        "число известных игроков не сохранено"
    assert (b.roster_known_radiant, b.roster_known_dire) == (5, 5)
    assert a.elo_mean_diff != b.elo_mean_diff, \
        "неполный состав дал то же значение, что полный — значит он дополнен"


def test_6b_roster_known_only_after_observed_at():
    """Состав, объявленный после T, невидим прогнозу на T."""
    store = TemporalRosterStore()
    for p in ("Yatoro", "Larl", "Collapse", "not_me", "rue"):
        store.add(RosterMembership(team_ref=REF, player_id=p, player_name=p,
                                   valid_from=T0 + timedelta(days=3), valid_to=None,
                                   provenance=Provenance("test", T0 + timedelta(days=12),
                                                         SourceConfidence.MEDIUM)))
    assert store.roster_as_of(REF, T0 + timedelta(days=10)) == [], \
        "состав, объявленный на 12-й день, виден прогнозу 10-го"
    assert len(store.roster_as_of(REF, T0 + timedelta(days=13))) == 5


# ------------------------------------------------------------------ 7
def test_7_current_role_cannot_rewrite_historical_role():
    """Роль — новая запись со своим observed_at, а не мутация старой."""
    store = TemporalRosterStore()
    for pos, (vf, vt, obs) in ((1, (0, 14, 0)), (5, (14, None, 14))):
        store.add(RosterMembership(
            team_ref=REF, player_id="Yatoro", player_name="Yatoro",
            valid_from=T0 + timedelta(days=vf),
            valid_to=None if vt is None else T0 + timedelta(days=vt),
            provenance=Provenance("test", T0 + timedelta(days=obs),
                                  SourceConfidence.MEDIUM),
            position=pos, role_confidence=RoleConfidence.CONFIRMED))

    assert [x.position for x in store.roster_as_of(REF, T0 + timedelta(days=10))] == [1], \
        "сегодняшняя роль переписала историческую"
    assert [x.position for x in store.roster_as_of(REF, T0 + timedelta(days=20))] == [5]


# ------------------------------------------------------------------ 8
def test_8_future_identity_mapping_cannot_affect_historical_prediction():
    """Связь идентичностей, доказанная позже, не действует задним числом."""
    other = ExternalTeamRef(source="other", external_id="TS", name="Team Spirit")
    store = TemporalRosterStore()
    store.add(RosterMembership(REF, "Yatoro", "Yatoro", None, None,
                               Provenance("test", T0, SourceConfidence.MEDIUM)))
    store.add(RosterMembership(other, "someone_else", None, None, None,
                               Provenance("other", T0 + timedelta(days=20),
                                          SourceConfidence.MEDIUM)))

    at10 = {x.player_id for x in store.roster_as_of(REF, T0 + timedelta(days=10))}
    assert at10 == {"Yatoro"}, f"чужая запись подмешалась по совпадению имени: {at10}"
    assert store.roster_as_of(other, T0 + timedelta(days=10)) == []
    assert REF != other, "контроль: одинаковое имя не делает ссылки равными"
    assert store.roster_as_of(other, T0 + timedelta(days=25)) != []


# ------------------------------------------------------------------ 9
def test_9_draft_cannot_affect_pre_draft_prediction():
    """В режиме DRAFT_T0 драфта не существует, а не «он нейтрален»."""
    hist = history()
    a = m(99, 20, picks=((1, 2, 3, 4, 5), (6, 7, 8, 9, 10)))
    b = m(99, 20, picks=((60, 61, 62, 63, 64), (70, 71, 72, 73, 74)))
    rp = rp_for(99)

    fa = feats(hist + [a], timedelta(hours=24), rp, include_draft=False)[99]
    fb = feats(hist + [b], timedelta(hours=24), rp, include_draft=False)[99]
    assert fa.hero_exp_decay_diff is None and fb.hero_exp_decay_diff is None, \
        "признак драфта заполнен там, где драфта ещё нет"
    assert same(fa, fb, [f for f in ALL_FIELDS if f != "pool_meta_diff"]), \
        "драфт повлиял на пре-драфт признаки"

    ga = feats(hist + [a], timedelta(0), rp, include_draft=True)[99]
    gb = feats(hist + [b], timedelta(0), rp, include_draft=True)[99]
    assert ga.hero_exp_decay_diff != gb.hero_exp_decay_diff, \
        "контроль: в DRAFT_T2 драфт обязан влиять"


# ------------------------------------------------------------------ 10
def test_10_future_patch_statistics_forbidden():
    """Будущий патч не активен, даже если анонсирован."""
    def p(name, rel, obs):
        return PatchInfo(name=name, released_at=T0 + timedelta(days=rel),
                         provenance=Provenance("test", T0 + timedelta(days=obs),
                                               SourceConfidence.HIGH))
    cur, nxt, late = p("7.41", 0, 0), p("7.42", 20, 5), p("7.41b", 2, 30)

    at10 = patch_as_of([cur, nxt, late], T0 + timedelta(days=10))
    assert at10 is not None and at10.name == "7.41", \
        f"на 10-й день активным признан {at10 and at10.name}"
    assert patch_as_of([cur, nxt, late], T0 + timedelta(days=25)).name == "7.42"
    assert patch_as_of([cur, late], T0 + timedelta(days=10),
                       require_known=False).name == "7.41b"


# ------------------------------------------------------------------ 11
def test_11_moving_prediction_time_backward_cannot_freshen_features():
    """Более ранний момент прогноза не может видеть БОЛЬШЕ матчей."""
    hist = history(14)
    target = m(99, 20)
    rp = rp_for(99)

    seen = {}
    for hours in (0, 1, 3, 6, 24, 72, 24 * 10):
        f = feats(hist + [target], timedelta(hours=hours), rp)[99]
        seen[hours] = f.radiant_matches_before

    ordered = [seen[h] for h in (0, 1, 3, 6, 24, 72, 240)]
    assert all(ordered[i] >= ordered[i + 1] for i in range(len(ordered) - 1)), \
        f"число видимых матчей выросло при отходе назад во времени: {seen}"
    assert ordered[0] > ordered[-1], \
        "контроль: горизонты обязаны различаться, иначе тест пуст"


# ------------------------------------------------------------------ 12
def test_12_rerun_after_revealing_result_is_identical():
    """ГЛАВНЫЙ ТЕСТ ФАЗЫ.

    Повторный прогон после того, как исход стал известен, обязан дать
    побитово тот же ответ. Проверяется двумя способами сразу:
    детерминизмом (одинаковый вход → одинаковый выход) и независимостью
    от самого исхода (перевёрнутый результат → тот же ответ).
    """
    hist = history()
    rp = rp_for(99)
    horizon = timedelta(hours=24)

    first = feats(hist + [m(99, 20, win=True)], horizon, rp)[99]
    rerun = feats(hist + [m(99, 20, win=True)], horizon, rp)[99]
    revealed_other = feats(hist + [m(99, 20, win=False)], horizon, rp)[99]

    assert same_exact(first, rerun), \
        "повторный прогон дал другой ответ — конвейер недетерминирован"
    assert same(first, revealed_other), \
        "ответ зависит от исхода матча — раскрытие результата меняет прогноз"
    assert first.target != revealed_other.target, "контроль: метки обязаны различаться"


# ------------------------------- структура серии Bo5 -----------------
def test_series_earlier_games_visible_only_at_small_horizon():
    """Игры одной серии идут через ~1.4 ч.

    При Δ = 3 ч момент прогноза для игры 3 лежит раньше старта игры 1,
    и та невидима. При Δ = 30 мин — видима. Это не утечка: игра
    действительно состоялась. Тест закрепляет, что «прогноз на игру 5 за
    30 минут» и «прогноз на серию» — разные вещи.
    """
    hist = history()
    g1 = m(101, 20, hours=0.0, win=True)
    g2 = m(102, 20, hours=1.4, win=False)
    g3 = m(103, 20, hours=3.0, win=True)
    rp = rp_for(101, 102, 103)
    seq = hist + [g1, g2, g3]

    far = feats(seq, timedelta(hours=3), rp)[103]
    near = feats(seq, timedelta(minutes=30), rp)[103]
    alone = feats(hist + [g3], timedelta(hours=3), rp)[103]

    assert same(far, alone), "при Δ=3ч ранние игры серии не должны быть видны"
    assert near.radiant_matches_before > far.radiant_matches_before, \
        "при Δ=30мин ранние игры серии обязаны быть видны"


# ------------------- находка фазы: чтение мутирует состояние ----------
def test_reading_hero_strength_mutates_decay_clock():
    """НАХОДКА PHASE 19, зафиксирована, а не исправлена.

    `PointInTimeState` объявляет: «читается на PREDICT, меняется на
    UPDATE». Это не так. `strength()` вызывает `_decay()`, который пишет
    в `_w`, `_g` и `_t`. Проверяется прямо: два чтения подряд сдвигают
    часы героя.

    Последствие — расхождение в последних битах между прогонами с разным
    составом выдачи (см. допуск в `same`). Утечкой оно быть не может:
    часы идут только вперёд и только до момента прогноза. Но контракт
    нарушен, и чинить это внутри Phase 19 нельзя: движок заморожен, а
    любая правка сдвинула бы округление во всех прошлых числах. Вынесено
    в RESEARCH FURTHER.
    """
    from src.pit.engine import _DecayHeroStrength

    h = _DecayHeroStrength()
    h.observe(1, True, 0.0)
    before = dict(h._t)
    h.strength(1, 86400.0 * 30)
    after = dict(h._t)

    assert before != after, \
        "если это упало — контракт наконец соблюдён, и допуск в same можно убирать"

    # математическая эквивалентность: два шага затухания == один
    a = _DecayHeroStrength(); a.observe(1, True, 0.0)
    b = _DecayHeroStrength(); b.observe(1, True, 0.0)
    a.strength(1, 86400.0 * 15)
    v_two = a.strength(1, 86400.0 * 30)
    v_one = b.strength(1, 86400.0 * 30)
    assert math.isclose(v_two, v_one, rel_tol=1e-12), \
        "затухание перестало быть мультипликативным — это уже настоящая ошибка"
