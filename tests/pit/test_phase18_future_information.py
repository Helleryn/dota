"""
PHASE 18 — adversarial-тесты «никакой информации из будущего» (PART L).

Восемь сценариев, по одному на каждый способ, которым будущее может
просочиться в pre-match прогноз. Устройство каждого теста одинаково:

    строим прогноз на момент T →
    добавляем факт, ставший известным ПОСЛЕ T →
    прогноз обязан остаться байт-в-байт прежним →
    контроль: при сдвиге T за этот факт прогноз обязан измениться

Второй пункт существеннее первого. Тест, который проверяет только
«ничего не изменилось», проходит и на коде, который вообще игнорирует
данные, — поэтому у каждого сценария есть парный контроль из шума.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.pit.engine import (
    PitMatch,
    RosterProvider,
    build_point_in_time_features,
)
from src.sources.base import Provenance, SourceConfidence
from src.sources.models import (
    ExternalTeamRef,
    PatchInfo,
    RoleConfidence,
    RosterMembership,
    UpcomingMatch,
)
from src.sources.resolution import resolve_roster
from src.sources.temporal import TemporalRosterStore, patch_as_of

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
NAVI = ExternalTeamRef(source="bo3gg", external_id="navi", name="Natus Vincere")


def prov(observed_days: float, source: str = "bo3gg",
         conf: SourceConfidence = SourceConfidence.MEDIUM) -> Provenance:
    return Provenance(source=source, observed_at=T0 + timedelta(days=observed_days),
                      confidence=conf)


def member(player: str, valid_from_days, valid_to_days, observed_days,
           ref: ExternalTeamRef = NAVI, position=None,
           role_confidence=RoleConfidence.UNKNOWN,
           source="bo3gg", conf=SourceConfidence.MEDIUM) -> RosterMembership:
    def d(x):
        return None if x is None else T0 + timedelta(days=x)
    return RosterMembership(team_ref=ref, player_id=player, player_name=player,
                            valid_from=d(valid_from_days), valid_to=d(valid_to_days),
                            provenance=prov(observed_days, source, conf),
                            position=position, role_confidence=role_confidence)


def pit(mid, day, r_team=100, d_team=200, win=True,
        rr=frozenset({1, 2, 3, 4, 5}), dd=frozenset({11, 12, 13, 14, 15}),
        picks=((1, 2, 3, 4, 5), (6, 7, 8, 9, 10))) -> PitMatch:
    return PitMatch(match_id=mid, start_time=T0 + timedelta(days=day),
                    radiant_team_id=r_team, dire_team_id=d_team, radiant_win=win,
                    radiant_roster=rr, dire_roster=dd,
                    radiant_picks=picks[0], dire_picks=picks[1])


def feats(matches, horizon, rp=None, **kw):
    return {f.match_id: f for f in
            build_point_in_time_features(matches, horizon, roster_provider=rp, **kw)}


# ---------------------------------------------------------------- 1
def test_future_roster_announcement_is_invisible():
    """1. Объявление состава, опубликованное ПОСЛЕ момента прогноза.

    Состав действовал уже неделю, но мы узнали о нём только на 12-й день.
    Прогноз на 10-й день права им пользоваться не имеет.
    """
    store = TemporalRosterStore()
    for p in ("a", "b", "c", "d", "e"):
        store.add(member(p, valid_from_days=3, valid_to_days=None, observed_days=12))

    t_predict = T0 + timedelta(days=10)
    assert store.roster_as_of(NAVI, t_predict) == [], \
        "состав, объявленный на 12-й день, виден прогнозу 10-го дня"

    # контроль: факт действительно есть — он просто ещё не известен
    assert len(store.roster_as_of(NAVI, t_predict, require_known=False)) == 5
    assert len(store.roster_as_of(NAVI, T0 + timedelta(days=13))) == 5


# ---------------------------------------------------------------- 2
def test_future_player_transfer_does_not_rewrite_past_roster():
    """2. Трансфер, произошедший после прогноза.

    Игрок `old` уходит на 15-й день, `new` приходит тогда же, и обе
    записи мы видим на 15-й день. Прогноз 10-го дня обязан видеть старый
    состав, а не новый и не оба сразу.
    """
    store = TemporalRosterStore()
    store.add(member("old", valid_from_days=0, valid_to_days=15, observed_days=0))
    store.add(member("new", valid_from_days=15, valid_to_days=None, observed_days=15))

    at10 = {m.player_id for m in store.roster_as_of(NAVI, T0 + timedelta(days=10))}
    at20 = {m.player_id for m in store.roster_as_of(NAVI, T0 + timedelta(days=20))}

    assert at10 == {"old"}, f"трансфер из будущего протёк в прошлое: {at10}"
    assert at20 == {"new"}, f"контроль: после трансфера состав обязан смениться: {at20}"


# ---------------------------------------------------------------- 3
def test_future_role_change_is_invisible():
    """3. Смена роли, ставшая известной после прогноза.

    Роль хранится на членстве, поэтому смена роли — это новая запись с
    более поздним `observed_at`, а не мутация старой.
    """
    store = TemporalRosterStore()
    store.add(member("carry", valid_from_days=0, valid_to_days=14, observed_days=0,
                     position=1, role_confidence=RoleConfidence.CONFIRMED))
    store.add(member("carry", valid_from_days=14, valid_to_days=None, observed_days=14,
                     position=5, role_confidence=RoleConfidence.CONFIRMED))

    at10 = store.roster_as_of(NAVI, T0 + timedelta(days=10))
    at20 = store.roster_as_of(NAVI, T0 + timedelta(days=20))

    assert [m.position for m in at10] == [1], "будущая роль видна прогнозу прошлого"
    assert [m.position for m in at20] == [5], "контроль: роль обязана смениться"


# ---------------------------------------------------------------- 4
def test_future_patch_is_never_active():
    """4. Патч, анонсированный, но не вышедший к моменту прогноза.

    Даже если релиз уже объявлен, матч играется на текущем патче. Два
    независимых барьера: `active_at` (ещё не вышел) и `observed_at`
    (ещё не знали).
    """
    cur = PatchInfo(name="7.40", released_at=T0, provenance=prov(0))
    nxt = PatchInfo(name="7.41", released_at=T0 + timedelta(days=20),
                    provenance=prov(5))          # анонсирован заранее
    late = PatchInfo(name="7.40b", released_at=T0 + timedelta(days=2),
                     provenance=prov(30))        # вышел давно, узнали поздно

    at10 = patch_as_of([cur, nxt, late], T0 + timedelta(days=10))
    assert at10 is not None and at10.name == "7.40", \
        f"на 10-й день активным признан {at10 and at10.name}"

    # контроли
    assert patch_as_of([cur, nxt, late], T0 + timedelta(days=25)).name == "7.41"
    assert patch_as_of([cur, late], T0 + timedelta(days=10),
                       require_known=False).name == "7.40b"


# ---------------------------------------------------------------- 5
def test_future_hero_statistics_do_not_enter_pool_meta():
    """5. Статистика героев из матчей, сыгранных после момента прогноза.

    `pool_meta_diff` — сила пула героев команды в текущей мете. Матч,
    начавшийся между T и стартом прогнозируемого, менять её не вправе.

    Асимметрия обязательна: если обе стороны играют одних героев,
    величина сокращается и тест проходит на любом коде.
    """
    STRONG = (1, 2, 3, 4, 5)
    WEAK = (21, 22, 23, 24, 25)

    hist = [pit(i, i, r_team=100, d_team=200, win=True,
                picks=(STRONG, WEAK)) for i in range(1, 9)]
    target = pit(99, 20, r_team=100, d_team=200)
    # матч за 12 часов до прогнозируемого: при горизонте 24 ч он ещё не сыгран
    between = PitMatch(match_id=50, start_time=T0 + timedelta(days=19, hours=12),
                       radiant_team_id=100, dire_team_id=300, radiant_win=False,
                       radiant_roster=frozenset({1, 2, 3, 4, 5}),
                       dire_roster=frozenset({31, 32, 33, 34, 35}),
                       radiant_picks=STRONG, dire_picks=WEAK)
    rp = RosterProvider({99: (frozenset({1, 2, 3, 4, 5}),
                              frozenset({11, 12, 13, 14, 15}))})

    with_between = feats(hist + [between, target], timedelta(hours=24), rp)[99]
    without = feats(hist + [target], timedelta(hours=24), rp)[99]
    h0 = feats(hist + [between, target], timedelta(0), rp)[99]

    assert without.pool_meta_diff is not None, \
        "контроль фикстуры: без значения тест проверял бы None == None"

    assert with_between.pool_meta_diff == without.pool_meta_diff, \
        "герои из ещё не сыгранного матча попали в pool_meta_diff"
    assert h0.pool_meta_diff != without.pool_meta_diff, \
        "контроль: при горизонте 0 этот матч обязан влиять"


# ---------------------------------------------------------------- 6
def test_future_match_result_does_not_enter_elo():
    """6. Результат матча, состоявшегося после момента прогноза.

    Тот же барьер, что в Phase 17, но проверенный на всех признаках
    сразу, а не только на `elo_difference`.
    """
    hist = [pit(i, i) for i in range(1, 6)]
    target = pit(99, 20)
    between = PitMatch(match_id=50, start_time=T0 + timedelta(days=19, hours=12),
                       radiant_team_id=100, dire_team_id=200, radiant_win=False,
                       radiant_roster=frozenset({1, 2, 3, 4, 5}),
                       dire_roster=frozenset({11, 12, 13, 14, 15}),
                       radiant_picks=(1, 2, 3, 4, 5), dire_picks=(6, 7, 8, 9, 10))
    rp = RosterProvider({99: (frozenset({1, 2, 3, 4, 5}),
                              frozenset({11, 12, 13, 14, 15}))})

    a = feats(hist + [between, target], timedelta(hours=24), rp)[99]
    b = feats(hist + [target], timedelta(hours=24), rp)[99]
    c = feats(hist + [between, target], timedelta(0), rp)[99]

    for field in ("elo_difference", "form_3_difference", "elo_mean_diff",
                  "five_vs_team_elo_diff", "radiant_matches_before"):
        assert getattr(a, field) == getattr(b, field), \
            f"результат будущего матча протёк в {field}"
    assert c.elo_difference != b.elo_difference, \
        "контроль: при горизонте 0 матч обязан влиять"


# ---------------------------------------------------------------- 7
def test_future_team_identity_resolution_is_not_retroactive():
    """7. Сопоставление идентичностей, выполненное после прогноза.

    Две записи об одной команде из разных источников хранятся раздельно
    до тех пор, пока связь не доказана. Доказательство, полученное на
    20-й день, не делает их одной командой на 10-й.

    Хранилище ключует команду парой (источник, внешний id) — именно
    поэтому запрос по одной ссылке не подхватывает вторую.
    """
    ref_bo3 = ExternalTeamRef(source="bo3gg", external_id="navi", name="Natus Vincere")
    ref_liq = ExternalTeamRef(source="liquipedia", external_id="Natus_Vincere",
                              name="Natus Vincere")

    store = TemporalRosterStore()
    for p in ("a", "b", "c", "d", "e"):
        store.add(member(p, 0, None, 0, ref=ref_bo3))
    for p in ("a", "b", "c", "d", "x"):
        store.add(member(p, 0, None, 20, ref=ref_liq, source="liquipedia"))

    at10 = {m.player_id for m in store.roster_as_of(ref_bo3, T0 + timedelta(days=10))}
    assert at10 == {"a", "b", "c", "d", "e"}, \
        f"состав другого источника подмешался по совпадению имени: {at10}"
    assert store.roster_as_of(ref_liq, T0 + timedelta(days=10)) == [], \
        "запись liquipedia, наблюдённая на 20-й день, видна на 10-й"

    # контроль: одинаковое имя НЕ делает ссылки равными
    assert ref_bo3 != ref_liq
    assert store.roster_as_of(ref_liq, T0 + timedelta(days=25)) != []


def test_conflicting_identities_end_in_explicit_unknown():
    """7b. Расхождение источников не разрешается молчанием.

    Два источника с ОДИНАКОВЫМ доверием называют разные составы. Система
    обязана вернуть UNKNOWN с описанием конфликта, а не выбрать один.
    """
    a = [member(p, 0, None, 0, source="bo3gg") for p in ("a", "b", "c", "d", "e")]
    b = [member(p, 0, None, 0, source="liquipedia") for p in ("a", "b", "c", "d", "x")]
    r = resolve_roster({"bo3gg": a, "liquipedia": b})

    assert r.is_unknown, f"конфликт разрешён молча правилом «{r.rule}»"
    assert r.conflict and "расходятся" in r.conflict

    # контроль: официальный источник конфликт разрешает
    hi = [member(p, 0, None, 0, source="valve", conf=SourceConfidence.HIGH)
          for p in ("a", "b", "c", "d", "z")]
    r2 = resolve_roster({"bo3gg": a, "valve": hi})
    assert not r2.is_unknown
    assert {m.player_id for m in r2.resolved} == {"a", "b", "c", "d", "z"}


# ---------------------------------------------------------------- 8
def test_future_schedule_correction_does_not_change_past_prediction():
    """8. Исправление расписания, пришедшее после прогноза.

    Матч перенесли на сутки, и мы узнали об этом на следующий день.
    Прогноз, сделанный до этого, строился на старом времени; переписать
    его задним числом нельзя — объекты неизменяемы, а исправление
    приходит отдельной записью со своим `observed_at`.
    """
    v1 = UpcomingMatch(source="bo3gg", external_id="m1",
                       scheduled_start=T0 + timedelta(days=10),
                       team_a=NAVI, team_b=ExternalTeamRef("bo3gg", "og"),
                       provenance=prov(8))
    v2 = UpcomingMatch(source="bo3gg", external_id="m1",
                       scheduled_start=T0 + timedelta(days=11),
                       team_a=NAVI, team_b=ExternalTeamRef("bo3gg", "og"),
                       provenance=prov(9), status="POSTPONED")

    t_predict = T0 + timedelta(days=8, hours=1)
    known = [m for m in (v1, v2) if m.provenance.observed_at <= t_predict]
    assert [m.scheduled_start for m in known] == [v1.scheduled_start], \
        "исправление расписания из будущего видно прогнозу"

    # объект неизменяем: «исправить» v1 на месте невозможно
    with pytest.raises(Exception):
        v1.scheduled_start = v2.scheduled_start          # type: ignore[misc]

    # контроль: после observed_at исправление видно и меняет запас времени
    later = T0 + timedelta(days=9, hours=1)
    known2 = [m for m in (v1, v2) if m.provenance.observed_at <= later]
    assert len(known2) == 2
    assert known2[-1].lead_hours(later) > v1.lead_hours(later)
    assert known2[-1].status == "POSTPONED"


# ------------------------------------------------- сквозная проверка
def test_partial_roster_is_never_completed_from_today():
    """Сквозной инвариант Phase 18: неполный состав НЕ дополняется.

    Рабочий режим фазы — k < 5 известных игроков. Соблазн подставить
    недостающих из сегодняшнего состава — это ровно утечка из будущего,
    поэтому провайдер обязан отдавать ровно то, что известно, а число
    известных обязано быть видимым в признаках.
    """
    # Пятёрка обязана быть НЕОДНОРОДНОЙ по рейтингу: player-Elo двигает всех
    # пятерых на общую дельту, поэтому у игроков с одинаковой историей
    # рейтинги совпадают, и среднее по двоим равнялось бы среднему по пятерым
    # на любом коде. Игроки 1–2 выигрывают, 3–5 проигрывают.
    hist = [
        pit(1, 1, win=True, rr=frozenset({1, 2, 90, 91, 92})),
        pit(2, 2, win=False, rr=frozenset({3, 4, 5, 93, 94})),
        pit(3, 3, win=True, rr=frozenset({1, 2, 90, 91, 92})),
    ]
    target = pit(99, 20)
    partial = RosterProvider({99: (frozenset({1, 2}), frozenset({11, 12, 13}))})
    full = RosterProvider({99: (frozenset({1, 2, 3, 4, 5}),
                                frozenset({11, 12, 13, 14, 15}))})

    p = feats(hist + [target], timedelta(hours=24), partial)[99]
    f = feats(hist + [target], timedelta(hours=24), full)[99]

    assert (p.roster_known_radiant, p.roster_known_dire) == (2, 3), \
        "число известных игроков не сохранено в признаках"
    assert (f.roster_known_radiant, f.roster_known_dire) == (5, 5)
    assert p.elo_mean_diff != f.elo_mean_diff, \
        "неполный состав дал тот же признак, что полный — значит он дополнен"
