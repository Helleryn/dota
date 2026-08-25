"""
Phase 8 — adversarial leakage tests для player_features.py и
meta_features.py (раздел 56 задания, все 5 требуемых сценариев).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import FrozenSet, Tuple

import pytest

from src.datasets.meta_features import build_meta_features
from src.datasets.player_features import build_player_features


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
    radiant_team_id: int
    dire_team_id: int
    radiant_picks: Tuple[int, ...]
    dire_picks: Tuple[int, ...]
    radiant_roster: Tuple[int, ...]
    dire_roster: Tuple[int, ...]
    radiant_win: bool


BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _roster_matches(n=12):
    a = frozenset({1, 2, 3, 4, 5})
    b = frozenset({10, 11, 12, 13, 14})
    return [MR(i, BASE + timedelta(days=i), 100, 200, a, b, i % 2 == 0) for i in range(n)]


def _draft_matches(n=12):
    return [
        MD(
            match_id=i,
            start_time=BASE + timedelta(days=i),
            radiant_team_id=100,
            dire_team_id=200,
            radiant_picks=(1, 2, 3, 4, 5),
            dire_picks=(6, 7, 8, 9, 10),
            radiant_roster=(1, 2, 3, 4, 5),
            dire_roster=(10, 11, 12, 13, 14),
            radiant_win=i % 2 == 0,
        )
        for i in range(n)
    ]


def _psnap(rows):
    return {
        r.match_id: (
            round(r.radiant_player_elo_mean, 9), round(r.dire_player_elo_mean, 9),
            round(r.radiant_player_elo_min, 9), round(r.dire_player_elo_min, 9),
            r.radiant_roster_vs_team_delta, r.dire_roster_vs_team_delta,
            r.radiant_pair_synergy, r.dire_pair_synergy,
            r.radiant_player_matches_min, r.dire_player_matches_min,
        )
        for r in rows
    }


def _msnap(rows):
    return {
        r.match_id: (
            round(r.radiant_hero_strength_decayed, 9), round(r.dire_hero_strength_decayed, 9),
            round(r.radiant_hero_synergy, 9), round(r.dire_hero_synergy, 9),
            round(r.counter_advantage, 9),
            round(r.radiant_player_hero_proficiency, 9), round(r.dire_player_hero_proficiency, 9),
        )
        for r in rows
    }


# --- Сценарий 1: изменение результата БУДУЩЕГО матча не меняет прошлые признаки ---

def test_player_future_result_change_does_not_affect_past():
    ms = _roster_matches()
    base = _psnap(build_player_features(ms))
    mid = 6
    mutated = list(ms)
    m = ms[mid]
    mutated[mid] = MR(m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id,
                      m.radiant_roster, m.dire_roster, not m.radiant_win)
    mut = _psnap(build_player_features(mutated))
    for x in ms[:mid]:
        assert base[x.match_id] == mut[x.match_id], f"УТЕЧКА player на match_id={x.match_id}"


def test_meta_future_result_change_does_not_affect_past():
    ms = _draft_matches()
    base = _msnap(build_meta_features(ms))
    mid = 6
    mutated = list(ms)
    m = ms[mid]
    mutated[mid] = MD(m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id,
                      m.radiant_picks, m.dire_picks, m.radiant_roster, m.dire_roster, not m.radiant_win)
    mut = _msnap(build_meta_features(mutated))
    for x in ms[:mid]:
        assert base[x.match_id] == mut[x.match_id], f"УТЕЧКА meta на match_id={x.match_id}"


# --- Сценарий 4: добавление матчей ПОСЛЕ prediction timestamp ничего не меняет ---

def test_player_appending_future_match_does_not_change_existing():
    ms = _roster_matches()
    base = _psnap(build_player_features(ms))
    future = MR(999, ms[-1].start_time + timedelta(days=30), 100, 200,
                frozenset({1, 2, 3, 4, 99}), frozenset({10, 11, 12, 13, 14}), True)
    mut = _psnap(build_player_features(ms + [future]))
    for x in ms:
        assert base[x.match_id] == mut[x.match_id]


def test_meta_appending_future_match_does_not_change_existing():
    ms = _draft_matches()
    base = _msnap(build_meta_features(ms))
    future = MD(999, ms[-1].start_time + timedelta(days=30), 100, 200,
                (11, 12, 13, 14, 15), (16, 17, 18, 19, 20), (1, 2, 3, 4, 5), (10, 11, 12, 13, 14), True)
    mut = _msnap(build_meta_features(ms + [future]))
    for x in ms:
        assert base[x.match_id] == mut[x.match_id]


# --- Сценарий 2: изменение БУДУЩЕГО состава/игроков не меняет прошлый prediction ---

def test_future_roster_change_does_not_affect_past_player_features():
    ms = _roster_matches()
    base = _psnap(build_player_features(ms))
    mid = 8
    mutated = list(ms)
    m = ms[mid]
    mutated[mid] = MR(m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id,
                      frozenset({1, 2, 3, 4, 77}), m.dire_roster, m.radiant_win)  # замена игрока в будущем
    mut = _psnap(build_player_features(mutated))
    for x in ms[:mid]:
        assert base[x.match_id] == mut[x.match_id]


# --- Сценарий 3: изменение будущего hero win rate не меняет прошлый draft feature ---

def test_future_hero_results_do_not_affect_past_meta_features():
    ms = _draft_matches()
    base = _msnap(build_meta_features(ms))
    mid = 7
    mutated = list(ms)
    m = ms[mid]
    # тот же герой, другой исход в БУДУЩЕМ -> его winrate изменится, но только вперёд
    mutated[mid] = MD(m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id,
                      m.radiant_picks, m.dire_picks, m.radiant_roster, m.dire_roster, not m.radiant_win)
    mut = _msnap(build_meta_features(mutated))
    for x in ms[:mid]:
        assert base[x.match_id] == mut[x.match_id]


# --- Сценарий 5: первый матч не имеет истории (нет тайного доступа к текущему матчу) ---

def test_first_match_has_no_information():
    rows = build_player_features(_roster_matches(3))
    r0 = rows[0]
    assert r0.radiant_player_elo_mean == 1000.0
    assert r0.dire_player_elo_mean == 1000.0
    assert r0.radiant_player_matches_min == 0
    assert r0.radiant_pair_synergy == pytest.approx(0.0)  # shrinkage при games=0 -> ровно 0.5 -> отклонение 0
    assert r0.radiant_roster_vs_team_delta == pytest.approx(0.0)

    mrows = build_meta_features(_draft_matches(3))
    m0 = mrows[0]
    assert m0.radiant_hero_strength_decayed == pytest.approx(0.0)
    assert m0.dire_hero_strength_decayed == pytest.approx(0.0)
    assert m0.counter_advantage == pytest.approx(0.0)
    assert m0.radiant_player_hero_proficiency == pytest.approx(0.0)


# --- Специальный тест раздела 57: prediction реагирует на замену игрока ---

def test_player_elo_follows_player_across_teams():
    """Ключевое свойство, ради которого построен player-Elo: рейтинг едет
    ЗА ИГРОКОМ при переходе в другую команду (team-Elo так не умеет)."""
    strong = frozenset({1, 2, 3, 4, 5})
    weak = frozenset({10, 11, 12, 13, 14})
    ms = [MR(i, BASE + timedelta(days=i), 100, 200, strong, weak, True) for i in range(20)]
    # Игрок 1 (из сильной команды) переходит в третью команду 300
    ms.append(MR(100, BASE + timedelta(days=40), 300, 400,
                 frozenset({1, 50, 51, 52, 53}), frozenset({60, 61, 62, 63, 64}), True))
    rows = build_player_features(ms)
    last = rows[-1]
    # Команда 300 никогда не играла (team-Elo = 1000), но в её составе игрок
    # с высоким личным рейтингом -> сила пятёрки ВЫШЕ базовой.
    assert last.radiant_player_elo_mean > 1000.0, "рейтинг игрока не перенёсся в новую команду"
    assert last.radiant_player_elo_max > 1050.0
    # И это отражается в roster_vs_team_delta (пятёрка сильнее, чем история team_id)
    assert last.radiant_roster_vs_team_delta > 0.0


def test_roster_replacement_changes_strength_estimate():
    """Раздел 57: prediction должен РЕАГИРОВАТЬ на замену (не обязательно в
    сторону улучшения — только реагировать)."""
    strong_five = frozenset({1, 2, 3, 4, 5})
    weak = frozenset({10, 11, 12, 13, 14})
    ms = [MR(i, BASE + timedelta(days=i), 100, 200, strong_five, weak, True) for i in range(20)]

    without_change = ms + [MR(100, BASE + timedelta(days=40), 100, 200, strong_five, weak, True)]
    # заменяем игрока 5 на новичка 90 (рейтинг 1000, ниже накопленного)
    with_change = ms + [MR(100, BASE + timedelta(days=40), 100, 200,
                           frozenset({1, 2, 3, 4, 90}), weak, True)]

    a = build_player_features(without_change)[-1]
    b = build_player_features(with_change)[-1]
    assert b.radiant_player_elo_mean < a.radiant_player_elo_mean, "замена не отразилась на силе пятёрки"
    assert b.radiant_player_matches_min == 0  # новичок без истории
