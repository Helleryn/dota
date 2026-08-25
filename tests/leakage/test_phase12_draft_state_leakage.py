"""
PHASE 12 — adversarial leakage tests для draft_state.py.

Покрывает карту утечек из reports/phase12-draft-feasibility.md, раздел 9.
Главный новый риск фазы — L1: при частичном раскрытии драфта в состояние
не должно попадать НИ ОДНО действие с большим `ord`.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence

import pytest

from src.datasets.draft_state import (
    CM_POST_734,
    CM_PRE_734,
    build_draft_state_features,
    classify_draft,
)

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class A:
    ord: int
    is_pick: bool
    hero_id: Optional[int]
    team: int


@dataclass(frozen=True)
class P:
    account_id: int
    hero_id: int
    is_radiant: bool
    lane_role: Optional[int] = 1


@dataclass(frozen=True)
class M:
    match_id: int
    start_time: datetime
    radiant_win: bool
    actions: Sequence[A]
    players: Sequence[P]


def _actions(sig: str, rad_picks, dire_picks, rad_bans, dire_bans) -> List[A]:
    """Строит действия по сигнатуре: пики/баны раздаются сторонам по кругу."""
    out, rp, dp, rb, db, turn = [], list(rad_picks), list(dire_picks), list(rad_bans), list(dire_bans), 0
    for i, ch in enumerate(sig):
        team = turn % 2
        turn += 1
        if ch == "P":
            src = rp if team == 0 else dp
        else:
            src = rb if team == 0 else db
        hero = src.pop(0) if src else None
        out.append(A(ord=i, is_pick=(ch == "P"), hero_id=hero, team=team))
    return out


def _match(mid, day, win=True, rad_bans=(60, 61, 62, 63, 64, 65),
           dire_bans=(70, 71, 72, 73, 74, 75), rad_lane=1):
    rad_picks = (10, 11, 12, 13, 14)
    dire_picks = (20, 21, 22, 23, 24)
    acts = _actions(CM_POST_734, rad_picks, dire_picks, list(rad_bans), list(dire_bans))
    players = ([P(1 + i, rad_picks[i], True, rad_lane) for i in range(5)]
               + [P(101 + i, dire_picks[i], False, 3) for i in range(5)])
    return M(mid, BASE + timedelta(days=day), win, acts, players)


def _matches(n=12):
    return [_match(i, i, win=(i % 2 == 0)) for i in range(n)]


def _snap(rows):
    return {
        r.match_id: (
            round(r.denied_comfort_diff, 9),
            round(r.denied_comfort_top_diff, 9),
            round(r.order_weighted_hero_strength_diff, 9),
            round(r.last_pick_counter_diff, 9),
            round(r.lane_balance_diff, 9),
            round(r.lane_prior_entropy_diff, 9),
        )
        for r in rows
    }


# ---------- формат драфта (раздел 3 аудита) ----------

def test_classify_recognises_both_captains_mode_eras():
    assert classify_draft(_actions(CM_PRE_734, (10, 11, 12, 13, 14), (20, 21, 22, 23, 24),
                                   [60] * 6, [70] * 6)) == "cm_pre_734"
    assert classify_draft(_actions(CM_POST_734, (10, 11, 12, 13, 14), (20, 21, 22, 23, 24),
                                   [60] * 6, [70] * 6)) == "cm_post_734"


def test_classify_rejects_degenerate_format():
    """Аудит нашёл 111 матчей с сигнатурой PPPPPPPPPP — это не Captains Mode,
    и порог `count(*) >= 20` их бы не отсеял."""
    acts = _actions("PPPPPPPPPP", (10, 11, 12, 13, 14), (20, 21, 22, 23, 24), [], [])
    assert classify_draft(acts) == "other"


def test_classify_is_order_independent_of_input_sequence():
    acts = _actions(CM_POST_734, (10, 11, 12, 13, 14), (20, 21, 22, 23, 24), [60] * 6, [70] * 6)
    assert classify_draft(list(reversed(acts))) == classify_draft(acts)


# ---------- L1: частичное раскрытие не читает будущие действия ----------

def test_hidden_actions_cannot_affect_revealed_state():
    """КЛЮЧЕВОЙ ТЕСТ ФАЗЫ. При reveal=k подмена всех действий с ord >= k
    не должна менять ни одного признака."""
    ms = _matches(8)
    K = 9
    base = _snap(build_draft_state_features(ms, reveal=K))

    mut = []
    for m in ms:
        acts = [a if a.ord < K else A(a.ord, a.is_pick, (a.hero_id or 0) + 40, 1 - a.team)
                for a in m.actions]
        mut.append(M(m.match_id, m.start_time, m.radiant_win, acts, m.players))
    after = _snap(build_draft_state_features(mut, reveal=K))

    assert base == after, "УТЕЧКА: скрытая часть драфта повлияла на раскрытое состояние"


def test_reveal_zero_gives_neutral_features():
    rows = build_draft_state_features(_matches(5), reveal=0)
    for r in rows:
        assert r.denied_comfort_diff == pytest.approx(0.0)
        assert r.order_weighted_hero_strength_diff == pytest.approx(0.0)
        assert r.last_pick_counter_diff == pytest.approx(0.0)
        assert r.revealed_actions == 0


def test_more_reveal_changes_state():
    """Обратная проверка: если раскрытие НЕ меняет признаки, значит драфт
    не читается вообще и тест на утечку проходил бы тривиально."""
    ms = _matches(10)
    early = _snap(build_draft_state_features(ms, reveal=4))
    full = _snap(build_draft_state_features(ms, reveal=None))
    assert early != full


# ---------- L3/L5: текущий матч не участвует в собственных признаках ----------

def test_future_result_does_not_change_past():
    ms = _matches(12)
    base = _snap(build_draft_state_features(ms))
    idx = 6
    m = ms[idx]
    mut = list(ms)
    mut[idx] = M(m.match_id, m.start_time, not m.radiant_win, m.actions, m.players)
    after = _snap(build_draft_state_features(mut))
    for x in ms[:idx]:
        assert base[x.match_id] == after[x.match_id], f"утечка на {x.match_id}"


def test_own_match_does_not_enter_own_pool():
    """Первый матч: у игроков нет истории, значит вырезать нечего и
    denied обязан быть ровно 0 — даже если соперник забанил ровно тех
    героев, на которых они играют В ЭТОМ матче."""
    m = _match(0, 0, rad_bans=(20, 21, 22, 23, 24, 25), dire_bans=(10, 11, 12, 13, 14, 15))
    r = build_draft_state_features([m])[0]
    assert r.denied_comfort_diff == pytest.approx(0.0)
    assert r.denied_comfort_top_diff == pytest.approx(0.0)


# ---------- L4: lane_role текущего матча — post-match ----------

def test_current_match_lane_role_does_not_affect_own_features():
    ms = _matches(10)
    base = _snap(build_draft_state_features(ms))
    idx = 9  # последний матч: влиять некуда, кроме себя самого
    m = ms[idx]
    scrambled = [P(p.account_id, p.hero_id, p.is_radiant, 2 if p.lane_role != 2 else 4)
                 for p in m.players]
    mut = list(ms)
    mut[idx] = M(m.match_id, m.start_time, m.radiant_win, m.actions, scrambled)
    after = _snap(build_draft_state_features(mut))
    assert base[m.match_id] == after[m.match_id], \
        "УТЕЧКА: post-match lane_role повлиял на признаки своего же матча"


# ---------- L6: time decay ----------

def test_appending_future_matches_changes_nothing_past():
    ms = _matches(10)
    base = _snap(build_draft_state_features(ms))
    future = [_match(900 + i, 300 + i, win=False) for i in range(4)]
    after = _snap(build_draft_state_features(ms + future))
    for x in ms:
        assert base[x.match_id] == after[x.match_id], "time-decay утечка"


def test_prefix_stability():
    ms = _matches(14)
    full = _snap(build_draft_state_features(ms))
    prefix = _snap(build_draft_state_features(ms[:7]))
    for x in ms[:7]:
        assert full[x.match_id] == prefix[x.match_id]


# ---------- содержательная проверка H1 ----------

def test_denied_comfort_has_correct_sign():
    """Radiant накапливает историю на героях 10-14. Затем соперник банит
    ровно их. denied_comfort_diff = dire_denied - rad_denied обязан стать
    ОТРИЦАТЕЛЬНЫМ: вырезали пул Radiant, значит Radiant хуже."""
    history = _matches(20)
    # решающий матч: dire (team=1) банит комфортных героев Radiant
    decisive = _match(500, 100, rad_bans=(80, 81, 82, 83, 84, 85),
                      dire_bans=(10, 11, 12, 13, 14, 15))
    rows = build_draft_state_features(history + [decisive])
    last = rows[-1]
    assert last.denied_comfort_diff < 0, "знак признака не соответствует смыслу"
    assert last.denied_comfort_top_diff < 0


def test_denied_comfort_is_symmetric():
    """Зеркальная ситуация даёт зеркальный знак — признак не смещён по сторонам."""
    history = _matches(20)
    mirrored = _match(501, 100, rad_bans=(20, 21, 22, 23, 24, 25),
                      dire_bans=(80, 81, 82, 83, 84, 85))
    rows = build_draft_state_features(history + [mirrored])
    assert rows[-1].denied_comfort_diff > 0


def test_first_match_is_neutral():
    r = build_draft_state_features([_match(0, 0)])[0]
    assert r.denied_comfort_diff == pytest.approx(0.0)
    assert r.order_weighted_hero_strength_diff == pytest.approx(0.0)
    assert r.last_pick_counter_diff == pytest.approx(0.0)
    assert r.draft_format == "cm_post_734"
