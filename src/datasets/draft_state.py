"""
PHASE 12 — состояние драфта: порядок действий, баны против конкретного
соперника, композиция ролей.

## Чем это отличается от Phase 7-9

Пять предыдущих фаз работали с драфтом как с НЕУПОРЯДОЧЕННЫМ множеством
пяти героев. Поле `picks_bans.ord` не использовалось ни одним модулем, а
баны учитывались только глобально (`ban_rate`, `contest_rate`, вердикт
Phase 9 — REMOVE). Здесь используется порядок и адресность действий.

## Почему нельзя опираться на сырой `ord` (аудит Phase 12, раздел 3)

Captains Mode менял формат на патче 7.34 (2023-08-08). Измерено:
`ord=4` — это ПИК в 2021-2022 и БАН в 2024-2026. Два формата делят выборку
почти пополам (58 959 против 57 444). Поэтому состояние драфта здесь
описывается парой **(раскрыто банов, раскрыто пиков)**, а не номером
действия: такое описание одинаково означает одно и то же в обеих эрах.

## Гипотезы, которые проверяет модуль (сформулированы до экспериментов,
   см. reports/phase12-draft-feasibility.md, раздел 8)

* **H1 `denied_comfort`** — доля пула героев команды, вырезанная банами
  СОПЕРНИКА. Свидетельство до моделирования: бан попадает в top-10 пула
  соперника в 23.09% случаев против 12.34% для собственного пула
  (случайный baseline 7.94%) — разрыв в 1.87 раза не объясняется общей
  популярностью героя.
* **H2 `order_weighted_hero_strength`** — ранний пик делается вслепую и по
  силе, поздний — под контрпик; вклад героя взвешивается порядком.
* **H3 `last_pick_counter`** — контрпик применим только там, где соперник
  уже раскрыт, то есть на последних пиках.
* **H4 `lane_composition`** — состав линий восстановим лишь вероятностно:
  только 53 из 127 героев закреплены за линией в >=80% случаев.

## Границы утечки

Все счётчики читаются ДО матча и обновляются строго ПОСЛЕ фиксации
признаков. `lane_role` — post-match (Phase 11), поэтому приор «герой ->
линия» строится только по матчам < t. При частичном раскрытии драфта
(`reveal`) действия с большим порядковым номером не читаются вообще —
формальная проверка в tests/leakage/test_phase12_draft_state_leakage.py.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

HALF_LIFE_DAYS = 90.0
HERO_PRIOR_GAMES = 50.0
MATCHUP_PRIOR_GAMES = 30.0
LANE_PRIOR_GAMES = 20.0
# tau для H2: вес пика падает вдвое примерно каждые 2 позиции
ORDER_WEIGHT_TAU = 2.0

# Сигнатуры Captains Mode, подтверждённые аудитом (P=pick, B=ban по ord).
CM_PRE_734 = "BBBBPPPPBBBBBBPPPPBBBBPP"
CM_POST_734 = "BBBBBBBPPBBBPPPPPPBBBBPP"
_KNOWN_CM = {
    CM_PRE_734: "cm_pre_734",
    CM_POST_734: "cm_post_734",
    "BBBBPPPPBBBBBBPPPPBBBPP": "cm_pre_734",    # 23 действия, тот же порядок фаз
    "BBBBBBBPPBBBPPPPPPBBBPP": "cm_post_734",
}


class DraftAction(Protocol):
    ord: int
    is_pick: bool
    hero_id: Optional[int]
    team: int          # 0 = radiant (проверено на 1 172 689 пиках)


class PlayerInMatch(Protocol):
    account_id: int
    hero_id: int
    is_radiant: bool
    lane_role: Optional[int]


class MatchWithDraftState(Protocol):
    match_id: int
    start_time: datetime
    radiant_win: bool
    actions: Sequence[DraftAction]
    players: Sequence[PlayerInMatch]


def classify_draft(actions: Sequence[DraftAction]) -> str:
    """Формат драфта по фактической сигнатуре P/B, а не по числу действий.

    Порог `count(*) >= 20` пропускает вырожденные форматы (аудит нашёл
    111 матчей `PPPPPPPPPP` — это не Captains Mode), поэтому проверяется
    именно последовательность.
    """
    sig = "".join("P" if a.is_pick else "B" for a in sorted(actions, key=lambda x: x.ord))
    return _KNOWN_CM.get(sig, "other")


class _DecayCounter:
    """(wins, games) с ленивым экспоненциальным затуханием.

    Ленивым — чтобы добавление будущего матча не могло задним числом
    изменить прошлое значение (тест time-decay утечки, Phase 9).
    """

    def __init__(self, half_life_days: float = HALF_LIFE_DAYS):
        self.hl = half_life_days
        self._w: Dict[object, float] = defaultdict(float)
        self._g: Dict[object, float] = defaultdict(float)
        self._t: Dict[object, float] = {}

    def _decay(self, k, ts: float) -> None:
        last = self._t.get(k)
        if last is None:
            self._t[k] = ts
            return
        dt = (ts - last) / 86400.0
        if dt > 0:
            f = 0.5 ** (dt / self.hl)
            self._w[k] *= f
            self._g[k] *= f
        self._t[k] = ts

    def games(self, k, ts: float) -> float:
        self._decay(k, ts)
        return self._g.get(k, 0.0)

    def rate(self, k, ts: float, prior: float) -> float:
        self._decay(k, ts)
        return (self._w.get(k, 0.0) + prior / 2.0) / (self._g.get(k, 0.0) + prior)

    def observe(self, k, won: bool, ts: float) -> None:
        self._decay(k, ts)
        self._g[k] += 1.0
        self._w[k] += 1.0 if won else 0.0


@dataclass(frozen=True)
class DraftStateRow:
    match_id: int
    as_of_timestamp: datetime
    # H1 — доля пула, вырезанная банами соперника (>0 => сопернику отрезали больше)
    denied_comfort_diff: float
    denied_comfort_top_diff: float
    # H2 — сила героев, взвешенная порядком пика
    order_weighted_hero_strength_diff: float
    # H3 — контрпик-адвантаж только на раскрытых последних пиках
    last_pick_counter_diff: float
    # H4 — вероятностный состав линий
    lane_balance_diff: float
    lane_prior_entropy_diff: float
    # служебное: сколько действий было раскрыто и какой формат
    revealed_actions: int
    draft_format: str
    radiant_win: bool


def _entropy(p: Sequence[float]) -> float:
    s = sum(p)
    if s <= 0:
        return 0.0
    return -sum((x / s) * math.log2(x / s) for x in p if x > 0)


def build_draft_state_features(
    matches: Iterable[MatchWithDraftState],
    reveal: Optional[int] = None,
    half_life_days: float = HALF_LIFE_DAYS,
    order_tau: float = ORDER_WEIGHT_TAU,
) -> List[DraftStateRow]:
    """
    `matches` ОБЯЗАН быть отсортирован по (start_time, match_id) — тот же
    контракт, что у всех Feature Set модулей проекта.

    `reveal`:
      * None — виден весь драфт (режим DRAFT_AWARE после драфта);
      * k — видны только действия с k наименьшими `ord` (прогноз ПО ХОДУ
        драфта). Действия с большим `ord` не читаются вообще.

    Признак «нет данных» всегда даёт ровно 0: возвращаются ОТКЛОНЕНИЯ от
    baseline, а не сырые доли.
    """
    pool = _DecayCounter(half_life_days)      # (account_id, hero_id) — пул игрока
    hero = _DecayCounter(half_life_days)      # hero_id — сила героя
    matchup = _DecayCounter(half_life_days)   # (hero_ours, hero_theirs) — очные
    lane = _DecayCounter(half_life_days)      # (hero_id, lane_role) — приор линии
    heroes_of: Dict[int, set] = defaultdict(set)   # account_id -> сыгранные герои

    rows: List[DraftStateRow] = []

    def pool_weights(accounts: Sequence[int], ts: float) -> Dict[int, float]:
        w: Dict[int, float] = defaultdict(float)
        for a in accounts:
            for h in heroes_of.get(a, ()):
                w[h] += pool.games((a, h), ts)
        return {h: v for h, v in w.items() if v > 0.0}

    def lane_distribution(hero_id: int, ts: float) -> List[float]:
        """P(линия | герой) со сглаживанием; равномерно при отсутствии истории."""
        g = [lane.games((hero_id, l), ts) for l in (1, 2, 3, 4)]
        tot = sum(g)
        if tot <= 0:
            return [0.25] * 4
        return [(x + LANE_PRIOR_GAMES / 4.0) / (tot + LANE_PRIOR_GAMES) for x in g]

    for m in matches:
        ts = m.start_time.timestamp()
        fmt = classify_draft(m.actions)
        acts = sorted(m.actions, key=lambda a: a.ord)
        visible = acts if reveal is None else acts[:reveal]

        # ---- раскрытое состояние драфта, строго по видимым действиям ----
        picks: Dict[bool, List[int]] = {True: [], False: []}   # is_radiant -> герои в порядке пика
        bans_by: Dict[bool, List[int]] = {True: [], False: []} # кто банил -> кого
        # ord последнего видимого пика каждой стороны — нужен для H3
        last_pick_ord: Dict[bool, Optional[int]] = {True: None, False: None}
        for a in visible:
            if a.hero_id is None:
                continue
            is_rad = (a.team == 0)
            if a.is_pick:
                picks[is_rad].append(a.hero_id)
                last_pick_ord[is_rad] = a.ord
            else:
                bans_by[is_rad].append(a.hero_id)

        accounts = {True: [p.account_id for p in m.players if p.is_radiant],
                    False: [p.account_id for p in m.players if not p.is_radiant]}

        side: Dict[bool, Dict[str, float]] = {}
        for is_rad in (True, False):
            # --- H1: сколько пула вырезал СОПЕРНИК ---
            w = pool_weights(accounts[is_rad], ts)
            total = sum(w.values())
            enemy_bans = bans_by[not is_rad]
            denied = sum(w.get(h, 0.0) for h in enemy_bans)
            denied_share = denied / total if total > 0 else 0.0
            # вариант с явным «комфортным ядром»: top-10 героев пятёрки
            top = sorted(w.items(), key=lambda x: -x[1])[:10]
            top_total = sum(v for _, v in top)
            top_set = {h for h, _ in top}
            denied_top = sum(v for h, v in top if h in set(enemy_bans))
            denied_top_share = denied_top / top_total if top_total > 0 else 0.0

            # --- H2: сила героев, взвешенная порядком пика ---
            ws, acc = 0.0, 0.0
            for i, h in enumerate(picks[is_rad]):
                wt = math.exp(-i / order_tau)
                acc += wt * (hero.rate(h, ts, HERO_PRIOR_GAMES) - 0.5)
                ws += wt
            order_strength = acc / ws if ws > 0 else 0.0

            # --- H4: вероятностный состав линий ---
            expected = [0.0, 0.0, 0.0, 0.0]
            ent = 0.0
            for h in picks[is_rad]:
                d = lane_distribution(h, ts)
                for i in range(4):
                    expected[i] += d[i]
                ent += _entropy(d)
            n_p = len(picks[is_rad])
            # каноничный состав Captains Mode: 2 safelane, 1 mid, 2 offlane, 0 jungle
            canonical = [2.0, 1.0, 2.0, 0.0]
            scale = (n_p / 5.0) if n_p else 0.0
            balance = -sum(abs(expected[i] - canonical[i] * scale) for i in range(4))
            side[is_rad] = {
                "denied": denied_share,
                "denied_top": denied_top_share,
                "order": order_strength,
                "balance": balance,
                "ent": ent / n_p if n_p else 0.0,
            }

        # --- H3: контрпик только против УЖЕ раскрытых героев соперника ---
        counter: Dict[bool, float] = {True: 0.0, False: 0.0}
        for is_rad in (True, False):
            lp = last_pick_ord[is_rad]
            if lp is None or not picks[is_rad]:
                continue
            our_hero = picks[is_rad][-1]
            # герои соперника, раскрытые СТРОГО РАНЬШЕ нашего последнего пика
            revealed_enemy = [a.hero_id for a in visible
                              if a.is_pick and a.hero_id is not None
                              and (a.team == 0) != is_rad and a.ord < lp]
            if not revealed_enemy:
                continue
            vals = [matchup.rate((our_hero, e), ts, MATCHUP_PRIOR_GAMES) - 0.5
                    for e in revealed_enemy]
            counter[is_rad] = sum(vals) / len(vals)

        r, d = side[True], side[False]
        rows.append(DraftStateRow(
            match_id=m.match_id,
            as_of_timestamp=m.start_time,
            # знак: положительное значение = лучше для Radiant.
            # У denied смысл обратный (вырезали пул = хуже), поэтому d - r.
            denied_comfort_diff=d["denied"] - r["denied"],
            denied_comfort_top_diff=d["denied_top"] - r["denied_top"],
            order_weighted_hero_strength_diff=r["order"] - d["order"],
            last_pick_counter_diff=counter[True] - counter[False],
            lane_balance_diff=r["balance"] - d["balance"],
            lane_prior_entropy_diff=r["ent"] - d["ent"],
            revealed_actions=len(visible),
            draft_format=fmt,
            radiant_win=m.radiant_win,
        ))

        # ---------- обновление состояния строго ПОСЛЕ фиксации ----------
        rad_heroes = [p.hero_id for p in m.players if p.is_radiant]
        dire_heroes = [p.hero_id for p in m.players if not p.is_radiant]
        for p in m.players:
            won = m.radiant_win if p.is_radiant else not m.radiant_win
            pool.observe((p.account_id, p.hero_id), won, ts)
            heroes_of[p.account_id].add(p.hero_id)
            hero.observe(p.hero_id, won, ts)
            if p.lane_role in (1, 2, 3, 4):
                lane.observe((p.hero_id, p.lane_role), won, ts)
        for a in rad_heroes:
            for b in dire_heroes:
                matchup.observe((a, b), m.radiant_win, ts)
                matchup.observe((b, a), not m.radiant_win, ts)

    return rows
