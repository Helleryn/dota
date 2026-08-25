"""
PHASE 11 — Player x Hero x Role x Patch/Meta.

## Чем это отличается от отвергнутого Team x Hero (Phase 10)

Phase 10 отвергла Team x Hero: медиана **3 матча на team_id ВСЕГО** —
выборки не существует. Здесь носитель — игрок, и плотность на порядок выше
(аудит Phase 11.0): к моменту матча у пары (игрок, герой) медиана **8**
прошлых игр, у 83.3% player-match записей есть хотя бы одна. Это другая по
плотности задача, а не возврат к отвергнутой гипотезе.

## Формулы (сырой winrate не используется нигде — прямой запрет задания)

Все оценки — **относительно baseline**, с экспоненциальным затуханием и
Beta-Binomial сглаживанием. Возвращается ОТКЛОНЕНИЕ, поэтому «нет данных»
даёт ровно 0, а не 0.5:

    ph_wr(P,H,t) = (decayed_wins(P,H) + m/2) / (decayed_games(P,H) + m)
    h_wr(H,t)    = (decayed_wins(H)   + m/2) / (decayed_games(H)   + m)
    PlayerHeroStrength(P,H,t) = ph_wr − h_wr

Смысл (PART X задания): не «игрок выиграл X% игр», а «насколько игрок
превосходит ожидаемый уровень ЭТОГО героя, с поправкой на объём выборки и
давность результатов».

## Роль: где проходит граница утечки

`lane_role` и GPM — **post-match**. Поэтому:

* роль в ТЕКУЩЕМ матче не используется НИКОГДА;
* роли матчей СТРОГО РАНЬШЕ t формируют профиль игрока — это законно,
  такая информация доступна к моменту прогноза;
* «ожидаемая роль» на текущий матч = мода прошлых ролей (аудит: точность
  83.5% walk-forward).

Позиция 1-5 выводится из lane_role + ранга GPM внутри пары одной линии
(safelane: выше GPM = поз.1, ниже = поз.5; offlane: поз.3 / поз.4; mid = 2).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

PLAYER_HERO_PRIOR = 10.0
ROLE_PRIOR = 20.0
DEFAULT_HALF_LIFE_DAYS = 90.0


class PlayerInMatch(Protocol):
    account_id: int
    hero_id: int
    is_radiant: bool
    lane_role: Optional[int]
    gold_per_min: Optional[int]


class MatchWithPlayers(Protocol):
    match_id: int
    start_time: datetime
    radiant_win: bool
    players: Sequence[PlayerInMatch]


def derive_positions(players: Sequence[PlayerInMatch]) -> Dict[int, Optional[int]]:
    """
    Позиция 1-5 из lane_role + ранга GPM. ВНИМАНИЕ: применять ТОЛЬКО к уже
    сыгранным матчам (< t). Для текущего матча это post-match информация.
    """
    out: Dict[int, Optional[int]] = {}
    lanes: Dict[Optional[int], List[PlayerInMatch]] = defaultdict(list)
    for p in players:
        lanes[p.lane_role].append(p)
    for lane, group in lanes.items():
        group = sorted(group, key=lambda x: -(x.gold_per_min or 0))
        if lane == 2:
            for p in group:
                out[p.account_id] = 2
        elif lane == 1:
            if len(group) >= 2:
                out[group[0].account_id] = 1
                out[group[1].account_id] = 5
                for p in group[2:]:
                    out[p.account_id] = None
            else:
                for p in group:
                    out[p.account_id] = 1
        elif lane == 3:
            if len(group) >= 2:
                out[group[0].account_id] = 3
                out[group[1].account_id] = 4
                for p in group[2:]:
                    out[p.account_id] = None
            else:
                for p in group:
                    out[p.account_id] = 3
        else:
            for p in group:
                out[p.account_id] = None
    return out


class _DecayCounter:
    """(wins, games) с ленивым экспоненциальным затуханием."""

    def __init__(self, half_life_days: float):
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

    def rate(self, k, ts: float, prior: float) -> float:
        self._decay(k, ts)
        return (self._w.get(k, 0.0) + prior / 2.0) / (self._g.get(k, 0.0) + prior)

    def games(self, k, ts: float) -> float:
        self._decay(k, ts)
        return self._g.get(k, 0.0)

    def observe(self, k, won: bool, ts: float) -> None:
        self._decay(k, ts)
        self._g[k] += 1.0
        self._w[k] += 1.0 if won else 0.0


@dataclass(frozen=True)
class PlayerHeroRow:
    match_id: int
    as_of_timestamp: datetime
    # F1 — Player x Hero относительно baseline героя
    player_hero_strength_diff: float
    # покрытие: средний объём прошлых игр пары (для интерпретации/uncertainty)
    player_hero_games_min: float
    # F3 — сила игрока на ОЖИДАЕМОЙ роли (мода прошлых ролей)
    player_role_strength_diff: float
    # F4 — Player x Hero x Role
    player_hero_role_strength_diff: float
    # F5 — энтропия ролей (мера нестабильности позиции игрока)
    role_entropy_diff: float
    # F6 — взаимодействие мастерства с силой героя в текущей мете
    meta_relative_ph_diff: float
    radiant_win: bool


def build_player_hero_features(
    matches: Iterable[MatchWithPlayers],
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> List[PlayerHeroRow]:
    """matches ОБЯЗАН быть отсортирован по (start_time, match_id) — тот же
    контракт, что у всех Feature Set модулей проекта."""
    ph = _DecayCounter(half_life_days)      # (player, hero)
    hero = _DecayCounter(half_life_days)    # hero
    prole = _DecayCounter(half_life_days)   # (player, role)
    phr = _DecayCounter(half_life_days)     # (player, hero, role)
    role_hist: Dict[int, Counter] = defaultdict(Counter)   # роли игрока по матчам < t

    rows: List[PlayerHeroRow] = []

    def expected_role(aid: int) -> Optional[int]:
        c = role_hist.get(aid)
        return c.most_common(1)[0][0] if c else None

    def role_entropy(aid: int) -> float:
        c = role_hist.get(aid)
        if not c:
            return 0.0
        n = sum(c.values())
        return -sum((v / n) * math.log2(v / n) for v in c.values() if v)

    for m in matches:
        ts = m.start_time.timestamp()
        sides = {True: [], False: []}
        for p in m.players:
            sides[bool(p.is_radiant)].append(p)

        agg: Dict[bool, Dict[str, float]] = {}
        for is_rad, ps in sides.items():
            if not ps:
                agg[is_rad] = {k: 0.0 for k in
                               ("ph", "ph_games", "prole", "phr", "entropy", "meta_rel")}
                continue
            ph_vals, prole_vals, phr_vals, ent_vals, meta_vals, games = [], [], [], [], [], []
            for p in ps:
                h_rate = hero.rate(p.hero_id, ts, PLAYER_HERO_PRIOR)
                ph_rate = ph.rate((p.account_id, p.hero_id), ts, PLAYER_HERO_PRIOR)
                ph_strength = ph_rate - h_rate
                ph_vals.append(ph_strength)
                games.append(ph.games((p.account_id, p.hero_id), ts))

                er = expected_role(p.account_id)
                prole_vals.append(
                    prole.rate((p.account_id, er), ts, ROLE_PRIOR) - 0.5 if er is not None else 0.0
                )
                phr_vals.append(
                    phr.rate((p.account_id, p.hero_id, er), ts, PLAYER_HERO_PRIOR) - h_rate
                    if er is not None else 0.0
                )
                ent_vals.append(role_entropy(p.account_id))
                # F6: мастерство взвешено силой героя в текущей мете
                meta_vals.append(ph_strength * (h_rate - 0.5))

            n = len(ps)
            agg[is_rad] = {
                "ph": sum(ph_vals) / n,
                "ph_games": min(games) if games else 0.0,
                "prole": sum(prole_vals) / n,
                "phr": sum(phr_vals) / n,
                "entropy": sum(ent_vals) / n,
                "meta_rel": sum(meta_vals) / n,
            }

        r, d = agg[True], agg[False]
        rows.append(PlayerHeroRow(
            match_id=m.match_id,
            as_of_timestamp=m.start_time,
            player_hero_strength_diff=r["ph"] - d["ph"],
            player_hero_games_min=min(r["ph_games"], d["ph_games"]),
            player_role_strength_diff=r["prole"] - d["prole"],
            player_hero_role_strength_diff=r["phr"] - d["phr"],
            role_entropy_diff=r["entropy"] - d["entropy"],
            meta_relative_ph_diff=r["meta_rel"] - d["meta_rel"],
            radiant_win=m.radiant_win,
        ))

        # --- обновление состояния строго ПОСЛЕ фиксации признаков ---
        # Позиции ЭТОГО матча вычисляются здесь и попадают только в ИСТОРИЮ,
        # то есть станут доступны начиная со СЛЕДУЮЩЕГО матча игрока.
        positions = derive_positions(list(m.players))
        for p in m.players:
            won = m.radiant_win if p.is_radiant else not m.radiant_win
            pos = positions.get(p.account_id)
            ph.observe((p.account_id, p.hero_id), won, ts)
            hero.observe(p.hero_id, won, ts)
            if pos is not None:
                prole.observe((p.account_id, pos), won, ts)
                phr.observe((p.account_id, p.hero_id, pos), won, ts)
                role_hist[p.account_id][pos] += 1

    return rows
