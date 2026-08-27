"""
PHASE 19 — player × hero, соотнесённый с текущей метой (PART O, H1).

Зачем отдельный модуль. Phase 11 проверяла player×hero в виде «winrate
игрока на герое, которого он взял в этом матче» — а какой герой достанется
какому игроку, известно только ПОСЛЕ драфта и распределения ролей, то есть
это post-match величина. Такой признак в pre-match режиме не существует
вовсе, сколько его ни улучшай.

Здесь проверяется форма, которая pre-match существует: не «игрок на этом
герое», а **пул игрока** — герои, на которых он играл раньше, и то,
насколько он на них силён ОТНОСИТЕЛЬНО текущей меты.

    raw(p)   = Σ g(p,h)·wr(p,h)              / Σ g(p,h)
    resid(p) = Σ g(p,h)·(wr(p,h) − meta(h))  / Σ g(p,h)

`g` — затухающее число игр, `wr` — затухающий winrate с усадкой к 0.5,
`meta(h)` — сила героя в мете на момент T (та же формула, что в Phase 9).

Гипотеза H1: `resid` информативнее `raw`, потому что высокий winrate на
герое, который сейчас силён у всех, говорит об игроке меньше, чем такой же
winrate на герое, который сейчас слаб.

Движок Phase 17 НЕ трогается: здесь повторена его лента событий, а не
изменена. Правило то же — при равном времени PREDICT раньше UPDATE.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from src.pit.engine import (
    HERO_HALF_LIFE_DAYS,
    HERO_PRIOR_GAMES,
    PitMatch,
    RosterProvider,
    _DecayHeroStrength,
)

# Усадка для пары «игрок × герой» больше, чем для героя вообще: у игрока
# на конкретном герое игр на порядок меньше, и без этого признак был бы
# шумом от нескольких матчей.
PH_PRIOR_GAMES = 10.0


@dataclass(frozen=True)
class PlayerMetaRow:
    match_id: int
    pool_raw_diff: Optional[float]
    pool_resid_diff: Optional[float]
    pool_players_min: int


class _PlayerHero:
    """Затухающие победы и игры для пары (игрок, герой)."""

    def __init__(self, half_life_days: float = HERO_HALF_LIFE_DAYS):
        self.hl = half_life_days
        self._w: Dict[Tuple[int, int], float] = defaultdict(float)
        self._g: Dict[Tuple[int, int], float] = defaultdict(float)
        self._t: Dict[Tuple[int, int], float] = {}
        self._by_player: Dict[int, set] = defaultdict(set)

    def _factor(self, key, ts: float) -> float:
        last = self._t.get(key)
        if last is None:
            return 1.0
        dt = (ts - last) / 86400.0
        return 0.5 ** (dt / self.hl) if dt > 0 else 1.0

    def observe(self, pid: int, hero: int, won: bool, ts: float) -> None:
        key = (pid, hero)
        f = self._factor(key, ts)
        self._w[key] = self._w[key] * f + (1.0 if won else 0.0)
        self._g[key] = self._g[key] * f + 1.0
        self._t[key] = ts
        self._by_player[pid].add(hero)

    def winrate(self, pid: int, hero: int, ts: float) -> Tuple[float, float]:
        """(усаженный winrate−0.5, затухающее число игр). Чтение НЕ пишет."""
        key = (pid, hero)
        f = self._factor(key, ts)
        g = self._g.get(key, 0.0) * f
        w = self._w.get(key, 0.0) * f
        return (w + PH_PRIOR_GAMES / 2.0) / (g + PH_PRIOR_GAMES) - 0.5, g

    def heroes(self, pid: int) -> set:
        return self._by_player.get(pid, set())


def _team_signals(ph: _PlayerHero, hero: _DecayHeroStrength,
                  roster: FrozenSet[int], ts: float
                  ) -> Tuple[Optional[float], Optional[float], int]:
    raws, resids = [], []
    for p in roster:
        num_r = num_x = tot = 0.0
        for h in ph.heroes(p):
            wr, g = ph.winrate(p, h, ts)
            if g <= 1e-6:
                continue
            meta = hero.strength(h, ts)
            tot += g
            num_r += g * wr
            num_x += g * (wr - meta)
        if tot > 0:
            raws.append(num_r / tot)
            resids.append(num_x / tot)
    if not raws:
        return None, None, 0
    return sum(raws) / len(raws), sum(resids) / len(resids), len(raws)


def build_player_meta_features(matches: Sequence[PitMatch],
                               horizon: timedelta,
                               roster_provider: Optional[RosterProvider] = None,
                               emit_only: Optional[set] = None
                               ) -> List[PlayerMetaRow]:
    """Признаки H1 на момент `start − horizon`.

    Возвращает разности «radiant минус dire». `None`, если хотя бы у одной
    стороны пул пуст: подставлять туда ноль означало бы утверждать
    «разницы нет», хотя её просто не измерили.
    """
    order = sorted(matches, key=lambda m: (m.start_time, m.match_id))
    events: List[Tuple[float, int, int]] = []
    for i, m in enumerate(order):
        events.append(((m.start_time - horizon).timestamp(), 0, i))
        events.append((m.start_time.timestamp(), 1, i))
    events.sort()

    ph = _PlayerHero()
    hero = _DecayHeroStrength()
    out: Dict[int, PlayerMetaRow] = {}

    for t, kind, i in events:
        m = order[i]
        if kind == 1:
            ts = m.start_time.timestamp()
            for h in m.radiant_picks:
                hero.observe(h, m.radiant_win, ts)
            for h in m.dire_picks:
                hero.observe(h, not m.radiant_win, ts)
            # Пара «игрок × герой» без распределения ролей неизвестна, поэтому
            # каждому игроку стороны засчитываются все пять героев своей
            # стороны. Это НЕ «кто на ком играл» — это пул пятёрки, и именно
            # он pre-match доступен. Ограничение названо прямо.
            for roster, picks, won in ((m.radiant_roster, m.radiant_picks, m.radiant_win),
                                       (m.dire_roster, m.dire_picks, not m.radiant_win)):
                for p in roster:
                    for h in picks:
                        ph.observe(p, h, won, ts)
            continue

        if emit_only is not None and m.match_id not in emit_only:
            continue
        rr = dr = frozenset()
        if roster_provider is not None:
            rr, dr = roster_provider.for_match(m.match_id)
        if not rr or not dr:
            out[m.match_id] = PlayerMetaRow(m.match_id, None, None, 0)
            continue
        ts_p = (m.start_time - horizon).timestamp()
        r_raw, r_res, nr = _team_signals(ph, hero, rr, ts_p)
        d_raw, d_res, nd = _team_signals(ph, hero, dr, ts_p)
        if r_raw is None or d_raw is None:
            out[m.match_id] = PlayerMetaRow(m.match_id, None, None, min(nr, nd))
        else:
            out[m.match_id] = PlayerMetaRow(m.match_id, r_raw - d_raw,
                                            r_res - d_res, min(nr, nd))

    return [out[m.match_id] for m in order if m.match_id in out]
