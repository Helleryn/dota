"""
Phase 9, PART H/I — как ПРАВИЛЬНО моделировать текущую силу героя.

Phase 8 установила эмпирически: мета течёт НЕПРЕРЫВНО, а не скачками на
границе патча (L1 внутри патча 0.35-0.44 против 0.46-0.76 между патчами).
Phase 7 пробовала жёсткий patch-scoping и получила результат ХУЖЕ
глобального win rate. Phase 8 взяла exp-decay и получила лучше — но
СРАВНЕНИЯ ВСЕХ СХЕМ НА ОДНОЙ ВЫБОРКЕ не проводилось ни разу.

Здесь все 5 схем считаются в ОДНОМ walk-forward проходе, чтобы сравнение
было честным (одни и те же матчи, один и тот же момент времени):

    S1 global        — вся история, Beta-Binomial shrinkage
    S2 same_patch    — только текущий патч
    S3 recent_window — скользящее окно 60 дней
    S4 exp_decay     — экспоненциальное затухание, полураспад 90 дней
    S5 patch_decay   — exp-decay, но счётчики СБРАСЫВАЮТСЯ на смене патча
                       (гибрид: свежесть по времени + разрыв на патче)

PART I — популярность (баны в проекте не использовались ни разу):
    pick_rate    — затухающая доля драфтов, где героя ВЫБРАЛИ
    ban_rate     — затухающая доля драфтов, где героя ЗАБАНИЛИ
    contest_rate — pick_rate + ban_rate («востребованность»)

Гипотеза Part I проверяется, а не постулируется: высокий ban_rate НЕ
объявляется автоматически признаком силы.

## Shrinkage

Все win-rate величины сглажены к 0.5:
    strength = (wins + m/2) / (games + m) - 0.5
Возвращается ОТКЛОНЕНИЕ от 0.5, чтобы «нет данных» == 0.

## Leakage
Все счётчики обновляются строго ПОСЛЕ фиксации признаков матча.
Затухание — ленивое, только от прошлых наблюдений к текущему моменту;
будущие матчи физически не могут повлиять (проверяется adversarial-тестом).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, Iterable, List, Optional, Protocol, Tuple

HALF_LIFE_DAYS = 90.0
RECENT_WINDOW_DAYS = 60.0
HERO_PRIOR_GAMES = 50.0
POPULARITY_HALF_LIFE_DAYS = 30.0

SCHEMES = ("global", "same_patch", "recent_window", "exp_decay", "patch_decay")


class MatchWithDraft(Protocol):
    match_id: int
    start_time: datetime
    patch_id: Optional[int]
    radiant_picks: Tuple[int, ...]
    dire_picks: Tuple[int, ...]
    radiant_bans: Tuple[int, ...]
    dire_bans: Tuple[int, ...]
    radiant_win: bool


@dataclass(frozen=True)
class HeroSchemeRow:
    match_id: int
    as_of_timestamp: datetime
    # разность (radiant - dire) средней силы 5 героев по каждой схеме
    strength_diff: Dict[str, float]
    # PART I
    pick_rate_diff: float
    ban_rate_diff: float
    contest_rate_diff: float
    radiant_win: bool


class _Counter:
    """(wins, games) со сглаживанием; поддерживает 3 режима старения."""

    def __init__(self, mode: str, half_life: float = HALF_LIFE_DAYS):
        self.mode = mode
        self.half_life = half_life
        self._w: Dict[int, float] = defaultdict(float)
        self._g: Dict[int, float] = defaultdict(float)
        self._last: Dict[int, float] = {}

    def _decay(self, hero: int, ts: float) -> None:
        if self.mode != "decay":
            return
        last = self._last.get(hero)
        if last is None:
            self._last[hero] = ts
            return
        dt = (ts - last) / 86400.0
        if dt > 0:
            f = 0.5 ** (dt / self.half_life)
            self._w[hero] *= f
            self._g[hero] *= f
        self._last[hero] = ts

    def strength(self, hero: int, ts: float) -> float:
        self._decay(hero, ts)
        g, w = self._g.get(hero, 0.0), self._w.get(hero, 0.0)
        return (w + HERO_PRIOR_GAMES / 2.0) / (g + HERO_PRIOR_GAMES) - 0.5

    def observe(self, hero: int, won: bool, ts: float) -> None:
        self._decay(hero, ts)
        self._g[hero] += 1.0
        self._w[hero] += 1.0 if won else 0.0

    def reset(self) -> None:
        self._w.clear()
        self._g.clear()
        self._last.clear()


class _WindowCounter:
    """Скользящее окно фиксированной длины в днях (S3)."""

    def __init__(self, window_days: float = RECENT_WINDOW_DAYS):
        self.window = window_days
        self._events: Dict[int, Deque[Tuple[float, bool]]] = defaultdict(lambda: deque(maxlen=4000))

    def _trim(self, hero: int, ts: float) -> None:
        ev = self._events.get(hero)
        if not ev:
            return
        cutoff = ts - self.window * 86400.0
        while ev and ev[0][0] < cutoff:
            ev.popleft()

    def strength(self, hero: int, ts: float) -> float:
        self._trim(hero, ts)
        ev = self._events.get(hero)
        if not ev:
            return 0.0
        g = len(ev)
        w = sum(1 for _, won in ev if won)
        return (w + HERO_PRIOR_GAMES / 2.0) / (g + HERO_PRIOR_GAMES) - 0.5

    def observe(self, hero: int, won: bool, ts: float) -> None:
        self._events[hero].append((ts, won))


class _PopularityCounter:
    """Затухающие доли пиков/банов от общего числа драфтов (PART I)."""

    def __init__(self, half_life: float = POPULARITY_HALF_LIFE_DAYS):
        self.half_life = half_life
        self._picks: Dict[int, float] = defaultdict(float)
        self._bans: Dict[int, float] = defaultdict(float)
        self._last: Dict[int, float] = {}
        self._total = 0.0
        self._total_last: Optional[float] = None

    def _decay_hero(self, hero: int, ts: float) -> None:
        last = self._last.get(hero)
        if last is None:
            self._last[hero] = ts
            return
        dt = (ts - last) / 86400.0
        if dt > 0:
            f = 0.5 ** (dt / self.half_life)
            self._picks[hero] *= f
            self._bans[hero] *= f
        self._last[hero] = ts

    def _decay_total(self, ts: float) -> None:
        if self._total_last is None:
            self._total_last = ts
            return
        dt = (ts - self._total_last) / 86400.0
        if dt > 0:
            self._total *= 0.5 ** (dt / self.half_life)
        self._total_last = ts

    def rates(self, hero: int, ts: float) -> Tuple[float, float]:
        self._decay_hero(hero, ts)
        self._decay_total(ts)
        if self._total <= 0:
            return 0.0, 0.0
        return self._picks.get(hero, 0.0) / self._total, self._bans.get(hero, 0.0) / self._total

    def observe(self, picks: Iterable[int], bans: Iterable[int], ts: float) -> None:
        self._decay_total(ts)
        self._total += 1.0
        for h in picks:
            self._decay_hero(h, ts)
            self._picks[h] += 1.0
        for h in bans:
            self._decay_hero(h, ts)
            self._bans[h] += 1.0


def _mean(v: List[float]) -> float:
    return sum(v) / len(v) if v else 0.0


def build_hero_scheme_features(matches: Iterable[MatchWithDraft]) -> List[HeroSchemeRow]:
    """matches ОБЯЗАН быть отсортирован по (start_time, match_id)."""
    c_global = _Counter("plain")
    c_patch = _Counter("plain")
    c_decay = _Counter("decay", HALF_LIFE_DAYS)
    c_patch_decay = _Counter("decay", HALF_LIFE_DAYS)
    c_window = _WindowCounter(RECENT_WINDOW_DAYS)
    popularity = _PopularityCounter()

    current_patch: Optional[int] = None
    rows: List[HeroSchemeRow] = []

    for m in matches:
        ts = m.start_time.timestamp()

        # Смена патча: обнуляем patch-scoped счётчики. Это происходит ДО
        # чтения признаков — патч текущего матча известен заранее (он
        # определён датой матча, см. docs/features.md, категория E).
        if m.patch_id != current_patch:
            c_patch.reset()
            c_patch_decay.reset()
            current_patch = m.patch_id

        strength_diff = {}
        for name, counter in (
            ("global", c_global), ("same_patch", c_patch),
            ("exp_decay", c_decay), ("patch_decay", c_patch_decay),
        ):
            r = _mean([counter.strength(h, ts) for h in m.radiant_picks])
            d = _mean([counter.strength(h, ts) for h in m.dire_picks])
            strength_diff[name] = r - d
        r_w = _mean([c_window.strength(h, ts) for h in m.radiant_picks])
        d_w = _mean([c_window.strength(h, ts) for h in m.dire_picks])
        strength_diff["recent_window"] = r_w - d_w

        r_rates = [popularity.rates(h, ts) for h in m.radiant_picks]
        d_rates = [popularity.rates(h, ts) for h in m.dire_picks]
        r_pick, r_ban = _mean([x[0] for x in r_rates]), _mean([x[1] for x in r_rates])
        d_pick, d_ban = _mean([x[0] for x in d_rates]), _mean([x[1] for x in d_rates])

        rows.append(HeroSchemeRow(
            match_id=m.match_id,
            as_of_timestamp=m.start_time,
            strength_diff=strength_diff,
            pick_rate_diff=r_pick - d_pick,
            ban_rate_diff=r_ban - d_ban,
            contest_rate_diff=(r_pick + r_ban) - (d_pick + d_ban),
            radiant_win=m.radiant_win,
        ))

        # --- обновление строго ПОСЛЕ фиксации признаков ---
        rw = m.radiant_win
        for h in m.radiant_picks:
            for c in (c_global, c_patch, c_decay, c_patch_decay):
                c.observe(h, rw, ts)
            c_window.observe(h, rw, ts)
        for h in m.dire_picks:
            for c in (c_global, c_patch, c_decay, c_patch_decay):
                c.observe(h, not rw, ts)
            c_window.observe(h, not rw, ts)
        popularity.observe(
            list(m.radiant_picks) + list(m.dire_picks),
            list(m.radiant_bans) + list(m.dire_bans),
            ts,
        )

    return rows
