"""
Phase 8 — meta-aware draft (M1, M2, D1, D2, D3, D4).

## Почему не patch-scoping (эмпирическое обоснование, не догадка)

Phase 7 попробовала жёсткий patch-scoped hero strength и получила результат
ХУЖЕ глобального win rate. Phase 8 измерила, почему:

    L1-дистанция распределения пиков МЕЖДУ соседними патчами: 0.46 - 0.76
    L1-дистанция ВНУТРИ одного патча (половина vs половина):   0.35 - 0.44

Внутрипатчевый дрейф почти такой же большой, как межпатчевый — то есть
мета «течёт» НЕПРЕРЫВНО, а не скачком на границе патча. Жёсткое разбиение
по патчу поэтому делает две ошибки сразу: выбрасывает всё ещё релевантные
данные предыдущего патча И считает начало/конец одного патча одинаково
актуальными. Экспоненциальное затухание по ВРЕМЕНИ снимает обе.

Дополнительно измерено: медианный сдвиг win rate героя между соседними
патчами (герои с n>=50 в обоих) = 2.9 п.п., p90 = 7.9 п.п. — то есть
сигнал реально устаревает, но не обнуляется.

## Затухание и сглаживание

Для каждого счётчика хранится (decayed_wins, decayed_games, last_t), при
обращении применяется ленивое затухание:

    f = 0.5 ** (dt_days / HALF_LIFE_DAYS)
    wins *= f ; games *= f

Затем Beta-Binomial shrinkage к 0.5 (раздел 25 задания — N=3 и 100% побед
не означает силу 100%):

    strength = (wins + m/2) / (games + m) - 0.5

Возвращается отклонение от 0.5, чтобы «нет данных» == 0, а не 0.5.

HALF_LIFE_DAYS=90 и m=... выбраны априорно (порядок величины: типичный
патч живёт ~3 месяца), НЕ подбирались по test (раздел 19/71 задания).

## Leakage-семантика

Все счётчики обновляются строго ПОСЛЕ фиксации признаков матча, поэтому
source_timestamp < prediction_timestamp по построению. `picks_bans` этого
матча используется как СОСТОЯНИЕ ДРАФТА (доступно после драфта, до начала
игры) — это MODE B / post-draft, не pre-draft.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

HALF_LIFE_DAYS = 90.0
HERO_PRIOR_GAMES = 50.0        # shrinkage для hero strength
PAIR_PRIOR_GAMES = 30.0        # shrinkage для synergy/counter (пар сильно больше -> меньше данных на пару)
PLAYER_HERO_PRIOR_GAMES = 10.0  # игрок-герой: выборки маленькие, prior мягче


class MatchWithDraft(Protocol):
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_picks: Tuple[int, ...]
    dire_picks: Tuple[int, ...]
    radiant_roster: Tuple[int, ...]
    dire_roster: Tuple[int, ...]
    radiant_win: bool


@dataclass(frozen=True)
class MetaFeatureRow:
    match_id: int
    as_of_timestamp: datetime

    # M1/M2 + D1 — сила драфта из затухающей, сглаженной силы героев
    radiant_hero_strength_decayed: float
    dire_hero_strength_decayed: float

    # D2 — синергия внутри своей пятёрки героев (сверх индивидуальной силы)
    radiant_hero_synergy: float
    dire_hero_synergy: float

    # D3 — контр-преимущество radiant над dire (5x5)
    counter_advantage: float

    # D4 — профильность игроков на выбранных ими героях
    radiant_player_hero_proficiency: float
    dire_player_hero_proficiency: float

    radiant_win: bool


class _DecayingCounter:
    """Пара (wins, games) с ленивым экспоненциальным затуханием по времени."""

    def __init__(self, half_life_days: float = HALF_LIFE_DAYS):
        self.half_life = half_life_days
        self._wins: Dict[object, float] = defaultdict(float)
        self._games: Dict[object, float] = defaultdict(float)
        self._last_t: Dict[object, float] = {}

    def _decay(self, key, now_ts: float) -> None:
        last = self._last_t.get(key)
        if last is None:
            self._last_t[key] = now_ts
            return
        dt_days = (now_ts - last) / 86400.0
        if dt_days > 0:
            f = 0.5 ** (dt_days / self.half_life)
            self._wins[key] *= f
            self._games[key] *= f
        self._last_t[key] = now_ts

    def strength(self, key, now_ts: float, prior_games: float) -> float:
        """Отклонение сглаженного win rate от 0.5. 0 == нет информации."""
        self._decay(key, now_ts)
        g = self._games.get(key, 0.0)
        w = self._wins.get(key, 0.0)
        return (w + prior_games / 2.0) / (g + prior_games) - 0.5

    def games(self, key, now_ts: float) -> float:
        self._decay(key, now_ts)
        return self._games.get(key, 0.0)

    def observe(self, key, won: bool, now_ts: float) -> None:
        self._decay(key, now_ts)
        self._games[key] += 1.0
        self._wins[key] += 1.0 if won else 0.0


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_meta_features(
    matches: Iterable[MatchWithDraft],
    half_life_days: float = HALF_LIFE_DAYS,
) -> List[MetaFeatureRow]:
    """matches ОБЯЗАН быть отсортирован по (start_time, match_id)."""
    hero = _DecayingCounter(half_life_days)
    pair_same = _DecayingCounter(half_life_days)   # два героя в ОДНОЙ команде -> synergy
    pair_vs = _DecayingCounter(half_life_days)     # герой A против героя B -> counter
    player_hero = _DecayingCounter(half_life_days)

    rows: List[MetaFeatureRow] = []

    for m in matches:
        ts = m.start_time.timestamp()

        r_hero_strength = _mean([hero.strength(h, ts, HERO_PRIOR_GAMES) for h in m.radiant_picks])
        d_hero_strength = _mean([hero.strength(h, ts, HERO_PRIOR_GAMES) for h in m.dire_picks])

        # D2: синергия сверх индивидуальной силы. Для пары (a,b) сравниваем
        # наблюдаемую совместную силу с суммой индивидуальных — иначе пара
        # из двух сильных героев выглядела бы «синергичной» просто потому,
        # что оба сильны (раздел 36 задания).
        def synergy(picks) -> float:
            vals = []
            for i in range(len(picks)):
                for j in range(i + 1, len(picks)):
                    a, b = picks[i], picks[j]
                    key = (a, b) if a <= b else (b, a)
                    observed = pair_same.strength(key, ts, PAIR_PRIOR_GAMES)
                    expected = hero.strength(a, ts, HERO_PRIOR_GAMES) + hero.strength(b, ts, HERO_PRIOR_GAMES)
                    vals.append(observed - expected / 2.0)
            return _mean(vals)

        r_syn, d_syn = synergy(m.radiant_picks), synergy(m.dire_picks)

        # D3: контры. pair_vs[(a,b)] = как часто сторона героя a побеждала
        # сторону героя b. Аналогично вычитаем индивидуальную силу, чтобы
        # не считать «контрой» просто перевес сильного героя над слабым.
        counter_vals = []
        for a in m.radiant_picks:
            for b in m.dire_picks:
                observed = pair_vs.strength((a, b), ts, PAIR_PRIOR_GAMES)
                expected = (hero.strength(a, ts, HERO_PRIOR_GAMES) - hero.strength(b, ts, HERO_PRIOR_GAMES)) / 2.0
                counter_vals.append(observed - expected)
        counter_adv = _mean(counter_vals)

        # D4: игрок-герой. Состав и пики известны, но КТО на КОМ играет —
        # это lane/hero assignment конкретного матча. Пары (игрок, герой)
        # берём из фактической связки этого матча только для СПРАВКИ о том,
        # какие герои выбраны; профильность считается как средняя по всем
        # (игрок команды x герой команды) парам — так мы не используем
        # знание «кто именно взял какого героя» внутри этого матча.
        def proficiency(roster, picks) -> float:
            if not roster or not picks:
                return 0.0
            vals = [player_hero.strength((p, h), ts, PLAYER_HERO_PRIOR_GAMES) for p in roster for h in picks]
            return _mean(vals)

        r_prof = proficiency(m.radiant_roster, m.radiant_picks)
        d_prof = proficiency(m.dire_roster, m.dire_picks)

        rows.append(
            MetaFeatureRow(
                match_id=m.match_id,
                as_of_timestamp=m.start_time,
                radiant_hero_strength_decayed=r_hero_strength,
                dire_hero_strength_decayed=d_hero_strength,
                radiant_hero_synergy=r_syn,
                dire_hero_synergy=d_syn,
                counter_advantage=counter_adv,
                radiant_player_hero_proficiency=r_prof,
                dire_player_hero_proficiency=d_prof,
                radiant_win=m.radiant_win,
            )
        )

        # --- обновление состояния строго ПОСЛЕ фиксации признаков ---
        rw = m.radiant_win
        for h in m.radiant_picks:
            hero.observe(h, rw, ts)
        for h in m.dire_picks:
            hero.observe(h, not rw, ts)

        for picks, won in ((m.radiant_picks, rw), (m.dire_picks, not rw)):
            for i in range(len(picks)):
                for j in range(i + 1, len(picks)):
                    a, b = picks[i], picks[j]
                    pair_same.observe((a, b) if a <= b else (b, a), won, ts)

        for a in m.radiant_picks:
            for b in m.dire_picks:
                pair_vs.observe((a, b), rw, ts)
                pair_vs.observe((b, a), not rw, ts)

        for p in m.radiant_roster:
            for h in m.radiant_picks:
                player_hero.observe((p, h), rw, ts)
        for p in m.dire_roster:
            for h in m.dire_picks:
                player_hero.observe((p, h), not rw, ts)

    return rows
