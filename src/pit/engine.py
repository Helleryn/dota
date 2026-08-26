"""
PHASE 17 — движок признаков «на момент T» (point-in-time).

## Зачем понадобился отдельный движок

Замороженные модули Phase 9 считают признаки walk-forward, но **на момент
начала матча**. Для настоящего pre-match прогноза этого мало: прогноз
делается за часы или сутки ДО старта, и за это время команды успевают
сыграть другие матчи.

Замер Phase 17: хотя бы одна из двух команд играет
* в последний **час** перед стартом — у **32.3%** матчей,
* в последние **сутки** — у **85.0%**,
* за последнюю неделю — у 96.7%.

То есть признаки на T−24ч и на момент старта — **разные величины**, и
разница касается большинства матчей. Phase 15 этого не измеряла: там
прогноз ставился за 30 минут до старта и признаки брались из кэша,
посчитанного по состоянию на старт.

## Как обеспечена точность «на момент T»

Один проход по СЛИТОЙ временной шкале двух видов событий:

    PREDICT(match_i) в момент T_i = start_i − horizon
    UPDATE(match_j)  в момент start_j

События сортируются по времени; при `PREDICT` состояние только читается,
при `UPDATE` — только изменяется. Это даёт точное состояние на T за один
проход, без снимков состояния и без изменения замороженного кода.

Тонкость, ради которой всё и затевалось: матч j может начаться ПОЗЖЕ
T_i, но РАНЬШЕ start_i. В обычном walk-forward он попал бы в признаки
матча i; здесь — не попадёт, потому что на момент T_i его ещё не было.

## Верность замороженным формулам

Все формулы скопированы из `src/ratings/engine.py`,
`src/datasets/multi_window_features.py`,
`src/datasets/player_features.py` и `src/datasets/roster_representation.py`
буква в букву. Проверка — тест
`test_pit_reproduces_frozen_features_at_zero_horizon`: при horizon=0
значения обязаны совпасть с кэшем frozen-конвейера. Если формулу здесь
случайно изменят, тест это поймает.
"""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

TEAM_ELO_K = 16.0
TEAM_ELO_BASE = 1000.0
PLAYER_ELO_K = 16.0
PLAYER_ELO_BASE = 1000.0
FORM_WINDOW = 3
FORM_MAXLEN = 20
HERO_HALF_LIFE_DAYS = 90.0
HERO_PRIOR_GAMES = 50.0


@dataclass(frozen=True)
class PitMatch:
    """Матч для point-in-time прохода."""
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_win: bool
    radiant_roster: FrozenSet[int] = frozenset()
    dire_roster: FrozenSet[int] = frozenset()
    radiant_picks: Tuple[int, ...] = ()
    dire_picks: Tuple[int, ...] = ()


@dataclass(frozen=True)
class PitFeatures:
    match_id: int
    prediction_at: datetime
    start_time: datetime
    horizon_hours: float
    # --- доступны БЕЗ состава и без драфта ---
    elo_difference: float
    form_3_difference: Optional[float]
    radiant_matches_before: int
    dire_matches_before: int
    # --- требуют состава ---
    elo_mean_diff: Optional[float]
    five_vs_team_elo_diff: Optional[float]
    roster_known_radiant: int
    roster_known_dire: int
    # --- требует драфта (post-draft) ---
    hero_exp_decay_diff: Optional[float]
    # --- PART G/H: сила ПУЛА героев команды в текущей мете ---
    # Драфт этого матча НЕ используется: берутся герои, на которых пятёрка
    # играла РАНЬШЕ, и их сила в мете на момент T. Величина известна до
    # драфта, в отличие от самого драфта.
    pool_meta_diff: Optional[float]
    pool_size_min: int
    target: int


class _DecayHeroStrength:
    """Сила героя с экспоненциальным затуханием и shrinkage к 0.5.

    Формула повторяет `hero_strength_schemes.exp_decay` (Phase 9).
    """

    def __init__(self, half_life_days: float = HERO_HALF_LIFE_DAYS,
                 prior: float = HERO_PRIOR_GAMES):
        self.hl = half_life_days
        self.prior = prior
        self._w: Dict[int, float] = defaultdict(float)
        self._g: Dict[int, float] = defaultdict(float)
        self._t: Dict[int, float] = {}

    def _decay(self, h: int, ts: float) -> None:
        last = self._t.get(h)
        if last is None:
            self._t[h] = ts
            return
        dt = (ts - last) / 86400.0
        if dt > 0:
            f = 0.5 ** (dt / self.hl)
            self._w[h] *= f
            self._g[h] *= f
        self._t[h] = ts

    def strength(self, h: int, ts: float) -> float:
        self._decay(h, ts)
        return (self._w.get(h, 0.0) + self.prior / 2.0) / (self._g.get(h, 0.0) + self.prior) - 0.5

    def observe(self, h: int, won: bool, ts: float) -> None:
        self._decay(h, ts)
        self._g[h] += 1.0
        self._w[h] += 1.0 if won else 0.0


class PointInTimeState:
    """Состояние системы. Читается на PREDICT, меняется на UPDATE."""

    def __init__(self):
        self.team_elo: Dict[int, float] = {}
        self.player_elo: Dict[int, float] = {}
        self.player_team_elo: Dict[int, float] = {}   # параллельный team-Elo движка игроков
        self.form: Dict[int, deque] = defaultdict(lambda: deque(maxlen=FORM_MAXLEN))
        self.played: Dict[int, int] = defaultdict(int)
        self.hero = _DecayHeroStrength()
        # пул героев игрока: hero_id -> затухающее число игр
        self.pool: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.pool_t: Dict[int, float] = {}

    # ---------- чтение ----------
    def elo(self, tid: int) -> float:
        return self.team_elo.get(tid, TEAM_ELO_BASE)

    def p_elo(self, pid: int) -> float:
        return self.player_elo.get(pid, PLAYER_ELO_BASE)

    def p_team_elo(self, tid: int) -> float:
        return self.player_team_elo.get(tid, PLAYER_ELO_BASE)

    def recent_winrate(self, tid: int, window: int = FORM_WINDOW) -> Optional[float]:
        d = self.form.get(tid)
        if not d:
            return None
        recent = list(d)[-window:]
        return sum(recent) / len(recent) if recent else None

    def roster_rating(self, roster: FrozenSet[int]) -> float:
        if not roster:
            return PLAYER_ELO_BASE
        return sum(self.p_elo(p) for p in roster) / len(roster)

    def pool_meta_strength(self, roster: FrozenSet[int], ts: float,
                           half_life_days: float = HERO_HALF_LIFE_DAYS
                           ) -> Tuple[Optional[float], int]:
        """Средняя сила героев из пула пятёрки, взвешенная частотой игры.

        Пул затухает так же, как сила героя: герой, на котором игрок не
        играл полгода, почти не влияет. Возвращается (значение, размер
        самого бедного пула) — второй элемент нужен, чтобы не выдавать
        число там, где пула фактически нет.
        """
        vals, sizes = [], []
        for p in roster:
            counts = self.pool.get(p)
            if not counts:
                sizes.append(0)
                continue
            last = self.pool_t.get(p, ts)
            f = 0.5 ** (max(ts - last, 0.0) / 86400.0 / half_life_days)
            tot = 0.0
            acc = 0.0
            for h, c in counts.items():
                w = c * f
                if w <= 1e-6:
                    continue
                tot += w
                acc += w * self.hero.strength(h, ts)
            sizes.append(int(sum(1 for c in counts.values() if c * f > 1e-6)))
            if tot > 0:
                vals.append(acc / tot)
        if not vals:
            return None, 0
        return sum(vals) / len(vals), (min(sizes) if sizes else 0)

    # ---------- изменение ----------
    def apply(self, m: PitMatch) -> None:
        # team Elo (src/ratings/engine.py)
        r_pre, d_pre = self.elo(m.radiant_team_id), self.elo(m.dire_team_id)
        exp_r = 1.0 / (1.0 + 10 ** ((d_pre - r_pre) / 400.0))
        s_r = 1.0 if m.radiant_win else 0.0
        self.team_elo[m.radiant_team_id] = r_pre + TEAM_ELO_K * (s_r - exp_r)
        self.team_elo[m.dire_team_id] = d_pre + TEAM_ELO_K * ((1.0 - s_r) - (1.0 - exp_r))

        # форма и счётчики
        self.form[m.radiant_team_id].append(bool(m.radiant_win))
        self.form[m.dire_team_id].append(not m.radiant_win)
        self.played[m.radiant_team_id] += 1
        self.played[m.dire_team_id] += 1

        # player-Elo (src/datasets/player_features.py) — общий delta на пятёрку
        if m.radiant_roster and m.dire_roster:
            rr = self.roster_rating(m.radiant_roster)
            dr = self.roster_rating(m.dire_roster)
            e_r = 1.0 / (1.0 + 10 ** ((dr - rr) / 400.0))
            delta = PLAYER_ELO_K * (s_r - e_r)
            for p in m.radiant_roster:
                self.player_elo[p] = self.p_elo(p) + delta
            for p in m.dire_roster:
                self.player_elo[p] = self.p_elo(p) - delta
            tr, td = self.p_team_elo(m.radiant_team_id), self.p_team_elo(m.dire_team_id)
            e_t = 1.0 / (1.0 + 10 ** ((td - tr) / 400.0))
            dt = PLAYER_ELO_K * (s_r - e_t)
            self.player_team_elo[m.radiant_team_id] = tr + dt
            self.player_team_elo[m.dire_team_id] = td - dt

        # сила героев
        ts = m.start_time.timestamp()
        for h in m.radiant_picks:
            self.hero.observe(h, m.radiant_win, ts)
        for h in m.dire_picks:
            self.hero.observe(h, not m.radiant_win, ts)

        # пул героев: кто на чём играл. Обновляется ПОСЛЕ матча, как и всё
        # остальное состояние.
        for roster, picks in ((m.radiant_roster, m.radiant_picks),
                              (m.dire_roster, m.dire_picks)):
            if not roster or not picks:
                continue
            for p in roster:
                last = self.pool_t.get(p)
                if last is not None and ts > last:
                    f = 0.5 ** ((ts - last) / 86400.0 / HERO_HALF_LIFE_DAYS)
                    for h in list(self.pool[p]):
                        self.pool[p][h] *= f
                self.pool_t[p] = ts
                for h in picks:
                    self.pool[p][h] += 1.0 / len(picks)


@dataclass(frozen=True)
class RosterProvider:
    """Откуда берётся состав на момент прогноза.

    Ключевое требование фазы: **если состав неизвестен — его нельзя
    подставить сегодняшним.** Провайдер возвращает то, что известно на T,
    и пустое множество, если не известно ничего.
    """
    rosters: Dict[int, Tuple[FrozenSet[int], FrozenSet[int]]] = field(default_factory=dict)

    def for_match(self, match_id: int) -> Tuple[FrozenSet[int], FrozenSet[int]]:
        return self.rosters.get(match_id, (frozenset(), frozenset()))


def build_point_in_time_features(
    matches: Sequence[PitMatch],
    horizon: timedelta,
    roster_provider: Optional[RosterProvider] = None,
    include_draft: bool = False,
    emit_only: Optional[set] = None,
) -> List[PitFeatures]:
    """
    Признаки на момент `start − horizon` для каждого матча.

    `include_draft=False` (по умолчанию) означает режим **PRE_DRAFT**:
    `hero_exp_decay_diff` не вычисляется вовсе, потому что до драфта
    выбранных героев не существует. Подставлять туда что-либо было бы
    выдумыванием данных.

    `roster_provider=None` означает **UNKNOWN_ROSTER**: признаки,
    требующие состава, возвращаются как None, а не как значения по
    умолчанию.
    """
    order = sorted(matches, key=lambda m: (m.start_time, m.match_id))
    events: List[Tuple[float, int, int]] = []
    for i, m in enumerate(order):
        t_pred = (m.start_time - horizon).timestamp()
        # 0 = PREDICT, 1 = UPDATE. При равном времени PREDICT идёт РАНЬШЕ
        # UPDATE: матч, стартующий ровно в момент прогноза, к этому моменту
        # исхода ещё не имеет.
        events.append((t_pred, 0, i))
        events.append((m.start_time.timestamp(), 1, i))
    events.sort()

    st = PointInTimeState()
    out: Dict[int, PitFeatures] = {}

    for t, kind, i in events:
        m = order[i]
        if kind == 1:
            st.apply(m)
            continue

        if emit_only is not None and m.match_id not in emit_only:
            continue          # матч участвует только в обновлении состояния
        pred_at = m.start_time - horizon
        r_form = st.recent_winrate(m.radiant_team_id)
        d_form = st.recent_winrate(m.dire_team_id)
        form_diff = (r_form - d_form) if (r_form is not None and d_form is not None) else None

        rr = dr = frozenset()
        if roster_provider is not None:
            rr, dr = roster_provider.for_match(m.match_id)

        elo_mean_diff = five_vs = None
        if rr and dr:
            r_mean, d_mean = st.roster_rating(rr), st.roster_rating(dr)
            elo_mean_diff = r_mean - d_mean
            five_vs = ((r_mean - st.p_team_elo(m.radiant_team_id))
                       - (d_mean - st.p_team_elo(m.dire_team_id)))

        hero_diff = None
        if include_draft and m.radiant_picks and m.dire_picks:
            ts = pred_at.timestamp()
            r_h = sum(st.hero.strength(h, ts) for h in m.radiant_picks) / len(m.radiant_picks)
            d_h = sum(st.hero.strength(h, ts) for h in m.dire_picks) / len(m.dire_picks)
            hero_diff = r_h - d_h

        pool_diff, pool_min = None, 0
        if rr and dr:
            ts_p = pred_at.timestamp()
            pr, sr = st.pool_meta_strength(rr, ts_p)
            pd_, sd = st.pool_meta_strength(dr, ts_p)
            if pr is not None and pd_ is not None:
                pool_diff = pr - pd_
                pool_min = min(sr, sd)

        out[m.match_id] = PitFeatures(
            match_id=m.match_id, prediction_at=pred_at, start_time=m.start_time,
            horizon_hours=horizon.total_seconds() / 3600.0,
            elo_difference=st.elo(m.radiant_team_id) - st.elo(m.dire_team_id),
            form_3_difference=form_diff,
            radiant_matches_before=st.played[m.radiant_team_id],
            dire_matches_before=st.played[m.dire_team_id],
            elo_mean_diff=elo_mean_diff, five_vs_team_elo_diff=five_vs,
            roster_known_radiant=len(rr), roster_known_dire=len(dr),
            hero_exp_decay_diff=hero_diff,
            pool_meta_diff=pool_diff, pool_size_min=pool_min,
            target=int(m.radiant_win))

    return [out[m.match_id] for m in order if m.match_id in out]
