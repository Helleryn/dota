"""
PHASE 20 — структура состава: неопределённость, позиции, актуальность,
сыгранность, новизна.

Пять механизмов, каждый со своим обоснованием, почему он может добавить
информацию сверх player-Elo (см. `reports/phase20-plan.md`, §3):

  H1  индивидуальная неопределённость игрока  — дисперсия оценки, не среднее
  H2  сила по позициям                        — сопоставление, не разброс
  H3  временная актуальность                  — разность затухающей и обычной
  H4  сыгранность                             — с кем накоплен опыт
  H5  новизна пятёрки                         — похожесть на свою историю

Движок Phase 17 НЕ изменяется: его лента событий повторена здесь, а не
переписана. Правило то же — при равном времени PREDICT строго раньше
UPDATE, поэтому матч не может попасть в собственные признаки.

Важно про H1: неопределённость **индивидуальна**. player-Elo двигает
пятёрку общей дельтой (Phase 19), поэтому любая мера, построенная на
приращениях рейтинга, вырождается в командную величину. Здесь она
строится на СОБСТВЕННОЙ истории игрока — числе его игр с затуханием и
давности последней, — и потому переживает смену команды.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

# Полураспад общий для всех затухающих счётчиков фазы. Значение взято из
# Phase 9 (сила героя) и НЕ подбиралось на VALIDATION: подбор полураспада
# был бы шестым скрытым экспериментом сверх пяти объявленных.
HALF_LIFE_DAYS = 90.0
DAY = 86400.0

PLAYER_ELO_BASE = 1000.0
PLAYER_ELO_K = 16.0

# RD₀ — отклонение рейтинга у игрока без истории. Величина условная:
# признак используется как разность между сторонами и как множитель, то
# есть её масштаб поглощается StandardScaler.
RD0 = 350.0
# κ в аттенюации a = n̄/(n̄+κ). 10 матчей — это p10 опыта по строкам
# (data audit), то есть граница «мало данных» взята из данных, а не из
# головы.
ATTENUATION_KAPPA = 10.0


@dataclass
class LineupMatch:
    """Матч с игроками. `lane_role` и `gpm` — post-match величины и
    применяются ТОЛЬКО к уже сыгранным матчам, при обновлении состояния."""
    match_id: int
    start_time: datetime
    radiant_team_id: Optional[int]
    dire_team_id: Optional[int]
    radiant_win: bool
    radiant: Tuple[int, ...]                  # account_id
    dire: Tuple[int, ...]
    lane_role: Dict[int, Optional[int]]       # account_id -> lane_role
    gpm: Dict[int, Optional[int]]


@dataclass
class LineupRow:
    match_id: int
    # --- H1 неопределённость ---
    rd_mean_diff: Optional[float]
    rd_max_diff: Optional[float]
    elo_mean_diff_attenuated: Optional[float]
    # --- H2 позиции ---
    pos_diff_1: Optional[float]
    pos_diff_2: Optional[float]
    pos_diff_3: Optional[float]
    pos_diff_4: Optional[float]
    pos_diff_5: Optional[float]
    pos_bottleneck: Optional[float]
    pos_best: Optional[float]
    pos_imbalance: Optional[float]
    pos_assigned_min: int
    # --- H3 актуальность ---
    elo_mean_decayed_diff: Optional[float]
    recency_gap: Optional[float]
    days_since_last_max_diff: Optional[float]
    # --- H4 сыгранность ---
    pair_games_mean_diff: Optional[float]
    pair_games_min_diff: Optional[float]
    core3_games_diff: Optional[float]
    synergy_residual_diff: Optional[float]
    # --- H5 новизна ---
    returning_players_diff: Optional[float]
    core_continuity_diff: Optional[float]
    lineup_novelty_diff: Optional[float]
    days_since_lineup_diff: Optional[float]
    # --- служебное ---
    prediction_at: datetime
    target: int


def _decay(prev: float, last_ts: Optional[float], ts: float,
           hl: float = HALF_LIFE_DAYS) -> float:
    if last_ts is None:
        return prev
    dt = (ts - last_ts) / DAY
    return prev * (0.5 ** (dt / hl)) if dt > 0 else prev


class _State:
    """Состояние walk-forward. Читается на PREDICT, меняется на UPDATE.

    В отличие от `PointInTimeState` (Phase 19 нашла там нарушение этого
    правила) все чтения здесь чистые: затухание вычисляется на лету и
    никуда не записывается.
    """

    def __init__(self):
        self.player_elo: Dict[int, float] = {}
        self.n_eff: Dict[int, float] = defaultdict(float)     # затухающие игры
        self.last_ts: Dict[int, float] = {}                   # последняя игра
        self.pos_hist: Dict[int, Counter] = defaultdict(Counter)
        self.pair_n: Dict[Tuple[int, int], float] = defaultdict(float)
        self.pair_ts: Dict[Tuple[int, int], float] = {}
        self.pair_w: Dict[Tuple[int, int], float] = defaultdict(float)  # победы пары
        self.trip_n: Dict[Tuple[int, int, int], float] = defaultdict(float)
        self.trip_ts: Dict[Tuple[int, int, int], float] = {}
        self.last_mates: Dict[int, FrozenSet[int]] = {}       # партнёры в прошлом матче
        self.lineup_since: Dict[FrozenSet[int], float] = {}   # когда пятёрка впервые

    # ---------- чистые чтения ----------
    def elo(self, p: int) -> float:
        return self.player_elo.get(p, PLAYER_ELO_BASE)

    def eff_games(self, p: int, ts: float) -> float:
        return _decay(self.n_eff.get(p, 0.0), self.last_ts.get(p), ts)

    def rd(self, p: int, ts: float) -> float:
        """Отклонение рейтинга: падает с числом игр, растёт с простоем.

        Затухание `n_eff` уже содержит рост от простоя: игрок, не игравший
        год, имеет меньше эффективных игр, а значит больше RD.
        """
        return RD0 / math.sqrt(1.0 + self.eff_games(p, ts))

    def elo_decayed(self, p: int, ts: float) -> float:
        """Рейтинг, притянутый к базе тем сильнее, чем дольше простой."""
        last = self.last_ts.get(p)
        if last is None:
            return PLAYER_ELO_BASE
        dt = max(ts - last, 0.0) / DAY
        f = 0.5 ** (dt / HALF_LIFE_DAYS)
        return PLAYER_ELO_BASE + (self.elo(p) - PLAYER_ELO_BASE) * f

    def days_since(self, p: int, ts: float) -> float:
        last = self.last_ts.get(p)
        return 3650.0 if last is None else max(ts - last, 0.0) / DAY

    def pair_games(self, a: int, b: int, ts: float) -> float:
        k = (a, b) if a < b else (b, a)
        return _decay(self.pair_n.get(k, 0.0), self.pair_ts.get(k), ts)

    def pair_wins(self, a: int, b: int, ts: float) -> float:
        k = (a, b) if a < b else (b, a)
        return _decay(self.pair_w.get(k, 0.0), self.pair_ts.get(k), ts)

    def trip_games(self, t: Tuple[int, int, int], ts: float) -> float:
        return _decay(self.trip_n.get(t, 0.0), self.trip_ts.get(t), ts)

    def predicted_position(self, roster: Sequence[int]
                           ) -> Tuple[Dict[int, Optional[int]], FrozenSet[int]]:
        """Позиция 1–5 как PREDICTED: мода прошлых позиций игрока.

        Статус НЕ повышается до CONFIRMED: источника роли на конкретный
        матч не существует (Phase 16/18), а `lane_role` текущего матча —
        post-match величина (Phase 11).

        Коллизии (двое с одной модой) разрешаются жадно по числу
        наблюдений; кто остался — получает свободные позиции в порядке
        `account_id`, детерминированно.

        Возвращает (назначения, множество игроков, чьё назначение
        ОПИРАЕТСЯ НА ИСТОРИЮ). Второе обязательно: запасное назначение
        игрока без истории — выдуманная величина, и строить на ней
        позиционную разницу нельзя. Тест
        `test_7_positions_of_current_match_are_predicted_not_observed`
        нашёл ровно эту ошибку в первой версии: разницы считались по
        произвольно сопоставленным парам.
        """
        cand: List[Tuple[float, int, int]] = []
        for p in roster:
            for pos, n in self.pos_hist.get(p, {}).items():
                cand.append((n, p, pos))
        cand.sort(key=lambda x: (-x[0], x[1], x[2]))
        out: Dict[int, Optional[int]] = {}
        used_pos, used_p = set(), set()
        for _, p, pos in cand:
            if p in used_p or pos in used_pos:
                continue
            out[p] = pos
            used_p.add(p)
            used_pos.add(pos)
        confident = frozenset(used_p)
        free = [x for x in (1, 2, 3, 4, 5) if x not in used_pos]
        for p in sorted(roster):
            if p not in out:
                out[p] = free.pop(0) if free else None
        return out, confident

    # ---------- изменение ----------
    def apply(self, m: LineupMatch, derive_positions) -> None:
        ts = m.start_time.timestamp()
        rr, dd = m.radiant, m.dire

        # player-Elo: та же формула, что в frozen (общая дельта на пятёрку).
        # Меняется НЕ здесь — здесь она лишь воспроизводится, чтобы H1/H3
        # опирались на тот же уровень, что и базовые признаки.
        if rr and dd:
            r_mean = sum(self.elo(p) for p in rr) / len(rr)
            d_mean = sum(self.elo(p) for p in dd) / len(dd)
            e_r = 1.0 / (1.0 + 10 ** ((d_mean - r_mean) / 400.0))
            delta = PLAYER_ELO_K * ((1.0 if m.radiant_win else 0.0) - e_r)
            for p in rr:
                self.player_elo[p] = self.elo(p) + delta
            for p in dd:
                self.player_elo[p] = self.elo(p) - delta

        # позиции прошлого матча — в историю
        pos = derive_positions(m)

        for roster, won in ((rr, m.radiant_win), (dd, not m.radiant_win)):
            for p in roster:
                self.n_eff[p] = _decay(self.n_eff.get(p, 0.0),
                                       self.last_ts.get(p), ts) + 1.0
                q = pos.get(p)
                if q is not None:
                    self.pos_hist[p][q] += 1
            rl = sorted(roster)
            for i in range(len(rl)):
                for j in range(i + 1, len(rl)):
                    k = (rl[i], rl[j])
                    self.pair_n[k] = _decay(self.pair_n.get(k, 0.0),
                                            self.pair_ts.get(k), ts) + 1.0
                    self.pair_w[k] = _decay(self.pair_w.get(k, 0.0),
                                            self.pair_ts.get(k), ts) + (1.0 if won else 0.0)
                    self.pair_ts[k] = ts
                    for l in range(j + 1, len(rl)):
                        t = (rl[i], rl[j], rl[l])
                        self.trip_n[t] = _decay(self.trip_n.get(t, 0.0),
                                                self.trip_ts.get(t), ts) + 1.0
                        self.trip_ts[t] = ts
            fs = frozenset(roster)
            self.lineup_since.setdefault(fs, ts)
            for p in roster:
                self.last_mates[p] = fs - {p}
        for p in list(rr) + list(dd):
            self.last_ts[p] = ts


def _side_stats(st: _State, roster: Sequence[int], ts: float) -> Dict[str, float]:
    """Все сторонние агрегаты одной пятёрки на момент ts."""
    n = len(roster)
    rds = [st.rd(p, ts) for p in roster]
    eff = [st.eff_games(p, ts) for p in roster]
    pairs = [(a, b) for i, a in enumerate(sorted(roster)) for b in sorted(roster)[i + 1:]]
    pg = [st.pair_games(a, b, ts) for a, b in pairs] or [0.0]
    trips = [(a, b, c) for i, a in enumerate(sorted(roster))
             for j, b in enumerate(sorted(roster)[i + 1:], i + 1)
             for c in sorted(roster)[j + 1:]]
    tg = [st.trip_games(t, ts) for t in trips] or [0.0]

    # сыгранность, очищенная от силы: сколько пара выиграла сверх того,
    # что предсказывает разница player-Elo. Без этого «сыгранность»
    # неотличима от «сильные игроки чаще играют вместе».
    resid = 0.0
    tot = 0.0
    for a, b in pairs:
        g = st.pair_games(a, b, ts)
        if g <= 1e-6:
            continue
        w = st.pair_wins(a, b, ts) / g
        # ожидание по среднему уровню пары относительно базы
        lvl = (st.elo(a) + st.elo(b)) / 2.0
        exp = 1.0 / (1.0 + 10 ** ((PLAYER_ELO_BASE - lvl) / 400.0))
        resid += g * (w - exp)
        tot += g
    synergy_resid = resid / tot if tot > 0 else 0.0

    # новизна: насколько пятёрка похожа на ту, в которой игроки были раньше
    fs = frozenset(roster)
    returning = 0
    best_core = 0
    for p in roster:
        mates = st.last_mates.get(p)
        if mates is None:
            continue
        shared = len(mates & (fs - {p}))
        if shared == len(fs) - 1:
            returning += 1
        best_core = max(best_core, shared + 1)
    novelty = 1.0 - (best_core / n if n else 0.0)
    since = st.lineup_since.get(fs)
    days_lineup = 0.0 if since is None else max(ts - since, 0.0) / DAY

    return {
        "rd_mean": sum(rds) / n, "rd_max": max(rds),
        "eff_mean": sum(eff) / n,
        "elo_mean": sum(st.elo(p) for p in roster) / n,
        "elo_dec_mean": sum(st.elo_decayed(p, ts) for p in roster) / n,
        "days_last_max": max(st.days_since(p, ts) for p in roster),
        "pair_mean": sum(pg) / len(pg), "pair_min": min(pg),
        "core3": sum(tg) / len(tg),
        "synergy_resid": synergy_resid,
        "returning": float(returning), "core_cont": float(best_core),
        "novelty": novelty, "days_lineup": days_lineup,
    }


def build_lineup_features(matches: Sequence[LineupMatch],
                          horizon: timedelta,
                          emit_only: Optional[set] = None,
                          derive_positions=None) -> List[LineupRow]:
    """Признаки H1–H5 на момент `start − horizon`.

    `emit_only` отделяет поток ОБНОВЛЕНИЙ от потока ВЫДАЧИ: состояние
    обновляется по всем матчам, признаки выдаются только для нужных. Это
    то же разделение, которое Phase 17 пришлось вводить после того, как
    единое множество развело team-Elo с замороженным на 2.1.
    """
    if derive_positions is None:
        from src.datasets.player_hero_features import derive_positions as _dp

        def derive_positions(m: LineupMatch) -> Dict[int, Optional[int]]:
            class _P:
                def __init__(self, aid, lr, g):
                    self.account_id, self.lane_role, self.gold_per_min = aid, lr, g
            out: Dict[int, Optional[int]] = {}
            for side in (m.radiant, m.dire):
                out.update(_dp([_P(a, m.lane_role.get(a), m.gpm.get(a)) for a in side]))
            return out

    order = sorted(matches, key=lambda m: (m.start_time, m.match_id))
    events: List[Tuple[float, int, int]] = []
    for i, m in enumerate(order):
        events.append(((m.start_time - horizon).timestamp(), 0, i))
        events.append((m.start_time.timestamp(), 1, i))
    events.sort()

    st = _State()
    out: Dict[int, LineupRow] = {}

    for _, kind, i in events:
        m = order[i]
        if kind == 1:
            st.apply(m, derive_positions)
            continue
        if emit_only is not None and m.match_id not in emit_only:
            continue
        pred_at = m.start_time - horizon
        ts = pred_at.timestamp()
        rr, dd = m.radiant, m.dire
        if len(rr) != 5 or len(dd) != 5:
            continue
        R, D = _side_stats(st, rr, ts), _side_stats(st, dd, ts)

        # H2: позиции сопоставляются попарно, а не усредняются
        pr, conf_r = st.predicted_position(rr)
        pd_, conf_d = st.predicted_position(dd)
        # В сопоставление попадают ТОЛЬКО игроки с историей: запасное
        # назначение — выдуманная пара, а не наблюдение.
        by_pos_r = {v: k for k, v in pr.items() if v and k in conf_r}
        by_pos_d = {v: k for k, v in pd_.items() if v and k in conf_d}
        pos_diffs: List[Optional[float]] = []
        for q in (1, 2, 3, 4, 5):
            a, b = by_pos_r.get(q), by_pos_d.get(q)
            pos_diffs.append(None if a is None or b is None
                             else st.elo(a) - st.elo(b))
        known = [x for x in pos_diffs if x is not None]
        assigned = min(len(conf_r), len(conf_d))   # только выведенные из истории

        att = (R["eff_mean"] + D["eff_mean"]) / 2.0
        a_mult = att / (att + ATTENUATION_KAPPA)

        out[m.match_id] = LineupRow(
            match_id=m.match_id,
            rd_mean_diff=R["rd_mean"] - D["rd_mean"],
            rd_max_diff=R["rd_max"] - D["rd_max"],
            elo_mean_diff_attenuated=(R["elo_mean"] - D["elo_mean"]) * a_mult,
            pos_diff_1=pos_diffs[0], pos_diff_2=pos_diffs[1], pos_diff_3=pos_diffs[2],
            pos_diff_4=pos_diffs[3], pos_diff_5=pos_diffs[4],
            pos_bottleneck=min(known) if known else None,
            pos_best=max(known) if known else None,
            pos_imbalance=(max(known) - min(known)) if known else None,
            pos_assigned_min=assigned,
            elo_mean_decayed_diff=R["elo_dec_mean"] - D["elo_dec_mean"],
            recency_gap=((R["elo_dec_mean"] - D["elo_dec_mean"])
                         - (R["elo_mean"] - D["elo_mean"])),
            days_since_last_max_diff=R["days_last_max"] - D["days_last_max"],
            pair_games_mean_diff=R["pair_mean"] - D["pair_mean"],
            pair_games_min_diff=R["pair_min"] - D["pair_min"],
            core3_games_diff=R["core3"] - D["core3"],
            synergy_residual_diff=R["synergy_resid"] - D["synergy_resid"],
            returning_players_diff=R["returning"] - D["returning"],
            core_continuity_diff=R["core_cont"] - D["core_cont"],
            lineup_novelty_diff=R["novelty"] - D["novelty"],
            days_since_lineup_diff=R["days_lineup"] - D["days_lineup"],
            prediction_at=pred_at, target=int(m.radiant_win))

    return [out[m.match_id] for m in order if m.match_id in out]


def load_lineup_matches(engine) -> List[LineupMatch]:
    """Матчи pro/premium с игроками, lane_role и GPM.

    `lane_role` и `gpm` нужны для вывода позиции прошлых матчей. К
    текущему матчу они не применяются — это post-match величины (Phase 11).
    """
    from datetime import timezone

    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id,
                   m.radiant_win
            FROM matches m
            LEFT JOIN leagues l ON l.league_id = m.league_id
            WHERE l.tier IN ('professional','premium')
              AND m.radiant_team_id IS NOT NULL AND m.dire_team_id IS NOT NULL
              AND m.radiant_win IS NOT NULL
            ORDER BY m.start_time, m.match_id
        """)).fetchall()
        players = conn.execute(text("""
            SELECT mp.match_id, mp.account_id, mp.is_radiant, mp.lane_role,
                   mp.gold_per_min
            FROM match_players mp
            WHERE mp.account_id IS NOT NULL
        """)).fetchall()

    by_match: Dict[int, Tuple[List[int], List[int], Dict[int, Optional[int]],
                              Dict[int, Optional[int]]]] = {}
    for p in players:
        r, d, lr, g = by_match.setdefault(p.match_id, ([], [], {}, {}))
        (r if p.is_radiant else d).append(p.account_id)
        lr[p.account_id] = p.lane_role
        g[p.account_id] = p.gold_per_min

    out: List[LineupMatch] = []
    for row in rows:
        r, d, lr, g = by_match.get(row.match_id, ([], [], {}, {}))
        st = (row.start_time if row.start_time.tzinfo
              else row.start_time.replace(tzinfo=timezone.utc))
        out.append(LineupMatch(match_id=row.match_id, start_time=st,
                               radiant_team_id=row.radiant_team_id,
                               dire_team_id=row.dire_team_id,
                               radiant_win=bool(row.radiant_win),
                               radiant=tuple(sorted(r)), dire=tuple(sorted(d)),
                               lane_role=lr, gpm=g))
    return out
