"""
PHASE 21 — контекст недавних соперников и история встреч (LEVEL 1).

Три класса механизмов, каждый со своим вопросом:

  H2  recent opponent-adjusted strength
      «как команда выступала ОТНОСИТЕЛЬНО силы тех, с кем играла»
  H3  common opponent signal
      «если обе команды играли с одними и теми же — кто справился лучше»
  H4-H6 head-to-head
      «есть ли у пары своя история, не сводимая к разнице сил»

Критическое требование §16: сила соперника берётся **на момент того
матча**, а не сегодняшняя. Поэтому каждый сыгранный матч сохраняется
вместе со снимком Elo обеих сторон, снятым ДО его применения. Взять
сегодняшний Elo было бы утечкой из будущего в прошлое — и её не поймал
бы ни один тест на порядок событий.

Драфт, герои, роли и любые post-match величины здесь отсутствуют: это
LEVEL 1, pre-match слой.

Движок Phase 17 не изменяется — его лента событий повторена.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

# Полураспад общий для всех затухающих величин фазы. Взят из Phase 9/20 и
# НЕ подбирается: подбор был бы скрытым экспериментом сверх объявленных.
HALF_LIFE_DAYS = 90.0
DAY = 86400.0
TEAM_ELO_BASE = 1000.0
TEAM_ELO_K = 16.0

WINDOWS = (7, 14, 30)          # объявлены в плане, не расширяются


@dataclass
class CtxMatch:
    """Матч в терминах LEVEL 1. Ни героев, ни ролей — их здесь нет."""
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_win: bool
    radiant_roster: FrozenSet[int] = frozenset()
    dire_roster: FrozenSet[int] = frozenset()


@dataclass
class _Played:
    """Сыгранный матч со снимком силы НА ТОТ МОМЕНТ.

    `opp_elo` и `own_elo` фиксируются перед применением матча к
    состоянию. Это и есть historical opponent strength: пересчитать её
    из текущего состояния нельзя, потому что оно уже ушло вперёд.
    """
    ts: float
    opponent: int
    won: bool
    own_elo: float
    opp_elo: float
    roster: FrozenSet[int]


@dataclass
class OpponentContextRow:
    match_id: int
    # --- H2: recent opponent-adjusted strength ---
    oas_diff_7: Optional[float]
    oas_diff_14: Optional[float]
    oas_diff_30: Optional[float]
    recent_n_min_7: int
    recent_n_min_14: int
    recent_n_min_30: int
    # --- H7: негативный контроль ---
    recent_winrate_diff_30: Optional[float]
    # --- H3: общие соперники ---
    common_opponent_delta: Optional[float]
    common_opponent_count: int
    # --- H4/H5/H6: история встреч ---
    h2h_residual_decayed: Optional[float]
    h2h_winrate_decayed: Optional[float]
    h2h_matches_decayed: float
    h2h_days_since_last: Optional[float]
    h2h_roster_overlap: Optional[float]
    h2h_residual_roster: Optional[float]
    # --- служебное ---
    prediction_at: datetime
    target: int


def _expected(own: float, opp: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opp - own) / 400.0))


def _w(ts: float, now: float) -> float:
    return 0.5 ** (max(now - ts, 0.0) / DAY / HALF_LIFE_DAYS)


class _Ctx:
    """Состояние walk-forward. Читается на PREDICT, меняется на UPDATE.

    Все чтения чистые: затухание вычисляется на лету и никуда не
    записывается — то же требование, что в Phase 20, после найденного в
    Phase 19 нарушения контракта.
    """

    def __init__(self):
        self.elo: Dict[int, float] = {}
        self.played: Dict[int, List[_Played]] = defaultdict(list)
        self.h2h: Dict[Tuple[int, int], List[_Played]] = defaultdict(list)

    def team_elo(self, t: int) -> float:
        return self.elo.get(t, TEAM_ELO_BASE)

    # ---------- H2 / H7 ----------
    def recent(self, team: int, now: float, window_days: int) -> List[_Played]:
        lo = now - window_days * DAY
        return [p for p in self.played.get(team, ()) if lo <= p.ts < now]

    def oas(self, team: int, now: float, window_days: int
            ) -> Tuple[Optional[float], int]:
        """Opponent-adjusted strength: (значение, число наблюдений).

        `None` при отсутствии матчей — это `missing`, а НЕ ноль. Разница
        принципиальна: ноль означает «сыграла ровно по ожиданию»,
        отсутствие — «не сыграла вовсе».
        """
        ms = self.recent(team, now, window_days)
        if not ms:
            return None, 0
        num = den = 0.0
        for p in ms:
            w = _w(p.ts, now)
            num += w * ((1.0 if p.won else 0.0) - _expected(p.own_elo, p.opp_elo))
            den += w
        return (num / den if den > 0 else None), len(ms)

    def raw_winrate(self, team: int, now: float, window_days: int) -> Optional[float]:
        """H7 — негативный контроль: winrate без поправки на силу соперников."""
        ms = self.recent(team, now, window_days)
        if not ms:
            return None
        return sum(1.0 for p in ms if p.won) / len(ms)

    # ---------- H3 ----------
    def _perf_vs(self, team: int, opp: int, now: float, window_days: int
                 ) -> Optional[float]:
        ms = [p for p in self.recent(team, now, window_days) if p.opponent == opp]
        if not ms:
            return None
        num = den = 0.0
        for p in ms:
            w = _w(p.ts, now)
            num += w * ((1.0 if p.won else 0.0) - _expected(p.own_elo, p.opp_elo))
            den += w
        return num / den if den > 0 else None

    def common_opponents(self, a: int, b: int, now: float, window_days: int = 30
                         ) -> Tuple[Optional[float], int]:
        """Средняя разница перформанса против ОБЩИХ соперников.

        Сравниваются residual'ы, а не winrate: иначе «обе обыграли B»
        и «обе проиграли B» неотличимы от того, насколько сильна была
        сама B в тот момент.
        """
        sa = {p.opponent for p in self.recent(a, now, window_days)}
        sb = {p.opponent for p in self.recent(b, now, window_days)}
        common = (sa & sb) - {a, b}
        if not common:
            return None, 0
        deltas = []
        for o in common:
            pa, pb = self._perf_vs(a, o, now, window_days), self._perf_vs(b, o, now, window_days)
            if pa is not None and pb is not None:
                deltas.append(pa - pb)
        if not deltas:
            return None, 0
        return sum(deltas) / len(deltas), len(deltas)

    # ---------- H4 / H5 / H6 ----------
    def head_to_head(self, a: int, b: int, now: float, roster_a: FrozenSet[int]
                     ) -> Dict[str, Optional[float]]:
        key = (a, b) if a < b else (b, a)
        ms = [p for p in self.h2h.get(key, ()) if p.ts < now]
        # берём записи со стороны a
        mine = [p for p in ms if p.opponent == b]
        if not mine:
            return {"residual": None, "winrate": None, "n": 0.0, "days": None,
                    "overlap": None, "residual_roster": None}
        num_r = num_w = den = 0.0
        num_ro = den_ro = 0.0
        ov_num = ov_den = 0.0
        for p in mine:
            w = _w(p.ts, now)
            act = 1.0 if p.won else 0.0
            num_r += w * (act - _expected(p.own_elo, p.opp_elo))
            num_w += w * act
            den += w
            if roster_a and p.roster:
                ov = len(roster_a & p.roster) / len(roster_a)
                ov_num += w * ov
                ov_den += w
                num_ro += w * ov * (act - _expected(p.own_elo, p.opp_elo))
                den_ro += w * ov
        return {
            "residual": num_r / den if den > 0 else None,
            "winrate": num_w / den if den > 0 else None,
            "n": den,
            "days": (now - max(p.ts for p in mine)) / DAY,
            "overlap": ov_num / ov_den if ov_den > 0 else None,
            "residual_roster": num_ro / den_ro if den_ro > 0 else None,
        }

    # ---------- изменение ----------
    def apply(self, m: CtxMatch) -> None:
        r_pre, d_pre = self.team_elo(m.radiant_team_id), self.team_elo(m.dire_team_id)
        ts = m.start_time.timestamp()

        # снимок силы ДО обновления — это и есть historical opponent strength
        pr = _Played(ts, m.dire_team_id, m.radiant_win, r_pre, d_pre, m.radiant_roster)
        pd_ = _Played(ts, m.radiant_team_id, not m.radiant_win, d_pre, r_pre, m.dire_roster)
        self.played[m.radiant_team_id].append(pr)
        self.played[m.dire_team_id].append(pd_)
        key = (m.radiant_team_id, m.dire_team_id)
        key = key if key[0] < key[1] else (key[1], key[0])
        self.h2h[key].extend((pr, pd_))

        exp_r = _expected(r_pre, d_pre)
        s_r = 1.0 if m.radiant_win else 0.0
        self.elo[m.radiant_team_id] = r_pre + TEAM_ELO_K * (s_r - exp_r)
        self.elo[m.dire_team_id] = d_pre + TEAM_ELO_K * ((1.0 - s_r) - (1.0 - exp_r))


def build_opponent_context(matches: Sequence[CtxMatch],
                           horizon: timedelta,
                           emit_only: Optional[set] = None
                           ) -> List[OpponentContextRow]:
    """Признаки контекста на момент `start − horizon`.

    Лента событий та же, что в Phase 17: при равном времени PREDICT идёт
    строго раньше UPDATE, поэтому матч не может попасть в собственные
    признаки, а матч, стартовавший между T и стартом прогнозируемого, —
    в чужие.
    """
    order = sorted(matches, key=lambda m: (m.start_time, m.match_id))
    events: List[Tuple[float, int, int]] = []
    for i, m in enumerate(order):
        events.append(((m.start_time - horizon).timestamp(), 0, i))
        events.append((m.start_time.timestamp(), 1, i))
    events.sort()

    st = _Ctx()
    out: Dict[int, OpponentContextRow] = {}

    for _, kind, i in events:
        m = order[i]
        if kind == 1:
            st.apply(m)
            continue
        if emit_only is not None and m.match_id not in emit_only:
            continue
        pred_at = m.start_time - horizon
        now = pred_at.timestamp()
        R, D = m.radiant_team_id, m.dire_team_id

        oas: Dict[int, Tuple[Optional[float], Optional[float], int, int]] = {}
        for wd in WINDOWS:
            ar, nr = st.oas(R, now, wd)
            ad, nd = st.oas(D, now, wd)
            oas[wd] = (ar, ad, nr, nd)

        def diff(wd: int) -> Optional[float]:
            ar, ad, _, _ = oas[wd]
            return None if ar is None or ad is None else ar - ad

        wr_r, wr_d = st.raw_winrate(R, now, 30), st.raw_winrate(D, now, 30)
        cod, con = st.common_opponents(R, D, now)
        h = st.head_to_head(R, D, now, m.radiant_roster)

        out[m.match_id] = OpponentContextRow(
            match_id=m.match_id,
            oas_diff_7=diff(7), oas_diff_14=diff(14), oas_diff_30=diff(30),
            recent_n_min_7=min(oas[7][2], oas[7][3]),
            recent_n_min_14=min(oas[14][2], oas[14][3]),
            recent_n_min_30=min(oas[30][2], oas[30][3]),
            recent_winrate_diff_30=(None if wr_r is None or wr_d is None else wr_r - wr_d),
            common_opponent_delta=cod, common_opponent_count=con,
            h2h_residual_decayed=h["residual"], h2h_winrate_decayed=h["winrate"],
            h2h_matches_decayed=h["n"], h2h_days_since_last=h["days"],
            h2h_roster_overlap=h["overlap"], h2h_residual_roster=h["residual_roster"],
            prediction_at=pred_at, target=int(m.radiant_win))

    return [out[m.match_id] for m in order if m.match_id in out]


def load_context_matches(engine) -> List[CtxMatch]:
    """Матчи pro/premium с составами. Ни героев, ни ролей — LEVEL 1."""
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
        players = conn.execute(text(
            "SELECT match_id, account_id, is_radiant FROM match_players "
            "WHERE account_id IS NOT NULL")).fetchall()

    by_match: Dict[int, Tuple[set, set]] = {}
    for p in players:
        r, d = by_match.setdefault(p.match_id, (set(), set()))
        (r if p.is_radiant else d).add(p.account_id)

    out: List[CtxMatch] = []
    for row in rows:
        r, d = by_match.get(row.match_id, (set(), set()))
        st = (row.start_time if row.start_time.tzinfo
              else row.start_time.replace(tzinfo=timezone.utc))
        out.append(CtxMatch(match_id=row.match_id, start_time=st,
                            radiant_team_id=row.radiant_team_id,
                            dire_team_id=row.dire_team_id,
                            radiant_win=bool(row.radiant_win),
                            radiant_roster=frozenset(r), dire_roster=frozenset(d)))
    return out
