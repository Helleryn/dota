"""PHASE 17 — загрузка матчей для point-in-time прохода."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Tuple

from sqlalchemy import text

from src.pit.engine import PitMatch, RosterProvider


def load_pit_matches(engine) -> Tuple[List[PitMatch], RosterProvider]:
    """Матчи pro/premium с составами и драфтом.

    Составы берутся из `match_players` — это ФАКТИЧЕСКИЙ состав, то есть
    post-match величина. Для исторического replay это единственное, что
    есть, и Phase 7/8 уже приняли это соглашение. Phase 17 не делает вид,
    что это pre-match знание: провайдер составов передаётся отдельно и
    может быть отключён, чтобы измерить, что будет БЕЗ него (режим
    UNKNOWN_ROSTER).
    """
    # ВАЖНО: поток ОБНОВЛЕНИЙ и поток ПРОГНОЗОВ — разные множества.
    # Замороженный конвейер обновляет team-Elo и форму по ВСЕМ pro-матчам
    # (`_load_pro_matches`: tier IN PRO_LEAGUE_TIERS, обе команды не NULL),
    # а признаки выдаёт только для подмножества с полным драфтом и десятью
    # игроками. Первая версия загрузчика брала одно множество для обоих, и
    # team-Elo разошёлся с замороженным на 2.1 в среднем. Формула была
    # верной — популяция нет.
    with engine.connect() as conn:
        all_pro = conn.execute(text("""
            SELECT m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id, m.radiant_win
            FROM matches m
            LEFT JOIN leagues l ON l.league_id = m.league_id
            WHERE l.tier IN ('professional','premium')
              AND m.radiant_team_id IS NOT NULL AND m.dire_team_id IS NOT NULL
            ORDER BY m.start_time, m.match_id
        """)).fetchall()
        base = conn.execute(text("""
            WITH d AS (
              SELECT pb.match_id,
                     array_agg(pb.hero_id ORDER BY pb.ord) FILTER (WHERE pb.is_pick AND pb.team=0) r_picks,
                     array_agg(pb.hero_id ORDER BY pb.ord) FILTER (WHERE pb.is_pick AND pb.team=1) d_picks
              FROM picks_bans pb GROUP BY 1)
            SELECT m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id,
                   m.radiant_win, d.r_picks, d.d_picks
            FROM matches m
            JOIN leagues l ON l.league_id = m.league_id
            JOIN d ON d.match_id = m.match_id
            WHERE l.tier IN ('professional','premium')
              AND m.radiant_team_id IS NOT NULL AND m.dire_team_id IS NOT NULL
              AND m.radiant_win IS NOT NULL
              AND array_length(d.r_picks,1)=5 AND array_length(d.d_picks,1)=5
            ORDER BY m.start_time, m.match_id
        """)).fetchall()
        players = conn.execute(text(
            "SELECT match_id, account_id, is_radiant FROM match_players "
            "WHERE account_id IS NOT NULL")).fetchall()

    by_match: Dict[int, Tuple[set, set]] = {}
    for p in players:
        r, d = by_match.setdefault(p.match_id, (set(), set()))
        (r if p.is_radiant else d).add(p.account_id)

    # матчи, для которых выдаём признаки
    emit: Dict[int, object] = {}
    rosters: Dict[int, Tuple[frozenset, frozenset]] = {}
    for row in base:
        r, d = by_match.get(row.match_id, (set(), set()))
        if len(r) != 5 or len(d) != 5:
            continue
        st = row.start_time if row.start_time.tzinfo else row.start_time.replace(tzinfo=timezone.utc)
        rr, dd = frozenset(r), frozenset(d)
        rosters[row.match_id] = (rr, dd)
        emit[row.match_id] = (rr, dd, tuple(row.r_picks), tuple(row.d_picks))

    out: List[PitMatch] = []
    for row in all_pro:
        st = row.start_time if row.start_time.tzinfo else row.start_time.replace(tzinfo=timezone.utc)
        e = emit.get(row.match_id)
        rr, dd, rp_, dp_ = e if e else (frozenset(), frozenset(), (), ())
        out.append(PitMatch(
            match_id=row.match_id, start_time=st,
            radiant_team_id=row.radiant_team_id, dire_team_id=row.dire_team_id,
            radiant_win=bool(row.radiant_win) if row.radiant_win is not None else False,
            radiant_roster=rr, dire_roster=dd,
            radiant_picks=rp_, dire_picks=dp_))
    return out, RosterProvider(rosters)
