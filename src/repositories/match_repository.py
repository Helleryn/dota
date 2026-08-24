"""
Repository для normalized-слоя: teams/players/leagues/matches/match_players/
picks_bans + справочники heroes/patches (Phase 5, раздел 6-7).

Все записи — upsert (INSERT ... ON CONFLICT DO UPDATE), идемпотентно по
natural key (Phase 5, раздел 16: повторный запуск ingestion не создаёт
дублей). Ничего здесь не решает, что нормализовать и как валидировать —
это src/normalization/ (Phase 5.7-5.8); repository только пишет уже готовые,
провалидированные значения.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

from sqlalchemy import Connection
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.schema import heroes, leagues, match_players, matches, patches, picks_bans, players, teams


def upsert_team(conn: Connection, team_id: int, name: Optional[str], tag: Optional[str], seen_at: datetime) -> None:
    stmt = pg_insert(teams).values(
        team_id=team_id, name=name, tag=tag, first_seen_at=seen_at, last_seen_at=seen_at
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["team_id"],
        set_={
            "name": stmt.excluded.name,
            "tag": stmt.excluded.tag,
            "last_seen_at": stmt.excluded.last_seen_at,
        },
    )
    conn.execute(stmt)


def upsert_player(conn: Connection, account_id: int, name: Optional[str], seen_at: datetime) -> None:
    stmt = pg_insert(players).values(
        account_id=account_id, name=name, first_seen_at=seen_at, last_seen_at=seen_at
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["account_id"],
        set_={"name": stmt.excluded.name, "last_seen_at": stmt.excluded.last_seen_at},
    )
    conn.execute(stmt)


def upsert_league(conn: Connection, league_id: int, name: Optional[str], tier: Optional[str]) -> None:
    stmt = pg_insert(leagues).values(league_id=league_id, name=name, tier=tier)
    stmt = stmt.on_conflict_do_update(
        index_elements=["league_id"], set_={"name": stmt.excluded.name, "tier": stmt.excluded.tier}
    )
    conn.execute(stmt)


def upsert_match(
    conn: Connection,
    match_id: int,
    start_time: datetime,
    duration_seconds: int,
    radiant_team_id: Optional[int],
    dire_team_id: Optional[int],
    radiant_win: bool,
    league_id: Optional[int],
    patch_id: Optional[int],
    series_id: Optional[int],
    series_type: Optional[int],
    source: str,
    ingested_at: datetime,
) -> None:
    stmt = pg_insert(matches).values(
        match_id=match_id,
        start_time=start_time,
        duration_seconds=duration_seconds,
        radiant_team_id=radiant_team_id,
        dire_team_id=dire_team_id,
        radiant_win=radiant_win,
        league_id=league_id,
        patch_id=patch_id,
        series_id=series_id,
        series_type=series_type,
        source=source,
        ingested_at=ingested_at,
    )
    update_cols = ["duration_seconds", "radiant_win", "league_id", "patch_id", "series_id", "series_type"]
    stmt = stmt.on_conflict_do_update(
        index_elements=["match_id"],
        set_={col: getattr(stmt.excluded, col) for col in update_cols},
    )
    conn.execute(stmt)


def upsert_match_players(conn: Connection, match_id: int, players: Iterable) -> None:
    """players — Iterable[RawPlayerMatch] (src/datasources/base.py). Каждый
    элемент — один player_slot; player_slot восстанавливается из is_radiant
    и позиции в списке (Steam API кодирует slot 0-4 radiant, 128-132 dire —
    мы не храним точный исходный slot, только is_radiant + порядковый номер
    в пределах команды, этого достаточно для наших целей: identity игрока
    даёт account_id, не player_slot)."""
    radiant_slot = 0
    dire_slot = 128
    for player in players:
        if player.is_radiant:
            slot = radiant_slot
            radiant_slot += 1
        else:
            slot = dire_slot
            dire_slot += 1

        stmt = pg_insert(match_players).values(
            match_id=match_id,
            player_slot=slot,
            account_id=player.account_id,
            is_radiant=player.is_radiant,
            hero_id=player.hero_id,
            kills=player.kills,
            deaths=player.deaths,
            assists=player.assists,
            gold_per_min=player.gold_per_min,
            xp_per_min=player.xp_per_min,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["match_id", "player_slot"],
            set_={
                "account_id": stmt.excluded.account_id,
                "hero_id": stmt.excluded.hero_id,
                "kills": stmt.excluded.kills,
                "deaths": stmt.excluded.deaths,
                "assists": stmt.excluded.assists,
                "gold_per_min": stmt.excluded.gold_per_min,
                "xp_per_min": stmt.excluded.xp_per_min,
            },
        )
        conn.execute(stmt)


def upsert_picks_bans(conn: Connection, match_id: int, picks: Iterable) -> None:
    """picks — Iterable[RawPickBan]."""
    for pb in picks:
        stmt = pg_insert(picks_bans).values(
            match_id=match_id, ord=pb.order, is_pick=pb.is_pick, hero_id=pb.hero_id, team=pb.team
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["match_id", "ord"],
            set_={"is_pick": stmt.excluded.is_pick, "hero_id": stmt.excluded.hero_id, "team": stmt.excluded.team},
        )
        conn.execute(stmt)


def sync_heroes(conn: Connection, heroes_data: dict) -> int:
    """heroes_data — распарсенный src/normalization/data/heroes.json (dotaconstants)."""
    count = 0
    for entry in heroes_data.values():
        stmt = pg_insert(heroes).values(
            hero_id=entry["id"],
            name=entry["name"],
            localized_name=entry["localized_name"],
            primary_attr=entry.get("primary_attr"),
            attack_type=entry.get("attack_type"),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["hero_id"],
            set_={
                "name": stmt.excluded.name,
                "localized_name": stmt.excluded.localized_name,
                "primary_attr": stmt.excluded.primary_attr,
                "attack_type": stmt.excluded.attack_type,
            },
        )
        conn.execute(stmt)
        count += 1
    return count


def sync_patches(conn: Connection, patches_data: list) -> int:
    """patches_data — распарсенный src/normalization/data/patches.json (dotaconstants)."""
    count = 0
    for entry in patches_data:
        released_at = datetime.fromisoformat(entry["date"].replace("Z", "+00:00"))
        stmt = pg_insert(patches).values(patch_id=entry["id"], name=entry["name"], released_at=released_at)
        stmt = stmt.on_conflict_do_update(
            index_elements=["patch_id"], set_={"name": stmt.excluded.name, "released_at": stmt.excluded.released_at}
        )
        conn.execute(stmt)
        count += 1
    return count
