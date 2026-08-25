"""
DatasetBuilder (Phase 5, раздел 22-24): читает матчи из PostgreSQL в
хронологическом порядке, строит Feature Set 0 через build_feature_set_0
(walk-forward, leakage-safe), сохраняет per-team признаки в match_features
(docs/database-design.md), возвращает готовый ML-датасет (pandas DataFrame,
одна строка на матч — team_a/team_b = radiant/dire, см. обоснование в
docs/database-design.md, раздел match_features).

Только про-матчи (PRO_LEAGUE_TIERS) — тот же quality-фильтр, что и
дальше по всему проекту (ADR-001).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
from sqlalchemy import Engine, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.datasets.feature_set_0 import FeatureRow, build_feature_set_0
from src.db.schema import feature_sets, leagues, match_features, matches
from src.normalization.validate import PRO_LEAGUE_TIERS

FEATURE_SET_VERSION = "v0_baseline"


@dataclass(frozen=True)
class DatasetBuildResult:
    feature_set_version: str
    rows: List[FeatureRow]
    dataframe: "pd.DataFrame"


def _load_pro_matches(engine: Engine) -> list:
    """
    Только матчи с league_tier из PRO_LEAGUE_TIERS, отсортированные по
    start_time — та же выборка, что использовалась бы для тренировки.

    Tie-breaker: `match_id` ASC при равном `start_time` (Phase 6.5, раздел
    10 задания). На непрерывном 2021-2026 датасете реально встречается 192
    группы матчей с идентичным `start_time` (одна и та же секунда —
    разрешение таймстампа OpenDota) — без вторичного детерминированного
    ключа порядок таких матчей внутри группы не гарантирован Postgres между
    прогонами (может отличаться при повторном запросе), что ломает
    воспроизводимость walk-forward Elo/recent-form. `match_id` — не имеет
    отношения к исходу матча, назначается OpenDota независимо от него, т.е.
    не вносит утечку, только детерминированность.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                matches.c.match_id,
                matches.c.start_time,
                matches.c.radiant_team_id,
                matches.c.dire_team_id,
                matches.c.radiant_win,
            )
            .select_from(matches.join(leagues, matches.c.league_id == leagues.c.league_id, isouter=True))
            .where(leagues.c.tier.in_(PRO_LEAGUE_TIERS))
            .where(matches.c.radiant_team_id.is_not(None))
            .where(matches.c.dire_team_id.is_not(None))
            .order_by(matches.c.start_time.asc(), matches.c.match_id.asc())
        ).all()
    return rows


def _rows_to_dataframe(rows: List[FeatureRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": r.match_id,
                "as_of_timestamp": r.as_of_timestamp,
                "radiant_team_id": r.radiant_team_id,
                "dire_team_id": r.dire_team_id,
                "radiant_elo": r.radiant_elo,
                "dire_elo": r.dire_elo,
                "elo_difference": r.elo_difference,
                "radiant_recent_winrate": r.radiant_recent_winrate,
                "dire_recent_winrate": r.dire_recent_winrate,
                "recent_winrate_difference": r.recent_winrate_difference,
                "radiant_days_since_last_match": r.radiant_days_since_last_match,
                "dire_days_since_last_match": r.dire_days_since_last_match,
                "radiant_matches_played_before": r.radiant_matches_played_before,
                "dire_matches_played_before": r.dire_matches_played_before,
                "radiant_win": r.radiant_win,
            }
            for r in rows
        ]
    )


def _persist_match_features(engine: Engine, rows: List[FeatureRow], feature_set_version: str) -> None:
    """Пишет per-team записи в match_features (docs/database-design.md) —
    одна строка на (match_id, team_id), не на пару. Идемпотентно (upsert)."""
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            pg_insert(feature_sets)
            .values(
                feature_set_version=feature_set_version,
                description="Team Elo (walk-forward) + recent form (Phase 5, Feature Set 0)",
                code_git_sha=None,
                config={"rating_engine": "elo_k32_base1000", "recent_form_window": 5},
                created_at=now,
            )
            .on_conflict_do_nothing(index_elements=["feature_set_version"])
        )

        for r in rows:
            for team_id, side in ((r.radiant_team_id, "radiant"), (r.dire_team_id, "dire")):
                features = (
                    {
                        "elo": r.radiant_elo,
                        "elo_difference": r.elo_difference,
                        "recent_winrate": r.radiant_recent_winrate,
                        "recent_winrate_difference": r.recent_winrate_difference,
                        "days_since_last_match": r.radiant_days_since_last_match,
                        "matches_played_before": r.radiant_matches_played_before,
                        "is_radiant": True,
                    }
                    if side == "radiant"
                    else {
                        "elo": r.dire_elo,
                        "elo_difference": -r.elo_difference,
                        "recent_winrate": r.dire_recent_winrate,
                        "recent_winrate_difference": (
                            -r.recent_winrate_difference if r.recent_winrate_difference is not None else None
                        ),
                        "days_since_last_match": r.dire_days_since_last_match,
                        "matches_played_before": r.dire_matches_played_before,
                        "is_radiant": False,
                    }
                )
                stmt = pg_insert(match_features).values(
                    match_id=r.match_id,
                    team_id=team_id,
                    feature_set_version=feature_set_version,
                    as_of_timestamp=r.as_of_timestamp,
                    calculated_at=now,
                    features=features,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["match_id", "team_id", "feature_set_version"],
                    set_={
                        "as_of_timestamp": stmt.excluded.as_of_timestamp,
                        "calculated_at": stmt.excluded.calculated_at,
                        "features": stmt.excluded.features,
                    },
                )
                conn.execute(stmt)


def build_dataset(engine: Engine, *, persist: bool = True) -> DatasetBuildResult:
    pro_matches = _load_pro_matches(engine)
    rows = build_feature_set_0(pro_matches)
    df = _rows_to_dataframe(rows)

    if persist and rows:
        _persist_match_features(engine, rows, FEATURE_SET_VERSION)

    return DatasetBuildResult(feature_set_version=FEATURE_SET_VERSION, rows=rows, dataframe=df)
