"""
Схема БД как SQLAlchemy Core MetaData (ADR-004: Core, не полный ORM — см.
docs/architecture.md, раздел Technology stack). Это единственный источник
истины для структуры таблиц: Alembic-миграции генерируются из этого файла
(`alembic revision --autogenerate`), а не пишутся дублирующим кодом.

Прямое соответствие docs/database-design.md — каждая таблица здесь имеет
там комментарий с обоснованием grain/PK/FK/индексов. Здесь эти обоснования
не повторяются, только их реализация.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()

organizations = Table(
    "organizations",
    metadata,
    Column("organization_id", BigInteger, primary_key=True, autoincrement=True),
    Column("canonical_name", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

teams = Table(
    "teams",
    metadata,
    Column("team_id", BigInteger, primary_key=True, autoincrement=False),  # natural key = OpenDota team_id
    Column("organization_id", BigInteger, ForeignKey("organizations.organization_id"), nullable=True),
    Column("name", Text, nullable=True),
    Column("tag", Text, nullable=True),
    Column("first_seen_at", DateTime(timezone=True), nullable=True),
    Column("last_seen_at", DateTime(timezone=True), nullable=True),
    Index("idx_teams_organization_id", "organization_id"),
)

players = Table(
    "players",
    metadata,
    Column("account_id", BigInteger, primary_key=True, autoincrement=False),  # natural key = Steam account_id
    Column("name", Text, nullable=True),
    Column("first_seen_at", DateTime(timezone=True), nullable=True),
    Column("last_seen_at", DateTime(timezone=True), nullable=True),
)

team_roster_periods = Table(
    "team_roster_periods",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("team_id", BigInteger, ForeignKey("teams.team_id"), nullable=False),
    Column("account_id", BigInteger, ForeignKey("players.account_id"), nullable=False),
    Column("valid_from", DateTime(timezone=True), nullable=False),
    Column("valid_to", DateTime(timezone=True), nullable=True),  # NULL = по настоящий момент
    Column("role", Text, nullable=True),
    Column("source", Text, nullable=False),  # 'reconstructed_from_matches' | 'liquipedia'
    Column("confidence", Float, nullable=True),
    Index("idx_roster_team_account_valid", "team_id", "account_id", "valid_from"),
    Index("idx_roster_valid_range", "team_id", "valid_from", "valid_to"),
)

leagues = Table(
    "leagues",
    metadata,
    Column("league_id", BigInteger, primary_key=True, autoincrement=False),
    Column("name", Text, nullable=True),
    Column("tier", Text, nullable=True),  # 'premium' | 'professional' | 'amateur' | ...
    Index("idx_leagues_tier", "tier"),
)

patches = Table(
    "patches",
    metadata,
    Column("patch_id", Integer, primary_key=True, autoincrement=False),  # соответствует id из dotaconstants
    Column("name", Text, nullable=False, unique=True),  # напр. "7.41"
    Column("released_at", DateTime(timezone=True), nullable=False),
)

heroes = Table(
    "heroes",
    metadata,
    Column("hero_id", Integer, primary_key=True, autoincrement=False),
    Column("name", Text, nullable=False),
    Column("localized_name", Text, nullable=False),
    Column("primary_attr", Text, nullable=True),
    Column("attack_type", Text, nullable=True),
)

matches = Table(
    "matches",
    metadata,
    Column("match_id", BigInteger, primary_key=True, autoincrement=False),
    Column("start_time", DateTime(timezone=True), nullable=False),
    Column("duration_seconds", Integer, nullable=False),
    Column("radiant_team_id", BigInteger, ForeignKey("teams.team_id"), nullable=True),
    Column("dire_team_id", BigInteger, ForeignKey("teams.team_id"), nullable=True),
    Column("radiant_win", Boolean, nullable=False),
    Column("league_id", BigInteger, ForeignKey("leagues.league_id"), nullable=True),
    Column("patch_id", Integer, ForeignKey("patches.patch_id"), nullable=True),
    Column("series_id", BigInteger, nullable=True),
    Column("series_type", SmallInteger, nullable=True),
    Column("source", Text, nullable=False),
    Column("ingested_at", DateTime(timezone=True), nullable=False),
    Index("idx_matches_start_time", "start_time"),
    Index("idx_matches_league_id", "league_id"),
    Index("idx_matches_radiant_team_id", "radiant_team_id"),
    Index("idx_matches_dire_team_id", "dire_team_id"),
)

match_players = Table(
    "match_players",
    metadata,
    Column("match_id", BigInteger, ForeignKey("matches.match_id", ondelete="CASCADE"), primary_key=True),
    Column("player_slot", SmallInteger, primary_key=True),
    Column("account_id", BigInteger, ForeignKey("players.account_id"), nullable=True),
    Column("is_radiant", Boolean, nullable=False),
    Column("hero_id", Integer, ForeignKey("heroes.hero_id"), nullable=True),
    Column("kills", Integer, nullable=True),
    Column("deaths", Integer, nullable=True),
    Column("assists", Integer, nullable=True),
    Column("gold_per_min", Integer, nullable=True),
    Column("xp_per_min", Integer, nullable=True),
    # Phase 11: линия из разбора реплея (1=safe, 2=mid, 3=off, 4=jungle).
    # КРИТИЧНО: это POST-MATCH величина — она появляется только после игры.
    # Использовать её для ТЕКУЩЕГО матча как признак запрещено; допустимо
    # только характеризовать игрока по матчам СТРОГО РАНЬШЕ прогнозируемого
    # (см. docs/features.md и reports/phase11-data-feasibility.md).
    # Это линия, а не позиция 1-5: safelane содержит и керри, и хард-саппорта.
    Column("lane_role", SmallInteger, nullable=True),
    Index("idx_match_players_account_id", "account_id"),
    Index("idx_match_players_hero_id", "hero_id"),
)

picks_bans = Table(
    "picks_bans",
    metadata,
    Column("match_id", BigInteger, ForeignKey("matches.match_id", ondelete="CASCADE"), primary_key=True),
    Column("ord", SmallInteger, primary_key=True),
    Column("is_pick", Boolean, nullable=False),
    Column("hero_id", Integer, ForeignKey("heroes.hero_id"), nullable=True),
    Column("team", SmallInteger, nullable=False),  # 0=radiant, 1=dire
    Index("idx_picks_bans_hero_id", "hero_id"),
)

team_ratings = Table(
    "team_ratings",
    metadata,
    Column("team_id", BigInteger, ForeignKey("teams.team_id"), primary_key=True),
    Column("match_id", BigInteger, ForeignKey("matches.match_id", ondelete="CASCADE"), primary_key=True),
    Column("rating_before", Float, nullable=False),
    Column("rating_after", Float, nullable=False),
    Column("k_factor", Float, nullable=False),
    Column("rating_engine_version", Text, nullable=False),
    Column("computed_at", DateTime(timezone=True), nullable=False),
    Index("idx_team_ratings_team_match", "team_id", "match_id"),
)

feature_sets = Table(
    "feature_sets",
    metadata,
    Column("feature_set_version", Text, primary_key=True),
    Column("description", Text, nullable=True),
    Column("code_git_sha", Text, nullable=True),
    Column("config", JSONB, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

match_features = Table(
    "match_features",
    metadata,
    Column("match_id", BigInteger, ForeignKey("matches.match_id", ondelete="CASCADE"), primary_key=True),
    Column("team_id", BigInteger, ForeignKey("teams.team_id"), primary_key=True),
    Column("feature_set_version", Text, ForeignKey("feature_sets.feature_set_version"), primary_key=True),
    Column("as_of_timestamp", DateTime(timezone=True), nullable=False),
    Column("calculated_at", DateTime(timezone=True), nullable=False),
    Column("features", JSONB, nullable=False),
    Index("idx_match_features_asof", "as_of_timestamp"),
)

models = Table(
    "models",
    metadata,
    Column("model_id", BigInteger, primary_key=True, autoincrement=True),
    Column("name", Text, nullable=False),
    Column("version", Text, nullable=False),
    Column("algorithm", Text, nullable=False),
    Column("feature_set_version", Text, ForeignKey("feature_sets.feature_set_version"), nullable=True),
    Column("train_start", DateTime(timezone=True), nullable=True),
    Column("train_end", DateTime(timezone=True), nullable=True),
    Column("val_start", DateTime(timezone=True), nullable=True),
    Column("val_end", DateTime(timezone=True), nullable=True),
    Column("test_start", DateTime(timezone=True), nullable=True),
    Column("test_end", DateTime(timezone=True), nullable=True),
    Column("config", JSONB, nullable=True),
    Column("metrics", JSONB, nullable=True),
    Column("artifact_path", Text, nullable=True),
    Column("is_active", Boolean, nullable=False, server_default="false"),
    Column("trained_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("name", "version", name="uq_models_name_version"),
    Index("idx_models_is_active", "is_active", postgresql_where=Column("is_active") == True),  # noqa: E712
)

predictions = Table(
    "predictions",
    metadata,
    Column("prediction_id", BigInteger, primary_key=True, autoincrement=True),
    Column("match_id", BigInteger, ForeignKey("matches.match_id"), nullable=False),
    Column("model_id", BigInteger, ForeignKey("models.model_id"), nullable=False),
    Column("predicted_at", DateTime(timezone=True), nullable=False),
    Column("data_cutoff", DateTime(timezone=True), nullable=False),
    Column("team_a_id", BigInteger, ForeignKey("teams.team_id"), nullable=True),
    Column("team_b_id", BigInteger, ForeignKey("teams.team_id"), nullable=True),
    Column("team_a_probability", Float, nullable=False),
    Column("explanation", JSONB, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("idx_predictions_match_id", "match_id"),
    Index("idx_predictions_predicted_at", "predicted_at"),
)

raw_responses = Table(
    "raw_responses",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("source", Text, nullable=False),
    Column("endpoint", Text, nullable=False),
    Column("request_params", JSONB, nullable=True),
    Column("fetched_at", DateTime(timezone=True), nullable=False),
    Column("http_status", Integer, nullable=False),
    Column("response_body", JSONB, nullable=True),
    Column("content_hash", Text, nullable=False),
    Index("idx_raw_responses_source_endpoint_fetched", "source", "endpoint", "fetched_at"),
    Index("idx_raw_responses_content_hash", "content_hash"),
)

ingestion_runs = Table(
    "ingestion_runs",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("source", Text, nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("status", Text, nullable=False),  # 'running' | 'succeeded' | 'failed'
    Column("records_fetched", Integer, nullable=True),
    Column("checkpoint", JSONB, nullable=True),
    Column("error", Text, nullable=True),
    Index("idx_ingestion_runs_source_status", "source", "status", "started_at"),
)


# =====================================================================
# PHASE 15 — shadow validation. Снимок прогноза и запись разрешения.
#
# Два раздельных объекта — это не удобство, а требование фазы: результат
# матча НЕ ДОЛЖЕН иметь возможности изменить прогноз задним числом.
# Снимок неизменяем (триггер в миграции разрешает менять только `state`
# и только вперёд по разрешённому порядку), исход живёт отдельно.
# =====================================================================

prediction_snapshots = Table(
    "prediction_snapshots",
    metadata,
    # Детерминированный идентификатор: хеш от (match_key, prediction_timestamp,
    # версии). Повторный запуск с теми же входами даёт тот же id, поэтому
    # дубликат отсекается первичным ключом, а не тихо создаётся.
    Column("prediction_id", Text, primary_key=True),
    Column("match_id", BigInteger, nullable=True),          # у fixture-режима неизвестен
    Column("match_key", Text, nullable=False),              # стабильный ключ матча
    Column("prediction_timestamp", DateTime(timezone=True), nullable=False),
    Column("match_start_time", DateTime(timezone=True), nullable=True),
    Column("radiant_team_id", BigInteger, nullable=True),
    Column("dire_team_id", BigInteger, nullable=True),
    Column("radiant_team_name", Text, nullable=True),
    Column("dire_team_name", Text, nullable=True),
    Column("patch_id", Integer, nullable=True),
    Column("patch_name", Text, nullable=True),
    Column("league_id", BigInteger, nullable=True),
    Column("tournament", Text, nullable=True),
    Column("features", JSONB, nullable=False),
    Column("raw_probability", Float, nullable=True),
    Column("calibrated_probability", Float, nullable=True),
    Column("confidence", Float, nullable=True),
    Column("decision", Text, nullable=True),                # PREDICT | ABSTAIN
    Column("state", Text, nullable=False),
    Column("invalid_reason", Text, nullable=True),
    Column("source", Text, nullable=False),                 # live_draft | replay | fixture
    Column("model_version", Text, nullable=False),
    Column("feature_version", Text, nullable=False),
    Column("calibration_version", Text, nullable=False),
    Column("prediction_version", Text, nullable=False),
    # --- отметки среза данных (PART C) ---
    Column("data_cutoff", DateTime(timezone=True), nullable=False),
    Column("feature_data_cutoff", DateTime(timezone=True), nullable=True),
    Column("rating_state_timestamp", DateTime(timezone=True), nullable=True),
    Column("roster_state_timestamp", DateTime(timezone=True), nullable=True),
    Column("hero_meta_state_timestamp", DateTime(timezone=True), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("idx_pred_snapshots_match", "match_id"),
    Index("idx_pred_snapshots_state", "state", "prediction_timestamp"),
    Index("idx_pred_snapshots_source", "source", "prediction_timestamp"),
)

prediction_resolutions = Table(
    "prediction_resolutions",
    metadata,
    Column("prediction_id", Text,
           ForeignKey("prediction_snapshots.prediction_id", ondelete="RESTRICT"),
           primary_key=True),
    Column("match_id", BigInteger, nullable=False),
    Column("resolved_at", DateTime(timezone=True), nullable=False),
    Column("radiant_win", Boolean, nullable=False),
    Column("actual_start_time", DateTime(timezone=True), nullable=True),
    Column("correct_raw", Boolean, nullable=True),
    Column("correct_calibrated", Boolean, nullable=True),
    Column("log_loss_raw", Float, nullable=True),
    Column("log_loss_calibrated", Float, nullable=True),
    Column("brier_raw", Float, nullable=True),
    Column("brier_calibrated", Float, nullable=True),
    Column("calibration_error_raw", Float, nullable=True),
    Column("calibration_error_calibrated", Float, nullable=True),
    Column("confidence_bucket", Text, nullable=True),
    Column("resolution_version", Text, nullable=False),
    Index("idx_pred_resolutions_match", "match_id"),
)
