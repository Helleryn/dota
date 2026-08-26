"""PHASE 15: снимки прогнозов и записи разрешения (shadow validation)

Ключевая часть миграции — не таблицы, а ТРИГГЕР неизменяемости.

Требование фазы: опубликованный прогноз нельзя изменить задним числом,
и результат матча не должен иметь такой возможности в принципе. Если бы
это обеспечивалось только дисциплиной кода, любая будущая правка
репозитория молча сломала бы гарантию. Поэтому запрет живёт в СУБД:
у снимка в состоянии PUBLISHED и позже разрешено менять ТОЛЬКО поле
`state`, и только вперёд по разрешённому порядку. Любая попытка изменить
вероятность, признаки, версии или отметки среза отклоняется.

Revision ID: a1b2c3d4e5f6
Revises: 3428dd53fcd0
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "a1b2c3d4e5f6"
down_revision = "3428dd53fcd0"
branch_labels = None
depends_on = None

# Порядок состояний. INVALID достижим из любого состояния — это отметка
# «прогноз непригоден», а не шаг вперёд.
STATE_ORDER = [
    "DISCOVERED", "FEATURES_READY", "PREDICTED", "CALIBRATED", "PUBLISHED",
    "MATCH_STARTED", "MATCH_FINISHED", "RESOLVED",
]

IMMUTABLE_TRIGGER = """
CREATE OR REPLACE FUNCTION prediction_snapshot_guard() RETURNS trigger AS $$
DECLARE
    old_rank int;
    new_rank int;
    order_arr text[] := ARRAY['DISCOVERED','FEATURES_READY','PREDICTED',
                              'CALIBRATED','PUBLISHED','MATCH_STARTED',
                              'MATCH_FINISHED','RESOLVED'];
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'снимок прогноза удалять нельзя: %', OLD.prediction_id;
    END IF;

    -- До публикации снимок ещё собирается: правки разрешены.
    IF array_position(order_arr, OLD.state) IS NOT NULL
       AND array_position(order_arr, OLD.state) < array_position(order_arr, 'PUBLISHED')
       AND OLD.state <> 'INVALID' THEN
        RETURN NEW;
    END IF;

    -- После публикации меняться может ТОЛЬКО состояние.
    IF ROW(NEW.*) IS DISTINCT FROM ROW(OLD.*) THEN
        IF NEW.prediction_id     IS DISTINCT FROM OLD.prediction_id
        OR NEW.match_key         IS DISTINCT FROM OLD.match_key
        OR NEW.prediction_timestamp IS DISTINCT FROM OLD.prediction_timestamp
        OR NEW.features          IS DISTINCT FROM OLD.features
        OR NEW.raw_probability   IS DISTINCT FROM OLD.raw_probability
        OR NEW.calibrated_probability IS DISTINCT FROM OLD.calibrated_probability
        OR NEW.confidence        IS DISTINCT FROM OLD.confidence
        OR NEW.decision          IS DISTINCT FROM OLD.decision
        OR NEW.model_version     IS DISTINCT FROM OLD.model_version
        OR NEW.feature_version   IS DISTINCT FROM OLD.feature_version
        OR NEW.calibration_version IS DISTINCT FROM OLD.calibration_version
        OR NEW.prediction_version  IS DISTINCT FROM OLD.prediction_version
        OR NEW.data_cutoff       IS DISTINCT FROM OLD.data_cutoff
        OR NEW.feature_data_cutoff IS DISTINCT FROM OLD.feature_data_cutoff
        OR NEW.rating_state_timestamp IS DISTINCT FROM OLD.rating_state_timestamp
        OR NEW.roster_state_timestamp IS DISTINCT FROM OLD.roster_state_timestamp
        OR NEW.hero_meta_state_timestamp IS DISTINCT FROM OLD.hero_meta_state_timestamp
        OR NEW.created_at        IS DISTINCT FROM OLD.created_at
        THEN
            RAISE EXCEPTION
              'опубликованный снимок % неизменяем: разрешено менять только state',
              OLD.prediction_id;
        END IF;
    END IF;

    -- Состояние двигается только вперёд; INVALID достижим всегда.
    IF NEW.state <> OLD.state AND NEW.state <> 'INVALID' THEN
        old_rank := array_position(order_arr, OLD.state);
        new_rank := array_position(order_arr, NEW.state);
        IF old_rank IS NULL OR new_rank IS NULL OR new_rank < old_rank THEN
            RAISE EXCEPTION 'недопустимый переход состояния % -> % (снимок %)',
                OLD.state, NEW.state, OLD.prediction_id;
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_prediction_snapshot_guard
BEFORE UPDATE OR DELETE ON prediction_snapshots
FOR EACH ROW EXECUTE FUNCTION prediction_snapshot_guard();
"""


def upgrade() -> None:
    op.create_table(
        "prediction_snapshots",
        sa.Column("prediction_id", sa.Text(), primary_key=True),
        sa.Column("match_id", sa.BigInteger(), nullable=True),
        sa.Column("match_key", sa.Text(), nullable=False),
        sa.Column("prediction_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("match_start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("radiant_team_id", sa.BigInteger(), nullable=True),
        sa.Column("dire_team_id", sa.BigInteger(), nullable=True),
        sa.Column("radiant_team_name", sa.Text(), nullable=True),
        sa.Column("dire_team_name", sa.Text(), nullable=True),
        sa.Column("patch_id", sa.Integer(), nullable=True),
        sa.Column("patch_name", sa.Text(), nullable=True),
        sa.Column("league_id", sa.BigInteger(), nullable=True),
        sa.Column("tournament", sa.Text(), nullable=True),
        sa.Column("features", JSONB(), nullable=False),
        sa.Column("raw_probability", sa.Float(), nullable=True),
        sa.Column("calibrated_probability", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("decision", sa.Text(), nullable=True),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("invalid_reason", sa.Text(), nullable=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("feature_version", sa.Text(), nullable=False),
        sa.Column("calibration_version", sa.Text(), nullable=False),
        sa.Column("prediction_version", sa.Text(), nullable=False),
        sa.Column("data_cutoff", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_data_cutoff", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rating_state_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("roster_state_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("hero_meta_state_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_pred_snapshots_match", "prediction_snapshots", ["match_id"])
    op.create_index("idx_pred_snapshots_state", "prediction_snapshots",
                    ["state", "prediction_timestamp"])
    op.create_index("idx_pred_snapshots_source", "prediction_snapshots",
                    ["source", "prediction_timestamp"])

    op.create_table(
        "prediction_resolutions",
        sa.Column("prediction_id", sa.Text(),
                  sa.ForeignKey("prediction_snapshots.prediction_id", ondelete="RESTRICT"),
                  primary_key=True),
        sa.Column("match_id", sa.BigInteger(), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("radiant_win", sa.Boolean(), nullable=False),
        sa.Column("actual_start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("correct_raw", sa.Boolean(), nullable=True),
        sa.Column("correct_calibrated", sa.Boolean(), nullable=True),
        sa.Column("log_loss_raw", sa.Float(), nullable=True),
        sa.Column("log_loss_calibrated", sa.Float(), nullable=True),
        sa.Column("brier_raw", sa.Float(), nullable=True),
        sa.Column("brier_calibrated", sa.Float(), nullable=True),
        sa.Column("calibration_error_raw", sa.Float(), nullable=True),
        sa.Column("calibration_error_calibrated", sa.Float(), nullable=True),
        sa.Column("confidence_bucket", sa.Text(), nullable=True),
        sa.Column("resolution_version", sa.Text(), nullable=False),
    )
    op.create_index("idx_pred_resolutions_match", "prediction_resolutions", ["match_id"])

    op.execute(IMMUTABLE_TRIGGER)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_prediction_snapshot_guard ON prediction_snapshots")
    op.execute("DROP FUNCTION IF EXISTS prediction_snapshot_guard()")
    op.drop_table("prediction_resolutions")
    op.drop_table("prediction_snapshots")
