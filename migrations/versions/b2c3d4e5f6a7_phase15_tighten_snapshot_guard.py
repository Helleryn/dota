"""PHASE 15: строгая неизменяемость снимка — сравнение всей строки

Первая версия триггера перечисляла защищённые колонки поимённо. Тест
`test_patch_change_does_not_mutate_historical_prediction` показал, что
перечисление неполно: `patch_name` в списке отсутствовал, и обновить его
у опубликованного снимка удавалось. Перечисление колонок в принципе
ненадёжно — при добавлении колонки о нём забудут.

Логика инвертирована: после публикации у снимка разрешено отличаться
ТОЛЬКО поле `state`, всё остальное сравнивается целиком через jsonb.
Новые колонки автоматически попадают под защиту.

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
"""

from alembic import op

revision = "b2c3d4e5f6a7"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None

STRICT = """
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

    -- После публикации отличаться может ТОЛЬКО state. Сравнение всей
    -- строки, а не перечня колонок: новые колонки защищены автоматически.
    IF (to_jsonb(NEW) - 'state') IS DISTINCT FROM (to_jsonb(OLD) - 'state') THEN
        RAISE EXCEPTION
          'опубликованный снимок % неизменяем: разрешено менять только state',
          OLD.prediction_id;
    END IF;

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
"""

LOOSE_ROLLBACK = """
CREATE OR REPLACE FUNCTION prediction_snapshot_guard() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'снимок прогноза удалять нельзя: %', OLD.prediction_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""


def upgrade() -> None:
    op.execute(STRICT)


def downgrade() -> None:
    op.execute(LOOSE_ROLLBACK)
