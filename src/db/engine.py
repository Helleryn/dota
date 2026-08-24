"""Создание SQLAlchemy Engine из Settings (src/config.py)."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine

from src.config import Settings


def make_engine(settings: Settings, *, use_test_database: bool = False) -> Engine:
    # use_test_database=True — только для integration-тестов
    # (tests/integration/test_database.py), которые очищают таблицы перед
    # каждым тестом. Намеренно НЕ падает обратно на database_url, если
    # test_database_url не задан — вызывающий код (тест) сам решает
    # пропустить тесты в этом случае, чтобы не задеть живые данные.
    url = settings.test_database_url if use_test_database else settings.database_url
    if not url:
        var_name = "TEST_DATABASE_URL" if use_test_database else "DATABASE_URL"
        raise ValueError(f"{var_name} не задан в конфигурации (.env)")
    return create_engine(url, future=True)
