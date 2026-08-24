"""Создание SQLAlchemy Engine из Settings (src/config.py)."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine

from src.config import Settings


def make_engine(settings: Settings) -> Engine:
    if not settings.database_url:
        raise ValueError("DATABASE_URL не задан в конфигурации (.env)")
    return create_engine(settings.database_url, future=True)
