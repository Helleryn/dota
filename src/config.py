"""
Конфигурация проекта (Phase 4, раздел 1 и 7 docs/architecture.md).

Ключевое требование: границы данных и time-based split — ПАРАМЕТРЫ, не
константы в коде. Историческая глубина OpenDota пока не подтверждена
live-запросом (см. docs/environment-constraints.md), поэтому архитектура не
должна жёстко зависеть от конкретных дат — они меняются здесь, без правок
кода pipeline/backtesting.

Реализовано на stdlib (dataclasses + os.environ), без Pydantic: в этой среде
разработки Pydantic не установлен (см. проверку зависимостей), а конфигурация
должна быть проверяемой прямо сейчас, а не только продекларированной. Проект
ЗАЯВЛЕННО выбирает Pydantic BaseSettings как целевую технологию
(docs/architecture.md, раздел Technology stack) — миграция с этого класса на
Pydantic в Phase 5 тривиальна (те же поля, тот же .env), это не смена
архитектуры, а смена библиотеки валидации.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Optional


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return date.fromisoformat(value)


def _parse_int(value: Optional[str], default: int) -> int:
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    # --- Границы данных (Phase 4, раздел 1 запроса пользователя) ---
    # Не хардкодятся: реальная историческая глубина OpenDota подтверждается
    # через scripts/verify_data_source.py, значения ниже — placeholders,
    # которые обязаны быть переопределены в .env после верификации.
    data_start_date: Optional[date] = None   # нижняя граница backfill
    data_end_date: Optional[date] = None     # верхняя граница backfill (None = "сейчас")
    min_match_date: Optional[date] = None    # нижняя граница включения в ML-датасет

    # --- Time-based split (ADR-005) ---
    train_start: Optional[date] = None
    validation_start: Optional[date] = None
    test_start: Optional[date] = None

    # --- Источники данных ---
    opendota_api_key: Optional[str] = None          # секрет, только из окружения
    opendota_rate_limit_per_min: int = 55            # ниже подтверждённого лимита 60/мин без ключа
    contact_email: Optional[str] = None              # для Liquipedia User-Agent (ToS), НЕ хардкодится

    # --- База данных ---
    database_url: Optional[str] = None               # секрет, только из окружения

    # --- Backtesting ---
    backtest_retrain_interval_days: int = 90         # см. docs/backtesting.md, подбирается эмпирически в Phase 8

    def validate(self) -> None:
        """
        Проверка порядка границ split. Явная ошибка конфигурации при
        старте — лучше, чем тихо обученная на некорректных границах модель.
        """
        dates = [
            ("train_start", self.train_start),
            ("validation_start", self.validation_start),
            ("test_start", self.test_start),
        ]
        known = [(name, d) for name, d in dates if d is not None]
        for (name_a, date_a), (name_b, date_b) in zip(known, known[1:]):
            if date_a >= date_b:
                raise ValueError(
                    f"Некорректный порядок time-based split: "
                    f"{name_a}={date_a} должен быть строго раньше {name_b}={date_b} "
                    f"(ADR-005 — walk-forward split запрещает пересечение периодов)"
                )

        if self.min_match_date and self.data_start_date and self.min_match_date < self.data_start_date:
            raise ValueError(
                f"min_match_date={self.min_match_date} раньше data_start_date="
                f"{self.data_start_date} — датасет не может включать матчи, "
                f"которые pipeline не загружает"
            )


def load_settings(env: Optional[dict] = None) -> Settings:
    """
    Читает конфигурацию из переменных окружения (или переданного dict —
    удобно для тестов, чтобы не трогать реальный os.environ).
    """
    source = env if env is not None else os.environ

    settings = Settings(
        data_start_date=_parse_date(source.get("DATA_START_DATE")),
        data_end_date=_parse_date(source.get("DATA_END_DATE")),
        min_match_date=_parse_date(source.get("MIN_MATCH_DATE")),
        train_start=_parse_date(source.get("TRAIN_START")),
        validation_start=_parse_date(source.get("VALIDATION_START")),
        test_start=_parse_date(source.get("TEST_START")),
        opendota_api_key=source.get("OPENDOTA_API_KEY") or None,
        opendota_rate_limit_per_min=_parse_int(source.get("OPENDOTA_RATE_LIMIT_PER_MIN"), 55),
        contact_email=source.get("CONTACT_EMAIL") or None,
        database_url=source.get("DATABASE_URL") or None,
        backtest_retrain_interval_days=_parse_int(source.get("BACKTEST_RETRAIN_INTERVAL_DAYS"), 90),
    )
    settings.validate()
    return settings


if __name__ == "__main__":
    # Смоук-тест: конфигурация с корректным и некорректным порядком дат.
    ok = load_settings({
        "TRAIN_START": "2018-01-01",
        "VALIDATION_START": "2023-01-01",
        "TEST_START": "2024-01-01",
    })
    print("OK:", ok)

    try:
        load_settings({
            "TRAIN_START": "2024-01-01",
            "VALIDATION_START": "2018-01-01",
        })
        raise SystemExit("ОШИБКА: некорректный порядок дат должен был вызвать ValueError")
    except ValueError as e:
        print("OK, ожидаемая ошибка валидации:", e)
