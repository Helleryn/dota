"""
PHASE 16 — общий контракт источника внешних данных.

Центральная идея фазы: **любой внешний факт несёт происхождение и три
временные метки**, и без них он в систему не попадает.

## Три метки (PART I)

| Метка | Смысл | Чем НЕ является |
|---|---|---|
| `observed_at` | когда НАША система увидела факт | не когда факт возник |
| `effective_at` | с какого момента факт действует | не когда мы о нём узнали |
| `event_at` | когда происходит само событие | не когда о нём объявили |

Пример: замена объявлена в 14:00, действует с матча в 18:00. Прогноз в
13:00 обязан её не знать, прогноз в 15:00 — обязан. Без разделения
`observed_at` и `effective_at` это выразить невозможно, и именно здесь
возникает самая незаметная утечка: выгруженный сегодня состав молча
выдаётся за состав месячной давности.

## Доверие к источнику ≠ уверенность модели (PART M)

`SourceConfidence` описывает, насколько надёжен ФАКТ (официальный анонс
против вывода по косвенным данным). Это не имеет отношения к `confidence`
прогноза, который равен `|p − 0.5|`. Смешивать их нельзя: модель может
быть очень уверена в матче, о составе которого мы почти ничего не знаем.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar


class SourceConfidence(str, Enum):
    """Насколько надёжен сам факт."""
    HIGH = "HIGH"        # официальный источник (Valve, анонс команды)
    MEDIUM = "MEDIUM"    # сообщество со структурированными данными
    LOW = "LOW"          # вывод по косвенным признакам
    UNKNOWN = "UNKNOWN"  # происхождение неизвестно — в признаки не идёт


class SourceStatus(str, Enum):
    OK = "OK"
    EMPTY = "EMPTY"                  # источник ответил, данных нет
    UNAVAILABLE = "UNAVAILABLE"      # источник не ответил
    RATE_LIMITED = "RATE_LIMITED"    # 429 — результат, а не препятствие
    FORBIDDEN = "FORBIDDEN"          # нужен ключ
    SCHEMA_CHANGED = "SCHEMA_CHANGED"


@dataclass(frozen=True)
class Provenance:
    """Происхождение факта. Обязательно у каждой записи."""
    source: str
    observed_at: datetime
    confidence: SourceConfidence
    url: Optional[str] = None
    raw_id: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "observed_at": self.observed_at.isoformat(),
                "confidence": self.confidence.value, "url": self.url,
                "raw_id": self.raw_id}


T = TypeVar("T")


@dataclass(frozen=True)
class SourceResult(Generic[T]):
    """Ответ источника. Пустой результат и отказ источника — РАЗНЫЕ вещи.

    Phase 15 показала, почему это важно: молчание источника выглядит как
    спокойный день, если статус не различать.
    """
    status: SourceStatus
    items: List[T] = field(default_factory=list)
    provenance: Optional[Provenance] = None
    error: Optional[str] = None
    retry_after_seconds: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.status == SourceStatus.OK


class SourceAdapter:
    """Базовый адаптер. Наследники не должны глотать ошибки: любая
    неудача превращается в SourceResult с явным статусом."""

    name: str = "abstract"
    confidence: SourceConfidence = SourceConfidence.UNKNOWN
    requires_key: bool = False

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "confidence": self.confidence.value,
                "requires_key": self.requires_key}
