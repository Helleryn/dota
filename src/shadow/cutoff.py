"""
PHASE 15 — проверка среза данных (PART C).

Единственное правило: **NO DATA AFTER prediction_timestamp**.

Нарушение не смягчается и не логируется как предупреждение — прогноз
получает состояние INVALID. Причина такая: в shadow-режиме цена ложного
«всё в порядке» несопоставимо выше цены потерянного прогноза. Прогноз,
про который нельзя доказать отсутствие будущего, бесполезен целиком.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Отметки состояния, каждая из которых обязана быть не позже прогноза.
CUTOFF_FIELDS = (
    "data_cutoff",
    "feature_data_cutoff",
    "rating_state_timestamp",
    "roster_state_timestamp",
    "hero_meta_state_timestamp",
)


@dataclass(frozen=True)
class CutoffViolation:
    field: str
    value: datetime
    prediction_timestamp: datetime

    @property
    def lateness_seconds(self) -> float:
        return (self.value - self.prediction_timestamp).total_seconds()

    def __str__(self) -> str:
        return (f"{self.field}={self.value.isoformat()} позже прогноза "
                f"{self.prediction_timestamp.isoformat()} на "
                f"{self.lateness_seconds:.0f} с")


def check_snapshot(snapshot) -> List[CutoffViolation]:
    """Возвращает список нарушений. Пустой список = прогноз чист."""
    out: List[CutoffViolation] = []
    pt = snapshot.prediction_timestamp
    for f in CUTOFF_FIELDS:
        v = getattr(snapshot, f, None)
        if v is None:
            continue
        if v > pt:
            out.append(CutoffViolation(f, v, pt))
    return out


def check_prediction_precedes_match(snapshot) -> Optional[str]:
    """PART W: прогноз обязан существовать ДО начала матча.

    Отдельная проверка, а не часть предыдущей: срез данных и момент
    начала матча — разные вещи, и путать их причины отказа не следует.
    """
    if snapshot.match_start_time is None:
        return None
    if snapshot.prediction_timestamp >= snapshot.match_start_time:
        return "match_already_started"
    return None


def validate(snapshot) -> Tuple[bool, Optional[str], List[CutoffViolation]]:
    """(пригоден, причина отказа, список нарушений)."""
    v = check_snapshot(snapshot)
    if v:
        return False, "future_data_detected", v
    late = check_prediction_precedes_match(snapshot)
    if late:
        return False, late, []
    return True, None, []


def audit_rows(rows) -> Dict[str, object]:
    """Массовая проверка уже сохранённых снимков — для отчёта и для
    команды `calibration-status`. Работает с любыми объектами, у которых
    есть нужные атрибуты."""
    total, clean, violated = 0, 0, []
    for r in rows:
        total += 1
        ok, reason, viol = validate(r)
        if ok:
            clean += 1
        else:
            violated.append({"prediction_id": getattr(r, "prediction_id", None),
                             "reason": reason,
                             "violations": [str(x) for x in viol]})
    return {"total": total, "clean": clean, "violated": len(violated),
            "details": violated[:50]}
