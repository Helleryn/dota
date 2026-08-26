"""
PHASE 15 — неизменяемый снимок прогноза (PART B) и запись разрешения (PART F).

Разделение на два объекта — требование фазы, а не удобство: исход матча
не должен иметь возможности изменить прогноз задним числом. Снимок
неизменяем, исход живёт отдельной записью и ссылается на снимок.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, Optional

from src.shadow import states, versions


def make_prediction_id(match_key: str, prediction_timestamp: datetime,
                       prediction_version: str) -> str:
    """Детерминированный идентификатор.

    Он намеренно НЕ случайный: повторный запуск с теми же входами обязан
    дать тот же id, чтобы дубликат отсекался первичным ключом, а не
    создавался молча (сценарий PART U-8). Время округляется до секунды —
    иначе два запуска в одну секунду дали бы разные id из-за микросекунд.
    """
    ts = prediction_timestamp.replace(microsecond=0).isoformat()
    raw = f"{match_key}|{ts}|{prediction_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class PredictionSnapshot:
    prediction_id: str
    match_key: str
    prediction_timestamp: datetime
    features: Dict[str, float]
    data_cutoff: datetime
    source: str
    state: str = states.DISCOVERED
    match_id: Optional[int] = None
    match_start_time: Optional[datetime] = None
    radiant_team_id: Optional[int] = None
    dire_team_id: Optional[int] = None
    radiant_team_name: Optional[str] = None
    dire_team_name: Optional[str] = None
    patch_id: Optional[int] = None
    patch_name: Optional[str] = None
    league_id: Optional[int] = None
    tournament: Optional[str] = None
    raw_probability: Optional[float] = None
    calibrated_probability: Optional[float] = None
    confidence: Optional[float] = None
    decision: Optional[str] = None
    invalid_reason: Optional[str] = None
    feature_data_cutoff: Optional[datetime] = None
    rating_state_timestamp: Optional[datetime] = None
    roster_state_timestamp: Optional[datetime] = None
    hero_meta_state_timestamp: Optional[datetime] = None
    model_version: str = versions.MODEL_VERSION
    feature_version: str = versions.FEATURE_VERSION
    calibration_version: str = versions.CALIBRATION_VERSION
    prediction_version: str = versions.PREDICTION_VERSION
    created_at: Optional[datetime] = None

    def to_row(self) -> dict:
        d = asdict(self)
        d["features"] = json.loads(json.dumps(self.features))
        return d

    def content_hash(self) -> str:
        """Хеш содержимого без служебных полей. Используется тестом
        неизменяемости: если после публикации хеш изменился, инвариант
        нарушен независимо от того, какое поле правили."""
        d = self.to_row()
        for k in ("state", "created_at"):
            d.pop(k, None)
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResolutionRecord:
    prediction_id: str
    match_id: int
    resolved_at: datetime
    radiant_win: bool
    actual_start_time: Optional[datetime] = None
    correct_raw: Optional[bool] = None
    correct_calibrated: Optional[bool] = None
    log_loss_raw: Optional[float] = None
    log_loss_calibrated: Optional[float] = None
    brier_raw: Optional[float] = None
    brier_calibrated: Optional[float] = None
    calibration_error_raw: Optional[float] = None
    calibration_error_calibrated: Optional[float] = None
    confidence_bucket: Optional[str] = None
    resolution_version: str = versions.RESOLUTION_VERSION

    def to_row(self) -> dict:
        return asdict(self)


def confidence_bucket(conf: Optional[float]) -> str:
    """Границы взяты из coverage-risk кривой Phase 13/14, а не назначены
    произвольно: 0.15 отделяет верхние ~27% прогнозов, 0.05 — нижнюю треть."""
    if conf is None:
        return "unknown"
    if conf >= 0.15:
        return "high"
    if conf >= 0.05:
        return "medium"
    return "low"


def score_resolution(snapshot: PredictionSnapshot, radiant_win: bool,
                     resolved_at: datetime, match_id: int,
                     actual_start_time: Optional[datetime] = None) -> ResolutionRecord:
    """Считает метрики исхода. Снимок при этом НЕ меняется — функция
    принимает его только для чтения и возвращает отдельный объект."""
    import math

    y = 1.0 if radiant_win else 0.0
    eps = 1e-15

    def score(p: Optional[float]):
        if p is None:
            return None, None, None, None
        pc = min(max(p, eps), 1 - eps)
        return (bool((p >= 0.5) == radiant_win),
                float(-(y * math.log(pc) + (1 - y) * math.log(1 - pc))),
                float((p - y) ** 2),
                float(abs(p - y)))

    c_r, ll_r, br_r, ce_r = score(snapshot.raw_probability)
    c_c, ll_c, br_c, ce_c = score(snapshot.calibrated_probability)
    return ResolutionRecord(
        prediction_id=snapshot.prediction_id,
        match_id=match_id,
        resolved_at=resolved_at,
        radiant_win=radiant_win,
        actual_start_time=actual_start_time,
        correct_raw=c_r, correct_calibrated=c_c,
        log_loss_raw=ll_r, log_loss_calibrated=ll_c,
        brier_raw=br_r, brier_calibrated=br_c,
        calibration_error_raw=ce_r, calibration_error_calibrated=ce_c,
        confidence_bucket=confidence_bucket(snapshot.confidence),
    )
