"""
PHASE 15 — вычисление замороженного прогноза на произвольный момент времени.

## Как считаются признаки без переписывания замороженного кода

Модули признаков Phase 9 устроены как walk-forward проход: состояние
читается ДО матча и обновляется строго ПОСЛЕ. Это свойство доказано
тестами утечки каждого модуля (prefix-stability).

Отсюда приём, позволяющий получить признаки нового матча, **не трогая
замороженный код**: к истории добавляется целевой матч с заглушкой
исхода, проход выполняется по всей последовательности, и берётся
последняя строка. Её признаки по построению зависят только от матчей
раньше неё, а подставленный исход в них попасть не может — он влияет
лишь на состояние ПОСЛЕ.

Тест `test_placeholder_outcome_cannot_reach_its_own_features` проверяет
это напрямую: перевёрнутая заглушка не меняет ни одного признака.

## Что заморожено

Модель обучается ОДИН раз на TRAIN-срезе исторического датасета и больше
не переобучается — это и есть frozen model. Слой калибровки, наоборот,
обновляется по мере накопления результатов, но только по матчам строго
раньше момента прогноза.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.evaluation.calibrators import CALIBRATORS
from src.shadow import versions


@dataclass(frozen=True)
class PredictionResult:
    raw_probability: float
    calibrated_probability: Optional[float]
    confidence: Optional[float]
    decision: str
    calibration_version: str
    calibration_sample_size: int
    feature_data_cutoff: Optional[datetime]
    rating_state_timestamp: Optional[datetime]
    roster_state_timestamp: Optional[datetime]
    hero_meta_state_timestamp: Optional[datetime]


class FrozenEngine:
    """Замороженная модель + скользящий слой калибровки.

    `history` — датафрейм со всеми историческими матчами: признаки,
    `target`, `as_of_timestamp`. Обучение модели происходит один раз на
    первых `train_fraction` строках (тот же срез, что в Phase 9–14).
    """

    def __init__(self, history: pd.DataFrame, train_fraction: float = 0.70):
        from src.models.sklearn_models import LogisticRegressionModel

        self.history = history.sort_values(
            ["as_of_timestamp", "match_id"]).reset_index(drop=True)
        n_train = int(len(self.history) * train_fraction)
        train = self.history.iloc[:n_train]
        self.model = LogisticRegressionModel(
            feature_names=list(versions.FROZEN_FEATURES),
            random_state=versions.RANDOM_SEED)
        self.model.fit(train, train["target"])
        self.train_cutoff = pd.to_datetime(train["as_of_timestamp"].max(), utc=True)

        # Сырые вероятности по всей истории — материал для слоя калибровки.
        self._hist_p = np.asarray(self.model.predict_proba(self.history))[:, 1]
        self._hist_y = self.history["target"].to_numpy(dtype=int)
        self._hist_ts = pd.to_datetime(self.history["as_of_timestamp"], utc=True)

    # ------------------------------------------------------------------
    def raw_probability(self, features: Dict[str, float]) -> float:
        row = pd.DataFrame([{k: features.get(k, np.nan)
                             for k in versions.FROZEN_FEATURES}])
        return float(np.asarray(self.model.predict_proba(row))[0, 1])

    def calibration_window(self, cutoff: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """Последние CALIBRATION_WINDOW матчей СТРОГО раньше cutoff.

        Строгое неравенство существенно: матч, начавшийся ровно в момент
        прогноза, к этому моменту ещё не имеет исхода.
        """
        mask = (self._hist_ts < pd.Timestamp(cutoff)).to_numpy()
        p, y = self._hist_p[mask], self._hist_y[mask]
        w = versions.CALIBRATION_WINDOW
        return p[-w:], y[-w:]

    def calibrate(self, p_raw: float, cutoff: datetime) -> Tuple[Optional[float], str, int]:
        p_hist, y_hist = self.calibration_window(cutoff)
        if len(p_hist) < versions.CALIBRATION_MIN_HISTORY:
            # Истории мало — калибровка НЕ применяется, и это фиксируется
            # версией "none", а не тихо подменяется сырым значением.
            return None, "none", int(len(p_hist))
        cal = CALIBRATORS[versions.CALIBRATION_METHOD]().fit(p_hist, y_hist)
        if not cal.fitted:
            return None, "none", int(len(p_hist))
        return (float(np.asarray(cal.transform(np.array([p_raw])))[0]),
                versions.CALIBRATION_VERSION, int(len(p_hist)))

    def predict(self, features: Dict[str, float], cutoff: datetime,
                abstain_threshold: float = 0.0,
                state_timestamps: Optional[Dict[str, datetime]] = None) -> PredictionResult:
        p_raw = self.raw_probability(features)
        p_cal, cal_ver, n_hist = self.calibrate(p_raw, cutoff)
        p_used = p_cal if p_cal is not None else p_raw
        conf = abs(p_used - 0.5)
        # В shadow-режиме отказ НЕ выбрасывает прогноз: он сохраняется с
        # пометкой ABSTAIN, иначе последующий анализ покрытия невозможен
        # (PART H).
        decision = "PREDICT" if conf >= abstain_threshold else "ABSTAIN"
        st = state_timestamps or {}
        return PredictionResult(
            raw_probability=p_raw,
            calibrated_probability=p_cal,
            confidence=conf,
            decision=decision,
            calibration_version=cal_ver,
            calibration_sample_size=n_hist,
            feature_data_cutoff=st.get("feature_data_cutoff"),
            rating_state_timestamp=st.get("rating_state_timestamp"),
            roster_state_timestamp=st.get("roster_state_timestamp"),
            hero_meta_state_timestamp=st.get("hero_meta_state_timestamp"),
        )
