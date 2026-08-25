"""
Baseline'ы и модели Phase 6 (раздел 7-11) — реализуют `BasePredictionModel`
(`src/models/base.py`, Phase 4 интерфейс, не переписан). X — pandas
DataFrame с именованными колонками (нужно для feature_importance()),
y — pandas Series/np.ndarray из 0/1 (radiant_win).

Все evaluate() делегируют в `src/evaluation/metrics.compute_metrics` — один
и тот же код метрик для каждой модели (docs/ml-architecture.md).
"""

from __future__ import annotations

import json
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.models.base import BasePredictionModel

RANDOM_SEED = 42


class RandomBaselineModel(BasePredictionModel):
    """P(radiant_win) = 0.5 всегда — reference point (раздел 7)."""

    name = "random_baseline"

    def fit(self, X: Any, y: Any) -> None:
        return None

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])

    def evaluate(self, X_test: Any, y_test: Any) -> Dict[str, float]:
        p = self.predict_proba(X_test)[:, 1]
        return compute_metrics(y_test, p)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write("random_baseline\n")

    @classmethod
    def load(cls, path: str) -> "RandomBaselineModel":
        return cls()


class MajorityBaselineModel(BasePredictionModel):
    """P(radiant_win) = доля radiant_win в TRAIN (раздел 8) — не argmax,
    а вероятность, чтобы log_loss/brier были осмысленными, не только accuracy."""

    name = "majority_baseline"

    def __init__(self):
        self._p_majority: Optional[float] = None

    def fit(self, X: Any, y: Any) -> None:
        y_arr = np.asarray(y, dtype=int)
        self._p_majority = float(y_arr.mean())

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._p_majority is None:
            raise RuntimeError("MajorityBaselineModel.fit() должен быть вызван до predict_proba()")
        n = len(X)
        p1 = np.full(n, self._p_majority)
        return np.column_stack([1 - p1, p1])

    def evaluate(self, X_test: Any, y_test: Any) -> Dict[str, float]:
        p = self.predict_proba(X_test)[:, 1]
        return compute_metrics(y_test, p)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"p_majority": self._p_majority}, f)

    @classmethod
    def load(cls, path: str) -> "MajorityBaselineModel":
        model = cls()
        with open(path) as f:
            model._p_majority = json.load(f)["p_majority"]
        return model


class EloOnlyModel(BasePredictionModel):
    """
    P(radiant_win) = sigmoid(elo_difference / 400) — стандартная Elo win
    probability formula (та же, что RatingEngine использует внутри себя для
    expected score, ADR-003), применённая НАПРЯМУЮ к elo_difference без
    обучения (раздел 9: "Никаких других features"). Не путать с
    `EloRuleModel` (src/models/base.py) — та принимает пары (radiant_elo,
    dire_elo) отдельно, эта — уже готовую разницу (согласовано с форматом X
    единой feature-матрицы Phase 6).
    """

    name = "elo_only"

    def fit(self, X: Any, y: Any) -> None:
        return None

    def predict_proba(self, X: Any) -> np.ndarray:
        elo_diff = np.asarray(X["elo_difference"], dtype=float)
        p_radiant = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
        return np.column_stack([1 - p_radiant, p_radiant])

    def evaluate(self, X_test: Any, y_test: Any) -> Dict[str, float]:
        p = self.predict_proba(X_test)[:, 1]
        return compute_metrics(y_test, p)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write("elo_only\n")

    @classmethod
    def load(cls, path: str) -> "EloOnlyModel":
        return cls()


class LogisticRegressionModel(BasePredictionModel):
    """
    sklearn LogisticRegression поверх заданного списка признаков (Phase 6,
    раздел 10). Пропуски (cold-start recent_form/days_since_last_match)
    заполняются 0.0 через SimpleImputer, обученный ТОЛЬКО на train
    (`fit_transform` в `fit()`, `transform` в `predict_proba()`) — 0.0
    выбран как нейтральное значение ДЛЯ РАЗНОСТНОГО признака ("нет известной
    разницы"), не медиана (медиана требовала бы протечки статистики фолда
    в признак и не имеет более ясной интерпретации здесь).
    """

    name = "logistic_regression"

    def __init__(self, feature_names: List[str], random_state: int = RANDOM_SEED):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.feature_names = feature_names
        self.random_state = random_state
        self._imputer = SimpleImputer(strategy="constant", fill_value=0.0)
        self._scaler = StandardScaler()
        self._clf = LogisticRegression(random_state=random_state, max_iter=1000)
        self._calibrator = None  # опциональный Platt/isotonic поверх raw predict_proba

    def _transform(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        Xn = X[self.feature_names].to_numpy(dtype=float)
        Xn = self._imputer.fit_transform(Xn) if fit else self._imputer.transform(Xn)
        Xn = self._scaler.fit_transform(Xn) if fit else self._scaler.transform(Xn)
        return Xn

    def fit(self, X: pd.DataFrame, y: Any) -> None:
        Xn = self._transform(X, fit=True)
        self._clf.fit(Xn, np.asarray(y, dtype=int))

    def _raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xn = self._transform(X, fit=False)
        return self._clf.predict_proba(Xn)[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = self._raw_proba(X)
        if self._calibrator is not None:
            p1 = self._calibrator.predict(p1.reshape(-1, 1)) if hasattr(self._calibrator, "predict") else p1
        return np.column_stack([1 - p1, p1])

    def calibrate(self, X_val: pd.DataFrame, y_val: Any) -> None:
        """Isotonic regression поверх raw predict_proba, обучена ТОЛЬКО на
        VALIDATION (раздел 13: "TEST нельзя использовать для выбора
        calibration method")."""
        from sklearn.isotonic import IsotonicRegression

        p_raw = self._raw_proba(X_val)
        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(p_raw, np.asarray(y_val, dtype=int))

    def evaluate(self, X_test: pd.DataFrame, y_test: Any) -> Dict[str, float]:
        p = self.predict_proba(X_test)[:, 1]
        return compute_metrics(y_test, p)

    def feature_importance(self) -> Optional[Dict[str, float]]:
        coefs = self._clf.coef_[0]
        return {name: float(c) for name, c in zip(self.feature_names, coefs)}

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "LogisticRegressionModel":
        with open(path, "rb") as f:
            return pickle.load(f)


# Фиксированные гиперпараметры CatBoost (Phase 6, раздел 11: "НЕ делай
# огромный hyperparameter search... все параметры зафиксированы и записаны").
CATBOOST_PARAMS = {
    "iterations": 300,
    "depth": 4,
    "learning_rate": 0.05,
    "loss_function": "Logloss",
    "random_seed": RANDOM_SEED,
    "verbose": False,
    "allow_writing_files": False,
}


class CatBoostModel(BasePredictionModel):
    """
    CatBoost (ADR-004) с фиксированными небольшими гиперпараметрами
    (CATBOOST_PARAMS). NaN в признаках (cold-start) передаются НАПРЯМУЮ —
    CatBoost нативно обрабатывает пропуски (раздел ADR-004: "нативная
    поддержка... без ручного encoding"), в отличие от LogisticRegressionModel.
    """

    name = "catboost"

    def __init__(self, feature_names: List[str], params: Optional[Dict[str, Any]] = None):
        from catboost import CatBoostClassifier

        self.feature_names = feature_names
        self.params = dict(params or CATBOOST_PARAMS)
        self._model = CatBoostClassifier(**self.params)
        self._calibrator = None

    def fit(self, X: pd.DataFrame, y: Any) -> None:
        self._model.fit(X[self.feature_names], np.asarray(y, dtype=int))

    def _raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X[self.feature_names])[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = self._raw_proba(X)
        if self._calibrator is not None:
            p1 = self._calibrator.predict(p1)
        return np.column_stack([1 - p1, p1])

    def calibrate(self, X_val: pd.DataFrame, y_val: Any) -> None:
        from sklearn.isotonic import IsotonicRegression

        p_raw = self._raw_proba(X_val)
        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(p_raw, np.asarray(y_val, dtype=int))

    def evaluate(self, X_test: pd.DataFrame, y_test: Any) -> Dict[str, float]:
        p = self.predict_proba(X_test)[:, 1]
        return compute_metrics(y_test, p)

    def feature_importance(self) -> Optional[Dict[str, float]]:
        importances = self._model.get_feature_importance()
        return {name: float(v) for name, v in zip(self.feature_names, importances)}

    def save(self, path: str) -> None:
        self._model.save_model(path)

    @classmethod
    def load(cls, path: str) -> "CatBoostModel":
        from catboost import CatBoostClassifier

        model = cls(feature_names=[])
        model._model = CatBoostClassifier()
        model._model.load_model(path)
        return model
