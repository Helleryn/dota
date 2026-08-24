"""
BasePredictionModel — единый интерфейс для сравнения ELO-правила,
LogisticRegression, RandomForest, CatBoost/LightGBM/XGBoost без переписывания
Dataset Builder / BacktestEngine / Evaluation (Phase 4, раздел 15;
docs/ml-architecture.md).

Не импортирует numpy/pandas на уровне модуля намеренно: в этой среде
разработки они не установлены (см. проверку зависимостей в Phase 4), а сам
контракт интерфейса от них не зависит — конкретные реализации (Phase 8)
будут использовать numpy-массивы для X/y, здесь типизировано через
TYPE_CHECKING, чтобы модуль оставался импортируемым и тестируемым уже сейчас.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    import numpy as np


class BasePredictionModel(ABC):
    """
    X — табличные признаки (n_samples, n_features), y — бинарная целевая
    переменная (radiant_win). Конкретная форма X/y (numpy array vs pandas
    DataFrame) фиксируется в Phase 8 при реализации Dataset Builder — здесь
    важен только контракт вызовов, не тип данных.
    """

    name: str

    @abstractmethod
    def fit(self, X: Any, y: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: Any) -> "np.ndarray":
        """
        Возвращает вероятности победы radiant для каждой строки X.
        Обязателен именно predict_proba, не predict — Log Loss/Brier/
        calibration требуют вероятностей, не только предсказанного класса
        (прямое требование задания).
        """
        raise NotImplementedError

    def calibrate(self, X_val: Any, y_val: Any) -> None:
        """
        Опциональная посттренировочная калибровка (Platt/isotonic,
        docs/ml-architecture.md). No-op по умолчанию — не каждая модель
        нуждается в отдельной калибровке (например, ELO-правило и так
        откалибровано по построению вероятностной интерпретации).
        """
        return None

    @abstractmethod
    def evaluate(self, X_test: Any, y_test: Any) -> Dict[str, float]:
        """
        Возвращает как минимум log_loss, brier_score, roc_auc, accuracy —
        набор метрик из docs/ml-architecture.md, единый для всех моделей,
        чтобы сравнение (Phase 8) было честным (один и тот же код метрик).
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BasePredictionModel":
        raise NotImplementedError

    def feature_importance(self) -> Optional[Dict[str, float]]:
        """
        None по умолчанию — не каждая модель (напр. простое ELO-правило)
        имеет содержательный feature importance. Модели, для которых он
        осмыслен (LogReg, деревья), переопределяют этот метод; SHAP
        используется в PredictionService поверх этого API как более точная
        альтернатива для explanation в API-ответе (docs/api.md), но
        feature_importance() остаётся быстрым fallback-путём.
        """
        return None


class EloRuleModel(BasePredictionModel):
    """
    Простейший небезосновательный baseline из задания ("всегда сильнее по
    Elo") — не ML-модель, а детерминированное правило поверх RatingEngine,
    выраженное как вероятность через ту же логистическую формулу, что и сам
    Elo (не просто argmax). Референсная реализация BasePredictionModel —
    доказывает, что интерфейс достаточен даже для нерегрессионной модели.
    """

    name = "elo_rule"

    def __init__(self):
        self._fitted = False

    def fit(self, X: Any, y: Any) -> None:
        # Правилу не требуется обучение — рейтинги уже посчитаны
        # RatingEngine и переданы как признак. Метод существует только
        # чтобы соответствовать интерфейсу (BacktestEngine вызывает fit()
        # единообразно для всех моделей).
        self._fitted = True

    def predict_proba(self, X: Any):
        # X — список/массив пар (radiant_elo, dire_elo). Реализация
        # намеренно без numpy (см. docstring модуля) — чистый Python,
        # достаточно для baseline-правила.
        if not self._fitted:
            raise RuntimeError("EloRuleModel.fit() должен быть вызван до predict_proba()")
        result = []
        for radiant_elo, dire_elo in X:
            p_radiant = 1.0 / (1.0 + 10 ** ((dire_elo - radiant_elo) / 400.0))
            result.append((1.0 - p_radiant, p_radiant))
        return result

    def evaluate(self, X_test: Any, y_test: Any) -> Dict[str, float]:
        import math

        probs = self.predict_proba(X_test)
        n = len(y_test)
        log_loss_sum = 0.0
        brier_sum = 0.0
        correct = 0
        eps = 1e-15
        for (p_dire, p_radiant), y in zip(probs, y_test):
            p = min(max(p_radiant, eps), 1 - eps)
            log_loss_sum += -(y * math.log(p) + (1 - y) * math.log(1 - p))
            brier_sum += (p_radiant - y) ** 2
            correct += int((p_radiant >= 0.5) == bool(y))
        return {
            "log_loss": log_loss_sum / n,
            "brier_score": brier_sum / n,
            "accuracy": correct / n,
        }

    def save(self, path: str) -> None:
        # Без обучаемых параметров — сохранять нечего, метод существует
        # ради единообразия интерфейса.
        with open(path, "w") as f:
            f.write("elo_rule\n")

    @classmethod
    def load(cls, path: str) -> "EloRuleModel":
        model = cls()
        model._fitted = True
        return model


def _self_test() -> None:
    model = EloRuleModel()
    model.fit(None, None)

    # Синтетика: команда с явно более высоким Elo должна получать P > 0.5
    X = [(1200, 1000), (1000, 1200), (1000, 1000)]
    y = [1, 0, 1]  # radiant_win

    probs = model.predict_proba(X)
    assert probs[0][1] > 0.5, "Команда с более высоким Elo должна иметь P(win) > 0.5"
    assert probs[1][1] < 0.5
    assert abs(probs[2][1] - 0.5) < 1e-9, "Равные рейтинги -> ровно 0.5"

    metrics = model.evaluate(X, y)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["log_loss"] > 0.0
    print("OK: BasePredictionModel/EloRuleModel self-test пройден:", metrics)


if __name__ == "__main__":
    _self_test()
