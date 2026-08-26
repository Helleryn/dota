"""
PHASE 14 — методы калибровки вероятностей.

Слой калибровки — монотонное (или почти монотонное) преобразование
`p_raw -> p_cal`. Базовая модель при этом не меняется вообще: калибровка
исправляет ОТОБРАЖЕНИЕ «сырая вероятность -> фактическая частота», а не
то, как модель ранжирует матчи.

Отсюда важное следствие, которое проверяется в тестах: строго монотонное
преобразование **не может изменить ROC-AUC**. Если после калибровки AUC
поменялся — это либо изотоническая регрессия (кусочно-постоянная, склеивает
разные значения), либо ошибка в коде.

Реализованы четыре метода. Сложнее — сознательно не рассматриваются:
у задачи два-три параметра отображения, и любой более гибкий метод будет
подгонять шум.

| Метод | Параметров | Когда уместен |
|---|---|---|
| `PlattCalibrator` | 2 (наклон, сдвиг) | общий случай; правит и форму, и смещение |
| `TemperatureCalibrator` | 1 (наклон) | только пере/недоуверенность, без смещения |
| `BetaCalibrator` | 3 | асимметричные искажения (разное поведение у краёв) |
| `IsotonicCalibrator` | непараметрический | много данных, произвольная форма искажения |

Все методы принимают веса наблюдений — это нужно для экспоненциального
затухания (свежие матчи важнее старых).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

EPS = 1e-12


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def _weighted_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                       max_iter: int = 100, ridge: float = 1e-6) -> Optional[np.ndarray]:
    """Ньютоновская логистическая регрессия с весами наблюдений.

    `ridge` — крошечный, только чтобы гессиан оставался обратимым при
    вырожденной выборке (например, все метки одного класса в маленьком
    окне). Он на порядки меньше сигнала и не смещает наклон заметно;
    сознательная регуляризация здесь была бы вредна — она тянула бы
    наклон к нулю и превращала бы хорошо откалиброванную модель в
    «переуверенную» (та же причина, что в src/evaluation/calibration.py).
    """
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        mu = _sigmoid(X @ beta)
        s = np.clip(mu * (1.0 - mu), 1e-10, None) * w
        grad = X.T @ (w * (y - mu)) - ridge * beta
        H = X.T @ (X * s[:, None]) + ridge * np.eye(k)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return None
        beta_new = beta + step
        if not np.all(np.isfinite(beta_new)):
            return None
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            return beta_new
        beta = beta_new
    return beta


class BaseCalibrator:
    name = "base"

    def fit(self, p: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> "BaseCalibrator":
        raise NotImplementedError

    def transform(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def fitted(self) -> bool:
        return getattr(self, "_ok", False)


class IdentityCalibrator(BaseCalibrator):
    """Ничего не делает. Нужен как явный вариант A, а не как заглушка."""

    name = "identity"

    def fit(self, p, y, w=None):
        self._ok = True
        return self

    def transform(self, p):
        return np.asarray(p, dtype=float)


class PlattCalibrator(BaseCalibrator):
    """p_cal = sigmoid(a + b * logit(p_raw)). Два параметра."""

    name = "platt"

    def __init__(self):
        self.a = 0.0
        self.b = 1.0
        self._ok = False

    def fit(self, p, y, w=None):
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.ones_like(p) if w is None else np.asarray(w, dtype=float)
        if len(p) < 20 or len(np.unique(y)) < 2:
            self._ok = False
            return self
        X = np.column_stack([np.ones_like(p), _logit(p)])
        beta = _weighted_logistic(X, y, w)
        if beta is None:
            self._ok = False
            return self
        self.a, self.b = float(beta[0]), float(beta[1])
        self._ok = True
        return self

    def transform(self, p):
        p = np.asarray(p, dtype=float)
        if not self._ok:
            return p
        return _sigmoid(self.a + self.b * _logit(p))


class TemperatureCalibrator(BaseCalibrator):
    """p_cal = sigmoid(logit(p_raw) / T). Один параметр: только
    пере/недоуверенность, систематический сдвиг не правится."""

    name = "temperature"

    def __init__(self):
        self.inv_t = 1.0
        self._ok = False

    def fit(self, p, y, w=None):
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.ones_like(p) if w is None else np.asarray(w, dtype=float)
        if len(p) < 20 or len(np.unique(y)) < 2:
            self._ok = False
            return self
        X = _logit(p)[:, None]
        beta = _weighted_logistic(X, y, w)
        if beta is None:
            self._ok = False
            return self
        self.inv_t = float(beta[0])
        self._ok = True
        return self

    def transform(self, p):
        p = np.asarray(p, dtype=float)
        if not self._ok:
            return p
        return _sigmoid(self.inv_t * _logit(p))


class BetaCalibrator(BaseCalibrator):
    """p_cal = sigmoid(c + a*log(p) - b*log(1-p)). Три параметра.

    В отличие от Platt допускает разное поведение у нуля и у единицы —
    полезно, когда искажение асимметрично (Phase 13: калибровка на
    фаворитах и на андердогах различалась).
    """

    name = "beta"

    def __init__(self):
        self.coef = np.array([0.0, 1.0, 1.0])
        self._ok = False

    def fit(self, p, y, w=None):
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        y = np.asarray(y, dtype=float)
        w = np.ones_like(p) if w is None else np.asarray(w, dtype=float)
        if len(p) < 50 or len(np.unique(y)) < 2:
            self._ok = False
            return self
        X = np.column_stack([np.ones_like(p), np.log(p), -np.log(1 - p)])
        beta = _weighted_logistic(X, y, w)
        if beta is None:
            self._ok = False
            return self
        self.coef = beta
        self._ok = True
        return self

    def transform(self, p):
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        if not self._ok:
            return p
        X = np.column_stack([np.ones_like(p), np.log(p), -np.log(1 - p)])
        return _sigmoid(X @ self.coef)


class IsotonicCalibrator(BaseCalibrator):
    """Непараметрическая монотонная подгонка (PAVA).

    ВНИМАНИЕ: кусочно-постоянная. Она склеивает разные сырые вероятности
    в одно значение, поэтому — единственный из четырёх методов — способна
    ИЗМЕНИТЬ ROC-AUC, обычно в худшую сторону. Это не дефект метода, а его
    свойство, и оно проверяется тестом.
    """

    name = "isotonic"

    def __init__(self, min_samples: int = 200):
        self.min_samples = min_samples
        self._x = None
        self._y = None
        self._ok = False

    def fit(self, p, y, w=None):
        from sklearn.isotonic import IsotonicRegression

        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.ones_like(p) if w is None else np.asarray(w, dtype=float)
        if len(p) < self.min_samples or len(np.unique(y)) < 2:
            self._ok = False
            return self
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir.fit(p, y, sample_weight=w)
        self._model = ir
        self._ok = True
        return self

    def transform(self, p):
        p = np.asarray(p, dtype=float)
        if not self._ok:
            return p
        return np.clip(self._model.predict(p), EPS, 1 - EPS)


CALIBRATORS = {
    "identity": IdentityCalibrator,
    "platt": PlattCalibrator,
    "temperature": TemperatureCalibrator,
    "beta": BetaCalibrator,
    "isotonic": IsotonicCalibrator,
}
