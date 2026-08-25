"""
PHASE 13 — метрики калибровки.

Калибровка отвечает на вопрос, который accuracy не задаёт вовсе: **если
модель говорит 70%, происходит ли это в 70% случаев?** Для продукта это
важнее дискриминации — прогноз 0.55 и прогноз 0.85 должны означать разное.

Реализованы:

* **reliability curve** — эмпирическая частота против среднего прогноза
  по бинам;
* **ECE** (Expected Calibration Error) — средневзвешенное отклонение,
  веса = размеры бинов;
* **MCE** (Maximum Calibration Error) — худший бин; ECE может быть мал
  при одном катастрофически плохом бине, MCE это ловит;
* **calibration slope / intercept** — логистическая регрессия
  `y ~ a + b * logit(p)`. Идеал: b = 1, a = 0.
  b < 1 означает переуверенность (прогнозы слишком крайние), b > 1 —
  недоуверенность. Это чувствительнее, чем бинированные метрики: не
  зависит от произвольного выбора числа бинов.

Бинирование по умолчанию — равной ширины (0.0-0.1, 0.1-0.2, ...), как в
Phase 12, чтобы числа были сопоставимы с уже опубликованными. Доступно и
бинирование по квантилям: при сильно скошенном распределении прогнозов
равноширинные бины на краях почти пусты.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    n: int
    mean_pred: float
    frac_positive: float

    @property
    def gap(self) -> float:
        return abs(self.mean_pred - self.frac_positive)


def _prep(y: Sequence[int], p: Sequence[float]):
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"несовпадение форм: y={y.shape}, p={p.shape}")
    return y, p


def reliability_curve(y, p, bins: int = 10, strategy: str = "uniform") -> List[Bin]:
    y, p = _prep(y, p)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, bins + 1)
    else:
        raise ValueError(f"неизвестная стратегия бинирования: {strategy}")

    out: List[Bin] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (p >= lo) & (p < hi) if i < len(edges) - 2 else (p >= lo) & (p <= hi)
        if not sel.any():
            continue
        out.append(Bin(float(lo), float(hi), int(sel.sum()),
                       float(p[sel].mean()), float(y[sel].mean())))
    return out


def expected_calibration_error(y, p, bins: int = 10, strategy: str = "uniform") -> float:
    curve = reliability_curve(y, p, bins, strategy)
    n = sum(b.n for b in curve)
    return float(sum(b.n * b.gap for b in curve) / n) if n else float("nan")


def maximum_calibration_error(y, p, bins: int = 10, strategy: str = "uniform",
                              min_bin_size: int = 30) -> float:
    """Худший бин. `min_bin_size` отсекает бины, где отклонение — шум:
    в бине из 5 матчей |0.9 − 0.6| = 0.3 не означает плохой калибровки."""
    curve = [b for b in reliability_curve(y, p, bins, strategy) if b.n >= min_bin_size]
    return float(max((b.gap for b in curve), default=float("nan")))


def calibration_slope_intercept(y, p) -> Dict[str, float]:
    """Логистическая регрессия y ~ a + b*logit(p) методом Ньютона.

    Реализовано напрямую, а не через sklearn: нужен именно вариант БЕЗ
    регуляризации (штраф сместил бы наклон к нулю и превратил бы хорошо
    откалиброванную модель в «переуверенную»).
    """
    y, p = _prep(y, p)
    if len(y) < 10 or len(np.unique(y)) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "n": int(len(y))}
    z = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1 - EPS))
    X = np.column_stack([np.ones_like(z), z])
    beta = np.zeros(2)
    for _ in range(100):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-10, None)
        grad = X.T @ (y - mu)
        H = X.T @ (X * w[:, None])
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return {"slope": float("nan"), "intercept": float("nan"), "n": int(len(y))}
        beta_new = beta + step
        if not np.all(np.isfinite(beta_new)):
            return {"slope": float("nan"), "intercept": float("nan"), "n": int(len(y))}
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            beta = beta_new
            break
        beta = beta_new
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "n": int(len(y))}


def calibration_report(y, p, bins: int = 10, strategy: str = "uniform") -> Dict[str, object]:
    y, p = _prep(y, p)
    si = calibration_slope_intercept(y, p)
    yc = np.clip(p, EPS, 1 - EPS)
    return {
        "n": int(len(y)),
        "accuracy": float(((p >= 0.5).astype(float) == y).mean()),
        "log_loss": float(-np.mean(y * np.log(yc) + (1 - y) * np.log(1 - yc))),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p, bins, strategy),
        "mce": maximum_calibration_error(y, p, bins, strategy),
        "slope": si["slope"],
        "intercept": si["intercept"],
        "mean_pred": float(p.mean()),
        "base_rate": float(y.mean()),
        "curve": [{"lo": b.lo, "hi": b.hi, "n": b.n,
                   "mean_pred": b.mean_pred, "frac_positive": b.frac_positive}
                  for b in reliability_curve(y, p, bins, strategy)],
    }
