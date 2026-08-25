"""
Единый набор метрик (Phase 6, раздел 12) — используется ОДИНАКОВО для всех
baseline'ов и моделей (docs/ml-architecture.md: "Random baseline и CatBoost
пробегают через ОДИН И ТОТ ЖЕ Evaluation... код — сравнение получается
честным"). Не дублируется в каждой модели отдельно.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def compute_metrics(y_true: Sequence[int], p_pred: Sequence[float]) -> Dict[str, float]:
    """
    y_true — 0/1 (radiant_win), p_pred — P(radiant_win) для каждой строки.

    ROC-AUC требует оба класса в y_true — если датасет вырожден (один
    класс), возвращается NaN с явной пометкой, а не падение и не тихая
    подстановка 0.5 (раздел 12: "Если dataset позволяет").
    """
    y_true = np.asarray(y_true, dtype=int)
    p_pred = np.asarray(p_pred, dtype=float)
    eps = 1e-15
    p_clipped = np.clip(p_pred, eps, 1 - eps)
    y_pred_class = (p_pred >= 0.5).astype(int)

    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred_class)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred_class)),
        "log_loss": float(log_loss(y_true, p_clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, p_pred)),
    }

    if len(set(y_true.tolist())) < 2:
        metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, p_pred))

    return metrics


def format_metrics_row(name: str, metrics: Dict[str, float]) -> str:
    return (
        f"{name:22s} n={metrics['n']:5d}  acc={metrics['accuracy']:.4f}  "
        f"bal_acc={metrics['balanced_accuracy']:.4f}  log_loss={metrics['log_loss']:.4f}  "
        f"brier={metrics['brier_score']:.4f}  roc_auc={metrics['roc_auc']:.4f}"
    )
