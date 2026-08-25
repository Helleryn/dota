"""
Phase 6.5 (раздел 16-18) — статистическая значимость на temporal-зависимых
данных. Матчи НЕ i.i.d. по времени (соседние матчи одной команды коррелируют
через Elo/форму) — обычный per-row bootstrap занижает дисперсию и завышает
уверенность. Используется BLOCK bootstrap: тестовая выборка (уже
хронологически отсортирована) делится на смежные блоки фиксированного
размера, ресэмплируются БЛОКИ, а не отдельные строки (раздел 17).
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest

RANDOM_SEED = 42


def _make_blocks(n: int, block_size: int) -> list:
    """Список массивов индексов — смежные блоки по block_size строк
    (последний блок может быть короче). Порядок строк ДОЛЖЕН быть
    хронологическим (ответственность вызывающего кода)."""
    return [np.arange(i, min(i + block_size, n)) for i in range(0, n, block_size)]


def block_bootstrap_metric(
    y_true: Sequence[int],
    p_pred: Sequence[float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    block_size: int = 20,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """
    Point estimate (на полном test) + 95% percentile CI через block bootstrap.
    """
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)
    n = len(y_true)
    point = metric_fn(y_true, p_pred)

    blocks = _make_blocks(n, block_size)
    n_blocks = len(blocks)
    rng = np.random.default_rng(seed)

    boot_values = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([blocks[i] for i in chosen])
        boot_values[b] = metric_fn(y_true[idx], p_pred[idx])

    ci_low, ci_high = np.percentile(boot_values, [2.5, 97.5])
    return {
        "point": float(point),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_blocks": n_blocks,
        "block_size": block_size,
        "n_bootstrap": n_bootstrap,
    }


def block_bootstrap_paired_diff(
    y_true: Sequence[int],
    p_pred_a: Sequence[float],
    p_pred_b: Sequence[float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    block_size: int = 20,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """
    Разница metric(A) - metric(B) на ОДНИХ И ТЕХ ЖЕ строках (paired) +
    95% CI через block bootstrap — те же resampled блоки применяются к
    ОБЕИМ моделям одновременно (не независимо), иначе CI разницы был бы
    некорректен (раздел 16: "difference + CI").
    """
    y_true = np.asarray(y_true)
    p_pred_a = np.asarray(p_pred_a)
    p_pred_b = np.asarray(p_pred_b)
    n = len(y_true)
    point_diff = metric_fn(y_true, p_pred_a) - metric_fn(y_true, p_pred_b)

    blocks = _make_blocks(n, block_size)
    n_blocks = len(blocks)
    rng = np.random.default_rng(seed)

    boot_diffs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([blocks[i] for i in chosen])
        boot_diffs[b] = metric_fn(y_true[idx], p_pred_a[idx]) - metric_fn(y_true[idx], p_pred_b[idx])

    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    return {
        "point_diff": float(point_diff),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_blocks": n_blocks,
        "block_size": block_size,
        "n_bootstrap": n_bootstrap,
    }


def mcnemar_exact(y_true: Sequence[int], pred_a: Sequence[int], pred_b: Sequence[int]) -> Dict[str, float]:
    """
    Exact McNemar test (paired classification, раздел 16) — сравнивает A и
    B на ОДНИХ И ТЕХ ЖЕ строках через discordant pairs (A верно/B неверно
    vs A неверно/B верно). Использует точный биномиальный тест
    (`scipy.stats.binomtest`), не chi-square approximation — корректно и
    при небольшом числе discordant pairs.
    """
    y_true = np.asarray(y_true)
    correct_a = (np.asarray(pred_a) == y_true)
    correct_b = (np.asarray(pred_b) == y_true)

    a_only = int(np.sum(correct_a & ~correct_b))  # A верно, B неверно
    b_only = int(np.sum(~correct_a & correct_b))  # B верно, A неверно
    n_discordant = a_only + b_only

    if n_discordant == 0:
        p_value = 1.0
    else:
        result = binomtest(a_only, n_discordant, p=0.5)
        p_value = float(result.pvalue)

    return {
        "a_correct_b_wrong": a_only,
        "b_correct_a_wrong": b_only,
        "n_discordant": n_discordant,
        "p_value": p_value,
    }


def accuracy_metric(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    return float(np.mean((p_pred >= 0.5).astype(int) == y_true))


def log_loss_metric(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    eps = 1e-15
    p = np.clip(p_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def brier_metric(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    return float(np.mean((p_pred - y_true) ** 2))
