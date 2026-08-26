"""
PHASE 15 — метрики shadow-потока (PART I/J/K/N/O).

Все метрики считаются раздельно для сырой и калиброванной вероятности:
именно это сравнение отвечает на вопрос, работает ли результат Phase 14
на потоке, а не только на историческом TEST.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.evaluation.calibration import calibration_report
from src.evaluation.metrics import compute_metrics
from src.evaluation.statistics import (
    accuracy_metric,
    block_bootstrap_metric,
    log_loss_metric,
)

# Размеры выборки фиксируются заранее (PART O): выводы не делаются раньше
# N=50, а на первых 10-20 матчах не делаются вовсе.
SAMPLE_MILESTONES = (50, 100, 250, 500)
MIN_SAMPLE_FOR_CLAIMS = 50


def to_frame(rows: Sequence) -> pd.DataFrame:
    """Пары (снимок, разрешение) из репозитория -> плоский датафрейм."""
    recs = []
    for r in rows:
        recs.append({
            "prediction_id": r.prediction_id,
            "match_id": r.match_id,
            "prediction_timestamp": r.prediction_timestamp,
            "match_start_time": r.match_start_time,
            "source": r.source,
            "patch_name": r.patch_name,
            "league_id": r.league_id,
            "p_raw": r.raw_probability,
            "p_cal": r.calibrated_probability,
            "confidence": r.confidence,
            "decision": r.decision,
            "confidence_bucket": r.confidence_bucket,
            "y": int(bool(r.radiant_win)),
            "elo_difference": (r.features or {}).get("elo_difference"),
            "hero_exp_decay_diff": (r.features or {}).get("hero_exp_decay_diff"),
            "elo_mean_diff": (r.features or {}).get("elo_mean_diff"),
            "five_vs_team_elo_diff": (r.features or {}).get("five_vs_team_elo_diff"),
        })
    df = pd.DataFrame(recs)
    if len(df):
        df = df.sort_values("prediction_timestamp").reset_index(drop=True)
    return df


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    m = compute_metrics(y, p)
    c = calibration_report(y, p)
    return {"n": m["n"], "accuracy": m["accuracy"], "roc_auc": m["roc_auc"],
            "log_loss": m["log_loss"], "brier": m["brier_score"],
            "ece": c["ece"], "slope": c["slope"], "intercept": c["intercept"]}


def dual_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """RAW против CALIBRATED на одной и той же выборке (PART G)."""
    if not len(df):
        return {}
    y = df["y"].to_numpy()
    out = {"raw": _metrics(y, df["p_raw"].to_numpy(dtype=float))}
    if df["p_cal"].notna().all():
        out["calibrated"] = _metrics(y, df["p_cal"].to_numpy(dtype=float))
    return out


def by_confidence(df: pd.DataFrame, column: str = "p_cal") -> List[Dict[str, object]]:
    out = []
    for b in ("high", "medium", "low"):
        sub = df[df["confidence_bucket"] == b]
        if len(sub) < 20:
            continue
        out.append({"bucket": b, **_metrics(sub["y"].to_numpy(),
                                            sub[column].to_numpy(dtype=float))})
    return out


def coverage_curve(df: pd.DataFrame, column: str = "p_cal",
                   grid: Sequence[float] = (1.0, 0.8, 0.5, 0.25, 0.1)) -> List[Dict[str, object]]:
    """Селективный прогноз (PART H). В shadow-режиме ни один прогноз не
    выбрасывается — покрытие применяется только на этапе анализа."""
    if not len(df):
        return []
    p = df[column].to_numpy(dtype=float)
    y = df["y"].to_numpy()
    conf = np.abs(p - 0.5)
    rows = []
    for cov in grid:
        thr = float(np.quantile(conf, 1 - cov)) if cov < 1 else float(conf.min() - 1)
        sel = conf >= thr
        if sel.sum() < 20:
            continue
        rows.append({"coverage": float(sel.mean()), **_metrics(y[sel], p[sel])})
    return rows


def sample_size_progression(df: pd.DataFrame, column: str = "p_cal") -> List[Dict[str, object]]:
    """Как метрики и их интервалы ведут себя по мере накопления выборки
    (PART O). Отвечает на вопрос «сколько матчей нужно»."""
    out = []
    for n in list(SAMPLE_MILESTONES) + [len(df)]:
        if n > len(df) or n < 20:
            continue
        sub = df.iloc[:n]
        y = sub["y"].to_numpy()
        p = sub[column].to_numpy(dtype=float)
        acc = block_bootstrap_metric(y, p, accuracy_metric, block_size=20, seed=42)
        ll = block_bootstrap_metric(y, p, log_loss_metric, block_size=20, seed=42)
        out.append({"n": int(n),
                    "accuracy": acc["point"],
                    "accuracy_ci": [acc["ci_low"], acc["ci_high"]],
                    "accuracy_ci_width": acc["ci_high"] - acc["ci_low"],
                    "log_loss": ll["point"],
                    "log_loss_ci": [ll["ci_low"], ll["ci_high"]],
                    "log_loss_ci_width": ll["ci_high"] - ll["ci_low"]})
    return out


def rolling(df: pd.DataFrame, column: str = "p_cal",
            window: int = 500, step: int = 100) -> List[Dict[str, float]]:
    """Скользящий мониторинг (PART J). Считается только по уже
    разрешённым матчам — то есть по информации, доступной в проде."""
    y = df["y"].to_numpy()
    p = df[column].to_numpy(dtype=float)
    out = []
    for end in range(window, len(df) + 1, step):
        sl = slice(end - window, end)
        r = calibration_report(y[sl], p[sl])
        m = compute_metrics(y[sl], p[sl])
        out.append({"end_index": end, "n": r["n"], "accuracy": r["accuracy"],
                    "roc_auc": m["roc_auc"], "log_loss": r["log_loss"],
                    "brier": r["brier"], "ece": r["ece"],
                    "slope": r["slope"], "intercept": r["intercept"]})
    return out


def drift_split(rolling_rows: List[Dict[str, float]]) -> Dict[str, object]:
    """Разделение дрейфа (PART K): падает ли ранжирование (predictive
    drift) или только качество вероятностей (calibration drift)."""
    if len(rolling_rows) < 2:
        return {"verdict": "недостаточно точек"}
    auc = [r["roc_auc"] for r in rolling_rows if r["roc_auc"] == r["roc_auc"]]
    ece = [r["ece"] for r in rolling_rows]
    slope = [r["slope"] for r in rolling_rows]
    auc_range = (max(auc) - min(auc)) if auc else float("nan")
    ece_range = max(ece) - min(ece)
    slope_range = max(slope) - min(slope)
    if auc_range > 0.05:
        verdict = "predictive drift: ранжирование нестабильно"
    elif ece_range > 0.02 or slope_range > 0.3:
        verdict = "calibration drift: ранжирование стабильно, вероятности плывут"
    else:
        verdict = "заметного дрейфа не обнаружено"
    return {"auc_range": auc_range, "ece_range": ece_range,
            "slope_range": slope_range, "verdict": verdict}
