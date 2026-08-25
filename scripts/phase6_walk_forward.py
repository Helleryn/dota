#!/usr/bin/env python3
"""
PHASE 6 — walk-forward backtest (раздел 15-17).

    train on past -> predict next period (STRICTLY out-of-sample) ->
    observe result -> expand training window -> predict next period -> ...

Признаки (Elo/recent form/rest) уже посчитаны walk-forward-безопасно один
раз при построении Feature Set 0 (RatingEngine/_TeamFormTracker читают
состояние ДО текущего матча, см. src/datasets/feature_set_0.py) — здесь
walk-forward применяется к ОБУЧЕНИЮ МОДЕЛИ (LogReg/CatBoost), не к
пересчёту Elo: на каждом шаге модель обучается заново на всех строках со
start_time СТРОГО раньше начала периода, затем предсказывает период
целиком, прежде чем увидеть его результаты (раздел 17).

Запуск:
    python3 scripts/phase6_walk_forward.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_settings
from src.db.engine import make_engine
from src.evaluation.metrics import compute_metrics
from src.models.sklearn_models import RANDOM_SEED, CatBoostModel, EloOnlyModel, LogisticRegressionModel
from scripts.phase6_pipeline import FEATURE_SET_0345, load_and_prepare_dataset, git_commit_sha

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")
MIN_TRAIN_ROWS = 200  # не предсказываем период, пока обучающее окно меньше этого (раздел 17: честный OOS)


def main() -> int:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    settings = load_settings()
    engine = make_engine(settings)
    df = load_and_prepare_dataset(engine)
    df["quarter"] = df["as_of_timestamp"].dt.to_period("Q")

    quarters = sorted(df["quarter"].unique())
    print(f"Матчей: {len(df)}, кварталов: {len(quarters)} ({quarters[0]} .. {quarters[-1]})")

    rows = []
    for q in quarters:
        train_df = df[df["quarter"] < q]
        period_df = df[df["quarter"] == q]
        if len(train_df) < MIN_TRAIN_ROWS or len(period_df) == 0:
            print(f"{q}: пропущено (train n={len(train_df)} < {MIN_TRAIN_ROWS} или период пуст)")
            continue

        y_train = train_df["target"]
        y_period = period_df["target"]

        elo_model = EloOnlyModel()
        elo_model.fit(train_df, y_train)
        p_elo = elo_model.predict_proba(period_df)[:, 1]
        m_elo = compute_metrics(y_period, p_elo)

        logreg = LogisticRegressionModel(feature_names=FEATURE_SET_0345, random_state=RANDOM_SEED)
        logreg.fit(train_df, y_train)
        p_logreg = logreg.predict_proba(period_df)[:, 1]
        m_logreg = compute_metrics(y_period, p_logreg)

        catboost_model = CatBoostModel(feature_names=FEATURE_SET_0345)
        catboost_model.fit(train_df, y_train)
        p_catboost = catboost_model.predict_proba(period_df)[:, 1]
        m_catboost = compute_metrics(y_period, p_catboost)

        rows.append({
            "period": str(q),
            "train_n": len(train_df),
            "period_n": len(period_df),
            "elo_only": m_elo,
            "logreg": m_logreg,
            "catboost": m_catboost,
        })
        print(
            f"{q}  train_n={len(train_df):5d}  period_n={len(period_df):4d}  "
            f"elo(acc={m_elo['accuracy']:.3f} ll={m_elo['log_loss']:.3f})  "
            f"logreg(acc={m_logreg['accuracy']:.3f} ll={m_logreg['log_loss']:.3f})  "
            f"catboost(acc={m_catboost['accuracy']:.3f} ll={m_catboost['log_loss']:.3f})"
        )

    # --- Figure: walk-forward performance over time ---
    periods = [r["period"] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for model_key, label in [("elo_only", "Elo only"), ("logreg", "LogReg"), ("catboost", "CatBoost")]:
        axes[0].plot(periods, [r[model_key]["accuracy"] for r in rows], marker="o", label=label)
        axes[1].plot(periods, [r[model_key]["log_loss"] for r in rows], marker="o", label=label)
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Walk-forward out-of-sample performance по кварталам")
    axes[0].legend()
    axes[1].set_ylabel("log loss")
    axes[1].set_xlabel("период")
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "walk_forward_performance.png"), dpi=110)
    plt.close(fig)
    print(f"\nСохранено: {os.path.join(FIGURES_DIR, 'walk_forward_performance.png')}")

    # --- Aggregate summary ---
    import numpy as np

    for model_key, label in [("elo_only", "Elo only"), ("logreg", "LogReg"), ("catboost", "CatBoost")]:
        accs = [r[model_key]["accuracy"] for r in rows]
        lls = [r[model_key]["log_loss"] for r in rows]
        print(f"{label:12s}: mean_acc={np.mean(accs):.4f} std_acc={np.std(accs):.4f} "
              f"mean_log_loss={np.mean(lls):.4f}")

    entry = {
        "experiment_id": f"phase6_walk_forward_{datetime.now(timezone.utc).isoformat()}",
        "git_commit": git_commit_sha(),
        "min_train_rows": MIN_TRAIN_ROWS,
        "feature_set": FEATURE_SET_0345,
        "random_seed": RANDOM_SEED,
        "periods": rows,
    }
    path = os.path.join(EXPERIMENTS_DIR, "phase6_walk_forward.json")
    with open(path, "w") as f:
        json.dump(entry, f, indent=2, default=str)
    print(f"Experiment registry записан: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
