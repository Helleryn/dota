#!/usr/bin/env python3
"""
Sanity-check построенного датасета (Phase 5, раздел 28) — dataset shape,
missing values, class balance, feature distributions, и ОДНА простая
LogisticRegression исключительно для проверки, что pipeline вообще
позволяет обучить модель на этих данных.

Это НЕ финальная модель и НЕ оценка качества (для этого — Phase 6,
walk-forward backtesting на полноценном датасете). На fixture-объёме
(единицы-десятки матчей) метрики не несут статистического смысла — здесь
проверяется механика, не предсказательная сила.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # позволяет запускать напрямую из scripts/

from src.config import load_settings
from src.datasets.builder import build_dataset
from src.db.engine import make_engine


def main() -> int:
    settings = load_settings()
    engine = make_engine(settings)

    result = build_dataset(engine, persist=False)
    df = result.dataframe

    print(f"=== Dataset shape ===\n{df.shape[0]} строк, {df.shape[1]} столбцов\n")

    print("=== Missing values (доля NaN по столбцу) ===")
    print(df.isna().mean().round(3).to_string())
    print()

    print("=== Class balance (radiant_win) ===")
    print(df["radiant_win"].value_counts(normalize=True).round(3).to_string())
    print()

    print("=== Feature distributions (числовые) ===")
    numeric_cols = ["radiant_elo", "dire_elo", "elo_difference", "radiant_recent_winrate", "dire_recent_winrate"]
    print(df[numeric_cols].describe().round(2).to_string())
    print()

    if len(df) < 10:
        print(f"ПРЕДУПРЕЖДЕНИЕ: всего {len(df)} строк — датасет слишком мал для содержательного "
              f"обучения модели (это ожидаемо на fixture-данных Phase 5, см. phase5-summary.md). "
              f"LogisticRegression ниже — ТОЛЬКО проверка механики pipeline, не оценка качества.")

    df_clean = df.dropna(subset=["radiant_elo", "dire_elo", "elo_difference"])
    if df_clean["radiant_win"].nunique() < 2 or len(df_clean) < 4:
        print("Недостаточно данных/классов для даже формального fit() LogisticRegression — пропуск.")
        return 0

    from sklearn.linear_model import LogisticRegression

    X = df_clean[["elo_difference"]].fillna(0.0)
    y = df_clean["radiant_win"].astype(int)

    model = LogisticRegression()
    model.fit(X, y)
    train_accuracy = model.score(X, y)

    print(f"\n=== Sanity-check LogisticRegression (НЕ финальная модель) ===")
    print(f"Обучена на {len(X)} строках (train == test, это проверка механики, не оценка качества)")
    print(f"Train accuracy: {train_accuracy:.3f}")
    print(f"Коэффициент при elo_difference: {model.coef_[0][0]:.5f} "
          f"({'положительный, как ожидается' if model.coef_[0][0] > 0 else 'ОТРИЦАТЕЛЬНЫЙ — проверить знак признака!'})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
