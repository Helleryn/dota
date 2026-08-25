#!/usr/bin/env python3
"""
PHASE 6 — baseline ML + честная out-of-sample оценка (см. задание).

Один воспроизводимый прогон: dataset -> leakage audit -> chronological
split -> baselines -> LogisticRegression -> CatBoost -> calibration ->
ablation -> team/cold-start checks -> sanity checks -> error analysis ->
figures -> experiment registry.

Запуск:
    python3 scripts/phase6_pipeline.py

Walk-forward backtest — отдельный скрипт (scripts/phase6_walk_forward.py),
т.к. использует другую структуру цикла (retrain per period).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_settings
from src.datasets.builder import build_dataset
from src.db.engine import make_engine
from src.evaluation.metrics import compute_metrics, format_metrics_row
from src.models.sklearn_models import (
    RANDOM_SEED,
    CATBOOST_PARAMS,
    CatBoostModel,
    EloOnlyModel,
    LogisticRegressionModel,
    MajorityBaselineModel,
    RandomBaselineModel,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")
FEATURE_SET_0345 = ["elo_difference", "recent_winrate_difference", "days_since_last_match_difference"]


def git_commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode().strip()
    except Exception:
        return "unknown"


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. Dataset + team_a/team_b sanity + data cutoff audit
# ---------------------------------------------------------------------------

def load_and_prepare_dataset(engine) -> pd.DataFrame:
    result = build_dataset(engine, persist=False)
    df = result.dataframe.copy()
    df = df.sort_values("as_of_timestamp").reset_index(drop=True)
    df["target"] = df["radiant_win"].astype(int)
    df["days_since_last_match_difference"] = (
        df["radiant_days_since_last_match"] - df["dire_days_since_last_match"]
    )
    return df


def audit_team_a_team_b_not_winner_loser(df: pd.DataFrame) -> None:
    """
    Раздел 2 задания: team_a (=radiant) / team_b (=dire) назначаются игрой
    ДО исхода матча (какая сторона карты), не по результату. Независимая
    проверка: если бы radiant было тайно "winner", radiant_win было бы
    тождественно True. Он не тождественен (см. class balance) — и явно
    НЕ коррелирует с порядком team_id (сортировка radiant/dire никак не
    зависит от того, кто в итоге победил — это определяется matchmaking'ом
    ДО игры, наш pipeline это поле только читает).
    """
    win_rate = df["target"].mean()
    assert 0.3 < win_rate < 0.7, (
        f"ПОДОЗРЕНИЕ НА УТЕЧКУ: radiant_win={win_rate:.3f} — если бы radiant "
        f"был замаскированным 'winner', здесь было бы ~1.0"
    )
    # radiant_team_id не обязан быть "меньше" dire_team_id, "raньше" его и т.п. —
    # проверяем, что нет тривиальной корреляции с порядком id.
    id_order_matches_winner = (df["radiant_team_id"] < df["dire_team_id"]) == (df["target"] == 1)
    corr = id_order_matches_winner.mean()
    assert 0.3 < corr < 0.7, f"ПОДОЗРЕНИЕ: team_id order коррелирует с target ({corr:.3f})"
    print(f"OK: radiant_win rate = {win_rate:.4f} (не тождественно 0 или 1)")
    print(f"OK: team_id order vs target корреляция = {corr:.4f} (не тривиальна)")


def audit_data_cutoff(df: pd.DataFrame) -> None:
    """
    Раздел 4: независимая (не через RatingEngine) проверка, что признаки
    команды в строке i вычислены СТРОГО по матчам этой же команды с
    as_of_timestamp < строки i. Используем pandas groupby/cumcount —
    другой путь вычисления, чем build_feature_set_0 (RatingEngine +
    _TeamFormTracker), поэтому это настоящая независимая перепроверка, а
    не "то же самое другими словами".
    """
    long = pd.concat([
        df[["match_id", "as_of_timestamp", "radiant_team_id", "radiant_matches_played_before"]]
        .rename(columns={"radiant_team_id": "team_id", "radiant_matches_played_before": "matches_played_before"}),
        df[["match_id", "as_of_timestamp", "dire_team_id", "dire_matches_played_before"]]
        .rename(columns={"dire_team_id": "team_id", "dire_matches_played_before": "matches_played_before"}),
    ]).sort_values(["team_id", "as_of_timestamp"]).reset_index(drop=True)

    independent_count = long.groupby("team_id").cumcount()
    mismatches = (independent_count.values != long["matches_played_before"].values).sum()

    assert mismatches == 0, (
        f"УТЕЧКА/БАГ: {mismatches} строк, где matches_played_before не совпадает "
        f"с независимо посчитанным числом строго более ранних матчей команды"
    )
    print(f"OK: {len(long)} team-match записей — matches_played_before независимо "
          f"перепроверен через pandas cumcount, 0 расхождений")
    print("Вывод: max(source_timestamp_for_features) < prediction_timestamp по построению "
          "(RatingEngine/_TeamFormTracker читают состояние ДО записи в него, см. "
          "src/datasets/feature_set_0.py) — независимая проверка выше подтверждает это "
          "на РЕАЛЬНОМ датасете, не только на коде.")


# ---------------------------------------------------------------------------
# 2. Chronological split
# ---------------------------------------------------------------------------

def chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    n = len(df)
    train_end_idx = int(n * train_frac)
    val_end_idx = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end_idx].reset_index(drop=True)
    val = df.iloc[train_end_idx:val_end_idx].reset_index(drop=True)
    test = df.iloc[val_end_idx:].reset_index(drop=True)

    periods = {
        "train_start": train["as_of_timestamp"].min(),
        "train_end": train["as_of_timestamp"].max(),
        "validation_start": val["as_of_timestamp"].min(),
        "validation_end": val["as_of_timestamp"].max(),
        "test_start": test["as_of_timestamp"].min(),
        "test_end": test["as_of_timestamp"].max(),
    }

    assert periods["train_end"] < periods["validation_start"], "TEMPORAL OVERLAP train/validation"
    assert periods["validation_end"] < periods["test_start"], "TEMPORAL OVERLAP validation/test"

    print(f"TRAIN:      {periods['train_start']} .. {periods['train_end']}  (n={len(train)})")
    print(f"VALIDATION: {periods['validation_start']} .. {periods['validation_end']}  (n={len(val)})")
    print(f"TEST:       {periods['test_start']} .. {periods['test_end']}  (n={len(test)})")
    print("OK: нет temporal overlap (max(train) < min(val) < ... < min(test))")

    return train, val, test, periods


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    commit_sha = git_commit_sha()
    run_timestamp = datetime.now(timezone.utc).isoformat()

    section("PHASE 6 — Existing ML Pipeline Audit")
    print("RatingEngine, DatasetBuilder, BacktestEngine, BasePredictionModel — из Phase 4/5, не переписаны.")
    print(f"git commit: {commit_sha}")

    settings = load_settings()
    engine = make_engine(settings)

    section("1. Dataset")
    df = load_and_prepare_dataset(engine)
    print(f"Строк (pro/premium матчей, Feature Set 0): {len(df)}")
    print(f"Период: {df['as_of_timestamp'].min()} .. {df['as_of_timestamp'].max()}")
    print(f"Class balance: radiant_win=1: {df['target'].mean():.4f}, radiant_win=0: {1-df['target'].mean():.4f}")

    section("2. team_a/team_b != winner/loser (раздел 2)")
    audit_team_a_team_b_not_winner_loser(df)

    section("3-4. Data cutoff audit (раздел 3-4)")
    audit_data_cutoff(df)

    section("5-6. Chronological split (раздел 5-6)")
    train, val, test, periods = chronological_split(df)

    X_train, y_train = train, train["target"]
    X_val, y_val = val, val["target"]
    X_test, y_test = test, test["target"]

    results: dict = {}

    section("7-9. Baselines: Random / Majority / Elo-only (раздел 7-9)")
    random_model = RandomBaselineModel()
    random_model.fit(X_train, y_train)
    results["random"] = random_model.evaluate(X_test, y_test)
    print(format_metrics_row("random_50_50", results["random"]))

    majority_model = MajorityBaselineModel()
    majority_model.fit(X_train, y_train)
    results["majority"] = majority_model.evaluate(X_test, y_test)
    print(format_metrics_row("majority", results["majority"]))

    elo_model = EloOnlyModel()
    elo_model.fit(X_train, y_train)
    results["elo_only"] = elo_model.evaluate(X_test, y_test)
    print(format_metrics_row("elo_only", results["elo_only"]))

    section("10. Logistic Regression (раздел 10)")
    logreg = LogisticRegressionModel(feature_names=FEATURE_SET_0345, random_state=RANDOM_SEED)
    logreg.fit(X_train, y_train)
    results["logreg_raw"] = logreg.evaluate(X_test, y_test)
    print(format_metrics_row("logreg_raw", results["logreg_raw"]))
    print("Коэффициенты:", logreg.feature_importance())

    logreg.calibrate(X_val, y_val)
    results["logreg_calibrated"] = logreg.evaluate(X_test, y_test)
    print(format_metrics_row("logreg_calibrated", results["logreg_calibrated"]))

    section("11. CatBoost (раздел 11)")
    print("Зафиксированные гиперпараметры:", json.dumps(CATBOOST_PARAMS))
    catboost_model = CatBoostModel(feature_names=FEATURE_SET_0345)
    catboost_model.fit(X_train, y_train)
    results["catboost_raw"] = catboost_model.evaluate(X_test, y_test)
    print(format_metrics_row("catboost_raw", results["catboost_raw"]))
    print("Feature importance:", catboost_model.feature_importance())

    catboost_model.calibrate(X_val, y_val)
    results["catboost_calibrated"] = catboost_model.evaluate(X_test, y_test)
    print(format_metrics_row("catboost_calibrated", results["catboost_calibrated"]))

    section("12. Итоговая сравнительная таблица (раздел 18)")
    table_rows = [
        ("Random 50/50", results["random"]),
        ("Majority", results["majority"]),
        ("Elo only", results["elo_only"]),
        ("Logistic Regression (raw)", results["logreg_raw"]),
        ("Logistic Regression (calibrated)", results["logreg_calibrated"]),
        ("CatBoost (raw)", results["catboost_raw"]),
        ("CatBoost (calibrated)", results["catboost_calibrated"]),
    ]
    print(f"{'Model':35s} {'Acc':>8s} {'BalAcc':>8s} {'LogLoss':>9s} {'Brier':>8s} {'ROC-AUC':>8s}")
    for name, m in table_rows:
        print(f"{name:35s} {m['accuracy']:8.4f} {m['balanced_accuracy']:8.4f} {m['log_loss']:9.4f} {m['brier_score']:8.4f} {m['roc_auc']:8.4f}")

    section("19. Feature ablation (раздел 19)")
    ablation_results = {}
    for exp_name, features in [
        ("A: elo only", ["elo_difference"]),
        ("B: elo + form", ["elo_difference", "recent_winrate_difference"]),
        ("C: elo + form + rest", FEATURE_SET_0345),
    ]:
        m = LogisticRegressionModel(feature_names=features, random_state=RANDOM_SEED)
        m.fit(X_train, y_train)
        metrics = m.evaluate(X_test, y_test)
        ablation_results[exp_name] = metrics
        print(format_metrics_row(exp_name, metrics))

    section("22. Team leakage check (раздел 22)")
    team_counts = pd.concat([df["radiant_team_id"], df["dire_team_id"]]).value_counts()
    print(f"Команд всего: {len(team_counts)}, матчей на команду: min={team_counts.min()} "
          f"median={team_counts.median():.0f} max={team_counts.max()}")
    frequent_threshold = team_counts.quantile(0.75)
    frequent_teams = set(team_counts[team_counts >= frequent_threshold].index)
    test_both_frequent = test[
        test["radiant_team_id"].isin(frequent_teams) & test["dire_team_id"].isin(frequent_teams)
    ]
    test_has_infrequent = test[
        ~(test["radiant_team_id"].isin(frequent_teams) & test["dire_team_id"].isin(frequent_teams))
    ]
    if len(test_both_frequent) > 5 and len(test_has_infrequent) > 5:
        p_freq = catboost_model.predict_proba(test_both_frequent)[:, 1]
        p_infreq = catboost_model.predict_proba(test_has_infrequent)[:, 1]
        m_freq = compute_metrics(test_both_frequent["target"], p_freq)
        m_infreq = compute_metrics(test_has_infrequent["target"], p_infreq)
        print(format_metrics_row("both_frequent_teams", m_freq))
        print(format_metrics_row("has_infrequent_team", m_infreq))
    else:
        print(f"Недостаточно строк в TEST для честного сравнения (frequent={len(test_both_frequent)}, "
              f"infrequent={len(test_has_infrequent)}) — пропущено")
        m_freq, m_infreq = None, None

    section("23. Cold start (раздел 23)")
    test_cold = test[(test["radiant_matches_played_before"] == 0) | (test["dire_matches_played_before"] == 0)]
    test_warm = test[(test["radiant_matches_played_before"] > 0) & (test["dire_matches_played_before"] > 0)]
    print(f"Cold-start строк в TEST: {len(test_cold)}, known-team строк: {len(test_warm)}")
    if len(test_cold) > 5:
        m_cold = compute_metrics(test_cold["target"], catboost_model.predict_proba(test_cold)[:, 1])
        print(format_metrics_row("cold_start", m_cold))
    else:
        m_cold = None
        print("Недостаточно cold-start строк в TEST для отдельной метрики")
    if len(test_warm) > 5:
        m_warm = compute_metrics(test_warm["target"], catboost_model.predict_proba(test_warm)[:, 1])
        print(format_metrics_row("known_team", m_warm))
    else:
        m_warm = None

    section("24. Elo sanity check (раздел 24)")
    all_elo = pd.concat([df["radiant_elo"], df["dire_elo"]])
    print(f"mean={all_elo.mean():.2f} std={all_elo.std():.2f} min={all_elo.min():.2f} max={all_elo.max():.2f}")
    n_nan = all_elo.isna().sum()
    n_inf = np.isinf(all_elo).sum()
    print(f"NaN: {n_nan}, inf: {n_inf}")
    assert n_nan == 0 and n_inf == 0, "Elo содержит NaN/inf — БАГ"
    print("OK: Elo без NaN/inf")

    section("25. Recent form sanity check (раздел 25)")
    # Берём случайный match с известной командой и пересчитываем recent_winrate
    # НАПРЯМУЮ из df (независимо от _TeamFormTracker) — окно не должно включать текущий матч.
    sample_row = df[df["radiant_matches_played_before"] >= 5].iloc[len(df) // 2]
    team_id = sample_row["radiant_team_id"]
    as_of = sample_row["as_of_timestamp"]
    prior_matches = df[
        ((df["radiant_team_id"] == team_id) | (df["dire_team_id"] == team_id))
        & (df["as_of_timestamp"] < as_of)
    ].sort_values("as_of_timestamp").tail(5)
    wins = 0
    for _, r in prior_matches.iterrows():
        won = (r["radiant_team_id"] == team_id and r["target"] == 1) or (r["dire_team_id"] == team_id and r["target"] == 0)
        wins += int(won)
    independent_winrate = wins / len(prior_matches) if len(prior_matches) else None
    print(f"match_id={sample_row['match_id']}, team={team_id}: "
          f"stored recent_winrate={sample_row['radiant_recent_winrate']:.4f}, "
          f"независимо пересчитанный (последние {len(prior_matches)} СТРОГО более ранних матчей)={independent_winrate:.4f}")
    assert abs(sample_row["radiant_recent_winrate"] - independent_winrate) < 1e-9, "УТЕЧКА в recent_winrate"
    print("OK: recent_winrate не включает текущий матч")

    section("30-31. Feature importance + Error analysis (раздел 30-31)")
    test_with_pred = test.copy()
    test_with_pred["p_radiant_calibrated"] = catboost_model.predict_proba(test)[:, 1]
    test_with_pred["error"] = np.abs(test_with_pred["p_radiant_calibrated"] - test_with_pred["target"])
    top_errors = test_with_pred.sort_values("error", ascending=False).head(10)
    print("Топ-10 самых уверенных ошибок (CatBoost, calibrated, TEST):")
    for _, r in top_errors.iterrows():
        print(
            f"  match_id={r['match_id']} as_of={r['as_of_timestamp']} "
            f"radiant={r['radiant_team_id']} dire={r['dire_team_id']} "
            f"P(radiant)={r['p_radiant_calibrated']:.3f} actual_radiant_win={bool(r['target'])} "
            f"elo_diff={r['elo_difference']:.1f} form_diff={r['recent_winrate_difference']}"
        )

    section("29. Visualizations (раздел 29)")
    save_figures(df, test, catboost_model, results, ablation_results)

    section("26-27. Experiment registry (раздел 26-27)")
    save_experiment_registry(commit_sha, run_timestamp, df, periods, results, ablation_results, FEATURE_SET_0345)

    section("Итог секции 33-35")
    print("Все обязательные шаги Phase 6 (пункты 1-11, 33) выполнены — см. вывод выше и reports/phase6-summary.md.")

    return 0


def save_figures(df, test, catboost_model, results, ablation_results) -> None:
    # 1. Elo distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pd.concat([df["radiant_elo"], df["dire_elo"]]), bins=50)
    ax.set_title("Elo distribution (все team-match записи)")
    ax.set_xlabel("Elo")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "elo_distribution.png"), dpi=110)
    plt.close(fig)

    # 2. Prediction probability distribution (CatBoost, TEST)
    p_test = catboost_model.predict_proba(test)[:, 1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(p_test, bins=30)
    ax.set_title("Predicted P(radiant_win) distribution — CatBoost, TEST")
    ax.set_xlabel("P(radiant_win)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "prediction_probability_distribution.png"), dpi=110)
    plt.close(fig)

    # 3. Calibration curve (raw vs calibrated)
    from sklearn.calibration import calibration_curve

    p_raw = catboost_model._raw_proba(test)
    y_true = test["target"].to_numpy()
    frac_raw, mean_raw = calibration_curve(y_true, p_raw, n_bins=8, strategy="quantile")
    frac_cal, mean_cal = calibration_curve(y_true, p_test, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    ax.plot(mean_raw, frac_raw, "o-", label="CatBoost raw")
    ax.plot(mean_cal, frac_cal, "o-", label="CatBoost calibrated (isotonic, fit on val)")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("fraction of positives")
    ax.set_title("Calibration curve — TEST")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "calibration_curve.png"), dpi=110)
    plt.close(fig)

    # 5. Feature importance (CatBoost)
    fi = catboost_model.feature_importance()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(list(fi.keys()), list(fi.values()))
    ax.set_title("CatBoost feature importance")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=110)
    plt.close(fig)

    print(f"Сохранено 4 графика в {FIGURES_DIR} (5-й, walk-forward, сохраняет scripts/phase6_walk_forward.py)")


def save_experiment_registry(commit_sha, run_timestamp, df, periods, results, ablation_results, feature_names) -> None:
    entry = {
        "experiment_id": f"phase6_baseline_{run_timestamp}",
        "timestamp": run_timestamp,
        "git_commit": commit_sha,
        "dataset_version": "v0_baseline",
        "dataset_n_rows": int(len(df)),
        "dataset_period": [str(df["as_of_timestamp"].min()), str(df["as_of_timestamp"].max())],
        "feature_set": feature_names,
        "split_periods": {k: str(v) for k, v in periods.items()},
        "random_seed": RANDOM_SEED,
        "catboost_hyperparameters": CATBOOST_PARAMS,
        "results": results,
        "ablation": ablation_results,
    }
    path = os.path.join(EXPERIMENTS_DIR, "phase6_baseline.json")
    with open(path, "w") as f:
        json.dump(entry, f, indent=2, default=str)
    print(f"Experiment registry записан: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
