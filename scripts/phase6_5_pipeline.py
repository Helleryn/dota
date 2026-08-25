#!/usr/bin/env python3
"""
PHASE 6.5 — continuous history + statistical validation.

Один прогон: непрерывный датасет (2021-2026, без квартальных разрывов) ->
Elo K-analysis (на validation) -> multi-window recent form/rest ablation
(один и тот же evaluation set для всех моделей, раздел 14) -> walk-forward ->
статистическая значимость (block bootstrap, McNemar) -> calibration ->
confidence buckets -> temporal/patch/team-frequency stability -> decision rule.

Запуск:
    python3 scripts/phase6_5_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.config import load_settings
from src.datasets.builder import _load_pro_matches
from src.datasets.multi_window_features import FORM_WINDOWS, REST_WINDOWS_DAYS, build_multi_window_features
from src.db.engine import make_engine
from src.evaluation.metrics import compute_metrics
from src.evaluation.statistics import (
    accuracy_metric,
    block_bootstrap_metric,
    block_bootstrap_paired_diff,
    brier_metric,
    log_loss_metric,
    mcnemar_exact,
)
from src.models.sklearn_models import RANDOM_SEED, EloOnlyModel, LogisticRegressionModel
from scripts.phase6_pipeline import chronological_split, git_commit_sha, section

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")

K_CANDIDATES = [16, 24, 32, 40, 64]
BLOCK_SIZE = 20  # ~несколько дней матчей подряд — единица block bootstrap


@dataclass
class RawMatchRow:
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_win: bool


def load_raw_matches() -> list:
    settings = load_settings()
    engine = make_engine(settings)
    rows = _load_pro_matches(engine)  # уже (start_time, match_id) tie-break отсортирован
    return [RawMatchRow(r.match_id, r.start_time, r.radiant_team_id, r.dire_team_id, r.radiant_win) for r in rows], engine


def rows_to_dataframe(rows) -> pd.DataFrame:
    records = []
    for r in rows:
        rec = {
            "match_id": r.match_id,
            "as_of_timestamp": r.as_of_timestamp,
            "radiant_team_id": r.radiant_team_id,
            "dire_team_id": r.dire_team_id,
            "elo_difference": r.elo_difference,
            "days_since_last_match_difference": r.days_since_last_match_difference,
            "radiant_matches_played_before": r.radiant_matches_played_before,
            "dire_matches_played_before": r.dire_matches_played_before,
            "target": int(r.radiant_win),
        }
        for w in FORM_WINDOWS:
            rec[f"form_{w}_difference"] = r.recent_winrate_difference[w]
        for n in REST_WINDOWS_DAYS:
            rec[f"matches_last_{n}d_difference"] = r.matches_last_n_days_difference[n]
        records.append(rec)
    return pd.DataFrame(records)


def split_by_index(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    n = len(df)
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:a].reset_index(drop=True), df.iloc[a:b].reset_index(drop=True), df.iloc[b:].reset_index(drop=True)


def main() -> int:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    commit_sha = git_commit_sha()
    run_ts = datetime.now(timezone.utc).isoformat()

    section("PHASE 6.5 — Existing Baseline Audit")
    print(f"git commit: {commit_sha}")

    raw_matches, engine = load_raw_matches()
    print(f"Матчей (pro+premium, непрерывная история): {len(raw_matches)}")
    print(f"Период: {raw_matches[0].start_time} .. {raw_matches[-1].start_time}")

    section("Elo K-analysis (раздел 19-20) — выбор K ТОЛЬКО на validation")
    # Разбиение по ИНДЕКСАМ фиксируется один раз по количеству матчей — не
    # пересчитывается для каждого K (одна и та же граница train/val/test).
    n = len(raw_matches)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    k_results = {}
    for k in K_CANDIDATES:
        rows_k = build_multi_window_features(raw_matches, k_factor=k)
        df_k = rows_to_dataframe(rows_k)
        val_k = df_k.iloc[train_end:val_end]
        elo_model = EloOnlyModel()
        p_val = elo_model.predict_proba(val_k.rename(columns={"elo_difference": "elo_difference"}))[:, 1]
        m = compute_metrics(val_k["target"], p_val)
        k_results[k] = m
        print(f"K={k:3d}: val accuracy={m['accuracy']:.4f} log_loss={m['log_loss']:.4f} brier={m['brier_score']:.4f}")

    best_k = min(k_results, key=lambda k: k_results[k]["log_loss"])
    print(f"\nВыбран K={best_k} (минимальный log_loss на VALIDATION) — используется дальше, "
          f"test ещё не тронут.")

    section("Построение финального multi-window feature set (раздел 10-12)")
    rows = build_multi_window_features(raw_matches, k_factor=best_k)
    df = rows_to_dataframe(rows)
    assert len(df) == n
    print(f"Строк: {len(df)}, tie-breaker: (start_time, match_id) — см. src/datasets/builder.py")

    train, val, test = split_by_index(df)
    print(f"TRAIN: {train['as_of_timestamp'].min()} .. {train['as_of_timestamp'].max()} (n={len(train)})")
    print(f"VAL:   {val['as_of_timestamp'].min()} .. {val['as_of_timestamp'].max()} (n={len(val)})")
    print(f"TEST:  {test['as_of_timestamp'].min()} .. {test['as_of_timestamp'].max()} (n={len(test)})")
    assert train["as_of_timestamp"].max() < val["as_of_timestamp"].min()
    assert val["as_of_timestamp"].max() < test["as_of_timestamp"].min()
    print("OK: нет temporal overlap")

    section("Раздел 14 — единый evaluation set (исключаем cold-start из ВСЕХ сравнений)")
    def not_cold_start(d):
        return d[(d["radiant_matches_played_before"] > 0) & (d["dire_matches_played_before"] > 0)]

    train_f, val_f, test_f = not_cold_start(train), not_cold_start(val), not_cold_start(test)
    print(f"После исключения cold-start: TRAIN n={len(train_f)} (было {len(train)}), "
          f"VAL n={len(val_f)} (было {len(val)}), TEST n={len(test_f)} (было {len(test)})")
    print("Эта же выборка (train_f/val_f/test_f) используется ВСЕМИ моделями A-G ниже — честное сравнение.")

    section("Раздел 13 — Models A-G ablation (LogisticRegression, единый TEST)")
    experiments = {
        "A: Elo only": ["elo_difference"],
        "B: Elo + Form3": ["elo_difference", "form_3_difference"],
        "C: Elo + Form5": ["elo_difference", "form_5_difference"],
        "D: Elo + Form10": ["elo_difference", "form_10_difference"],
        "E: Elo + Form20": ["elo_difference", "form_20_difference"],
        "F: Elo + Rest": ["elo_difference", "days_since_last_match_difference"],
    }
    ablation_results = {}
    fitted_models = {}
    for name, features in experiments.items():
        m = LogisticRegressionModel(feature_names=features, random_state=RANDOM_SEED)
        m.fit(train_f, train_f["target"])
        metrics = m.evaluate(test_f, test_f["target"])
        ablation_results[name] = metrics
        fitted_models[name] = m
        print(f"{name:22s} n={metrics['n']:5d} acc={metrics['accuracy']:.4f} bal_acc={metrics['balanced_accuracy']:.4f} "
              f"log_loss={metrics['log_loss']:.4f} brier={metrics['brier_score']:.4f} roc_auc={metrics['roc_auc']:.4f}")

    # Лучшее окно формы выбирается по VALIDATION (не test) для составной модели G.
    form_val_logloss = {}
    for w in FORM_WINDOWS:
        feats = ["elo_difference", f"form_{w}_difference"]
        m = LogisticRegressionModel(feature_names=feats, random_state=RANDOM_SEED)
        m.fit(train_f, train_f["target"])
        val_metrics = m.evaluate(val_f, val_f["target"])
        form_val_logloss[w] = val_metrics["log_loss"]
    best_form_window = min(form_val_logloss, key=lambda w: form_val_logloss[w])
    print(f"\nЛучшее окно формы по VALIDATION log_loss: {best_form_window} ({form_val_logloss})")

    features_g = ["elo_difference", f"form_{best_form_window}_difference", "days_since_last_match_difference"]
    model_g = LogisticRegressionModel(feature_names=features_g, random_state=RANDOM_SEED)
    model_g.fit(train_f, train_f["target"])
    metrics_g = model_g.evaluate(test_f, test_f["target"])
    ablation_results[f"G: Elo + Form{best_form_window} + Rest"] = metrics_g
    fitted_models[f"G: Elo + Form{best_form_window} + Rest"] = model_g
    print(f"{'G: Elo + Form'+str(best_form_window)+' + Rest':22s} n={metrics_g['n']:5d} acc={metrics_g['accuracy']:.4f} "
          f"bal_acc={metrics_g['balanced_accuracy']:.4f} log_loss={metrics_g['log_loss']:.4f} "
          f"brier={metrics_g['brier_score']:.4f} roc_auc={metrics_g['roc_auc']:.4f}")

    elo_model_final = EloOnlyModel()
    p_elo_test = elo_model_final.predict_proba(test_f)[:, 1]
    metrics_elo = compute_metrics(test_f["target"], p_elo_test)
    print(f"{'Elo-only (raw formula)':22s} n={metrics_elo['n']:5d} acc={metrics_elo['accuracy']:.4f} "
          f"log_loss={metrics_elo['log_loss']:.4f} brier={metrics_elo['brier_score']:.4f} roc_auc={metrics_elo['roc_auc']:.4f}")

    section("Раздел 15 — Walk-forward (по годам, единый evaluation set)")
    df["year"] = df["as_of_timestamp"].dt.year
    walk_forward_rows = []
    years = sorted(df["year"].unique())
    for y in years[1:]:  # первый год — только warm-up
        tr = not_cold_start(df[df["year"] < y])
        pe = not_cold_start(df[df["year"] == y])
        if len(tr) < 500 or len(pe) < 20:
            print(f"{y}: пропущено (train n={len(tr)}, period n={len(pe)})")
            continue
        p_elo = EloOnlyModel().predict_proba(pe)[:, 1]
        m_elo_y = compute_metrics(pe["target"], p_elo)

        m_full = LogisticRegressionModel(feature_names=features_g, random_state=RANDOM_SEED)
        m_full.fit(tr, tr["target"])
        p_full = m_full.predict_proba(pe)[:, 1]
        m_full_y = compute_metrics(pe["target"], p_full)

        walk_forward_rows.append({"year": int(y), "n": len(pe), "elo": m_elo_y, "full_model": m_full_y})
        print(f"{y}  n={len(pe):5d}  elo(acc={m_elo_y['accuracy']:.3f} ll={m_elo_y['log_loss']:.3f})  "
              f"full(acc={m_full_y['accuracy']:.3f} ll={m_full_y['log_loss']:.3f})")

    section("Раздел 16-18 — статистическая значимость (block bootstrap, block_size=20)")
    y_test = test_f["target"].to_numpy()
    p_elo = p_elo_test
    p_logreg_full = fitted_models[f"G: Elo + Form{best_form_window} + Rest"].predict_proba(test_f)[:, 1]

    acc_ci = block_bootstrap_metric(y_test, p_elo, accuracy_metric, block_size=BLOCK_SIZE)
    ll_ci = block_bootstrap_metric(y_test, p_elo, log_loss_metric, block_size=BLOCK_SIZE)
    brier_ci = block_bootstrap_metric(y_test, p_elo, brier_metric, block_size=BLOCK_SIZE)
    print(f"Elo accuracy:  point={acc_ci['point']:.4f}  95% CI=[{acc_ci['ci_low']:.4f}, {acc_ci['ci_high']:.4f}]  (n_blocks={acc_ci['n_blocks']})")
    print(f"Elo log_loss:  point={ll_ci['point']:.4f}  95% CI=[{ll_ci['ci_low']:.4f}, {ll_ci['ci_high']:.4f}]")
    print(f"Elo brier:     point={brier_ci['point']:.4f}  95% CI=[{brier_ci['ci_low']:.4f}, {brier_ci['ci_high']:.4f}]")

    effect_vs_random = block_bootstrap_paired_diff(
        y_test, p_elo, np.full_like(p_elo, 0.5), accuracy_metric, block_size=BLOCK_SIZE
    )
    print(f"\nEffect size (Elo accuracy - 0.5): {effect_vs_random['point_diff']:+.4f} pp "
          f"95% CI=[{effect_vs_random['ci_low']:+.4f}, {effect_vs_random['ci_high']:+.4f}]")
    elo_beats_random = effect_vs_random["ci_low"] > 0
    print(f"CI исключает 0 -> Elo статистически лучше 0.5: {elo_beats_random}")

    diff_elo_vs_full = block_bootstrap_paired_diff(y_test, p_elo, p_logreg_full, log_loss_metric, block_size=BLOCK_SIZE)
    print(f"\nlog_loss(Elo) - log_loss(Full model): {diff_elo_vs_full['point_diff']:+.4f} "
          f"95% CI=[{diff_elo_vs_full['ci_low']:+.4f}, {diff_elo_vs_full['ci_high']:+.4f}] "
          f"(отрицательное = Elo ЛУЧШЕ, т.к. log_loss меньше=лучше)")

    pred_class_elo = (p_elo >= 0.5).astype(int)
    pred_class_full = (p_logreg_full >= 0.5).astype(int)
    mcnemar_result = mcnemar_exact(y_test, pred_class_elo, pred_class_full)
    print(f"\nMcNemar (Elo vs Full model): a_only={mcnemar_result['a_correct_b_wrong']} "
          f"b_only={mcnemar_result['b_correct_a_wrong']} n_discordant={mcnemar_result['n_discordant']} "
          f"p_value={mcnemar_result['p_value']:.4f}")

    section("Раздел 21-22 — Elo probability vs constant 0.5, calibration")
    const_ci = block_bootstrap_metric(y_test, np.full_like(p_elo, 0.5), log_loss_metric, block_size=BLOCK_SIZE)
    print(f"Constant P=0.5:  log_loss point={const_ci['point']:.4f}  95% CI=[{const_ci['ci_low']:.4f}, {const_ci['ci_high']:.4f}]")
    print(f"Elo P:           log_loss point={ll_ci['point']:.4f}  95% CI=[{ll_ci['ci_low']:.4f}, {ll_ci['ci_high']:.4f}]")

    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_test, p_elo, n_bins=8, strategy="quantile")
    print("Calibration curve (Elo, TEST): mean_predicted -> fraction_actual_positive")
    for mp, fp in zip(mean_pred, frac_pos):
        print(f"  {mp:.3f} -> {fp:.3f}")

    section("Раздел 23 — Confidence buckets (Elo, TEST)")
    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 1.01)]
    bucket_rows = []
    p_elo_symmetric = np.where(p_elo >= 0.5, p_elo, 1 - p_elo)  # "уверенность в фаворите", симметрично
    y_favored_correct = np.where(p_elo >= 0.5, y_test == 1, y_test == 0).astype(int)
    for lo, hi in buckets:
        mask = (p_elo_symmetric >= lo) & (p_elo_symmetric < hi)
        n_b = int(mask.sum())
        if n_b == 0:
            print(f"  [{lo:.2f}, {hi:.2f}): n=0")
            continue
        acc_b = float(y_favored_correct[mask].mean())
        mean_p_b = float(p_elo_symmetric[mask].mean())
        bucket_rows.append({"range": f"[{lo:.2f},{hi:.2f})", "n": n_b, "mean_predicted": mean_p_b, "actual_favored_winrate": acc_b})
        print(f"  [{lo:.2f}, {hi:.2f}): n={n_b:4d}  mean_predicted={mean_p_b:.3f}  actual_favorite_winrate={acc_b:.3f}")

    section("Раздел 24 — Temporal stability по годам (Elo, вся история — формула без обучения)")
    df_nc = not_cold_start(df)
    temporal_rows = []
    for y in years:
        d_y = df_nc[df_nc["year"] == y]
        if len(d_y) < 20:
            continue
        p_y = EloOnlyModel().predict_proba(d_y)[:, 1]
        m_y = compute_metrics(d_y["target"], p_y)
        temporal_rows.append({"year": int(y), **m_y})
        print(f"{y}: n={m_y['n']:5d} acc={m_y['accuracy']:.4f} log_loss={m_y['log_loss']:.4f} roc_auc={m_y['roc_auc']:.4f}")

    section("Раздел 26 — Patch stability (Elo, analysis only — patch НЕ добавляется в модель)")
    with engine.connect() as conn:
        patch_rows = conn.execute(text("SELECT match_id, patch_id FROM matches")).fetchall()
    patch_map = {r.match_id: r.patch_id for r in patch_rows}
    df_nc = df_nc.copy()
    df_nc["patch_id"] = df_nc["match_id"].map(patch_map)
    patch_stability_rows = []
    for patch_id, d_p in df_nc.groupby("patch_id"):
        if len(d_p) < 200 or pd.isna(patch_id):
            continue
        p_p = EloOnlyModel().predict_proba(d_p)[:, 1]
        m_p = compute_metrics(d_p["target"], p_p)
        patch_stability_rows.append({"patch_id": int(patch_id), **m_p})
    patch_stability_rows.sort(key=lambda r: r["patch_id"])
    for r in patch_stability_rows:
        print(f"patch_id={r['patch_id']}: n={r['n']:5d} acc={r['accuracy']:.4f} log_loss={r['log_loss']:.4f}")

    section("Раздел 25 — Team frequency analysis (Elo, analysis only)")
    team_counts = pd.concat([df["radiant_team_id"], df["dire_team_id"]]).value_counts()
    print(f"Команд: {len(team_counts)}, матчей на команду: min={team_counts.min()} median={team_counts.median():.0f} max={team_counts.max()}")
    freq_threshold = team_counts.quantile(0.75)
    frequent_teams = set(team_counts[team_counts >= freq_threshold].index)
    test_both_freq = test_f[test_f["radiant_team_id"].isin(frequent_teams) & test_f["dire_team_id"].isin(frequent_teams)]
    test_has_infreq = test_f[~(test_f["radiant_team_id"].isin(frequent_teams) & test_f["dire_team_id"].isin(frequent_teams))]
    for label, d in [("both_frequent", test_both_freq), ("has_infrequent", test_has_infreq)]:
        if len(d) < 10:
            print(f"{label}: n={len(d)} — недостаточно")
            continue
        p_d = EloOnlyModel().predict_proba(d)[:, 1]
        m_d = compute_metrics(d["target"], p_d)
        print(f"{label}: n={m_d['n']:5d} acc={m_d['accuracy']:.4f} log_loss={m_d['log_loss']:.4f}")

    section("Визуализации")
    save_figures(df, test_f, p_elo, y_test, walk_forward_rows, temporal_rows, patch_stability_rows, bucket_rows, mean_pred, frac_pos)

    section("Реестр экспериментов")
    entry = {
        "experiment_id": f"phase6_5_{run_ts}",
        "git_commit": commit_sha,
        "timestamp": run_ts,
        "dataset_n_rows_total": int(n),
        "dataset_n_rows_no_cold_start": {"train": len(train_f), "val": len(val_f), "test": len(test_f)},
        "chosen_elo_k": best_k,
        "elo_k_validation_results": {str(k): v for k, v in k_results.items()},
        "best_form_window": best_form_window,
        "ablation": ablation_results,
        "elo_only_test": metrics_elo,
        "statistical_tests": {
            "elo_accuracy_ci": acc_ci,
            "elo_log_loss_ci": ll_ci,
            "elo_brier_ci": brier_ci,
            "effect_vs_random": effect_vs_random,
            "elo_vs_full_log_loss_diff": diff_elo_vs_full,
            "mcnemar_elo_vs_full": mcnemar_result,
        },
        "confidence_buckets": bucket_rows,
        "temporal_stability_by_year": temporal_rows,
        "patch_stability": patch_stability_rows,
        "walk_forward_by_year": walk_forward_rows,
    }
    path = os.path.join(EXPERIMENTS_DIR, "phase6_5_continuous.json")
    with open(path, "w") as f:
        json.dump(entry, f, indent=2, default=str)
    print(f"Записано: {path}")

    return 0


def save_figures(df, test_f, p_elo, y_test, walk_forward_rows, temporal_rows, patch_stability_rows, bucket_rows, mean_pred, frac_pos):
    # Elo distribution уже есть из Phase 6 (не пересоздаём) — здесь: walk-forward by year, calibration, confidence buckets, temporal stability
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", label="Elo (raw formula)")
    ax.set_xlabel("mean predicted P(radiant_win)")
    ax.set_ylabel("fraction of positives")
    ax.set_title("Calibration curve — Elo only, TEST (Phase 6.5)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase6_5_elo_calibration.png"), dpi=110)
    plt.close(fig)

    years = [r["year"] for r in temporal_rows]
    accs = [r["accuracy"] for r in temporal_rows]
    lls = [r["log_loss"] for r in temporal_rows]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(years, accs, marker="o")
    axes[0].axhline(0.5, color="gray", linestyle="--")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Elo temporal stability по годам (Phase 6.5)")
    axes[1].plot(years, lls, marker="o", color="tab:orange")
    axes[1].set_ylabel("log loss")
    axes[1].set_xlabel("год")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase6_5_temporal_stability.png"), dpi=110)
    plt.close(fig)

    if patch_stability_rows:
        patches_ids = [r["patch_id"] for r in patch_stability_rows]
        p_accs = [r["accuracy"] for r in patch_stability_rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([str(p) for p in patches_ids], p_accs)
        ax.axhline(0.5, color="gray", linestyle="--")
        ax.set_title("Elo accuracy по patch_id (Phase 6.5)")
        ax.set_xlabel("patch_id")
        ax.set_ylabel("accuracy")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "phase6_5_patch_stability.png"), dpi=110)
        plt.close(fig)

    if bucket_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [b["range"] for b in bucket_rows]
        predicted = [b["mean_predicted"] for b in bucket_rows]
        actual = [b["actual_favored_winrate"] for b in bucket_rows]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, predicted, width=0.4, label="mean predicted")
        ax.bar(x + 0.2, actual, width=0.4, label="actual favorite winrate")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title("Confidence buckets — Elo, TEST (Phase 6.5)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "phase6_5_confidence_buckets.png"), dpi=110)
        plt.close(fig)

    print(f"Сохранены графики Phase 6.5 в {FIGURES_DIR}")


if __name__ == "__main__":
    raise SystemExit(main())
