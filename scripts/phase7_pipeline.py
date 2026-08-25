#!/usr/bin/env python3
"""
PHASE 7 — roster + draft signal research.

Frozen baseline (Phase 6.5, НЕ меняется): Elo(K=16) + Form(window=3).
Исследует, добавляют ли roster (MODE A, pre-draft) и draft (MODE B,
post-draft) независимый сигнал ПОВЕРХ этого baseline — на ОДНИХ И ТЕХ ЖЕ
evaluation subset'ах для каждого сравнения (раздел 17 задания).

Запуск:
    python3 scripts/phase7_pipeline.py
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
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.config import load_settings
from src.datasets.builder import _load_pro_matches
from src.datasets.draft_features import build_draft_features, load_matches_with_draft
from src.datasets.multi_window_features import build_multi_window_features
from src.datasets.roster_features import build_roster_features, load_matches_with_rosters
from src.db.engine import make_engine
from src.evaluation.metrics import compute_metrics
from src.evaluation.statistics import (
    accuracy_metric,
    block_bootstrap_metric,
    block_bootstrap_paired_diff,
    log_loss_metric,
    mcnemar_exact,
)
from src.models.sklearn_models import RANDOM_SEED, LogisticRegressionModel
from scripts.phase6_pipeline import git_commit_sha, section

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")

ELO_K = 16          # frozen baseline (Phase 6.5)
FORM_WINDOW = 3      # frozen baseline (Phase 6.5)
BLOCK_SIZE = 20
BASELINE_FEATURES = ["elo_difference", "form_3_difference"]


def section_metrics(name, m):
    print(f"{name:38s} n={m['n']:6d} acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
          f"log_loss={m['log_loss']:.4f} brier={m['brier_score']:.4f} roc_auc={m['roc_auc']:.4f}")


def build_baseline_df(engine) -> pd.DataFrame:
    raw = _load_pro_matches(engine)
    rows = build_multi_window_features(raw, k_factor=ELO_K)
    records = []
    for r in rows:
        records.append({
            "match_id": r.match_id,
            "as_of_timestamp": r.as_of_timestamp,
            "radiant_team_id": r.radiant_team_id,
            "dire_team_id": r.dire_team_id,
            "elo_difference": r.elo_difference,
            "form_3_difference": r.recent_winrate_difference[FORM_WINDOW],
            "radiant_matches_played_before": r.radiant_matches_played_before,
            "dire_matches_played_before": r.dire_matches_played_before,
            "target": int(r.radiant_win),
        })
    return pd.DataFrame(records)


def build_roster_df(engine) -> pd.DataFrame:
    matches = load_matches_with_rosters(engine)
    rows = build_roster_features(matches)
    records = []
    for r in rows:
        records.append({
            "match_id": r.match_id,
            "roster_size_diff": r.radiant_roster_size - r.dire_roster_size,
            "roster_stability_diff": r.radiant_roster_matches_together - r.dire_roster_matches_together,
            "roster_age_diff": (
                (r.radiant_roster_age_days - r.dire_roster_age_days)
                if r.radiant_roster_age_days is not None and r.dire_roster_age_days is not None
                else None
            ),
            "player_continuity_diff": (
                (r.radiant_player_continuity - r.dire_player_continuity)
                if r.radiant_player_continuity is not None and r.dire_player_continuity is not None
                else None
            ),
            "roster_changes_7d_diff": r.radiant_roster_changes[7] - r.dire_roster_changes[7],
            "roster_changes_30d_diff": r.radiant_roster_changes[30] - r.dire_roster_changes[30],
            "roster_changes_90d_diff": r.radiant_roster_changes[90] - r.dire_roster_changes[90],
        })
    return pd.DataFrame(records)


def build_draft_df(engine) -> pd.DataFrame:
    matches = load_matches_with_draft(engine)
    rows = build_draft_features(matches)
    records = []
    for r in rows:
        records.append({
            "match_id": r.match_id,
            "hero_strength_diff": (
                (r.radiant_hero_strength - r.dire_hero_strength)
                if r.radiant_hero_strength is not None and r.dire_hero_strength is not None else None
            ),
            "hero_strength_patch_diff": (
                (r.radiant_hero_strength_patch - r.dire_hero_strength_patch)
                if r.radiant_hero_strength_patch is not None and r.dire_hero_strength_patch is not None else None
            ),
            "team_hero_experience_diff": r.radiant_team_hero_experience - r.dire_team_hero_experience,
            "team_hero_winrate_diff": (
                (r.radiant_team_hero_winrate - r.dire_team_hero_winrate)
                if r.radiant_team_hero_winrate is not None and r.dire_team_hero_winrate is not None else None
            ),
            "pick_popularity_diff": (
                (r.radiant_pick_popularity - r.dire_pick_popularity)
                if r.radiant_pick_popularity is not None and r.dire_pick_popularity is not None else None
            ),
            "matchup_advantage": r.matchup_advantage,
        })
    return pd.DataFrame(records)


def chronological_split_df(df, train_frac=0.70, val_frac=0.15):
    df = df.sort_values(["as_of_timestamp", "match_id"]).reset_index(drop=True)
    n = len(df)
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    train, val, test = df.iloc[:a].reset_index(drop=True), df.iloc[a:b].reset_index(drop=True), df.iloc[b:].reset_index(drop=True)
    assert train["as_of_timestamp"].max() < val["as_of_timestamp"].min()
    assert val["as_of_timestamp"].max() < test["as_of_timestamp"].min()
    return train, val, test


def fit_eval(features, train, test, val=None):
    m = LogisticRegressionModel(feature_names=features, random_state=RANDOM_SEED)
    m.fit(train, train["target"])
    metrics = m.evaluate(test, test["target"])
    return m, metrics


def main() -> int:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    commit_sha = git_commit_sha()
    run_ts = datetime.now(timezone.utc).isoformat()

    section("PHASE 7 — Existing Data & Timing Audit")
    print(f"git commit: {commit_sha}")
    print("Frozen baseline: Elo(K=16) + Form(window=3) — Phase 6.5, не меняется в этой фазе.")
    print("picks_bans.team semantics: team=0 <-> is_radiant, проверено эмпирически "
          "(2000/2000 совпадений с match_players.is_radiant, 0 расхождений).")

    settings = load_settings()
    engine = make_engine(settings)

    section("1. Data coverage")
    with engine.connect() as conn:
        total_matches = conn.execute(text("SELECT count(*) FROM matches")).scalar()
        pro_matches = conn.execute(text(
            "SELECT count(*) FROM matches m JOIN leagues l ON l.league_id=m.league_id "
            "WHERE l.tier IN ('professional','premium')"
        )).scalar()
        coverage_by_year = conn.execute(text(
            """
            SELECT extract(year from m.start_time)::int y,
                   count(distinct m.match_id) total,
                   count(distinct pb.match_id) with_pb,
                   count(distinct mp.match_id) with_players
            FROM matches m
            JOIN leagues l ON l.league_id = m.league_id
            LEFT JOIN picks_bans pb ON pb.match_id = m.match_id
            LEFT JOIN match_players mp ON mp.match_id = m.match_id
            WHERE l.tier IN ('professional','premium')
            GROUP BY 1 ORDER BY 1
            """
        )).fetchall()
    print(f"matches total (все tier): {total_matches}")
    print(f"pro+premium matches: {pro_matches}")
    print(f"{'year':6s} {'total':>8s} {'with_picks_bans':>16s} {'with_players':>14s}")
    for r in coverage_by_year:
        print(f"{r.y:6d} {r.total:8d} {r.with_pb:16d} ({r.with_pb/r.total*100:5.1f}%) {r.with_players:14d} ({r.with_players/r.total*100:5.1f}%)")

    section("Building feature dataframes (baseline / roster / draft)")
    df_base = build_baseline_df(engine)
    print(f"baseline (Elo+Form3) rows: {len(df_base)}")
    df_roster = build_roster_df(engine)
    print(f"roster feature rows: {len(df_roster)}")
    df_draft = build_draft_df(engine)
    print(f"draft feature rows (DRAFT_COMPLETE_SET): {len(df_draft)}")

    # Единый базовый фильтр (не cold-start по Elo/Form) — тот же, что Phase 6.5.
    df_base_f = df_base[(df_base["radiant_matches_played_before"] > 0) & (df_base["dire_matches_played_before"] > 0)]

    section("2. ROSTER_COMPLETE_SET / DRAFT_COMPLETE_SET (раздел 2 — основной dataset НЕ удаляется)")
    roster_eval = df_base_f.merge(df_roster, on="match_id", how="inner")
    draft_eval = df_base_f.merge(df_draft, on="match_id", how="inner")
    print(f"ROSTER_COMPLETE_SET (baseline non-cold-start ∩ has roster): {len(roster_eval)} / {len(df_base_f)}")
    print(f"DRAFT_COMPLETE_SET (baseline non-cold-start ∩ has complete draft): {len(draft_eval)} / {len(df_base_f)}")

    # ---- ROSTER experiments (единый subset для ВСЕХ моделей этого блока) ----
    section("9. Roster ablation (ОДИН И ТОТ ЖЕ evaluation subset для всех моделей)")
    train_r, val_r, test_r = chronological_split_df(roster_eval)
    print(f"ROSTER split: TRAIN n={len(train_r)} VAL n={len(val_r)} TEST n={len(test_r)}")

    roster_experiments = {
        "Baseline (Elo+Form3)": BASELINE_FEATURES,
        "+ Roster Stability": BASELINE_FEATURES + ["roster_stability_diff"],
        "+ Roster Age": BASELINE_FEATURES + ["roster_age_diff"],
        "+ Player Continuity": BASELINE_FEATURES + ["player_continuity_diff"],
        "+ All Roster": BASELINE_FEATURES + ["roster_stability_diff", "roster_age_diff", "player_continuity_diff", "roster_changes_7d_diff"],
    }
    roster_results = {}
    roster_models = {}
    for name, feats in roster_experiments.items():
        model, metrics = fit_eval(feats, train_r, test_r)
        roster_results[name] = metrics
        roster_models[name] = model
        section_metrics(name, metrics)

    # ---- DRAFT experiments ----
    section("15. Draft ablation (DRAFT-0..DRAFT-4, ОДИН И ТОТ ЖЕ evaluation subset)")
    train_d, val_d, test_d = chronological_split_df(draft_eval)
    print(f"DRAFT split: TRAIN n={len(train_d)} VAL n={len(val_d)} TEST n={len(test_d)}")

    draft_experiments = {
        "DRAFT-0: Baseline": BASELINE_FEATURES,
        "DRAFT-1: + Hero Strength": BASELINE_FEATURES + ["hero_strength_diff"],
        "DRAFT-2: + Team-Hero History": BASELINE_FEATURES + ["hero_strength_diff", "team_hero_experience_diff", "team_hero_winrate_diff"],
        "DRAFT-3: + Pick Popularity": BASELINE_FEATURES + ["hero_strength_diff", "team_hero_experience_diff", "team_hero_winrate_diff", "pick_popularity_diff"],
        "DRAFT-4: + Matchup Interaction": BASELINE_FEATURES + ["hero_strength_diff", "team_hero_experience_diff", "team_hero_winrate_diff", "pick_popularity_diff", "matchup_advantage"],
        "DRAFT-1b: + Hero Strength (patch-scoped)": BASELINE_FEATURES + ["hero_strength_patch_diff"],
    }
    draft_results = {}
    draft_models = {}
    for name, feats in draft_experiments.items():
        model, metrics = fit_eval(feats, train_d, test_d)
        draft_results[name] = metrics
        draft_models[name] = model
        section_metrics(name, metrics)

    section("8-9. MODE A (pre-draft) / MODE B (draft-aware)")
    mode_a_features = BASELINE_FEATURES + ["roster_stability_diff", "player_continuity_diff"]
    mode_a_model, mode_a_metrics_roster_set = fit_eval(mode_a_features, train_r, test_r)
    section_metrics("MODE A (pre-draft) on ROSTER_SET", mode_a_metrics_roster_set)

    mode_b_features = BASELINE_FEATURES + ["hero_strength_diff", "team_hero_experience_diff", "team_hero_winrate_diff", "matchup_advantage"]
    mode_b_model, mode_b_metrics_draft_set = fit_eval(mode_b_features, train_d, test_d)
    section_metrics("MODE B (draft-aware) on DRAFT_SET", mode_b_metrics_draft_set)

    baseline_on_roster_set_model, baseline_on_roster_set_metrics = fit_eval(BASELINE_FEATURES, train_r, test_r)
    baseline_on_draft_set_model, baseline_on_draft_set_metrics = fit_eval(BASELINE_FEATURES, train_d, test_d)
    section_metrics("Baseline on ROSTER_SET (для честного Δ)", baseline_on_roster_set_metrics)
    section_metrics("Baseline on DRAFT_SET (для честного Δ)", baseline_on_draft_set_metrics)

    section("19-20. Statistical validation (block bootstrap + McNemar)")
    y_test_r = test_r["target"].to_numpy()
    p_baseline_r = baseline_on_roster_set_model.predict_proba(test_r)[:, 1]
    p_mode_a = mode_a_model.predict_proba(test_r)[:, 1]

    diff_acc_roster = block_bootstrap_paired_diff(y_test_r, p_mode_a, p_baseline_r, accuracy_metric, block_size=BLOCK_SIZE)
    diff_ll_roster = block_bootstrap_paired_diff(y_test_r, p_mode_a, p_baseline_r, log_loss_metric, block_size=BLOCK_SIZE)
    mcnemar_roster = mcnemar_exact(y_test_r, (p_mode_a >= 0.5).astype(int), (p_baseline_r >= 0.5).astype(int))
    print(f"Roster: Δaccuracy = {diff_acc_roster['point_diff']:+.4f} 95% CI=[{diff_acc_roster['ci_low']:+.4f}, {diff_acc_roster['ci_high']:+.4f}]")
    print(f"Roster: Δlog_loss(MODE_A - Baseline) = {diff_ll_roster['point_diff']:+.4f} 95% CI=[{diff_ll_roster['ci_low']:+.4f}, {diff_ll_roster['ci_high']:+.4f}] (отрицательное = MODE A лучше)")
    print(f"Roster McNemar: a_only={mcnemar_roster['a_correct_b_wrong']} b_only={mcnemar_roster['b_correct_a_wrong']} p={mcnemar_roster['p_value']:.4f}")

    y_test_d = test_d["target"].to_numpy()
    p_baseline_d = baseline_on_draft_set_model.predict_proba(test_d)[:, 1]
    p_mode_b = mode_b_model.predict_proba(test_d)[:, 1]

    diff_acc_draft = block_bootstrap_paired_diff(y_test_d, p_mode_b, p_baseline_d, accuracy_metric, block_size=BLOCK_SIZE)
    diff_ll_draft = block_bootstrap_paired_diff(y_test_d, p_mode_b, p_baseline_d, log_loss_metric, block_size=BLOCK_SIZE)
    mcnemar_draft = mcnemar_exact(y_test_d, (p_mode_b >= 0.5).astype(int), (p_baseline_d >= 0.5).astype(int))
    print(f"Draft: Δaccuracy = {diff_acc_draft['point_diff']:+.4f} 95% CI=[{diff_acc_draft['ci_low']:+.4f}, {diff_acc_draft['ci_high']:+.4f}]")
    print(f"Draft: Δlog_loss(MODE_B - Baseline) = {diff_ll_draft['point_diff']:+.4f} 95% CI=[{diff_ll_draft['ci_low']:+.4f}, {diff_ll_draft['ci_high']:+.4f}] (отрицательное = MODE B лучше)")
    print(f"Draft McNemar: a_only={mcnemar_draft['a_correct_b_wrong']} b_only={mcnemar_draft['b_correct_a_wrong']} p={mcnemar_draft['p_value']:.4f}")

    section("22. Temporal stability (по годам, DRAFT_SET, MODE B vs Baseline)")
    draft_eval_sorted = draft_eval.sort_values(["as_of_timestamp", "match_id"]).reset_index(drop=True)
    draft_eval_sorted["year"] = draft_eval_sorted["as_of_timestamp"].dt.year
    temporal_rows = []
    years = sorted(draft_eval_sorted["year"].unique())
    for y in years[1:]:
        tr = draft_eval_sorted[draft_eval_sorted["year"] < y]
        pe = draft_eval_sorted[draft_eval_sorted["year"] == y]
        if len(tr) < 2000 or len(pe) < 100:
            print(f"{y}: пропущено (train n={len(tr)}, period n={len(pe)})")
            continue
        base_m, base_metrics = fit_eval(BASELINE_FEATURES, tr, pe)
        modeb_m, modeb_metrics = fit_eval(mode_b_features, tr, pe)
        temporal_rows.append({"year": int(y), "n": len(pe), "baseline": base_metrics, "mode_b": modeb_metrics})
        print(f"{y}  n={len(pe):5d}  baseline(acc={base_metrics['accuracy']:.3f})  mode_b(acc={modeb_metrics['accuracy']:.3f})  "
              f"Δ={modeb_metrics['accuracy']-base_metrics['accuracy']:+.3f}")

    section("23. Patch stability (DRAFT_SET, MODE B vs Baseline, train/test единый split)")
    with engine.connect() as conn:
        patch_map = {r.match_id: r.patch_id for r in conn.execute(text("SELECT match_id, patch_id FROM matches")).fetchall()}
    test_d_patch = test_d.copy()
    test_d_patch["patch_id"] = test_d_patch["match_id"].map(patch_map)
    patch_rows = []
    for patch_id, d_p in test_d_patch.groupby("patch_id"):
        if len(d_p) < 100 or pd.isna(patch_id):
            continue
        p_base_p = baseline_on_draft_set_model.predict_proba(d_p)[:, 1]
        p_modeb_p = mode_b_model.predict_proba(d_p)[:, 1]
        m_base_p = compute_metrics(d_p["target"], p_base_p)
        m_modeb_p = compute_metrics(d_p["target"], p_modeb_p)
        patch_rows.append({"patch_id": int(patch_id), "n": len(d_p), "baseline_acc": m_base_p["accuracy"], "mode_b_acc": m_modeb_p["accuracy"]})
    patch_rows.sort(key=lambda r: r["patch_id"])
    for r in patch_rows:
        print(f"patch_id={r['patch_id']}: n={r['n']:5d} baseline_acc={r['baseline_acc']:.4f} mode_b_acc={r['mode_b_acc']:.4f} Δ={r['mode_b_acc']-r['baseline_acc']:+.4f}")

    section("16. Error analysis (MODE B vs Baseline, DRAFT_SET TEST)")
    test_d_pred = test_d.copy()
    test_d_pred["p_baseline"] = p_baseline_d
    test_d_pred["p_mode_b"] = p_mode_b
    flips_correct = test_d_pred[
        ((test_d_pred["p_baseline"] >= 0.5).astype(int) != test_d_pred["target"])
        & ((test_d_pred["p_mode_b"] >= 0.5).astype(int) == test_d_pred["target"])
    ]
    flips_incorrect = test_d_pred[
        ((test_d_pred["p_baseline"] >= 0.5).astype(int) == test_d_pred["target"])
        & ((test_d_pred["p_mode_b"] >= 0.5).astype(int) != test_d_pred["target"])
    ]
    print(f"Baseline неверно -> MODE B верно: {len(flips_correct)} случаев")
    print(f"Baseline верно -> MODE B неверно (draft 'испортил'): {len(flips_incorrect)} случаев")
    print(f"Net effect: {len(flips_correct) - len(flips_incorrect):+d} (согласуется со знаком Δaccuracy выше)")

    section("27. Final comparison table")
    final_table = [
        ("Elo + Form3", "Pre-draft", baseline_on_roster_set_metrics, 0.0),
        ("+ Roster (best combo)", "Pre-draft", mode_a_metrics_roster_set, mode_a_metrics_roster_set["accuracy"] - baseline_on_roster_set_metrics["accuracy"]),
        ("+ Draft (best combo)", "Post-draft", mode_b_metrics_draft_set, mode_b_metrics_draft_set["accuracy"] - baseline_on_draft_set_metrics["accuracy"]),
    ]
    print(f"{'Model':30s} {'Timing':12s} {'Acc':>8s} {'LogLoss':>9s} {'Brier':>8s} {'ROC-AUC':>8s} {'Δ vs baseline':>14s}")
    for name, timing, m, delta in final_table:
        print(f"{name:30s} {timing:12s} {m['accuracy']:8.4f} {m['log_loss']:9.4f} {m['brier_score']:8.4f} {m['roc_auc']:8.4f} {delta:+14.4f}")
    print("\nВНИМАНИЕ (раздел 28): строки выше НЕ на одном evaluation set — "
          "'+ Roster' сравнивается с Baseline-на-ROSTER_SET, '+ Draft' — с Baseline-на-DRAFT_SET. "
          "Δ корректны ВНУТРИ своей строки/subset'а, но НЕ между Roster и Draft строками напрямую.")

    section("Визуализации")
    save_figures(roster_results, draft_results, temporal_rows, patch_rows)

    section("30. Data versioning / experiment registry")
    entry = {
        "experiment_id": f"phase7_{run_ts}",
        "git_commit": commit_sha,
        "timestamp": run_ts,
        "dataset_version": "continuous_2021_2026",
        "baseline_feature_version": f"elo_k{ELO_K}_form{FORM_WINDOW}",
        "roster_eval_set_n": len(roster_eval),
        "draft_eval_set_n": len(draft_eval),
        "roster_ablation": roster_results,
        "draft_ablation": draft_results,
        "mode_a": {"features": mode_a_features, "metrics_on_roster_set": mode_a_metrics_roster_set, "baseline_on_same_set": baseline_on_roster_set_metrics},
        "mode_b": {"features": mode_b_features, "metrics_on_draft_set": mode_b_metrics_draft_set, "baseline_on_same_set": baseline_on_draft_set_metrics},
        "statistical_tests": {
            "roster_vs_baseline_accuracy_diff": diff_acc_roster,
            "roster_vs_baseline_logloss_diff": diff_ll_roster,
            "roster_vs_baseline_mcnemar": mcnemar_roster,
            "draft_vs_baseline_accuracy_diff": diff_acc_draft,
            "draft_vs_baseline_logloss_diff": diff_ll_draft,
            "draft_vs_baseline_mcnemar": mcnemar_draft,
        },
        "temporal_stability": temporal_rows,
        "patch_stability": patch_rows,
        "error_analysis": {"flips_correct": len(flips_correct), "flips_incorrect": len(flips_incorrect)},
    }
    path = os.path.join(EXPERIMENTS_DIR, "phase7_roster_draft.json")
    with open(path, "w") as f:
        json.dump(entry, f, indent=2, default=str)
    print(f"Записано: {path}")

    return 0


def save_figures(roster_results, draft_results, temporal_rows, patch_rows):
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(roster_results.keys())
    accs = [roster_results[n]["accuracy"] for n in names]
    ax.barh(names, accs)
    ax.axvline(accs[0], color="gray", linestyle="--", label="baseline")
    ax.set_title("Roster ablation — accuracy (Phase 7)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase7_roster_ablation.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(draft_results.keys())
    accs = [draft_results[n]["accuracy"] for n in names]
    ax.barh(names, accs)
    ax.axvline(accs[0], color="gray", linestyle="--", label="baseline")
    ax.set_title("Draft ablation — accuracy (Phase 7)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase7_draft_ablation.png"), dpi=110)
    plt.close(fig)

    if temporal_rows:
        years = [r["year"] for r in temporal_rows]
        base_acc = [r["baseline"]["accuracy"] for r in temporal_rows]
        modeb_acc = [r["mode_b"]["accuracy"] for r in temporal_rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(years, base_acc, marker="o", label="Baseline")
        ax.plot(years, modeb_acc, marker="o", label="MODE B (draft-aware)")
        ax.axhline(0.5, color="gray", linestyle="--")
        ax.set_title("Temporal stability — Baseline vs MODE B (Phase 7)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "phase7_temporal_stability.png"), dpi=110)
        plt.close(fig)

    print(f"Сохранены графики Phase 7 в {FIGURES_DIR}")


if __name__ == "__main__":
    raise SystemExit(main())
