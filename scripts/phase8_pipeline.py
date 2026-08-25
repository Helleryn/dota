#!/usr/bin/env python3
"""
PHASE 8 — evaluation of redesigned roster/meta features.

Frozen baseline (Phase 6.5, НЕ меняется): Elo(K=16) + Form(window=3).

Ключевое отличие от Phase 7 по протоколу: ВСЕ модели (baseline, MODE A
pre-draft, MODE B post-draft) оцениваются на ОДНОМ И ТОМ ЖЕ
COMMON_DRAFT_SET (раздел 54 задания) — в Phase 7 roster и draft
сравнивались на разных subset'ах, из-за чего их Δ нельзя было сравнивать
между собой. Здесь можно.

Запуск:
    python3 scripts/phase8_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import FrozenSet, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.config import load_settings
from src.datasets.builder import _load_pro_matches
from src.datasets.meta_features import build_meta_features
from src.datasets.multi_window_features import build_multi_window_features
from src.datasets.player_features import build_player_features
from src.db.engine import make_engine
from src.evaluation.metrics import compute_metrics
from src.evaluation.statistics import (
    accuracy_metric,
    block_bootstrap_paired_diff,
    log_loss_metric,
    mcnemar_exact,
)
from src.models.sklearn_models import RANDOM_SEED, CatBoostModel, LogisticRegressionModel
from scripts.phase6_pipeline import git_commit_sha, section

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")

ELO_K = 16
FORM_WINDOW = 3
BLOCK_SIZE = 20
BASELINE = ["elo_difference", "form_3_difference"]


@dataclass(frozen=True)
class FullMatch:
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_roster: FrozenSet[int]
    dire_roster: FrozenSet[int]
    radiant_picks: Tuple[int, ...]
    dire_picks: Tuple[int, ...]
    radiant_win: bool


def load_full_matches(engine):
    """Матчи с ПОЛНЫМ составом (10 игроков) И полным драфтом (5+5 пиков).
    Основной continuous dataset не изменяется — это отдельная выборка."""
    sql = text(
        """
        WITH rosters AS (
          SELECT mp.match_id,
                 array_agg(mp.account_id) FILTER (WHERE mp.is_radiant AND mp.account_id IS NOT NULL) AS r_roster,
                 array_agg(mp.account_id) FILTER (WHERE NOT mp.is_radiant AND mp.account_id IS NOT NULL) AS d_roster
          FROM match_players mp GROUP BY 1
        ), drafts AS (
          SELECT pb.match_id,
                 array_agg(pb.hero_id ORDER BY pb.ord) FILTER (WHERE pb.is_pick AND pb.team = 0) AS r_picks,
                 array_agg(pb.hero_id ORDER BY pb.ord) FILTER (WHERE pb.is_pick AND pb.team = 1) AS d_picks
          FROM picks_bans pb GROUP BY 1
        )
        SELECT m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id, m.radiant_win,
               r.r_roster, r.d_roster, d.r_picks, d.d_picks
        FROM matches m
        JOIN leagues l ON l.league_id = m.league_id
        JOIN rosters r ON r.match_id = m.match_id
        JOIN drafts  d ON d.match_id = m.match_id
        WHERE l.tier IN ('professional','premium')
          AND m.radiant_team_id IS NOT NULL AND m.dire_team_id IS NOT NULL
          AND m.radiant_win IS NOT NULL
          AND array_length(r.r_roster,1) = 5 AND array_length(r.d_roster,1) = 5
          AND array_length(d.r_picks,1) = 5 AND array_length(d.d_picks,1) = 5
        ORDER BY m.start_time ASC, m.match_id ASC
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    out = []
    for r in rows:
        st = r.start_time if r.start_time.tzinfo else r.start_time.replace(tzinfo=timezone.utc)
        out.append(FullMatch(
            match_id=r.match_id, start_time=st,
            radiant_team_id=r.radiant_team_id, dire_team_id=r.dire_team_id,
            radiant_roster=frozenset(r.r_roster), dire_roster=frozenset(r.d_roster),
            radiant_picks=tuple(r.r_picks), dire_picks=tuple(r.d_picks),
            radiant_win=r.radiant_win,
        ))
    return out


def build_common_dataframe(engine):
    full = load_full_matches(engine)
    full_ids = {m.match_id for m in full}
    print(f"COMMON_DRAFT_SET (полный ростер + полный драфт): {len(full)}")

    # --- Baseline (frozen), считается на ВСЕЙ pro-истории, не только на common set,
    # чтобы Elo/Form имели полную историю; отбирается по common set в конце.
    raw = _load_pro_matches(engine)
    base_rows = build_multi_window_features(raw, k_factor=ELO_K)
    base = pd.DataFrame([{
        "match_id": r.match_id,
        "as_of_timestamp": r.as_of_timestamp,
        "radiant_team_id": r.radiant_team_id,
        "dire_team_id": r.dire_team_id,
        "elo_difference": r.elo_difference,
        "form_3_difference": r.recent_winrate_difference[FORM_WINDOW],
        "radiant_matches_played_before": r.radiant_matches_played_before,
        "dire_matches_played_before": r.dire_matches_played_before,
        "target": int(r.radiant_win),
    } for r in base_rows])

    # --- Player/roster features (Phase 8) ---
    p_rows = build_player_features(full, k_factor=ELO_K)
    player = pd.DataFrame([{
        "match_id": r.match_id,
        "player_elo_mean_diff": r.radiant_player_elo_mean - r.dire_player_elo_mean,
        "player_elo_min_diff": r.radiant_player_elo_min - r.dire_player_elo_min,
        "player_elo_max_diff": r.radiant_player_elo_max - r.dire_player_elo_max,
        "roster_vs_team_delta_diff": (r.radiant_roster_vs_team_delta or 0.0) - (r.dire_roster_vs_team_delta or 0.0),
        "pair_synergy_diff": (r.radiant_pair_synergy or 0.0) - (r.dire_pair_synergy or 0.0),
        "player_matches_min_diff": r.radiant_player_matches_min - r.dire_player_matches_min,
        "player_matches_mean_diff": r.radiant_player_matches_mean - r.dire_player_matches_mean,
        "player_matches_min_lower": min(r.radiant_player_matches_min, r.dire_player_matches_min),
    } for r in p_rows])

    # --- Meta/draft features (Phase 8) ---
    m_rows = build_meta_features(full)
    meta = pd.DataFrame([{
        "match_id": r.match_id,
        "hero_strength_decayed_diff": r.radiant_hero_strength_decayed - r.dire_hero_strength_decayed,
        "hero_synergy_diff": r.radiant_hero_synergy - r.dire_hero_synergy,
        "counter_advantage": r.counter_advantage,
        "player_hero_prof_diff": r.radiant_player_hero_proficiency - r.dire_player_hero_proficiency,
    } for r in m_rows])

    df = base[base["match_id"].isin(full_ids)].merge(player, on="match_id").merge(meta, on="match_id")
    df = df[(df["radiant_matches_played_before"] > 0) & (df["dire_matches_played_before"] > 0)]
    df = df.sort_values(["as_of_timestamp", "match_id"]).reset_index(drop=True)
    print(f"После исключения team cold-start: {len(df)}")
    return df


def split(df, train_frac=0.70, val_frac=0.15):
    n = len(df)
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    tr, va, te = df.iloc[:a].reset_index(drop=True), df.iloc[a:b].reset_index(drop=True), df.iloc[b:].reset_index(drop=True)
    assert tr["as_of_timestamp"].max() < va["as_of_timestamp"].min()
    assert va["as_of_timestamp"].max() < te["as_of_timestamp"].min()
    return tr, va, te


def fit_eval(features, train, test):
    m = LogisticRegressionModel(feature_names=features, random_state=RANDOM_SEED)
    m.fit(train, train["target"])
    return m, m.evaluate(test, test["target"])


def show(name, m, base_acc=None):
    delta = f"  Δacc={m['accuracy']-base_acc:+.4f}" if base_acc is not None else ""
    print(f"{name:44s} acc={m['accuracy']:.4f} ll={m['log_loss']:.4f} "
          f"brier={m['brier_score']:.4f} auc={m['roc_auc']:.4f}{delta}")


def main() -> int:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    commit = git_commit_sha()
    run_ts = datetime.now(timezone.utc).isoformat()

    section("PHASE 8 — Deep Feature Redesign: evaluation")
    print(f"git commit: {commit}")
    print("Frozen baseline: Elo(K=16) + Form(3). Player-Elo использует ТОТ ЖЕ K=16 (не подбирался).")

    engine = make_engine(load_settings())
    df = build_common_dataframe(engine)
    train, val, test = split(df)
    print(f"TRAIN n={len(train)} ({train['as_of_timestamp'].min().date()}..{train['as_of_timestamp'].max().date()})")
    print(f"VAL   n={len(val)} ({val['as_of_timestamp'].min().date()}..{val['as_of_timestamp'].max().date()})")
    print(f"TEST  n={len(test)} ({test['as_of_timestamp'].min().date()}..{test['as_of_timestamp'].max().date()})")

    results = {}
    models = {}

    section("MODEL A — PRE-DRAFT (Elo + Form + roster/player strength)")
    base_model, base_metrics = fit_eval(BASELINE, train, test)
    results["Baseline (Elo+Form3)"] = base_metrics
    models["Baseline (Elo+Form3)"] = base_model
    show("Baseline (Elo+Form3)", base_metrics)
    b_acc = base_metrics["accuracy"]

    a_experiments = {
        "A1 +player_elo_mean": BASELINE + ["player_elo_mean_diff"],
        "A2 +player_elo mean/min/max": BASELINE + ["player_elo_mean_diff", "player_elo_min_diff", "player_elo_max_diff"],
        "A3 +roster_vs_team_delta": BASELINE + ["roster_vs_team_delta_diff"],
        "A4 +pair_synergy": BASELINE + ["pair_synergy_diff"],
        "A5 +experience (cold-start)": BASELINE + ["player_matches_mean_diff", "player_matches_min_lower"],
        "A6 ALL roster/player": BASELINE + ["player_elo_mean_diff", "player_elo_min_diff", "player_elo_max_diff",
                                            "roster_vs_team_delta_diff", "pair_synergy_diff",
                                            "player_matches_mean_diff", "player_matches_min_lower"],
    }
    for name, feats in a_experiments.items():
        mdl, met = fit_eval(feats, train, test)
        results[name] = met
        models[name] = mdl
        show(name, met, b_acc)

    section("MODEL B — POST-DRAFT (+ meta-aware draft)")
    best_a_name = max(a_experiments, key=lambda k: -results[k]["log_loss"])
    best_a_feats = a_experiments[best_a_name]
    print(f"Лучшая A-конфигурация по log_loss: {best_a_name}")

    b_experiments = {
        "B1 +hero_strength_decayed": BASELINE + ["hero_strength_decayed_diff"],
        "B2 +hero synergy/counter": BASELINE + ["hero_strength_decayed_diff", "hero_synergy_diff", "counter_advantage"],
        "B3 +player_hero_proficiency": BASELINE + ["hero_strength_decayed_diff", "player_hero_prof_diff"],
        "B4 ALL meta/draft": BASELINE + ["hero_strength_decayed_diff", "hero_synergy_diff",
                                          "counter_advantage", "player_hero_prof_diff"],
        "B5 ALL roster + ALL draft": best_a_feats + ["hero_strength_decayed_diff", "hero_synergy_diff",
                                                      "counter_advantage", "player_hero_prof_diff"],
    }
    for name, feats in b_experiments.items():
        mdl, met = fit_eval(feats, train, test)
        results[name] = met
        models[name] = mdl
        show(name, met, b_acc)

    section("CatBoost проверка (нелинейность может использовать признаки иначе)")
    catboost_results = {}
    for name, feats in [("Baseline", BASELINE), (best_a_name, best_a_feats),
                        ("B5 ALL roster + ALL draft", b_experiments["B5 ALL roster + ALL draft"])]:
        cb = CatBoostModel(feature_names=feats)
        cb.fit(train, train["target"])
        met = cb.evaluate(test, test["target"])
        catboost_results[name] = met
        show(f"CatBoost {name}", met, b_acc)

    section("Статистическая значимость (block bootstrap + McNemar), vs frozen baseline")
    y = test["target"].to_numpy()
    p_base = base_model.predict_proba(test)[:, 1]
    stats = {}
    for name in ["A1 +player_elo_mean", "A6 ALL roster/player", "B1 +hero_strength_decayed",
                 "B4 ALL meta/draft", "B5 ALL roster + ALL draft"]:
        p = models[name].predict_proba(test)[:, 1]
        d_acc = block_bootstrap_paired_diff(y, p, p_base, accuracy_metric, block_size=BLOCK_SIZE)
        d_ll = block_bootstrap_paired_diff(y, p, p_base, log_loss_metric, block_size=BLOCK_SIZE)
        mc = mcnemar_exact(y, (p >= 0.5).astype(int), (p_base >= 0.5).astype(int))
        stats[name] = {"acc_diff": d_acc, "log_loss_diff": d_ll, "mcnemar": mc}
        print(f"{name:32s} Δacc={d_acc['point_diff']:+.4f} CI=[{d_acc['ci_low']:+.4f},{d_acc['ci_high']:+.4f}]  "
              f"Δll={d_ll['point_diff']:+.4f} CI=[{d_ll['ci_low']:+.4f},{d_ll['ci_high']:+.4f}]  "
              f"McNemar p={mc['p_value']:.4f}")
    print("(Δll отрицательное = модель ЛУЧШЕ baseline)")

    section("Walk-forward по годам (устойчивость во времени)")
    df_y = df.copy()
    df_y["year"] = df_y["as_of_timestamp"].dt.year
    wf_rows = []
    years = sorted(df_y["year"].unique())
    for yv in years[1:]:
        tr = df_y[df_y["year"] < yv]
        pe = df_y[df_y["year"] == yv]
        if len(tr) < 3000 or len(pe) < 200:
            print(f"{yv}: пропущено (train={len(tr)}, period={len(pe)})")
            continue
        _, mb = fit_eval(BASELINE, tr, pe)
        _, ma = fit_eval(a_experiments["A6 ALL roster/player"], tr, pe)
        _, mbb = fit_eval(b_experiments["B5 ALL roster + ALL draft"], tr, pe)
        wf_rows.append({"year": int(yv), "n": len(pe), "baseline": mb, "modelA": ma, "modelB": mbb})
        print(f"{yv}  n={len(pe):6d}  base={mb['accuracy']:.4f}  A6={ma['accuracy']:.4f} ({ma['accuracy']-mb['accuracy']:+.4f})  "
              f"B5={mbb['accuracy']:.4f} ({mbb['accuracy']-mb['accuracy']:+.4f})")

    section("Cold-start анализ (раздел 24 задания)")
    cold_rows = []
    for label, mask in [
        ("debutant (min prior = 0)", test["player_matches_min_lower"] == 0),
        ("1-5 prior", (test["player_matches_min_lower"] >= 1) & (test["player_matches_min_lower"] <= 5)),
        ("5-20 prior", (test["player_matches_min_lower"] > 5) & (test["player_matches_min_lower"] <= 20)),
        ("20+ prior", test["player_matches_min_lower"] > 20),
    ]:
        sub = test[mask]
        if len(sub) < 50:
            print(f"{label:28s} n={len(sub)} — слишком мало")
            continue
        mb = compute_metrics(sub["target"], base_model.predict_proba(sub)[:, 1])
        ma = compute_metrics(sub["target"], models["A6 ALL roster/player"].predict_proba(sub)[:, 1])
        cold_rows.append({"bucket": label, "n": len(sub), "baseline_acc": mb["accuracy"], "modelA_acc": ma["accuracy"]})
        print(f"{label:28s} n={len(sub):6d}  base={mb['accuracy']:.4f}  A6={ma['accuracy']:.4f} ({ma['accuracy']-mb['accuracy']:+.4f})")

    section("Коэффициенты лучшей модели (направление эффекта)")
    print("A6:", {k: round(v, 5) for k, v in models["A6 ALL roster/player"].feature_importance().items()})
    print("B5:", {k: round(v, 5) for k, v in models["B5 ALL roster + ALL draft"].feature_importance().items()})

    section("Визуализация")
    save_figures(results, wf_rows, b_acc)

    section("Реестр экспериментов")
    entry = {
        "experiment_id": f"phase8_{run_ts}",
        "git_commit": commit,
        "timestamp": run_ts,
        "dataset": "COMMON_DRAFT_SET (roster5+5 & draft5+5, team non-cold-start)",
        "n_total": int(len(df)), "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "frozen_baseline": BASELINE,
        "player_elo_k": ELO_K,
        "results_logreg": results,
        "results_catboost": catboost_results,
        "statistics_vs_baseline": stats,
        "walk_forward": wf_rows,
        "cold_start": cold_rows,
        "coefficients": {
            "A6": models["A6 ALL roster/player"].feature_importance(),
            "B5": models["B5 ALL roster + ALL draft"].feature_importance(),
        },
    }
    path = os.path.join(EXPERIMENTS_DIR, "phase8_feature_redesign.json")
    with open(path, "w") as f:
        json.dump(entry, f, indent=2, default=str)
    print(f"Записано: {path}")
    return 0


def save_figures(results, wf_rows, base_acc):
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["tab:gray" if n.startswith("Baseline") else ("tab:blue" if n.startswith("A") else "tab:orange") for n in names]
    ax.barh(names, accs, color=colors)
    ax.axvline(base_acc, color="black", linestyle="--", linewidth=1, label="frozen baseline")
    ax.set_xlim(min(accs) - 0.005, max(accs) + 0.005)
    ax.set_title("Phase 8 — accuracy на COMMON_DRAFT_SET (синий=pre-draft, оранж=post-draft)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase8_ablation.png"), dpi=110)
    plt.close(fig)

    if wf_rows:
        yrs = [r["year"] for r in wf_rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(yrs, [r["baseline"]["accuracy"] for r in wf_rows], marker="o", label="Baseline")
        ax.plot(yrs, [r["modelA"]["accuracy"] for r in wf_rows], marker="o", label="MODEL A (pre-draft)")
        ax.plot(yrs, [r["modelB"]["accuracy"] for r in wf_rows], marker="o", label="MODEL B (post-draft)")
        ax.set_title("Phase 8 — walk-forward по годам")
        ax.set_ylabel("accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "phase8_walk_forward.png"), dpi=110)
        plt.close(fig)
    print(f"Графики сохранены в {FIGURES_DIR}")


if __name__ == "__main__":
    raise SystemExit(main())
