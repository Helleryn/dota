#!/usr/bin/env python3
"""
PHASE 9 — Roster Strength 2.0 + Meta-Aware Draft.

ИСПРАВЛЕНИЕ ПРОТОКОЛА относительно Phase 8 (см. reports/phase9-research-plan.md,
раздел 0): Phase 8 выбирала лучшую конфигурацию по метрике на TEST. Здесь:

  * ВСЕ решения (какая агрегация лучше, какая мета-схема лучше, состав
    финальной модели) принимаются ИСКЛЮЧИТЕЛЬНО по VALIDATION;
  * TEST используется РОВНО ОДИН РАЗ, в самом конце, для заранее
    зафиксированного списка моделей;
  * для каждого семейства печатаются И VAL, И TEST, чтобы был виден
    масштаб оптимистического смещения отбора.

Запуск:
    python3 scripts/phase9_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import FrozenSet, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.config import load_settings
from src.datasets.builder import _load_pro_matches
from src.datasets.hero_strength_schemes import SCHEMES, build_hero_scheme_features
from src.datasets.meta_features import build_meta_features
from src.datasets.multi_window_features import build_multi_window_features
from src.datasets.roster_representation import build_roster_representation, to_feature_dict
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

TEST_ACCESS_COUNT = {"n": 0}  # аудит числа обращений к TEST


@dataclass(frozen=True)
class FullMatch:
    match_id: int
    start_time: datetime
    patch_id: Optional[int]
    radiant_team_id: int
    dire_team_id: int
    radiant_roster: FrozenSet[int]
    dire_roster: FrozenSet[int]
    radiant_picks: Tuple[int, ...]
    dire_picks: Tuple[int, ...]
    radiant_bans: Tuple[int, ...]
    dire_bans: Tuple[int, ...]
    radiant_win: bool


def load_full_matches(engine):
    """Полный состав (10) + полный драфт (5+5 пиков). Баны берутся как есть
    (в части матчей их меньше 10) — они не входят в критерий отбора."""
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
                 array_agg(pb.hero_id ORDER BY pb.ord) FILTER (WHERE pb.is_pick AND pb.team = 1) AS d_picks,
                 array_agg(pb.hero_id ORDER BY pb.ord) FILTER (WHERE NOT pb.is_pick AND pb.team = 0) AS r_bans,
                 array_agg(pb.hero_id ORDER BY pb.ord) FILTER (WHERE NOT pb.is_pick AND pb.team = 1) AS d_bans
          FROM picks_bans pb GROUP BY 1
        )
        SELECT m.match_id, m.start_time, m.patch_id, m.radiant_team_id, m.dire_team_id, m.radiant_win,
               r.r_roster, r.d_roster, d.r_picks, d.d_picks, d.r_bans, d.d_bans
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
            match_id=r.match_id, start_time=st, patch_id=r.patch_id,
            radiant_team_id=r.radiant_team_id, dire_team_id=r.dire_team_id,
            radiant_roster=frozenset(r.r_roster), dire_roster=frozenset(r.d_roster),
            radiant_picks=tuple(r.r_picks), dire_picks=tuple(r.d_picks),
            radiant_bans=tuple(r.r_bans or ()), dire_bans=tuple(r.d_bans or ()),
            radiant_win=r.radiant_win,
        ))
    return out


def build_dataset(engine):
    full = load_full_matches(engine)
    ids = {m.match_id for m in full}
    print(f"COMMON_SET (роcтер 5+5, драфт 5+5): {len(full)}")

    raw = _load_pro_matches(engine)
    base_rows = build_multi_window_features(raw, k_factor=ELO_K)
    base = pd.DataFrame([{
        "match_id": r.match_id, "as_of_timestamp": r.as_of_timestamp,
        "radiant_team_id": r.radiant_team_id, "dire_team_id": r.dire_team_id,
        "elo_difference": r.elo_difference,
        "form_3_difference": r.recent_winrate_difference[FORM_WINDOW],
        "radiant_matches_played_before": r.radiant_matches_played_before,
        "dire_matches_played_before": r.dire_matches_played_before,
        "target": int(r.radiant_win),
    } for r in base_rows])

    roster = pd.DataFrame([to_feature_dict(r) for r in build_roster_representation(full, k_factor=ELO_K)])

    hs = build_hero_scheme_features(full)
    hero = pd.DataFrame([{
        "match_id": r.match_id,
        **{f"hero_{s}_diff": r.strength_diff[s] for s in SCHEMES},
        "pick_rate_diff": r.pick_rate_diff,
        "ban_rate_diff": r.ban_rate_diff,
        "contest_rate_diff": r.contest_rate_diff,
    } for r in hs])

    # Phase 8 модуль: synergy / counters / player-hero (переиспользуем, не дублируем)
    mf = build_meta_features(full)
    meta = pd.DataFrame([{
        "match_id": r.match_id,
        "hero_synergy_diff": r.radiant_hero_synergy - r.dire_hero_synergy,
        "counter_advantage": r.counter_advantage,
        "player_hero_prof_diff": r.radiant_player_hero_proficiency - r.dire_player_hero_proficiency,
    } for r in mf])

    df = (base[base["match_id"].isin(ids)]
          .merge(roster, on="match_id").merge(hero, on="match_id").merge(meta, on="match_id"))
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


def fit_on_train(features, train):
    m = LogisticRegressionModel(feature_names=features, random_state=RANDOM_SEED)
    m.fit(train, train["target"])
    return m


def eval_val(model, val):
    """ОТБОР идёт только через эту функцию — VALIDATION."""
    return model.evaluate(val, val["target"])


def eval_test(model, test):
    """Единственная точка обращения к TEST. Считается для аудита."""
    TEST_ACCESS_COUNT["n"] += 1
    return model.evaluate(test, test["target"])


def show(name, m, ref=None):
    d = f"  Δacc={m['accuracy']-ref:+.4f}" if ref is not None else ""
    print(f"{name:42s} acc={m['accuracy']:.4f} ll={m['log_loss']:.4f} auc={m['roc_auc']:.4f}{d}")


def main() -> int:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    commit = git_commit_sha()
    run_ts = datetime.now(timezone.utc).isoformat()

    section("PHASE 9 — Roster Strength 2.0 + Meta-Aware Draft")
    print(f"git commit: {commit}")
    print("ПРОТОКОЛ: весь отбор — по VALIDATION; TEST — один раз в конце.")

    engine = make_engine(load_settings())
    df = build_dataset(engine)
    train, val, test = split(df)
    print(f"TRAIN n={len(train)} ({train['as_of_timestamp'].min().date()}..{train['as_of_timestamp'].max().date()})")
    print(f"VAL   n={len(val)} ({val['as_of_timestamp'].min().date()}..{val['as_of_timestamp'].max().date()})")
    print(f"TEST  n={len(test)} ({test['as_of_timestamp'].min().date()}..{test['as_of_timestamp'].max().date()})")

    registry = {}

    # ---------- 9.1 PART A ----------
    section("9.1 PART A — представление силы пятёрки (отбор по VALIDATION)")
    base_model = fit_on_train(BASELINE, train)
    base_val = eval_val(base_model, val)
    show("Baseline (Elo+Form3)", base_val)
    bv = base_val["accuracy"]

    part_a = {
        "A-mean": "elo_mean_diff", "A-sum": "elo_sum_diff", "A-median": "elo_median_diff",
        "A-min": "elo_min_diff", "A-max": "elo_max_diff", "A-std": "elo_std_diff",
        "A-spread": "elo_spread_diff", "A-weakest_link": "weakest_link_diff",
        "A-star_power": "star_power_diff",
    }
    a_val = {}
    for name, feat in part_a.items():
        m = fit_on_train(BASELINE + [feat], train)
        a_val[name] = eval_val(m, val)
        show(name, a_val[name], bv)
    best_a = min(a_val, key=lambda k: a_val[k]["log_loss"])
    print(f"\n>>> ПОБЕДИТЕЛЬ PART A по VAL log_loss: {best_a} ({part_a[best_a]})")
    registry["part_a_val"] = a_val
    registry["part_a_winner"] = best_a

    # ---------- 9.2 PART B/C ----------
    section("9.2 PART B/C — roster delta и current five vs team Elo (VALIDATION)")
    part_bc = {
        "B-roster_strength_delta": ["roster_strength_delta_diff"],
        "B-player_replacement_delta": ["player_replacement_delta_diff"],
        "B-n_replacements": [f"n_replacements_{w}d_diff" for w in (7, 14, 30)],
        "B-cum_roster_delta": ["cum_roster_delta_7d_diff", "cum_roster_delta_30d_diff"],
        "C-five_vs_team_elo": ["five_vs_team_elo_diff"],
    }
    bc_val = {}
    for name, feats in part_bc.items():
        m = fit_on_train(BASELINE + feats, train)
        bc_val[name] = eval_val(m, val)
        show(name, bc_val[name], bv)
    registry["part_bc_val"] = bc_val

    # ---------- 9.4 PART E ----------
    section("9.4 PART E — synergy игроков и героев (VALIDATION)")
    part_e = {
        "E-hero_synergy": ["hero_synergy_diff"],
        "E-counters": ["counter_advantage"],
        "E-player_hero_prof": ["player_hero_prof_diff"],
    }
    e_val = {}
    for name, feats in part_e.items():
        m = fit_on_train(BASELINE + feats, train)
        e_val[name] = eval_val(m, val)
        show(name, e_val[name], bv)
    registry["part_e_val"] = e_val

    # ---------- 9.5 PART H ----------
    section("9.5 PART H — 5 схем силы героя на ОДНОЙ выборке (VALIDATION)")
    h_val = {}
    for s in SCHEMES:
        m = fit_on_train(BASELINE + [f"hero_{s}_diff"], train)
        h_val[s] = eval_val(m, val)
        show(f"H-{s}", h_val[s], bv)
    best_scheme = min(h_val, key=lambda k: h_val[k]["log_loss"])
    print(f"\n>>> ПОБЕДИТЕЛЬ PART H по VAL log_loss: {best_scheme}")
    registry["part_h_val"] = h_val
    registry["part_h_winner"] = best_scheme

    # ---------- 9.6 PART I ----------
    section("9.6 PART I — pick/ban rate (ранее не использовались) (VALIDATION)")
    part_i = {
        "I-pick_rate": ["pick_rate_diff"],
        "I-ban_rate": ["ban_rate_diff"],
        "I-contest_rate": ["contest_rate_diff"],
    }
    i_val = {}
    for name, feats in part_i.items():
        m = fit_on_train(BASELINE + feats, train)
        i_val[name] = eval_val(m, val)
        show(name, i_val[name], bv)
    registry["part_i_val"] = i_val

    # ---------- 9.7 сборка финальных моделей (по VAL) ----------
    section("9.7 PART N — сборка финальных моделей (состав выбран по VALIDATION)")
    roster_feats = [part_a[best_a]]
    for name, feats in part_bc.items():
        if bc_val[name]["log_loss"] < base_val["log_loss"]:
            roster_feats += feats
    roster_feats = list(dict.fromkeys(roster_feats))

    draft_feats = [f"hero_{best_scheme}_diff"]
    for name, feats in list(part_e.items()) + list(part_i.items()):
        src = e_val if name in part_e else i_val
        if src[name]["log_loss"] < base_val["log_loss"]:
            draft_feats += feats
    draft_feats = list(dict.fromkeys(draft_feats))

    print(f"ROSTER набор (по VAL): {roster_feats}")
    print(f"DRAFT  набор (по VAL): {draft_feats}")

    FINAL = {
        "Baseline (frozen)": BASELINE,
        "Baseline + ROSTER": BASELINE + roster_feats,
        "Baseline + DRAFT": BASELINE + draft_feats,
        "Baseline + ROSTER + DRAFT": BASELINE + roster_feats + draft_feats,
    }
    for name, feats in FINAL.items():
        m = fit_on_train(feats, train)
        show(f"[VAL] {name}", eval_val(m, val), bv)

    # ---------- ФИНАЛ: единственное обращение к TEST ----------
    section("ФИНАЛЬНАЯ ОЦЕНКА НА TEST (список моделей зафиксирован выше по VAL)")
    final_models, final_test = {}, {}
    for name, feats in FINAL.items():
        m = fit_on_train(feats, train)
        final_models[name] = m
        final_test[name] = eval_test(m, test)
    tb = final_test["Baseline (frozen)"]["accuracy"]
    for name in FINAL:
        show(name, final_test[name], tb)

    print()
    for name, feats in [("Baseline", BASELINE), ("ROSTER+DRAFT", FINAL["Baseline + ROSTER + DRAFT"])]:
        cb = CatBoostModel(feature_names=feats)
        cb.fit(train, train["target"])
        met = eval_test(cb, test)
        final_test[f"CatBoost {name}"] = met
        show(f"CatBoost {name}", met, tb)

    section("Статистическая значимость на TEST (block bootstrap + McNemar)")
    y = test["target"].to_numpy()
    p_base = final_models["Baseline (frozen)"].predict_proba(test)[:, 1]
    stats = {}
    for name in ["Baseline + ROSTER", "Baseline + DRAFT", "Baseline + ROSTER + DRAFT"]:
        p = final_models[name].predict_proba(test)[:, 1]
        da = block_bootstrap_paired_diff(y, p, p_base, accuracy_metric, block_size=BLOCK_SIZE)
        dl = block_bootstrap_paired_diff(y, p, p_base, log_loss_metric, block_size=BLOCK_SIZE)
        mc = mcnemar_exact(y, (p >= 0.5).astype(int), (p_base >= 0.5).astype(int))
        stats[name] = {"acc": da, "ll": dl, "mcnemar": mc}
        print(f"{name:30s} Δacc={da['point_diff']:+.4f} CI=[{da['ci_low']:+.4f},{da['ci_high']:+.4f}]  "
              f"Δll={dl['point_diff']:+.4f} CI=[{dl['ci_low']:+.4f},{dl['ci_high']:+.4f}]  p={mc['p_value']:.4f}")

    section("Walk-forward по годам")
    dy = df.copy()
    dy["year"] = dy["as_of_timestamp"].dt.year
    wf = []
    for yv in sorted(dy["year"].unique())[1:]:
        tr, pe = dy[dy["year"] < yv], dy[dy["year"] == yv]
        if len(tr) < 3000 or len(pe) < 200:
            continue
        row = {"year": int(yv), "n": len(pe)}
        for name, feats in FINAL.items():
            mm = fit_on_train(feats, tr)
            row[name] = compute_metrics(pe["target"], mm.predict_proba(pe)[:, 1])
        wf.append(row)
        print(f"{yv}  n={row['n']:6d}  base={row['Baseline (frozen)']['accuracy']:.4f}  "
              f"R={row['Baseline + ROSTER']['accuracy']:.4f}  "
              f"D={row['Baseline + DRAFT']['accuracy']:.4f}  "
              f"R+D={row['Baseline + ROSTER + DRAFT']['accuracy']:.4f}")

    section("PART D — диагностика team identity (без причинных выводов)")
    t2 = test.copy()
    t2["min_hist"] = t2[["radiant_matches_played_before", "dire_matches_played_before"]].min(axis=1)
    ident = []
    for label, mask in [("<20 матчей", t2["min_hist"] < 20),
                        ("20-100", (t2["min_hist"] >= 20) & (t2["min_hist"] < 100)),
                        (">=100", t2["min_hist"] >= 100)]:
        sub = t2[mask]
        if len(sub) < 100:
            continue
        mb = compute_metrics(sub["target"], final_models["Baseline (frozen)"].predict_proba(sub)[:, 1])
        mr = compute_metrics(sub["target"], final_models["Baseline + ROSTER"].predict_proba(sub)[:, 1])
        ident.append({"bucket": label, "n": len(sub), "base": mb["accuracy"], "roster": mr["accuracy"]})
        print(f"{label:12s} n={len(sub):6d}  base={mb['accuracy']:.4f}  +ROSTER={mr['accuracy']:.4f}  "
              f"Δ={mr['accuracy']-mb['accuracy']:+.4f}")

    section("Коэффициенты финальной модели")
    coefs = final_models["Baseline + ROSTER + DRAFT"].feature_importance()
    for k, v in sorted(coefs.items(), key=lambda x: -abs(x[1])):
        print(f"  {k:34s} {v:+.5f}")

    print()
    print(f"ЧИСЛО ОБРАЩЕНИЙ К TEST ЗА ВЕСЬ ПРОГОН: {TEST_ACCESS_COUNT['n']} "
          f"(все — после фиксации состава моделей по VAL)")

    save_figures(a_val, h_val, wf, bv)

    entry = {
        "experiment_id": f"phase9_{run_ts}", "git_commit": commit, "timestamp": run_ts,
        "protocol": "selection on VALIDATION only; TEST touched once at the end",
        "test_access_count": TEST_ACCESS_COUNT["n"],
        "n_total": len(df), "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "frozen_baseline": BASELINE,
        "selected_roster_features": roster_feats,
        "selected_draft_features": draft_feats,
        **registry,
        "final_test": final_test,
        "statistics": stats,
        "walk_forward": wf,
        "identity_diagnostic": ident,
        "coefficients": coefs,
    }
    path = os.path.join(EXPERIMENTS_DIR, "phase9_roster_draft.json")
    with open(path, "w") as f:
        json.dump(entry, f, indent=2, default=str)
    print(f"Записано: {path}")
    return 0


def save_figures(a_val, h_val, wf, base_val_acc):
    names = list(a_val.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, [a_val[n]["log_loss"] for n in names], color="tab:blue")
    ax.set_xlim(min(a_val[n]["log_loss"] for n in names) - 0.002,
                max(a_val[n]["log_loss"] for n in names) + 0.002)
    ax.set_title("PART A — сравнение агрегаций player-Elo (VAL log loss, меньше=лучше)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "phase9_part_a.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ks = list(h_val.keys())
    ax.barh(ks, [h_val[k]["log_loss"] for k in ks], color="tab:orange")
    ax.set_xlim(min(h_val[k]["log_loss"] for k in ks) - 0.002,
                max(h_val[k]["log_loss"] for k in ks) + 0.002)
    ax.set_title("PART H — 5 схем силы героя (VAL log loss, меньше=лучше)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "phase9_part_h.png"), dpi=110); plt.close(fig)

    if wf:
        yrs = [r["year"] for r in wf]
        fig, ax = plt.subplots(figsize=(8, 4))
        for key, lbl in [("Baseline (frozen)", "Baseline"), ("Baseline + ROSTER", "+ROSTER"),
                         ("Baseline + DRAFT", "+DRAFT"), ("Baseline + ROSTER + DRAFT", "+BOTH")]:
            ax.plot(yrs, [r[key]["accuracy"] for r in wf], marker="o", label=lbl)
        ax.set_title("Phase 9 — walk-forward по годам"); ax.set_ylabel("accuracy"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "phase9_walk_forward.png"), dpi=110); plt.close(fig)
    print(f"Графики сохранены в {FIGURES_DIR}")


if __name__ == "__main__":
    raise SystemExit(main())
