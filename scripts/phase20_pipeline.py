#!/usr/bin/env python3
"""
PHASE 20 — пять механизмов сверх player-Elo: эксперименты TRAIN/VAL.

Гипотезы, механизмы и критерии зафиксированы в reports/phase20-plan.md ДО
запуска. Здесь ничего не подбирается: сценарий считает по объявленным
правилам.

  M0 = PREMATCH_BASELINE (4 входа)
  M1 = M0 + H1 неопределённость      M4 = M0 + H4 сыгранность
  M2 = M0 + H2 позиции               M5 = M0 + H5 новизна
  M3 = M0 + H3 актуальность          M6 = M0 + прошедшие поодиночке

Порог отбора Δlog loss > 1e-4 на VALIDATION. Основная метрика — log loss.

Запуск: python3 scripts/phase20_pipeline.py [--final-test]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.config import load_settings
from src.db.engine import make_engine
from src.evaluation.calibration import calibration_report
from src.evaluation.metrics import compute_metrics
from src.evaluation.statistics import (block_bootstrap_paired_diff, log_loss_metric,
                                       mcnemar_exact)
from src.models.sklearn_models import RANDOM_SEED, LogisticRegressionModel
from src.pit.engine import build_point_in_time_features
from src.pit.lineup_structure import build_lineup_features, load_lineup_matches
from src.pit.loader import load_pit_matches
from scripts.phase6_pipeline import git_commit_sha, section
from scripts.phase13_pipeline import load_common_set, split

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")
CACHE = os.path.join(EXPERIMENTS_DIR, "cache", "phase20_frame.csv")
HORIZON = timedelta(hours=24)
THRESHOLD = 1e-4

M0 = ["elo_difference", "form_3_difference", "elo_mean_diff", "five_vs_team_elo_diff"]
H1 = ["rd_mean_diff", "rd_max_diff", "elo_mean_diff_attenuated"]
H2 = ["pos_diff_1", "pos_diff_2", "pos_diff_3", "pos_diff_4", "pos_diff_5",
      "pos_bottleneck", "pos_best", "pos_imbalance"]
H3 = ["elo_mean_decayed_diff", "recency_gap", "days_since_last_max_diff"]
H4 = ["pair_games_mean_diff", "pair_games_min_diff", "core3_games_diff",
      "synergy_residual_diff"]
H5 = ["returning_players_diff", "core_continuity_diff", "lineup_novelty_diff",
      "days_since_lineup_diff"]
GROUPS = {"H1 неопределённость": H1, "H2 позиции": H2, "H3 актуальность": H3,
          "H4 сыгранность": H4, "H5 новизна": H5}


def build_frame() -> pd.DataFrame:
    if os.path.exists(CACHE):
        df = pd.read_csv(CACHE, parse_dates=["as_of_timestamp"])
        print(f"кэш: {CACHE} ({len(df):,} строк)")
        return df
    ref = load_common_set()
    emit = set(ref["match_id"])
    eng = make_engine(load_settings())

    print("проход 1/2: базовые признаки (движок Phase 17)…", flush=True)
    matches, rp = load_pit_matches(eng)
    base = pd.DataFrame([{
        "match_id": r.match_id, "as_of_timestamp": r.start_time,
        "elo_difference": r.elo_difference, "form_3_difference": r.form_3_difference,
        "elo_mean_diff": r.elo_mean_diff, "five_vs_team_elo_diff": r.five_vs_team_elo_diff,
        "target": r.target}
        for r in build_point_in_time_features(matches, HORIZON, roster_provider=rp,
                                              include_draft=False, emit_only=emit)])
    print(f"  {len(base):,} строк", flush=True)

    print("проход 2/2: структура состава (H1–H5)…", flush=True)
    lm = load_lineup_matches(eng)
    rows = build_lineup_features(lm, HORIZON, emit_only=emit)
    extra = pd.DataFrame([r.__dict__ for r in rows]).drop(
        columns=["prediction_at", "target"])
    print(f"  {len(extra):,} строк", flush=True)

    df = base.merge(extra, on="match_id", how="left").sort_values(
        ["as_of_timestamp", "match_id"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def ev(y, p):
    m = compute_metrics(y, p)
    c = calibration_report(y, p)
    return {"n": int(m["n"]), "accuracy": m["accuracy"], "roc_auc": m["roc_auc"],
            "log_loss": m["log_loss"], "brier": m["brier_score"], "ece": c["ece"]}


def fit(feats, tr):
    m = LogisticRegressionModel(feature_names=feats, random_state=RANDOM_SEED)
    m.fit(tr, tr["target"])
    return m


def proba(m, df):
    return np.asarray(m.predict_proba(df))[:, 1]


def show(name, r, ref=None):
    d = f"  Δll={r['log_loss'] - ref['log_loss']:+.5f}" if ref else ""
    print(f"{name:26s} acc={r['accuracy']:.4f} auc={r['roc_auc']:.4f} "
          f"ll={r['log_loss']:.4f} brier={r['brier']:.4f} ECE={r['ece']:.5f}{d}",
          flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args(argv)

    section("PHASE 20 — пять механизмов сверх player-Elo (VALIDATION)")
    print(f"git commit: {git_commit_sha()}")
    print(f"Порог отбора Δlog loss > {THRESHOLD:g}, объявлен заранее.\n")

    df = build_frame()
    payload = {"phase": 20, "commit": git_commit_sha(), "threshold": THRESHOLD,
               "test_accesses": 0}

    # ---------- PART L: покрытие ----------
    section("PART L — покрытие признаков")
    cov = {}
    for g, feats in GROUPS.items():
        for f in feats:
            cov[f] = float(df[f].notna().mean())
        print(f"  {g}: " + ", ".join(f"{f}={cov[f]:.1%}" for f in feats))
    print(f"\n  позиции назначены обеим сторонам полностью: "
          f"{(df['pos_assigned_min'] == 5).mean():.1%}")
    payload["coverage"] = cov
    payload["pos_full"] = float((df["pos_assigned_min"] == 5).mean())

    tr, va, te = split(df)
    y = va["target"].to_numpy()
    print(f"\nTRAIN {len(tr):,} до {tr['as_of_timestamp'].max():%Y-%m-%d} | "
          f"VAL {len(va):,} | TEST {len(te):,} (не трогается)")

    # ---------- PART J: ablation ----------
    section("PART J — ablation на VALIDATION")
    m0 = fit(M0, tr)
    r0 = ev(y, proba(m0, va))
    show("M0 PREMATCH_BASELINE", r0)
    results = {"M0": r0}
    survivors, verdicts = [], {}
    for i, (g, feats) in enumerate(GROUPS.items(), 1):
        r = ev(y, proba(fit(M0 + feats, tr), va))
        results[f"M{i}"] = r
        gain = r0["log_loss"] - r["log_loss"]
        ok = gain > THRESHOLD
        verdicts[g] = {"gain": gain, "passed": ok, "features": feats, **r}
        show(f"M{i} {g}", r, r0)
        if ok:
            survivors.extend(feats)
    print()
    for g, v in verdicts.items():
        print(f"  {g:24s} Δll={v['gain']:+.5f}  "
              f"{'ПРОШЛА порог' if v['passed'] else 'НЕ прошла'}")
    payload["ablation"] = {"M0": r0, "groups": verdicts}

    # ---------- PART K: redundancy ----------
    section("PART K — redundancy audit")
    print("  корреляции кандидатов с базовыми признаками:\n")
    print(f"  {'признак':28s} {'elo_mean':>9s} {'elo_diff':>9s} {'form_3':>9s} {'исход':>9s}")
    red = {}
    for g, feats in GROUPS.items():
        for f in feats:
            c = {b: float(df[[f, b]].corr().iloc[0, 1]) for b in
                 ("elo_mean_diff", "elo_difference", "form_3_difference")}
            c["target"] = float(df[[f, "target"]].corr().iloc[0, 1])
            red[f] = c
            print(f"  {f:28s} {c['elo_mean_diff']:+9.4f} {c['elo_difference']:+9.4f} "
                  f"{c['form_3_difference']:+9.4f} {c['target']:+9.4f}")
    payload["redundancy"] = red

    # ---------- M6 ----------
    section("M6 — только прошедшие поодиночке")
    if survivors:
        r6 = ev(y, proba(fit(M0 + survivors, tr), va))
        show("M6", r6, r0)
        print(f"\n  вошли: {survivors}")
        payload["M6"] = {"features": survivors, **r6,
                         "gain": r0["log_loss"] - r6["log_loss"]}
    else:
        print("  Ни один механизм не прошёл порог. M6 не строится.")
        print("  Это результат фазы, а не сбой: см. reports/phase20-summary.md.")
        payload["M6"] = None

    # ---------- единственное обращение к TEST ----------
    if args.final_test:
        section("ЕДИНСТВЕННОЕ обращение к TEST")
        if not survivors:
            print("  Обращение НЕ производится: на VALIDATION не выжил ни один")
            print("  механизм, сравнивать нечего. Обращений к TEST: 0.")
        else:
            print(f"  Конфигурация зафиксирована на VALIDATION: {survivors}\n")
            yt = te["target"].to_numpy()
            p0 = proba(fit(M0, tr), te)
            p6 = proba(fit(M0 + survivors, tr), te)
            show("M0 на TEST", ev(yt, p0))
            show("M6 на TEST", ev(yt, p6), ev(yt, p0))
            d = block_bootstrap_paired_diff(yt, p6, p0, log_loss_metric,
                                            block_size=20, seed=RANDOM_SEED)
            mc = mcnemar_exact(yt, (p6 > 0.5).astype(int), (p0 > 0.5).astype(int))
            print(f"\n  Δlog loss (M6 − M0): {d['point_diff']:+.5f}  "
                  f"95% CI [{d['ci_low']:+.5f}, {d['ci_high']:+.5f}]")
            print(f"  McNemar p = {mc['p_value']:.4f}")
            print("  " + ("ЗНАЧИМО лучше" if d["ci_high"] < 0 else "интервал включает 0"))
            payload["test"] = {"features": survivors, "m0": ev(yt, p0),
                               "m6": ev(yt, p6), "paired_log_loss": d, "mcnemar": mc}
            payload["test_accesses"] = 1

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    out = os.path.join(EXPERIMENTS_DIR, "phase20_pipeline.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nОбращений к TEST: {payload['test_accesses']}\nРезультаты: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
