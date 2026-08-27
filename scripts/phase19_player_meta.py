#!/usr/bin/env python3
"""
PHASE 19 — H1: player×hero, соотнесённый с текущей метой (PART O).

Гипотеза объявлена в docs/phase19-plan.md ДО запуска:

  H1  сигнал player×hero, соотнесённый с текущей метой (`resid`),
      информативнее сырого player×hero (`raw`).

Порог отбора Δlog loss > 1e-4 на VALIDATION — тот же, что в Phase 18.
TEST не используется.

Запуск: python3 scripts/phase19_player_meta.py
"""

from __future__ import annotations

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
from src.models.sklearn_models import RANDOM_SEED, LogisticRegressionModel
from src.pit.engine import build_point_in_time_features
from src.pit.loader import load_pit_matches
from src.pit.player_meta import build_player_meta_features
from scripts.phase6_pipeline import git_commit_sha, section
from scripts.phase13_pipeline import load_common_set, split

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")
HORIZON = timedelta(hours=24)
BASE = ["elo_difference", "form_3_difference", "elo_mean_diff", "five_vs_team_elo_diff"]
THRESHOLD = 1e-4


def ev(y, p):
    m = compute_metrics(y, p)
    c = calibration_report(y, p)
    return {"n": int(m["n"]), "accuracy": m["accuracy"], "roc_auc": m["roc_auc"],
            "log_loss": m["log_loss"], "brier": m["brier_score"], "ece": c["ece"]}


def fit(feats, tr):
    m = LogisticRegressionModel(feature_names=feats, random_state=RANDOM_SEED)
    m.fit(tr, tr["target"])
    return m


def show(name, r, ref=None):
    d = f"  Δll={r['log_loss'] - ref['log_loss']:+.5f}" if ref else ""
    print(f"{name:38s} acc={r['accuracy']:.4f} auc={r['roc_auc']:.4f} "
          f"ll={r['log_loss']:.4f} ECE={r['ece']:.5f}{d}", flush=True)


def main() -> int:
    section("PHASE 19 — H1: player×hero относительно текущей меты (VALIDATION)")
    print(f"git commit: {git_commit_sha()}")
    print(f"Порог отбора Δlog loss > {THRESHOLD:g}, объявлен заранее.\n")

    ref = load_common_set()
    emit = set(ref["match_id"])
    matches, rp = load_pit_matches(make_engine(load_settings()))
    print(f"матчей в ленте: {len(matches):,}, в выдаче: {len(emit):,}\n")

    rows = build_point_in_time_features(matches, HORIZON, roster_provider=rp,
                                        include_draft=False, emit_only=emit)
    base = pd.DataFrame([{
        "match_id": r.match_id, "as_of_timestamp": r.start_time,
        "elo_difference": r.elo_difference, "form_3_difference": r.form_3_difference,
        "elo_mean_diff": r.elo_mean_diff, "five_vs_team_elo_diff": r.five_vs_team_elo_diff,
        "target": r.target} for r in rows])

    pm = build_player_meta_features(matches, HORIZON, roster_provider=rp, emit_only=emit)
    extra = pd.DataFrame([{"match_id": r.match_id, "pool_raw_diff": r.pool_raw_diff,
                           "pool_resid_diff": r.pool_resid_diff,
                           "pool_players_min": r.pool_players_min} for r in pm])

    df = base.merge(extra, on="match_id", how="left").sort_values(
        ["as_of_timestamp", "match_id"]).reset_index(drop=True)
    cov = df["pool_resid_diff"].notna().mean()
    print(f"покрытие признака: {cov:.1%} матчей "
          f"({int(df['pool_resid_diff'].notna().sum()):,} из {len(df):,})")
    print(f"корреляция raw и resid: {df[['pool_raw_diff','pool_resid_diff']].corr().iloc[0,1]:.4f}\n")

    tr, va, _ = split(df)
    y = va["target"].to_numpy()

    section("Результаты на VALIDATION")
    r_base = ev(y, np.asarray(fit(BASE, tr).predict_proba(va))[:, 1])
    show("PREMATCH_BASELINE", r_base)
    r_raw = ev(y, np.asarray(fit(BASE + ["pool_raw_diff"], tr).predict_proba(va))[:, 1])
    show("  + сырой player×hero (raw)", r_raw, r_base)
    r_res = ev(y, np.asarray(fit(BASE + ["pool_resid_diff"], tr).predict_proba(va))[:, 1])
    show("  + относительно меты (resid)", r_res, r_base)
    r_both = ev(y, np.asarray(
        fit(BASE + ["pool_raw_diff", "pool_resid_diff"], tr).predict_proba(va))[:, 1])
    show("  + оба сразу", r_both, r_base)

    g_raw = r_base["log_loss"] - r_raw["log_loss"]
    g_res = r_base["log_loss"] - r_res["log_loss"]
    h1 = g_res > g_raw + THRESHOLD
    print(f"\n  выигрыш raw к базе:    {g_raw:+.5f}")
    print(f"  выигрыш resid к базе:  {g_res:+.5f}")
    print(f"  H1 (resid лучше raw на > {THRESHOLD:g}): "
          f"{'ПОДТВЕРЖДЕНА' if h1 else 'ОПРОВЕРГНУТА'}")
    keep = max(g_raw, g_res) > THRESHOLD
    print(f"  хоть один вариант превышает порог к базе: "
          f"{'ДА -> RESEARCH FURTHER' if keep else 'НЕТ -> REMOVE'}")

    payload = {"phase": 19, "commit": git_commit_sha(), "hypothesis": "H1",
               "coverage": float(cov), "threshold": THRESHOLD,
               "base": r_base, "raw": r_raw, "resid": r_res, "both": r_both,
               "gain_raw": g_raw, "gain_resid": g_res,
               "H1_verdict": "CONFIRMED" if h1 else "REFUTED",
               "feature_verdict": "RESEARCH_FURTHER" if keep else "REMOVE",
               "test_accesses": 0}
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    out = os.path.join(EXPERIMENTS_DIR, "phase19_player_meta.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nОбращений к TEST: 0\nРезультаты: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
