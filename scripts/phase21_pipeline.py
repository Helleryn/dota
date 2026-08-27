#!/usr/bin/env python3
"""
PHASE 21 — контекст соперников: TRAIN/VALIDATION (LEVEL 1).

Гипотезы, окна, полураспад и критерии зафиксированы в
reports/phase21-plan.md ДО запуска. Здесь ничего не подбирается.

Базы сравнения (§1.2 плана):
  B0 = фактический frozen pre-match      (4 признака)
  B1 = B0 + H1                            (кандидат Phase 20)
  B2 = B0 + H1 + H3                       (подтверждено на TEST в Phase 20)

Основная база — B2. Признак, бьющий B1, но не B2, избыточен с давностью.

Запуск: python3 scripts/phase21_pipeline.py
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
from src.pit.lineup_structure import build_lineup_features, load_lineup_matches
from src.pit.opponent_context import build_opponent_context, load_context_matches
from scripts.phase6_pipeline import git_commit_sha, section
from scripts.phase13_pipeline import load_common_set, split
from scripts.phase20_pipeline import H1 as P20_H1
from scripts.phase20_pipeline import H3 as P20_H3
from scripts.phase20_pipeline import M0 as P20_M0
from scripts.phase20_pipeline import CACHE as P20_CACHE

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")
CACHE = os.path.join(EXPERIMENTS_DIR, "cache", "phase21_frame.csv")
HORIZON = timedelta(hours=24)
THRESHOLD = 1e-4

B0 = list(P20_M0)
B1 = B0 + list(P20_H1)
B2 = B1 + list(P20_H3)

# --- гипотезы Phase 21 ---
H2_OAS = ["oas_diff_7", "oas_diff_14", "oas_diff_30",
          "recent_n_min_7", "recent_n_min_14", "recent_n_min_30"]
H3_COMMON = ["common_opponent_delta", "common_opponent_count"]
H4_H2H = ["h2h_residual_decayed", "h2h_matches_decayed", "h2h_days_since_last"]
H5_H2H_WR = ["h2h_winrate_decayed", "h2h_matches_decayed"]
H6_ROSTER = ["h2h_residual_roster", "h2h_roster_overlap", "h2h_matches_decayed"]
H7_RAW = ["recent_winrate_diff_30"]

GROUPS = {
    "H2 opponent-adjusted": H2_OAS,
    "H3 общие соперники": H3_COMMON,
    "H4 H2H residual": H4_H2H,
    "H5 H2H winrate (decay)": H5_H2H_WR,
    "H6 roster-aware H2H": H6_ROSTER,
    "H7 сырой winrate (контроль)": H7_RAW,
}
ALL_NEW = ["oas_diff_7", "oas_diff_14", "oas_diff_30", "recent_n_min_7",
           "recent_n_min_14", "recent_n_min_30", "recent_winrate_diff_30",
           "common_opponent_delta", "common_opponent_count",
           "h2h_residual_decayed", "h2h_winrate_decayed", "h2h_matches_decayed",
           "h2h_days_since_last", "h2h_roster_overlap", "h2h_residual_roster"]


def build_frame() -> pd.DataFrame:
    if os.path.exists(CACHE):
        df = pd.read_csv(CACHE, parse_dates=["as_of_timestamp"])
        print(f"кэш: {CACHE} ({len(df):,} строк)")
        return df
    if not os.path.exists(P20_CACHE):
        raise SystemExit(f"нет кэша Phase 20: {P20_CACHE}; сначала phase20_pipeline.py")
    base = pd.read_csv(P20_CACHE, parse_dates=["as_of_timestamp"])
    print(f"база Phase 20: {len(base):,} строк, {len(base.columns)} колонок")

    eng = make_engine(load_settings())
    emit = set(base["match_id"])
    print("проход: контекст соперников…", flush=True)
    rows = build_opponent_context(load_context_matches(eng), HORIZON, emit_only=emit)
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
    print(f"{name:32s} acc={r['accuracy']:.4f} auc={r['roc_auc']:.4f} "
          f"ll={r['log_loss']:.4f} brier={r['brier']:.4f} ECE={r['ece']:.5f}{d}",
          flush=True)


def main() -> int:
    section("PHASE 21 — контекст соперников (TRAIN/VALIDATION)")
    print(f"git commit: {git_commit_sha()}")
    print(f"Порог отбора Δlog loss > {THRESHOLD:g}, объявлен заранее.")
    print("TEST не открывается.\n")

    df = build_frame()
    payload = {"phase": 21, "commit": git_commit_sha(), "threshold": THRESHOLD,
               "test_accesses": 0}

    # ---------- §14 покрытие и политика пропусков ----------
    section("Покрытие и различение missing / zero (§14)")
    cov = {}
    for f in ALL_NEW:
        cov[f] = float(df[f].notna().mean())
        print(f"  {f:28s} покрытие {cov[f]:6.1%}")
    print()
    for w in (7, 14, 30):
        z = (df[f"recent_n_min_{w}"] == 0).mean()
        print(f"  окно {w:>2}д: хотя бы у одной стороны 0 матчей — {z:.1%}")
    co0 = (df["common_opponent_count"] == 0).mean()
    h0 = (df["h2h_matches_decayed"] == 0).mean()
    print(f"  общих соперников нет — {co0:.1%}")
    print(f"  прошлых встреч нет   — {h0:.1%}")
    payload["coverage"] = cov
    payload["missing"] = {"common_none": float(co0), "h2h_none": float(h0)}

    tr, va, te = split(df)
    y = va["target"].to_numpy()
    print(f"\nTRAIN {len(tr):,} до {tr['as_of_timestamp'].max():%Y-%m-%d} | "
          f"VAL {len(va):,} | TEST {len(te):,} (не трогается)")

    # ---------- базы ----------
    section("Базы сравнения")
    r_b0 = ev(y, proba(fit(B0, tr), va))
    r_b1 = ev(y, proba(fit(B1, tr), va))
    r_b2 = ev(y, proba(fit(B2, tr), va))
    show("B0 фактический frozen", r_b0)
    show("B1 = B0 + H1", r_b1, r_b0)
    show("B2 = B0 + H1 + H3 (основная)", r_b2, r_b0)
    payload["bases"] = {"B0": r_b0, "B1": r_b1, "B2": r_b2}

    # ---------- §20 ablation ----------
    section("Ablation: каждый механизм поверх B2 (основная база)")
    verdicts = {}
    survivors = []
    for g, feats in GROUPS.items():
        r = ev(y, proba(fit(B2 + feats, tr), va))
        gain = r_b2["log_loss"] - r["log_loss"]
        ok = gain > THRESHOLD
        verdicts[g] = {"gain_over_B2": gain, "passed": ok, "features": feats, **r}
        show(f"B2 + {g}", r, r_b2)
        if ok and not g.startswith("H7"):
            survivors.extend([f for f in feats if f not in survivors])
    print()
    for g, v in verdicts.items():
        print(f"  {g:30s} Δll={v['gain_over_B2']:+.5f}  "
              f"{'ПРОШЛА' if v['passed'] else 'не прошла'}")

    section("То же поверх B1 (как называет база в §20 задания)")
    for g, feats in GROUPS.items():
        r = ev(y, proba(fit(B1 + feats, tr), va))
        gain = r_b1["log_loss"] - r["log_loss"]
        verdicts[g]["gain_over_B1"] = gain
        print(f"  {g:30s} Δll поверх B1 = {gain:+.5f}"
              + ("   <- бьёт B1, но не B2: избыточен с давностью"
                 if gain > THRESHOLD and not verdicts[g]["passed"] else ""))
    payload["ablation"] = verdicts

    # ---------- §13 форма против контекста ----------
    section("§13 — recent form против opponent-adjusted strength")
    no_form = [f for f in B0 if f != "form_3_difference"]
    combos = {
        "A: без формы вообще": no_form,
        "B: + форма (это и есть B0)": B0,
        "C: + OAS вместо формы": no_form + H2_OAS,
        "D: + обе": B0 + H2_OAS,
    }
    r_a = None
    for name, feats in combos.items():
        r = ev(y, proba(fit(feats, tr), va))
        if r_a is None:
            r_a = r
        show(name, r, r_a)
    payload["form_vs_context"] = {k: ev(y, proba(fit(v, tr), va))
                                  for k, v in combos.items()}

    # ---------- M5 ----------
    section("M5 — только выжившие механизмы")
    if survivors:
        r5 = ev(y, proba(fit(B2 + survivors, tr), va))
        show("M5", r5, r_b2)
        print(f"\n  вошли: {survivors}")
        payload["M5"] = {"features": survivors, **r5,
                         "gain": r_b2["log_loss"] - r5["log_loss"]}
    else:
        print("  Ни один механизм не прошёл порог поверх B2.")
        print("  M5 не строится. Это результат фазы, а не сбой.")
        payload["M5"] = None

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    out = os.path.join(EXPERIMENTS_DIR, "phase21_pipeline.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nОбращений к TEST: 0\nРезультаты: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
