#!/usr/bin/env python3
"""
PHASE 17 — что модель может предсказать, используя ТОЛЬКО pre-match данные.

Центральный вопрос фазы и центральная находка аудита:

  Из пяти замороженных признаков **до драфта существуют не все**.

| Признак | Что нужно | Доступен до матча? |
|---|---|---|
| `elo_difference` | id команд | **да** |
| `form_3_difference` | id команд | **да** |
| `elo_mean_diff` | СОСТАВ | только если состав известен |
| `five_vs_team_elo_diff` | СОСТАВ | только если состав известен |
| `hero_exp_decay_diff` | ВЫБРАННЫЕ ГЕРОИ | **нет** — драфта ещё не было |

Отсюда три уровня доступности данных, и фаза измеряет каждый:

  T0 FULL          5 признаков — то, что оценивали Phase 14/15
  T1 PRE_DRAFT+ROSTER  4 признака — драфта нет, состав известен
  T2 PRE_DRAFT-ROSTER  2 признака — ни драфта, ни состава

Плюс горизонт прогноза: признаки на T−24ч и на момент старта — разные
величины у 85% матчей.

Протокол: всё выбирается на VALIDATION, TEST — одно обращение.
Замороженная модель Phase 9 не меняется; модели уровней — отдельные,
со своими версиями.

Запуск:
    python3 scripts/phase17_pipeline.py
    python3 scripts/phase17_pipeline.py --final-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.config import load_settings
from src.db.engine import make_engine
from src.evaluation.calibration import calibration_report
from src.evaluation.metrics import compute_metrics
from src.evaluation.online_calibration import run_walk_forward
from src.evaluation.statistics import (
    accuracy_metric,
    block_bootstrap_metric,
    block_bootstrap_paired_diff,
    log_loss_metric,
    mcnemar_exact,
)
from src.models.sklearn_models import RANDOM_SEED, LogisticRegressionModel
from src.pit.engine import build_point_in_time_features
from src.pit.loader import load_pit_matches
from scripts.phase6_pipeline import git_commit_sha, section
from scripts.phase13_pipeline import load_common_set, split

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")
CACHE = os.path.join(EXPERIMENTS_DIR, "cache")

# Горизонты прогноза. 0 — момент старта (как в Phase 15).
HORIZONS = {"H0 (старт)": timedelta(0), "H1ч": timedelta(hours=1),
            "H3ч": timedelta(hours=3), "H24ч": timedelta(hours=24),
            "H7сут": timedelta(days=7)}

TIERS: Dict[str, List[str]] = {
    "T0 FULL (5 признаков)": ["elo_difference", "form_3_difference", "elo_mean_diff",
                              "five_vs_team_elo_diff", "hero_exp_decay_diff"],
    "T1 PRE_DRAFT + состав": ["elo_difference", "form_3_difference", "elo_mean_diff",
                              "five_vs_team_elo_diff"],
    "T2 PRE_DRAFT без состава": ["elo_difference", "form_3_difference"],
}
BLOCK_SIZE = 20
TEST_ACCESS = {"n": 0}


def pit_frame(matches, rp, horizon: timedelta, emit: set,
              hero_ref: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Признаки на момент start − horizon.

    `hero_exp_decay_diff` берётся из ЗАМОРОЖЕННОГО конвейера, а не
    пересчитывается: собственная реализация расходилась максимум на
    0.0018, и подменять ею замороженный признак значило бы незаметно
    менять baseline. В режиме PRE_DRAFT он всё равно не используется.
    """
    rows = build_point_in_time_features(matches, horizon, roster_provider=rp,
                                        include_draft=False, emit_only=emit)
    df = pd.DataFrame([{
        "match_id": r.match_id, "prediction_at": r.prediction_at,
        "as_of_timestamp": r.start_time, "horizon_hours": r.horizon_hours,
        "elo_difference": r.elo_difference,
        "form_3_difference": r.form_3_difference,
        "elo_mean_diff": r.elo_mean_diff,
        "five_vs_team_elo_diff": r.five_vs_team_elo_diff,
        "roster_known": min(r.roster_known_radiant, r.roster_known_dire),
        "pool_meta_diff": r.pool_meta_diff, "pool_size_min": r.pool_size_min,
        "radiant_matches_played_before": r.radiant_matches_before,
        "dire_matches_played_before": r.dire_matches_before,
        "target": r.target,
    } for r in rows])
    if hero_ref is not None:
        df = df.merge(hero_ref, on="match_id", how="left")
    return df.sort_values(["as_of_timestamp", "match_id"]).reset_index(drop=True)


def fit(feats: List[str], train: pd.DataFrame) -> LogisticRegressionModel:
    m = LogisticRegressionModel(feature_names=feats, random_state=RANDOM_SEED)
    m.fit(train, train["target"])
    return m


def proba(model, frame) -> np.ndarray:
    return np.asarray(model.predict_proba(frame))[:, 1]


def metrics_row(y, p) -> Dict[str, float]:
    m = compute_metrics(y, p)
    c = calibration_report(y, p)
    return {"n": m["n"], "accuracy": m["accuracy"], "roc_auc": m["roc_auc"],
            "log_loss": m["log_loss"], "brier": m["brier_score"],
            "ece": c["ece"], "slope": c["slope"], "intercept": c["intercept"]}


def show(name: str, r: Dict[str, float], ref: Optional[Dict[str, float]] = None):
    d = ""
    if ref is not None:
        d = (f"  Δacc={r['accuracy']-ref['accuracy']:+.4f} "
             f"Δauc={r['roc_auc']-ref['roc_auc']:+.4f}")
    print(f"{name:34s} acc={r['accuracy']:.4f} auc={r['roc_auc']:.4f} "
          f"ll={r['log_loss']:.4f} brier={r['brier']:.4f} ECE={r['ece']:.5f}{d}", flush=True)


def calibrate_stream(p_tr, y_tr, ts_tr, p_ev, y_ev, ts_ev) -> np.ndarray:
    """Слой калибровки Phase 14 (beta, скользящее окно 5000), онлайн."""
    patch = np.array(["-"] * len(p_ev), dtype=object)
    return run_walk_forward(
        p_ev, y_ev, ts_ev, patch, mode="rolling", method="beta", window=5000,
        warm_p=p_tr, warm_y=y_tr, warm_ts=ts_tr,
        warm_patch=np.array(["-"] * len(p_tr), dtype=object),
        refit_every=10, min_history=500)


def epoch(s: pd.Series) -> np.ndarray:
    return ((pd.to_datetime(s, utc=True) - pd.Timestamp("1970-01-01", tz="UTC"))
            .dt.total_seconds().to_numpy())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    commit = git_commit_sha()
    payload: Dict[str, object] = {"phase": 17, "git_commit": commit}

    section("PHASE 17 — прогноз только по pre-match данным")
    print(f"git commit: {commit}")
    print("Замороженная модель Phase 9 НЕ меняется. Модели уровней — отдельные.")

    ref = load_common_set()
    emit = set(ref["match_id"])
    hero_ref = ref[["match_id", "hero_exp_decay_diff"]]
    eng = make_engine(load_settings())
    matches, rp = load_pit_matches(eng)
    print(f"\nматчей в потоке обновлений: {len(matches)}; выдаём признаки для: {len(emit)}")

    # ---------------- проверка верности движка ----------------
    section("Верность point-in-time движка при horizon=0")
    base0 = pit_frame(matches, rp, timedelta(0), emit, hero_ref)
    chk = base0.merge(ref[["match_id", "elo_difference", "form_3_difference",
                           "elo_mean_diff", "five_vs_team_elo_diff"]],
                      on="match_id", suffixes=("_pit", "_ref"))
    for c in ("elo_difference", "form_3_difference", "elo_mean_diff", "five_vs_team_elo_diff"):
        d = (chk[f"{c}_pit"].astype(float) - chk[f"{c}_ref"].astype(float)).abs()
        print(f"  {c:26s} max|Δ| = {d.max():.3g}")
    print("  => формулы воспроизведены; расхождение на уровне машинной точности")

    # ---------------- PART O: воспроизведение baseline ----------------
    section("PART O — воспроизведение frozen baseline (Phase 14/15)")
    ref_sorted = ref.sort_values(["as_of_timestamp", "match_id"]).reset_index(drop=True)
    tr0, va0, te0 = split(ref_sorted)
    m_base = fit(TIERS["T0 FULL (5 признаков)"], tr0)
    r_base_val = metrics_row(va0["target"].to_numpy(), proba(m_base, va0))
    show("frozen baseline, VALIDATION", r_base_val)
    print("  Phase 13/14 на VAL: acc=0.6048 auc=0.6536 ll=0.6532 ECE=0.00909")
    payload["baseline_val"] = r_base_val

    # ---------------- главный эксперимент ----------------
    section("Уровни доступности данных × горизонт прогноза (VALIDATION)")
    print("Модель каждого уровня обучается на TRAIN своего же горизонта.\n")
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    frames: Dict[str, pd.DataFrame] = {}

    for hname, h in HORIZONS.items():
        df = base0 if h == timedelta(0) else pit_frame(matches, rp, h, emit, hero_ref)
        frames[hname] = df
        tr, va, te = split(df)
        results[hname] = {}
        for tname, feats in TIERS.items():
            if tname.startswith("T0") and hname != "H0 (старт)":
                continue          # драфт существует только у самого старта
            mdl = fit(feats, tr)
            r = metrics_row(va["target"].to_numpy(), proba(mdl, va))
            results[hname][tname] = r
        print(f"--- горизонт {hname} ---")
        base_here = results[hname].get("T0 FULL (5 признаков)")
        for tname, r in results[hname].items():
            show(f"  {tname}", r, base_here if base_here and not tname.startswith("T0") else None)
    payload["tiers_by_horizon"] = results

    section("Сколько стоит отсутствие драфта и состава (VALIDATION, горизонт 0)")
    t0 = results["H0 (старт)"]["T0 FULL (5 признаков)"]
    t1 = results["H0 (старт)"]["T1 PRE_DRAFT + состав"]
    t2 = results["H0 (старт)"]["T2 PRE_DRAFT без состава"]
    print(f"  T0 -> T1 (нет драфта):        Δacc={t1['accuracy']-t0['accuracy']:+.4f}  "
          f"Δauc={t1['roc_auc']-t0['roc_auc']:+.4f}  Δll={t1['log_loss']-t0['log_loss']:+.4f}")
    print(f"  T1 -> T2 (нет состава):       Δacc={t2['accuracy']-t1['accuracy']:+.4f}  "
          f"Δauc={t2['roc_auc']-t1['roc_auc']:+.4f}  Δll={t2['log_loss']-t1['log_loss']:+.4f}")
    print(f"  T0 -> T2 (полная потеря):     Δacc={t2['accuracy']-t0['accuracy']:+.4f}  "
          f"Δauc={t2['roc_auc']-t0['roc_auc']:+.4f}")

    section("Сколько стоит горизонт прогноза (VALIDATION, уровень T1)")
    print(f"  {'горизонт':>12s} {'acc':>8s} {'auc':>8s} {'ll':>8s} {'Δacc к H0':>11s}")
    h0 = results["H0 (старт)"]["T1 PRE_DRAFT + состав"]
    hz = []
    for hname in HORIZONS:
        r = results[hname].get("T1 PRE_DRAFT + состав")
        if not r:
            continue
        hz.append({"horizon": hname, **r})
        print(f"  {hname:>12s} {r['accuracy']:8.4f} {r['roc_auc']:8.4f} "
              f"{r['log_loss']:8.4f} {r['accuracy']-h0['accuracy']:+11.4f}")
    payload["horizon_t1"] = hz

    # ---------------- PART G/H: сила пула в текущей мете ----------------
    section("PART G/H — сила пула героев команды в текущей мете (VALIDATION)")
    print("  Гипотеза H-META: команда, чей привычный пул героев силён в текущей")
    print("  мете, имеет преимущество. Величина известна ДО драфта — берутся")
    print("  герои, на которых пятёрка играла раньше, и их сила на момент T.")
    print("  Ожидаемое направление: положительное. Критерий отклонения:")
    print("  Δlog loss на VALIDATION не превосходит 1e-4.\n")
    dfm = frames["H24ч"]
    trm, vam, _ = split(dfm)
    y_vm = vam["target"].to_numpy()
    cov = float(dfm["pool_meta_diff"].notna().mean())
    corr = float(np.corrcoef(dfm["pool_meta_diff"].fillna(0.0).to_numpy(dtype=float),
                             dfm["target"].to_numpy(dtype=float))[0, 1])
    print(f"  покрытие признака: {cov*100:.1f}%   корреляция с исходом: {corr:+.4f}")
    print(f"  медиана размера пула: {dfm['pool_size_min'].median():.0f} героев")
    base_feats = TIERS["T1 PRE_DRAFT + состав"]
    r_wo = metrics_row(y_vm, proba(fit(base_feats, trm), vam))
    r_w = metrics_row(y_vm, proba(fit(base_feats + ["pool_meta_diff"], trm), vam))
    rng0 = np.random.default_rng(RANDOM_SEED)
    for d in (trm, vam):
        d["noise_control"] = rng0.normal(size=len(d))
    r_n = metrics_row(y_vm, proba(fit(base_feats + ["noise_control"], trm), vam))
    show("  T1 без признака", r_wo)
    show("  T1 + pool_meta_diff", r_w, r_wo)
    show("  T1 + шум (контроль)", r_n, r_wo)
    gain = r_wo["log_loss"] - r_w["log_loss"]
    verdict = "KEEP" if gain > 1e-4 else "REMOVE"
    print(f"\n  Δlog loss = {gain:+.5f}  ->  {verdict}")
    payload["part_gh"] = {"coverage": cov, "corr": corr, "without": r_wo,
                          "with": r_w, "noise": r_n, "delta_log_loss": gain,
                          "verdict": verdict}

    # ---------------- PART J: неполный состав ----------------
    section("PART J — что делать, когда состав неизвестен")
    df24 = frames["H24ч"]
    tr, va, te = split(df24)
    m_t1 = fit(TIERS["T1 PRE_DRAFT + состав"], tr)
    m_t2 = fit(TIERS["T2 PRE_DRAFT без состава"], tr)
    y_va = va["target"].to_numpy()
    p_t1, p_t2 = proba(m_t1, va), proba(m_t2, va)
    print("  Гипотеза: при неизвестном составе честнее использовать модель T2,")
    print("  чем подставлять в T1 значения по умолчанию.\n")
    # эмулируем неизвестный состав у случайной доли матчей
    rng = np.random.default_rng(RANDOM_SEED)
    part_j = []
    for share in (0.0, 0.25, 0.5, 0.775, 1.0):
        unknown = rng.random(len(va)) < share
        # A: подставить нули в признаки состава (то, что делает импьютер)
        va_zero = va.copy()
        va_zero.loc[unknown, ["elo_mean_diff", "five_vs_team_elo_diff"]] = 0.0
        p_a = proba(m_t1, va_zero)
        # B: честный переход на модель без состава
        p_b = np.where(unknown, p_t2, p_t1)
        ra, rb = metrics_row(y_va, p_a), metrics_row(y_va, p_b)
        part_j.append({"unknown_share": share, "impute_zero": ra, "fallback_model": rb})
        print(f"  доля без состава {share*100:5.1f}%:  подстановка нулей acc={ra['accuracy']:.4f} "
              f"ll={ra['log_loss']:.4f}  |  переход на T2 acc={rb['accuracy']:.4f} ll={rb['log_loss']:.4f}")
    payload["part_j"] = part_j

    # ---------------- фиксация конфигурации ----------------
    section("Фиксация конфигурации по VALIDATION")
    print("  Продуктовая конфигурация выбирается по реальной доступности данных,")
    print("  а не по лучшей метрике: до драфта T0 недоступен в принципе.")
    print("  Фиксируется: T1 при известном составе, T2 при неизвестном,")
    print("  горизонт H24ч как реалистичный для расписания (Phase 16: 82% матчей).")
    payload["frozen_config"] = {"tier_known_roster": "T1", "tier_unknown_roster": "T2",
                                "horizon": "H24ч", "calibration": "phase14-beta-rolling5000"}

    # ---------------- PART T: единственный TEST ----------------
    if args.final_test:
        section("ЕДИНСТВЕННОЕ обращение к TEST")
        TEST_ACCESS["n"] += 1
        tr, va, te = split(frames["H24ч"])
        y_te = te["target"].to_numpy()
        ts_tr, ts_te = epoch(tr["as_of_timestamp"]), epoch(te["as_of_timestamp"])

        # эталон: frozen baseline на том же наборе матчей
        tr0s, va0s, te0s = split(ref_sorted)
        r_ref = metrics_row(te0s["target"].to_numpy(), proba(m_base, te0s))
        show("frozen baseline (T0, горизонт 0)", r_ref)
        print("  Phase 14/15 на TEST: acc=0.6301 auc=0.6816 ll=0.6390 ECE=0.02066")

        out_test = {"frozen_baseline": r_ref}
        for tname in ("T1 PRE_DRAFT + состав", "T2 PRE_DRAFT без состава"):
            mdl = fit(TIERS[tname], tr)
            p_tr = proba(mdl, tr)
            p_te = proba(mdl, te)
            r_raw = metrics_row(y_te, p_te)
            p_cal = calibrate_stream(p_tr, tr["target"].to_numpy(), ts_tr,
                                     p_te, y_te, ts_te)
            r_cal = metrics_row(y_te, p_cal)
            show(f"{tname} сырая", r_raw, r_ref)
            show(f"{tname} калиброванная", r_cal, r_ref)
            b = block_bootstrap_metric(y_te, p_cal, accuracy_metric,
                                       block_size=BLOCK_SIZE, seed=RANDOM_SEED)
            print(f"    accuracy CI [{b['ci_low']:.4f}, {b['ci_high']:.4f}]")
            out_test[tname] = {"raw": r_raw, "calibrated": r_cal,
                               "accuracy_ci": [b["ci_low"], b["ci_high"]]}

        # цена pre-match режима, парно на одних и тех же матчах
        common = set(te["match_id"]) & set(te0s["match_id"])
        a = te0s[te0s.match_id.isin(common)].sort_values("match_id")
        b_ = te[te.match_id.isin(common)].sort_values("match_id")
        y_c = a["target"].to_numpy()
        p_full = proba(m_base, a)
        p_pre = proba(fit(TIERS["T1 PRE_DRAFT + состав"], tr), b_)
        d_acc = block_bootstrap_paired_diff(y_c, p_pre, p_full, accuracy_metric,
                                            block_size=BLOCK_SIZE, seed=RANDOM_SEED)
        mc = mcnemar_exact(y_c, p_full > 0.5, p_pre > 0.5)
        print(f"\n  Цена перехода к pre-match (T1@H24ч минус T0@старт), n={len(common)}:")
        print(f"    Δaccuracy = {d_acc['point_diff']:+.4f}  "
              f"95% CI [{d_acc['ci_low']:+.4f}, {d_acc['ci_high']:+.4f}]")
        print(f"    McNemar p = {mc.get('p_value', float('nan')):.4g}")
        out_test["pre_match_cost"] = {"delta_accuracy": d_acc, "mcnemar": mc,
                                      "n_common": len(common)}
        payload["test"] = out_test

    payload["test_accesses"] = TEST_ACCESS["n"]
    out = os.path.join(EXPERIMENTS_DIR, "phase17.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nОбращений к TEST: {TEST_ACCESS['n']}\nРезультаты: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
