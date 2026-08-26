#!/usr/bin/env python3
"""
PHASE 14 — калибровка во времени и онлайн-адаптация.

Протокол: все решения на VALIDATION; TEST — одно обращение, только с
флагом --final-test. Базовая модель заморожена и не меняется.

Главная цель — не поднять accuracy, а сделать вероятность стабильно
калиброванной во времени.

Запуск:
    python3 scripts/phase14_pipeline.py                 # только VALIDATION
    python3 scripts/phase14_pipeline.py --final-test    # + единственный TEST
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.config import load_settings
from src.db.engine import make_engine
from src.evaluation.calibration import calibration_report, reliability_curve
from src.evaluation.metrics import compute_metrics
from src.evaluation.online_calibration import run_walk_forward, rolling_drift_metrics
from src.evaluation.statistics import (
    accuracy_metric,
    block_bootstrap_metric,
    log_loss_metric,
)
from src.models.sklearn_models import RANDOM_SEED, CatBoostModel, LogisticRegressionModel
from scripts.phase6_pipeline import git_commit_sha, section
from scripts.phase13_dataset import PHASE9_FULL, load_or_build
from scripts.phase13_pipeline import auc, load_common_set, split

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
BLOCK_SIZE = 20
REFIT_EVERY = 10

WINDOWS = [500, 1000, 2000, 5000, 10000]
HALF_LIVES = [7, 14, 30, 60, 90, 180]
METHODS = ["platt", "temperature", "beta", "isotonic"]
TEST_ACCESS = {"n": 0}


def load_league_info(engine) -> pd.DataFrame:
    """tier и league_id нужны для контроля состава выборки (PART B).
    Загружаются отдельным запросом, чтобы не пересобирать кэш датасета."""
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT m.match_id, m.league_id, l.tier "
            "FROM matches m LEFT JOIN leagues l ON l.league_id = m.league_id")).fetchall()
    return pd.DataFrame([{"match_id": r.match_id, "league_id": r.league_id,
                          "tier": r.tier} for r in rows])


def fit_base(train: pd.DataFrame) -> LogisticRegressionModel:
    m = LogisticRegressionModel(feature_names=PHASE9_FULL, random_state=RANDOM_SEED)
    m.fit(train, train["target"])
    return m


def proba(model, frame) -> np.ndarray:
    return np.asarray(model.predict_proba(frame))[:, 1]


def metrics_row(y, p) -> Dict[str, float]:
    m = compute_metrics(y, p)
    c = calibration_report(y, p)
    return {"n": m["n"], "accuracy": m["accuracy"], "roc_auc": m["roc_auc"],
            "log_loss": m["log_loss"], "brier": m["brier_score"],
            "ece": c["ece"], "mce": c["mce"], "slope": c["slope"],
            "intercept": c["intercept"]}


def show(name: str, r: Dict[str, float], ref: Optional[Dict[str, float]] = None):
    d = ""
    if ref is not None:
        d = (f"  ΔECE={r['ece']-ref['ece']:+.5f} Δll={r['log_loss']-ref['log_loss']:+.5f} "
             f"Δacc={r['accuracy']-ref['accuracy']:+.4f}")
    print(f"{name:40s} acc={r['accuracy']:.4f} auc={r['roc_auc']:.4f} "
          f"ll={r['log_loss']:.4f} brier={r['brier']:.4f} ECE={r['ece']:.5f} "
          f"slope={r['slope']:.3f} int={r['intercept']:+.3f}{d}", flush=True)


def bucket_patch_age(days: np.ndarray) -> np.ndarray:
    out = np.full(len(days), "D31+", dtype=object)
    out[days < 31] = "D15-30"
    out[days < 15] = "D8-14"
    out[days < 8] = "D4-7"
    out[days < 4] = "D1-3"
    out[days < 1] = "D0"
    return out


BUCKET_ORDER = ["D0", "D1-3", "D4-7", "D8-14", "D15-30", "D31+"]


def stratum_weights(sub: pd.DataFrame, ref: pd.DataFrame, keys: List[str]) -> np.ndarray:
    """Веса, приводящие состав `sub` к составу `ref` по ключам `keys`.

    Это и есть ответ на вопрос «эффект патча или состав выборки»: если
    после приведения состава эффект исчезает, он был составом.
    """
    ref_share = ref.groupby(keys, observed=True).size() / len(ref)
    sub_share = sub.groupby(keys, observed=True).size() / len(sub)
    w = np.ones(len(sub))
    idx = pd.MultiIndex.from_frame(sub[keys]) if len(keys) > 1 else pd.Index(sub[keys[0]])
    for i, k in enumerate(idx):
        rs = ref_share.get(k, 0.0)
        ss = sub_share.get(k, 0.0)
        w[i] = (rs / ss) if ss > 0 else 0.0
    s = w.sum()
    return w * (len(sub) / s) if s > 0 else w


def weighted_calibration(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    """ECE, Brier и остаточное смещение со взвешиванием наблюдений."""
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    tot = w.sum()
    if tot <= 0:
        return {"n_eff": 0.0, "ece": float("nan"), "brier": float("nan"),
                "accuracy": float("nan"), "bias": float("nan")}
    edges = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        sel = (p >= edges[i]) & (p < edges[i + 1] if i < 9 else p <= edges[i + 1])
        ws = w[sel].sum()
        if ws <= 0:
            continue
        ece += ws / tot * abs((w[sel] * p[sel]).sum() / ws - (w[sel] * y[sel]).sum() / ws)
    return {
        "n_eff": float(tot ** 2 / np.sum(w ** 2)),
        "ece": float(ece),
        "brier": float((w * (p - y) ** 2).sum() / tot),
        "accuracy": float((w * ((p >= 0.5).astype(float) == y)).sum() / tot),
        "bias": float((w * (y - p)).sum() / tot),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    commit = git_commit_sha()
    payload: Dict[str, object] = {"phase": 14, "git_commit": commit}

    section("PHASE 14 — калибровка во времени и онлайн-адаптация")
    print(f"git commit: {commit}")
    print("Базовая модель ЗАМОРОЖЕНА (5 входов Phase 9). Все решения — на VALIDATION.")

    df = load_common_set()
    engine = make_engine(load_settings())
    df = df.merge(load_league_info(engine), on="match_id", how="left")
    # ВНИМАНИЕ: .astype("int64") на datetime зависит от разрешения столбца.
    # После чтения кэша оно микросекундное, и деление на 1e9 давало время в
    # 1000 раз меньше настоящего — все разности дат схлопывались почти в ноль,
    # а экспоненциальное затухание переставало работать (веса при half-life
    # 7 дней выходили практически единичными). Разность от эпохи от
    # разрешения не зависит.
    df["ts"] = (pd.to_datetime(df["as_of_timestamp"], utc=True)
                - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    df["patch_bucket"] = bucket_patch_age(df["days_since_patch"].to_numpy(dtype=float))
    df["elo_decile"] = pd.qcut(df["elo_difference"].abs().rank(method="first"),
                               10, labels=False)
    df["tier"] = df["tier"].fillna("unknown")
    train, val, test = split(df)
    print(f"\nСтрок: {len(df)}   TRAIN {len(train)} | VAL {len(val)} | TEST {len(test)}")

    base = fit_base(train)
    p_tr = proba(base, train)
    p_val = proba(base, val)
    y_tr = train["target"].to_numpy()
    y_val = val["target"].to_numpy()

    # =================== PART A ===================
    section("PART A — воспроизведение Phase 13")
    r_raw_val = metrics_row(y_val, p_val)
    show("Сырая модель, VALIDATION", r_raw_val)
    print("  Phase 13 на VAL: acc=0.6048 ll=0.6532 brier=0.2311 ECE=0.00909 "
          "slope=0.9932 int=+0.0195")
    payload["part_a_val"] = r_raw_val

    # =================== PART B ===================
    section("PART B — калибровка как функция возраста патча")
    print("  1) СЫРОЕ сравнение (как в Phase 13):")
    print(f"  {'окно':>8s} {'n':>7s} {'acc':>8s} {'ll':>8s} {'brier':>8s} "
          f"{'ECE':>8s} {'slope':>8s} {'int':>8s}")
    part_b_raw = []
    for b in BUCKET_ORDER:
        sel = (val["patch_bucket"] == b).to_numpy()
        if sel.sum() < 100:
            continue
        r = metrics_row(y_val[sel], p_val[sel])
        part_b_raw.append({"bucket": b, **r})
        print(f"  {b:>8s} {r['n']:7d} {r['accuracy']:8.4f} {r['log_loss']:8.4f} "
              f"{r['brier']:8.4f} {r['ece']:8.5f} {r['slope']:8.3f} {r['intercept']:+8.3f}")

    print("\n  2) СОСТАВ выборки по окнам — то, что могло всё объяснить:")
    print(f"  {'окно':>8s} {'|Elo|':>8s} {'фавориты':>9s} {'premium':>8s} "
          f"{'team_min':>9s} {'новых команд':>13s} {'матчей/день':>12s}")
    part_b_comp = []
    for b in BUCKET_ORDER:
        s = val[val["patch_bucket"] == b]
        if len(s) < 100:
            continue
        days = max(1.0, (s["ts"].max() - s["ts"].min()) / 86400.0)
        comp = {"bucket": b, "n": len(s),
                "abs_elo": float(s["elo_difference"].abs().mean()),
                "favorite_rate": float((np.abs(p_val[(val["patch_bucket"] == b).to_numpy()] - 0.5) > 0.15).mean()),
                "premium_share": float((s["tier"] == "premium").mean()),
                "team_matches_min": float(s["team_matches_min"].mean()),
                "new_team_share": float((s["team_matches_min"] < 5).mean()),
                "matches_per_day": float(len(s) / days)}
        part_b_comp.append(comp)
        print(f"  {b:>8s} {comp['abs_elo']:8.1f} {comp['favorite_rate']*100:8.1f}% "
              f"{comp['premium_share']*100:7.1f}% {comp['team_matches_min']:9.1f} "
              f"{comp['new_team_share']*100:12.1f}% {comp['matches_per_day']:12.1f}")

    print("\n  3) ПОСЛЕ приведения состава к остальному периоду")
    print("     (перевзвешивание по дециль |Elo| x tier):")
    ref = val[val["patch_bucket"] == "D31+"]
    print(f"  {'окно':>8s} {'n_эфф':>8s} {'acc':>8s} {'brier':>8s} {'ECE':>8s} {'смещение':>10s}")
    part_b_w = []
    r_ref = weighted_calibration(y_val[(val["patch_bucket"] == "D31+").to_numpy()],
                                 p_val[(val["patch_bucket"] == "D31+").to_numpy()],
                                 np.ones(len(ref)))
    part_b_w.append({"bucket": "D31+ (эталон)", **r_ref})
    print(f"  {'D31+':>8s} {r_ref['n_eff']:8.0f} {r_ref['accuracy']:8.4f} "
          f"{r_ref['brier']:8.4f} {r_ref['ece']:8.5f} {r_ref['bias']:+10.4f}")
    for b in BUCKET_ORDER[:-1]:
        m = (val["patch_bucket"] == b).to_numpy()
        if m.sum() < 100:
            continue
        sub = val[m]
        w = stratum_weights(sub, ref, ["elo_decile", "tier"])
        r = weighted_calibration(y_val[m], p_val[m], w)
        part_b_w.append({"bucket": b, **r})
        print(f"  {b:>8s} {r['n_eff']:8.0f} {r['accuracy']:8.4f} {r['brier']:8.4f} "
              f"{r['ece']:8.5f} {r['bias']:+10.4f}")
    payload["part_b"] = {"raw": part_b_raw, "composition": part_b_comp,
                         "reweighted": part_b_w}

    # =================== PART C/D/E/M ===================
    section("PART C/D/E/M — сравнение слоёв калибровки (VALIDATION)")
    warm = dict(warm_p=p_tr, warm_y=y_tr, warm_ts=train["ts"].to_numpy(),
                warm_patch=train["patch_name"].to_numpy(dtype=object))
    ev = (p_val, y_val, val["ts"].to_numpy(), val["patch_name"].to_numpy(dtype=object))

    configs: List[Tuple[str, dict]] = [("A: без калибровки", {"mode": "none"})]
    for meth in METHODS:
        configs.append((f"B: глобальная / {meth}", {"mode": "frozen", "method": meth}))
    for meth in METHODS:
        for w_ in WINDOWS:
            configs.append((f"C: окно {w_} / {meth}", {"mode": "rolling", "method": meth, "window": w_}))
    for meth in METHODS:
        for hl in HALF_LIVES:
            configs.append((f"E: затухание {hl}д / {meth}", {"mode": "decay", "method": meth, "half_life": hl}))
    for meth in METHODS:
        configs.append((f"D: патч-локальная / {meth}", {"mode": "patch_local", "method": meth, "half_life": 60}))

    results: Dict[str, Dict[str, float]] = {}
    calibrated: Dict[str, np.ndarray] = {}
    for name, kw in configs:
        pc = run_walk_forward(*ev, refit_every=REFIT_EVERY, **warm, **kw)
        results[name] = metrics_row(y_val, pc)
        calibrated[name] = pc
    ref_a = results["A: без калибровки"]

    print("  Лучшие 12 конфигураций по ECE на VALIDATION:")
    for name in sorted(results, key=lambda k: results[k]["ece"])[:12]:
        show(name, results[name], ref_a)
    print("\n  Худшие 3 (для полноты картины):")
    for name in sorted(results, key=lambda k: -results[k]["ece"])[:3]:
        show(name, results[name], ref_a)

    print("\n  Сводка по режимам (лучший представитель каждого):")
    for prefix in ("A:", "B:", "C:", "D:", "E:"):
        cand = [k for k in results if k.startswith(prefix)]
        best = min(cand, key=lambda k: results[k]["ece"])
        show(f"  лучший {prefix} {best.split('/')[-1].strip()}", results[best], ref_a)

    print("\n  Сводка по методам (лучший представитель каждого):")
    for meth in METHODS:
        cand = [k for k in results if k.endswith(meth)]
        best = min(cand, key=lambda k: results[k]["ece"])
        show(f"  {meth}", results[best], ref_a)

    # negative control: слой, обученный на перемешанных метках
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled_warm = dict(warm, warm_y=rng.permutation(y_tr))
    pc_neg = run_walk_forward(ev[0], rng.permutation(y_val), ev[2], ev[3],
                              mode="decay", method="platt", half_life=60,
                              refit_every=REFIT_EVERY, **shuffled_warm)
    results["NEG: перемешанные метки"] = metrics_row(y_val, pc_neg)
    print()
    show("NEG: перемешанные метки (контроль)", results["NEG: перемешанные метки"], ref_a)

    # Гибрид. PART F показывает разделение труда: патч-локальный слой лучше
    # в первые дни патча, скользящее окно — в стабильный период. Правило
    # переключения использует ТОЛЬКО возраст патча, который известен до
    # матча, поэтому гибрид не нарушает walk-forward.
    best_pure = min((k for k in results if not k.startswith(("A:", "NEG"))),
                    key=lambda k: results[k]["ece"])
    HYBRID_DAYS = 14
    hyb_src = "D: патч-локальная / platt"
    if hyb_src in calibrated:
        young = (val["days_since_patch"].to_numpy(dtype=float) < HYBRID_DAYS)
        p_hyb = np.where(young, calibrated[hyb_src], calibrated[best_pure])
        results[f"F: гибрид (<{HYBRID_DAYS}д патч-лок., иначе {best_pure})"] = metrics_row(y_val, p_hyb)
        calibrated[f"F: гибрид (<{HYBRID_DAYS}д патч-лок., иначе {best_pure})"] = p_hyb
        print()
        show(f"F: гибрид (<{HYBRID_DAYS}д патч-локальная)",
             results[f"F: гибрид (<{HYBRID_DAYS}д патч-лок., иначе {best_pure})"], ref_a)
        print(f"    доля матчей под патч-локальным слоем: {young.mean()*100:.1f}%")

    best_name = min((k for k in results if not k.startswith(("A:", "NEG"))),
                    key=lambda k: results[k]["ece"])
    is_hybrid = best_name.startswith("F:")
    best_kw = ({} if is_hybrid
               else dict(next(kw for n, kw in configs if n == best_name)))
    print(f"\n  >>> ВЫБРАНО по VALIDATION (минимум ECE): {best_name}")
    payload["part_cdem"] = {"all": results, "best": best_name, "best_kw": best_kw}

    # честная проверка выбранной конфигурации без разрежения пересчёта
    if not is_hybrid:
        pc_exact = run_walk_forward(*ev, refit_every=1, **warm, **best_kw)
        r_exact = metrics_row(y_val, pc_exact)
        print("  Проверка выбранной конфигурации при пересчёте после КАЖДОГО матча:")
        show("    refit_every=1", r_exact, results[best_name])
        payload["part_cdem"]["exact_refit"] = r_exact
    payload["part_cdem"]["is_hybrid"] = is_hybrid
    payload["part_cdem"]["best_pure"] = best_pure
    p_cal_val = calibrated[best_name]

    # =================== PART F ===================
    section("PART F — патч-осведомлённая калибровка")
    print("  Гипотеза задания: патч меняет не сигнал, а ОТОБРАЖЕНИЕ")
    print("  сырой вероятности в фактическую. Проверка — калибровка по окнам")
    print("  патча ПОСЛЕ применения выбранного слоя:")
    # Сравниваются несколько ведущих вариантов, а не только победитель по
    # общему ECE: вопрос фазы — чинится ли ИМЕННО переход патча, и здесь
    # победитель в среднем может проигрывать патч-локальному слою.
    f_variants = {"сырая": p_val, f"победитель ({best_name})": p_cal_val}
    for nm in ("D: патч-локальная / platt", "E: затухание 30д / platt",
               "B: глобальная / platt"):
        if nm in calibrated:
            f_variants[nm] = calibrated[nm]
    part_f = []
    header = "  " + f"{'окно':>8s} {'n':>7s}" + "".join(
        f" {k[:22]:>22s}" for k in f_variants)
    print(header)
    for b in BUCKET_ORDER:
        sel = (val["patch_bucket"] == b).to_numpy()
        if sel.sum() < 100:
            continue
        row = {"bucket": b, "n": int(sel.sum())}
        line = f"  {b:>8s} {sel.sum():7d}"
        for k, pp in f_variants.items():
            r = calibration_report(y_val[sel], pp[sel])
            row[f"ece::{k}"] = r["ece"]
            row[f"slope::{k}"] = r["slope"]
            line += f" {r['ece']:10.5f}/{r['slope']:<11.3f}"
        part_f.append(row)
        print(line)
    print("  (в каждой ячейке: ECE / calibration slope)")
    payload["part_f"] = part_f

    # =================== PART G ===================
    section("PART G — детектор дрейфа калибровки (только прошлые матчи)")
    drift = rolling_drift_metrics(p_val, y_val, window=2000, step=500)
    print(f"  {'до матча №':>11s} {'ECE':>8s} {'brier':>8s} {'slope':>8s} {'смещение':>10s}")
    for d in drift:
        print(f"  {d['end_index']:11d} {d['ece']:8.5f} {d['brier']:8.4f} "
              f"{d['slope']:8.3f} {d['residual_bias']:+10.4f}")
    eces = [d["ece"] for d in drift]
    print(f"\n  Размах скользящего ECE: {min(eces):.5f} .. {max(eces):.5f}")
    slopes = [d["slope"] for d in drift]
    print(f"  Размах скользящего наклона: {min(slopes):.3f} .. {max(slopes):.3f}")
    payload["part_g"] = drift

    # =================== PART I ===================
    section("PART I — рекалибровка против переобучения")
    print("  Вариант A: ничего не делать | B: только калибровка |")
    print("  C: полное переобучение на TRAIN+первая половина VAL |")
    print("  D: переобучение только hero-компоненты\n")
    half = len(val) // 2
    val_a, val_b = val.iloc[:half], val.iloc[half:]
    y_b = val_b["target"].to_numpy()
    p_b_raw = p_val[half:]

    opt = {}
    opt["A: ничего"] = metrics_row(y_b, p_b_raw)
    opt["B: только калибровка"] = metrics_row(y_b, p_cal_val[half:])

    retrain_src = pd.concat([train, val_a], ignore_index=True)
    m_full = fit_base(retrain_src)
    opt["C: полное переобучение"] = metrics_row(y_b, proba(m_full, val_b))

    # D: переобучается ТОЛЬКО hero-компонента. Коэффициенты остальных
    # четырёх признаков берутся из замороженной модели и входят как
    # смещение (offset); свободно подгоняются лишь свободный член и
    # коэффициент при hero_exp_decay_diff. Если бы здесь просто обучалась
    # полная модель, вариант D был бы копией C — а сравнение потеряло бы смысл.
    hero_idx = PHASE9_FULL.index("hero_exp_decay_diff")
    coef = base._clf.coef_.ravel()
    b0 = float(base._clf.intercept_[0])

    def offset_and_hero(frame):
        Xn = base._transform(frame, fit=False)
        hero = Xn[:, hero_idx]
        z_all = b0 + Xn @ coef
        return z_all - coef[hero_idx] * hero, hero

    off_tr, hero_tr = offset_and_hero(retrain_src)
    yy = retrain_src["target"].to_numpy(dtype=float)
    from src.evaluation.calibrators import _sigmoid, _weighted_logistic
    Xd = np.column_stack([np.ones_like(hero_tr), hero_tr])
    beta_d = np.zeros(2)
    for _ in range(100):
        eta = off_tr + Xd @ beta_d
        mu = _sigmoid(eta)
        w_ = np.clip(mu * (1 - mu), 1e-10, None)
        H = Xd.T @ (Xd * w_[:, None]) + 1e-8 * np.eye(2)
        step = np.linalg.solve(H, Xd.T @ (yy - mu))
        beta_d = beta_d + step
        if np.max(np.abs(step)) < 1e-10:
            break
    off_b, hero_b = offset_and_hero(val_b)
    p_d = _sigmoid(off_b + beta_d[0] + beta_d[1] * hero_b)
    opt["D: переобучение hero-части"] = metrics_row(y_b, p_d)
    print(f"  Коэффициент при hero_exp_decay_diff: {coef[hero_idx]:+.6f} (заморожен) -> "
          f"{beta_d[1]:+.6f} (переобучен)\n")

    for k, v in opt.items():
        show(k, v, opt["A: ничего"])
    payload["part_i"] = opt

    # =================== PART J ===================
    section("PART J — меняется ли ВЕЛИЧИНА hero-сигнала после патча")
    print("  Гипотеза задания: сигнал не исчезает, меняется его масштаб.")
    print("  Коэффициент при hero_exp_decay_diff оценивается отдельно внутри")
    print("  каждого режима меты — обучение только на TRAIN-части режима.\n")
    part_j = []
    print(f"  {'режим':>22s} {'n':>7s} {'коэфф hero':>12s} {'corr с исходом':>15s}")
    regimes = {
        "старая мета (D31+)": (train["patch_bucket"] == "D31+").to_numpy(),
        "переход (D0-7)": np.isin(train["patch_bucket"].to_numpy(), ["D0", "D1-3", "D4-7"]),
        "новая мета (D8-30)": np.isin(train["patch_bucket"].to_numpy(), ["D8-14", "D15-30"]),
    }
    for name, sel in regimes.items():
        sub = train[sel]
        if len(sub) < 500:
            continue
        m = LogisticRegressionModel(feature_names=PHASE9_FULL, random_state=RANDOM_SEED)
        m.fit(sub, sub["target"])
        c = float(m._clf.coef_.ravel()[PHASE9_FULL.index("hero_exp_decay_diff")])
        corr = float(np.corrcoef(sub["hero_exp_decay_diff"].to_numpy(dtype=float),
                                 sub["target"].to_numpy(dtype=float))[0, 1])
        part_j.append({"regime": name, "n": len(sub), "hero_coef": c, "corr": corr})
        print(f"  {name:>22s} {len(sub):7d} {c:+12.4f} {corr:+15.4f}")
    payload["part_j"] = part_j

    # =================== PART K ===================
    section("PART K — смена состава: есть ли дрейф калибровки после контроля")
    new_roster = (val["new_players_total"].to_numpy(dtype=float) >= 1)
    ref_stable = val[~new_roster]
    print(f"  {'группа':>26s} {'n/n_эфф':>9s} {'acc':>8s} {'brier':>8s} "
          f"{'ECE':>8s} {'смещение':>10s}")
    r_stable = weighted_calibration(y_val[~new_roster], p_val[~new_roster],
                                    np.ones(int((~new_roster).sum())))
    print(f"  {'стабильный состав':>26s} {r_stable['n_eff']:9.0f} {r_stable['accuracy']:8.4f} "
          f"{r_stable['brier']:8.4f} {r_stable['ece']:8.5f} {r_stable['bias']:+10.4f}")
    r_new_raw = weighted_calibration(y_val[new_roster], p_val[new_roster],
                                     np.ones(int(new_roster.sum())))
    print(f"  {'новый состав (сырое)':>26s} {r_new_raw['n_eff']:9.0f} {r_new_raw['accuracy']:8.4f} "
          f"{r_new_raw['brier']:8.4f} {r_new_raw['ece']:8.5f} {r_new_raw['bias']:+10.4f}")
    w_k = stratum_weights(val[new_roster], ref_stable, ["elo_decile", "tier"])
    r_new_w = weighted_calibration(y_val[new_roster], p_val[new_roster], w_k)
    print(f"  {'новый состав (состав ==)':>26s} {r_new_w['n_eff']:9.0f} {r_new_w['accuracy']:8.4f} "
          f"{r_new_w['brier']:8.4f} {r_new_w['ece']:8.5f} {r_new_w['bias']:+10.4f}")
    payload["part_k"] = {"stable": r_stable, "new_raw": r_new_raw, "new_reweighted": r_new_w}

    # =================== PART L ===================
    section("PART L — смещение на андердогах: откуда оно")
    und = p_val < 0.5
    print(f"  Сырая модель:  андердоги n={und.sum()}  смещение(y-p)="
          f"{np.mean(y_val[und] - p_val[und]):+.4f}  "
          f"фавориты смещение={np.mean(y_val[~und] - p_val[~und]):+.4f}")
    print(f"  После калибровки: андердоги смещение="
          f"{np.mean(y_val[und] - p_cal_val[und]):+.4f}  "
          f"фавориты={np.mean(y_val[~und] - p_cal_val[~und]):+.4f}")
    print("\n  Разрез смещения по силе разрыва (сырая модель):")
    print(f"  {'дециль |Elo|':>14s} {'n':>7s} {'смещение андердог':>18s} {'смещение фаворит':>18s}")
    part_l = []
    for d in range(10):
        m = (val["elo_decile"] == d).to_numpy()
        bu = float(np.mean(y_val[m & und] - p_val[m & und])) if (m & und).sum() > 30 else float("nan")
        bf = float(np.mean(y_val[m & ~und] - p_val[m & ~und])) if (m & ~und).sum() > 30 else float("nan")
        part_l.append({"decile": d + 1, "n": int(m.sum()), "bias_underdog": bu, "bias_favorite": bf})
        print(f"  {d+1:14d} {m.sum():7d} {bu:+18.4f} {bf:+18.4f}")
    print("\n  Разрез по tier:")
    for t in sorted(val["tier"].unique()):
        m = (val["tier"] == t).to_numpy()
        if m.sum() < 200:
            continue
        print(f"    {t:>14s} n={m.sum():6d}  смещение андердог="
              f"{np.mean(y_val[m & und] - p_val[m & und]):+.4f}  "
              f"фаворит={np.mean(y_val[m & ~und] - p_val[m & ~und]):+.4f}")
    payload["part_l"] = {"deciles": part_l,
                         "raw_underdog_bias": float(np.mean(y_val[und] - p_val[und])),
                         "cal_underdog_bias": float(np.mean(y_val[und] - p_cal_val[und]))}

    # =================== PART O/P ===================
    section("PART O/P — селективный прогноз и ранжирование уверенности")
    grid = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.25, 0.1]

    def cov_curve(p_used, p_score):
        rows = []
        sc = np.abs(p_score - 0.5)
        for cov in grid:
            thr = float(np.quantile(sc, 1 - cov)) if cov < 1 else float(sc.min() - 1)
            sel = sc >= thr
            r = calibration_report(y_val[sel], p_used[sel])
            rows.append({"coverage": float(sel.mean()), "n": r["n"],
                         "accuracy": r["accuracy"], "log_loss": r["log_loss"],
                         "brier": r["brier"], "ece": r["ece"]})
        return rows

    raw_curve = cov_curve(p_val, p_val)
    cal_curve = cov_curve(p_cal_val, p_cal_val)
    print(f"  {'покрытие':>9s} | {'СЫРАЯ acc':>10s} {'ECE':>8s} | "
          f"{'КАЛИБР acc':>11s} {'ECE':>8s}")
    for a, b in zip(raw_curve, cal_curve):
        print(f"  {a['coverage']*100:8.1f}% | {a['accuracy']:10.4f} {a['ece']:8.5f} | "
              f"{b['accuracy']:11.4f} {b['ece']:8.5f}")
    print(f"\n  PART P — ранжирование уверенности:")
    err = ((p_val > 0.5).astype(int) != y_val).astype(int)
    print(f"    AUC |p-0.5| как предиктора ошибки, сырая:     "
          f"{auc(err, -np.abs(p_val - 0.5)):.4f}")
    print(f"    то же после калибровки:                        "
          f"{auc(err, -np.abs(p_cal_val - 0.5)):.4f}")
    print("    Онлайн-слой меняется во времени, поэтому он монотонен для")
    print("    КАЖДОГО момента, но не как единое преобразование всей выборки —")
    print("    отсюда крошечное расхождение AUC. Заметное расхождение означало")
    print("    бы ошибку; проверка неизменности AUC при замороженном слое —")
    print("    в tests/calibration.")
    payload["part_op"] = {"raw": raw_curve, "calibrated": cal_curve}

    # =================== PART R ===================
    section("PART R — режимы отказа: что видно ДО матча")
    conf = np.abs(p_cal_val - 0.5)
    err_cal = ((p_cal_val > 0.5).astype(int) != y_val).astype(int)
    modes_r = {
        "high confidence + wrong": (conf > 0.15) & (err_cal == 1),
        "low confidence + correct": (conf < 0.03) & (err_cal == 0),
        "первые 7 дней патча": np.isin(val["patch_bucket"].to_numpy(), ["D0", "D1-3", "D4-7"]),
        "новый состав": new_roster,
    }
    print(f"  {'режим':>26s} {'n':>7s} {'доля':>7s} {'виден до матча?':>17s}")
    known_before = {"high confidence + wrong": "нет (нужен исход)",
                    "low confidence + correct": "нет (нужен исход)",
                    "первые 7 дней патча": "ДА (дата патча)",
                    "новый состав": "ДА (заявка)"}
    part_r = []
    for k, sel in modes_r.items():
        sel = np.asarray(sel)
        part_r.append({"mode": k, "n": int(sel.sum()), "share": float(sel.mean()),
                       "known_before": known_before[k]})
        print(f"  {k:>26s} {sel.sum():7d} {sel.mean()*100:6.1f}% {known_before[k]:>17s}")
    payload["part_r"] = part_r

    # =================== PART Q ===================
    section("PART Q — симуляция продакшена")
    print("  Полный проход VALIDATION в режиме потока: признаки уже посчитаны")
    print("  walk-forward, модель заморожена, слой калибровки обновляется")
    print("  после каждого исхода. Ниже — сводка по кварталам потока.\n")
    q = np.array_split(np.arange(len(val)), 4)
    print(f"  {'четверть':>10s} {'n':>7s} {'acc':>8s} {'ECE сырая':>10s} "
          f"{'ECE калибр.':>12s} {'slope сырая':>12s} {'slope калибр.':>14s}")
    part_q = []
    for i, idx in enumerate(q, 1):
        r0 = calibration_report(y_val[idx], p_val[idx])
        r1 = calibration_report(y_val[idx], p_cal_val[idx])
        part_q.append({"quarter": i, "n": r0["n"], "ece_raw": r0["ece"], "ece_cal": r1["ece"],
                       "slope_raw": r0["slope"], "slope_cal": r1["slope"]})
        print(f"  {i:10d} {r0['n']:7d} {r1['accuracy']:8.4f} {r0['ece']:10.5f} "
              f"{r1['ece']:12.5f} {r0['slope']:12.3f} {r1['slope']:14.3f}")
    payload["part_q"] = part_q

    # =================== PART T — финальный TEST ===================
    if args.final_test:
        section("PART T — ЕДИНСТВЕННОЕ обращение к TEST")
        TEST_ACCESS["n"] += 1
        y_test = test["target"].to_numpy()
        p_test = proba(base, test)
        r_raw_t = metrics_row(y_test, p_test)
        show("BASELINE: сырая модель", r_raw_t)
        print("  Phase 13 сообщала acc=0.6301 auc=0.6816 ll=0.6390 ECE=0.02066 — воспроизведение.")

        # История слоя на момент TEST — TRAIN + VALIDATION, как в проде.
        warm_t = dict(warm_p=np.concatenate([p_tr, p_val]),
                      warm_y=np.concatenate([y_tr, y_val]),
                      warm_ts=np.concatenate([train["ts"].to_numpy(), val["ts"].to_numpy()]),
                      warm_patch=np.concatenate([train["patch_name"].to_numpy(dtype=object),
                                                 val["patch_name"].to_numpy(dtype=object)]))
        ev_t = (p_test, y_test, test["ts"].to_numpy(),
                test["patch_name"].to_numpy(dtype=object))
        if is_hybrid:
            kw_pure = dict(next(kw for n, kw in configs if n == best_pure))
            kw_patch = dict(next(kw for n, kw in configs if n == hyb_src))
            p_pure_t = run_walk_forward(*ev_t, refit_every=REFIT_EVERY, **warm_t, **kw_pure)
            p_patch_t = run_walk_forward(*ev_t, refit_every=REFIT_EVERY, **warm_t, **kw_patch)
            young_t = test["days_since_patch"].to_numpy(dtype=float) < HYBRID_DAYS
            p_cal_test = np.where(young_t, p_patch_t, p_pure_t)
        else:
            p_cal_test = run_walk_forward(*ev_t, refit_every=REFIT_EVERY, **warm_t, **best_kw)
        r_cal_t = metrics_row(y_test, p_cal_test)
        show(f"CANDIDATE: {best_name}", r_cal_t, r_raw_t)

        # Парные интервалы для ECE. Без них "ECE упал в 2.6 раза" —
        # точечная оценка без указания, отличима ли она от нуля.
        def _ece_metric(yy, pp):
            from src.evaluation.calibration import expected_calibration_error
            return expected_calibration_error(yy, pp)

        from src.evaluation.statistics import block_bootstrap_paired_diff
        d_ece, ci_ece = block_bootstrap_paired_diff(
            y_test, p_test, p_cal_test, _ece_metric,
            block_size=BLOCK_SIZE, seed=RANDOM_SEED)
        d_ll, ci_ll = block_bootstrap_paired_diff(
            y_test, p_test, p_cal_test, log_loss_metric,
            block_size=BLOCK_SIZE, seed=RANDOM_SEED)
        print(f"\n  Парный block bootstrap (калиброванная минус сырая):")
        print(f"    ΔECE      = {d_ece:+.5f}  95% CI [{ci_ece[0]:+.5f}, {ci_ece[1]:+.5f}]")
        print(f"    Δlog loss = {d_ll:+.5f}  95% CI [{ci_ll[0]:+.5f}, {ci_ll[1]:+.5f}]")
        print(f"    ECE: {'ЗНАЧИМО лучше' if ci_ece[1] < 0 else 'интервал включает 0'}")
        print(f"    log loss: {'значимо лучше' if ci_ll[1] < 0 else 'интервал включает 0'}")

        bt_raw = block_bootstrap_metric(y_test, p_test, log_loss_metric,
                                        block_size=BLOCK_SIZE, seed=RANDOM_SEED)
        bt_cal = block_bootstrap_metric(y_test, p_cal_test, log_loss_metric,
                                        block_size=BLOCK_SIZE, seed=RANDOM_SEED)
        print(f"\n  Log loss сырая:     {bt_raw['point']:.5f} CI [{bt_raw['ci_low']:.5f}, {bt_raw['ci_high']:.5f}]")
        print(f"  Log loss калибр.:   {bt_cal['point']:.5f} CI [{bt_cal['ci_low']:.5f}, {bt_cal['ci_high']:.5f}]")

        print("\n  Reliability curve на TEST после калибровки:")
        print(f"  {'бин':>12s} {'n':>7s} {'прогноз':>9s} {'факт':>8s} {'разрыв':>8s}")
        for b in reliability_curve(y_test, p_cal_test):
            print(f"  [{b.lo:.1f},{b.hi:.1f}){'':>2s} {b.n:7d} {b.mean_pred:9.4f} "
                  f"{b.frac_positive:8.4f} {b.gap:8.4f}")

        print("\n  По окнам патча на TEST:")
        tb = bucket_patch_age(test["days_since_patch"].to_numpy(dtype=float))
        print(f"  {'окно':>8s} {'n':>7s} {'ECE сырая':>10s} {'ECE калибр.':>12s} "
              f"{'slope сырая':>12s} {'slope калибр.':>14s}")
        part_t_patch = []
        for b in BUCKET_ORDER:
            sel = tb == b
            if sel.sum() < 100:
                continue
            r0 = calibration_report(y_test[sel], p_test[sel])
            r1 = calibration_report(y_test[sel], p_cal_test[sel])
            part_t_patch.append({"bucket": b, "n": r0["n"], "ece_raw": r0["ece"],
                                 "ece_cal": r1["ece"], "slope_raw": r0["slope"],
                                 "slope_cal": r1["slope"]})
            print(f"  {b:>8s} {r0['n']:7d} {r0['ece']:10.5f} {r1['ece']:12.5f} "
                  f"{r0['slope']:12.3f} {r1['slope']:14.3f}")

        print("\n  Селективный прогноз на TEST (сырая против калиброванной):")
        print(f"  {'покрытие':>9s} | {'СЫРАЯ acc':>10s} {'ECE':>8s} | {'КАЛИБР acc':>11s} {'ECE':>8s}")
        sel_t = []
        for cov in grid:
            sc_r = np.abs(p_test - 0.5)
            sc_c = np.abs(p_cal_test - 0.5)
            tr_ = float(np.quantile(sc_r, 1 - cov)) if cov < 1 else float(sc_r.min() - 1)
            tc_ = float(np.quantile(sc_c, 1 - cov)) if cov < 1 else float(sc_c.min() - 1)
            a = calibration_report(y_test[sc_r >= tr_], p_test[sc_r >= tr_])
            b_ = calibration_report(y_test[sc_c >= tc_], p_cal_test[sc_c >= tc_])
            sel_t.append({"coverage": cov, "raw_acc": a["accuracy"], "raw_ece": a["ece"],
                          "cal_acc": b_["accuracy"], "cal_ece": b_["ece"]})
            print(f"  {cov*100:8.1f}% | {a['accuracy']:10.4f} {a['ece']:8.5f} | "
                  f"{b_['accuracy']:11.4f} {b_['ece']:8.5f}")

        und_t = p_test < 0.5
        print(f"\n  Смещение на андердогах: сырая {np.mean(y_test[und_t]-p_test[und_t]):+.4f}"
              f"  ->  калиброванная {np.mean(y_test[und_t]-p_cal_test[und_t]):+.4f}")

        payload["test"] = {"raw": r_raw_t, "calibrated": r_cal_t,
                           "delta_ece": d_ece, "ci_ece": list(ci_ece),
                           "delta_log_loss": d_ll, "ci_log_loss": list(ci_ll),
                           "ll_ci_raw": bt_raw, "ll_ci_cal": bt_cal,
                           "patch_buckets": part_t_patch, "selective": sel_t,
                           "underdog_bias_raw": float(np.mean(y_test[und_t]-p_test[und_t])),
                           "underdog_bias_cal": float(np.mean(y_test[und_t]-p_cal_test[und_t]))}

    payload["test_accesses"] = TEST_ACCESS["n"]
    out = os.path.join(EXPERIMENTS_DIR, "phase14.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nОбращений к TEST: {TEST_ACCESS['n']}")
    print(f"Результаты: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
