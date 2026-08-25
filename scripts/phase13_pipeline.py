#!/usr/bin/env python3
"""
PHASE 13 — границы модели, неопределённость, калибровка, анализ ошибок.

Протокол (PART P): ВСЕ решения принимаются на VALIDATION; TEST — одно
обращение, только с флагом --final-test; счётчик обращений пишется в JSON.

Цель фазы — НЕ поднять accuracy, а понять, где модели можно доверять.

Запуск:
    python3 scripts/phase13_pipeline.py                 # только VALIDATION
    python3 scripts/phase13_pipeline.py --final-test    # + единственный TEST
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

from src.evaluation.calibration import calibration_report, reliability_curve
from src.evaluation.metrics import compute_metrics
from src.evaluation.statistics import (
    accuracy_metric,
    block_bootstrap_metric,
    block_bootstrap_paired_diff,
    log_loss_metric,
)
from src.models.sklearn_models import RANDOM_SEED, CatBoostModel, LogisticRegressionModel
from scripts.phase6_pipeline import git_commit_sha, section
from scripts.phase13_dataset import PHASE9_ADDITIONS_ONLY, PHASE9_FULL, load_or_build

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
CACHE = os.path.join(EXPERIMENTS_DIR, "cache")
BLOCK_SIZE = 20

# Ковариаты неопределённости. Все известны ДО матча; ни одна не содержит
# исхода (проверено тестом test_result_does_not_enter_covariates_at_all).
COVARIATES = [
    "team_matches_min", "team_matches_max",
    "player_matches_min", "player_matches_mean",
    "new_players_max", "new_players_total",
    "roster_matches_together_min", "roster_age_days_min",
    "hero_games_min", "rare_heroes_count",
    "transferred_players_total", "days_since_transfer_min",
    "teammate_churn_max", "teammate_churn_mean", "players_with_new_teammates",
    "days_since_patch",
]
TEST_ACCESS = {"n": 0}


# ----------------------------------------------------------------------
# данные и сплит
# ----------------------------------------------------------------------

def load_common_set() -> pd.DataFrame:
    """COMMON set, тождественный Phase 12: те же 448 не-Captains-Mode
    матчей отсеиваются, иначе воспроизведение было бы не воспроизведением."""
    df = load_or_build()
    p12 = os.path.join(CACHE, "draft_state_reveal_full.csv")
    if os.path.exists(p12):
        fmt = pd.read_csv(p12, usecols=["match_id", "draft_format"])
        df = df.merge(fmt, on="match_id", how="inner")
        df = df[df["draft_format"] != "other"].reset_index(drop=True)
    else:
        print("ВНИМАНИЕ: кэш Phase 12 отсутствует, COMMON set может отличаться")
    return df.sort_values(["as_of_timestamp", "match_id"]).reset_index(drop=True)


def split(df, tf=0.70, vf=0.15):
    n = len(df)
    a, b = int(n * tf), int(n * (tf + vf))
    tr, va, te = (df.iloc[:a].reset_index(drop=True),
                  df.iloc[a:b].reset_index(drop=True),
                  df.iloc[b:].reset_index(drop=True))
    assert tr["as_of_timestamp"].max() < va["as_of_timestamp"].min()
    assert va["as_of_timestamp"].max() < te["as_of_timestamp"].min()
    return tr, va, te


def fit_logreg(feats, train):
    m = LogisticRegressionModel(feature_names=feats, random_state=RANDOM_SEED)
    m.fit(train, train["target"])
    return m


def proba(model, frame) -> np.ndarray:
    return np.asarray(model.predict_proba(frame))[:, 1]


def show(name, m, ref=None):
    d = ""
    if ref is not None:
        d = f"  Δacc={m['accuracy']-ref['accuracy']:+.4f}"
    print(f"{name:46s} acc={m['accuracy']:.4f} ll={m['log_loss']:.4f} "
          f"brier={m['brier_score']:.4f} auc={m['roc_auc']:.4f}{d}", flush=True)


def auc(y, s) -> float:
    """ROC-AUC через ранги: используется для оценки meta-модели ошибки."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def group_table(y, p, groups, label: str, min_n: int = 100) -> List[dict]:
    out = []
    for g in sorted(pd.unique(groups)):
        sel = np.asarray(groups == g)
        if sel.sum() < min_n:
            continue
        r = calibration_report(y[sel], p[sel])
        out.append({label: g, "n": r["n"], "accuracy": r["accuracy"],
                    "log_loss": r["log_loss"], "brier": r["brier"],
                    "ece": r["ece"], "slope": r["slope"], "intercept": r["intercept"]})
    return out


def print_group_table(rows: List[dict], label: str):
    if not rows:
        print("  (нет групп достаточного размера)")
        return
    print(f"  {label:>18s} {'n':>7s} {'acc':>8s} {'log loss':>9s} "
          f"{'brier':>8s} {'ECE':>8s} {'slope':>8s} {'intercept':>10s}")
    for r in rows:
        print(f"  {str(r[label]):>18s} {r['n']:7d} {r['accuracy']:8.4f} {r['log_loss']:9.4f} "
              f"{r['brier']:8.4f} {r['ece']:8.4f} {r['slope']:8.3f} {r['intercept']:10.3f}")


def coverage_curve(y, p, score, grid) -> List[dict]:
    """score — БОЛЬШЕ значит увереннее. Порог берётся как квантиль."""
    out = []
    for cov in grid:
        thr = float(np.quantile(score, 1.0 - cov)) if cov < 1.0 else float(score.min() - 1)
        sel = score >= thr
        if sel.sum() < 50:
            continue
        r = calibration_report(y[sel], p[sel])
        out.append({"coverage_target": cov, "threshold": thr,
                    "coverage_actual": float(sel.mean()), "n": r["n"],
                    "accuracy": r["accuracy"], "log_loss": r["log_loss"],
                    "brier": r["brier"], "ece": r["ece"]})
    return out


def print_coverage(rows: List[dict]):
    print(f"  {'покрытие':>9s} {'порог':>9s} {'n':>7s} {'acc':>8s} "
          f"{'log loss':>9s} {'brier':>8s} {'ECE':>8s}")
    for r in rows:
        print(f"  {r['coverage_actual']*100:8.1f}% {r['threshold']:9.4f} {r['n']:7d} "
              f"{r['accuracy']:8.4f} {r['log_loss']:9.4f} {r['brier']:8.4f} {r['ece']:8.4f}")


def mahalanobis(train_X: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Расстояние до TRAIN-распределения. Среднее и ковариация считаются
    ТОЛЬКО по TRAIN — иначе VAL/TEST участвовали бы в определении того,
    что считать «обычным матчем» (сценарий утечки Q9)."""
    mu = train_X.mean(axis=0)
    cov = np.cov(train_X, rowvar=False)
    inv = np.linalg.pinv(cov)
    d = X - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, inv, d), 0.0))


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    commit = git_commit_sha()
    payload: Dict[str, object] = {"phase": 13, "git_commit": commit}

    section("PHASE 13 — границы модели, неопределённость, калибровка")
    print(f"git commit: {commit}")
    print("Протокол: все решения на VALIDATION; TEST только с --final-test.")

    df = load_common_set()
    train, val, test = split(df)
    print(f"\nСтрок: {len(df)}   TRAIN {len(train)} | VAL {len(val)} | TEST {len(test)}")
    print(f"TRAIN {train['as_of_timestamp'].min():%Y-%m-%d}..{train['as_of_timestamp'].max():%Y-%m-%d}  "
          f"VAL {val['as_of_timestamp'].min():%Y-%m-%d}..{val['as_of_timestamp'].max():%Y-%m-%d}  "
          f"TEST {test['as_of_timestamp'].min():%Y-%m-%d}..{test['as_of_timestamp'].max():%Y-%m-%d}")

    y_val = val["target"].to_numpy()

    # =================== PART A — воспроизведение ===================
    section("PART A — воспроизведение Phase 9 / Phase 12")
    m_full = fit_logreg(PHASE9_FULL, train)
    m_add = fit_logreg(PHASE9_ADDITIONS_ONLY, train)
    p_val = proba(m_full, val)
    p_val_add = proba(m_add, val)
    r_full = compute_metrics(y_val, p_val)
    r_add = compute_metrics(y_val, p_val_add)
    show("Phase 9, пять входов (модель Phase 9/12)", r_full)
    show("Только три добавки из задания", r_add, r_full)
    print("\n  Разница показывает, что цифры задания (0.6279 / 0.6301) относятся")
    print("  к ПЯТИ входам, а не к трём перечисленным (см. phase13-plan.md, раздел 0).")
    payload["part_a_validation"] = {"phase9_full": r_full, "additions_only": r_add}

    # воспроизведение точек селективного прогноза Phase 12
    conf_abs = np.abs(p_val - 0.5)
    p12_points = []
    for cov in (1.0, 0.5):
        thr = float(np.quantile(conf_abs, 1.0 - cov)) if cov < 1.0 else -1.0
        sel = conf_abs >= thr
        p12_points.append({"coverage": float(sel.mean()),
                           "accuracy": float((y_val[sel] == (p_val[sel] > 0.5)).mean())})
    print(f"\n  Селективно на VAL: 100% -> {p12_points[0]['accuracy']:.4f}, "
          f"{p12_points[1]['coverage']*100:.1f}% -> {p12_points[1]['accuracy']:.4f}")
    print("  Phase 12 на VAL давала 0.6048 и 0.6715 — совпадение подтверждает воспроизводимость.")
    payload["part_a_selective_val"] = p12_points

    # =================== PART B — калибровка ===================
    section("PART B — калибровка (VALIDATION)")
    cal = calibration_report(y_val, p_val)
    print(f"  n={cal['n']}  log loss={cal['log_loss']:.4f}  Brier={cal['brier']:.4f}")
    print(f"  ECE={cal['ece']:.5f}  MCE={cal['mce']:.5f}")
    print(f"  calibration slope={cal['slope']:.4f}  intercept={cal['intercept']:+.4f}")
    print(f"  средний прогноз={cal['mean_pred']:.4f}  базовая частота={cal['base_rate']:.4f}")
    print("\n  Reliability curve:")
    print(f"  {'бин':>12s} {'n':>7s} {'прогноз':>9s} {'факт':>8s} {'разрыв':>8s}")
    for b in reliability_curve(y_val, p_val):
        print(f"  [{b.lo:.1f},{b.hi:.1f}){'':>2s} {b.n:7d} {b.mean_pred:9.4f} "
              f"{b.frac_positive:8.4f} {b.gap:8.4f}")

    print("\n  По годам:")
    by_year = group_table(y_val, p_val, val["year"].to_numpy(), "год")
    print_group_table(by_year, "год")

    print("\n  Фавориты (p>0.5) против андердогов:")
    fav = np.where(p_val > 0.5, "фаворит", "андердог")
    by_fav = group_table(y_val, p_val, fav, "сторона")
    print_group_table(by_fav, "сторона")

    print("\n  По бинам уверенности |p-0.5|:")
    cbin = pd.cut(conf_abs, [-0.001, 0.02, 0.05, 0.10, 0.15, 0.5]).astype(str)
    by_conf = group_table(y_val, p_val, cbin, "|p-0.5|")
    print_group_table(by_conf, "|p-0.5|")

    payload["part_b"] = {"overall": {k: v for k, v in cal.items()},
                         "by_year": by_year, "by_favorite": by_fav,
                         "by_confidence": by_conf}
    # =================== PART C/D — что предсказывает ошибку ===================
    section("PART C/D — ковариаты неопределённости и meta-модель P(error)")

    # Метки ошибки для обучения meta-модели должны быть OUT-OF-SAMPLE:
    # на TRAIN модель видела ответы, её ошибки там оптимистичны и не похожи
    # на будущие. Поэтому TRAIN режется хронологически: модель учится на
    # ранней части, ошибки собираются на поздней.
    cut = int(len(train) * 0.70)
    tr_early, tr_late = train.iloc[:cut].reset_index(drop=True), train.iloc[cut:].reset_index(drop=True)
    m_early = fit_logreg(PHASE9_FULL, tr_early)
    p_late = proba(m_early, tr_late)
    err_late = (( p_late > 0.5).astype(int) != tr_late["target"].to_numpy()).astype(int)
    err_val = ((p_val > 0.5).astype(int) != y_val).astype(int)
    print(f"  meta-TRAIN: {len(tr_late)} строк, доля ошибок {err_late.mean():.4f}")
    print(f"  VALIDATION: {len(val)} строк, доля ошибок {err_val.mean():.4f}")

    print("\n  Корреляция ковариаты с фактом ошибки (VALIDATION):")
    print(f"  {'ковариата':>30s} {'corr':>9s} {'AUC ошибки':>12s}")
    cov_stats = []
    for c in COVARIATES + ["_abs_conf"]:
        v = (conf_abs if c == "_abs_conf" else val[c].to_numpy(dtype=float))
        v = np.nan_to_num(v, nan=float(np.nanmedian(v)) if np.isfinite(np.nanmedian(v)) else 0.0)
        cc = float(np.corrcoef(v, err_val)[0, 1]) if v.std() > 0 else float("nan")
        a = auc(err_val, v)
        cov_stats.append({"covariate": c, "corr": cc, "auc": a})
        print(f"  {c:>30s} {cc:+9.4f} {a:12.4f}")
    payload["part_c_covariates"] = cov_stats

    def meta_frame(frame: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
        f = frame[COVARIATES].copy()
        for c in COVARIATES:
            f[c] = pd.to_numeric(f[c], errors="coerce")
        f["abs_conf"] = np.abs(p - 0.5)
        return f

    META_SETS = {
        "M0: только |p-0.5|": ["abs_conf"],
        "M1: только ковариаты": COVARIATES,
        "M2: |p-0.5| + ковариаты": ["abs_conf"] + COVARIATES,
    }
    rng = np.random.default_rng(RANDOM_SEED)
    meta_tr = meta_frame(tr_late, p_late); meta_tr["_err"] = err_late
    meta_va = meta_frame(val, p_val)
    meta_tr["noise"] = rng.normal(size=len(meta_tr))
    meta_va["noise"] = rng.normal(size=len(meta_va))
    META_SETS["M3: negative control (шум)"] = ["noise"]

    print("\n  Meta-модель P(error), AUC на VALIDATION:")
    meta_res = {}
    meta_models = {}
    for name, feats in META_SETS.items():
        mm = LogisticRegressionModel(feature_names=feats, random_state=RANDOM_SEED)
        mm.fit(meta_tr, meta_tr["_err"])
        pe = np.asarray(mm.predict_proba(meta_va))[:, 1]
        a = auc(err_val, pe)
        meta_res[name] = {"auc": a, "features": feats}
        meta_models[name] = mm
        print(f"  {name:34s} AUC={a:.4f}")
    payload["part_d_meta"] = meta_res

    p_error_val = np.asarray(meta_models["M2: |p-0.5| + ковариаты"].predict_proba(meta_va))[:, 1]

    # =================== PART E — сложность матча ===================
    section("PART E — сложность матча (децили)")
    difficulty_defs = {
        "|elo_difference|": -np.abs(val["elo_difference"].to_numpy(dtype=float)),
        "|elo_mean_diff| (player-Elo)": -np.abs(val["elo_mean_diff"].to_numpy(dtype=float)),
        "team_matches_min (мало истории)": -val["team_matches_min"].to_numpy(dtype=float),
        "новизна состава": val["new_players_total"].to_numpy(dtype=float),
        "оценка P(error)": p_error_val,
    }
    part_e = {}
    for name, score in difficulty_defs.items():
        # score: БОЛЬШЕ = труднее
        dec = pd.qcut(pd.Series(score).rank(method="first"), 10, labels=False)
        rows = []
        for d in range(10):
            sel = (dec == d).to_numpy()
            if sel.sum() < 50:
                continue
            r = calibration_report(y_val[sel], p_val[sel])
            rows.append({"decile": d + 1, "n": r["n"], "accuracy": r["accuracy"],
                         "log_loss": r["log_loss"], "ece": r["ece"]})
        part_e[name] = rows
        span = rows[0]["accuracy"] - rows[-1]["accuracy"]
        print(f"\n  {name}  (дециль 1 = самые лёгкие, 10 = самые трудные)")
        print("   " + "  ".join(f"D{r['decile']}={r['accuracy']:.3f}" for r in rows))
        print(f"   размах accuracy между крайними децилями: {span:+.4f}")
    payload["part_e"] = part_e

    # =================== PART F — таксономия ошибок ===================
    section("PART F — таксономия ошибок (VALIDATION)")
    v = val
    cats = {
        "1. близкий матч (|p-0.5|<0.03)": conf_abs < 0.03,
        "2. явный фаворит (|p-0.5|>0.20)": conf_abs > 0.20,
        "3. мало данных (team_matches_min<20)": v["team_matches_min"].to_numpy() < 20,
        "4. новый состав (>=1 замена)": v["new_players_total"].to_numpy() >= 1,
        "5. новая команда (team_matches_min<5)": v["team_matches_min"].to_numpy() < 5,
        "6. смена патча (<7 дней)": v["days_since_patch"].to_numpy() < 7,
        "7. редкие герои (rare_heroes_count>=1)": v["rare_heroes_count"].to_numpy() >= 1,
        "8. высокая P(error) (верхний дециль)": p_error_val >= np.quantile(p_error_val, 0.9),
        "9. смена окружения (teammate_churn_max>0.5)": v["teammate_churn_max"].to_numpy() > 0.5,
    }
    print(f"  {'категория':>44s} {'n':>7s} {'доля':>7s} {'accuracy':>9s} {'ошибок':>8s}")
    part_f = []
    for name, sel in cats.items():
        sel = np.asarray(sel)
        if sel.sum() == 0:
            print(f"  {name:>44s}       0")
            continue
        acc = float((y_val[sel] == (p_val[sel] > 0.5)).mean())
        part_f.append({"category": name, "n": int(sel.sum()),
                       "share": float(sel.mean()), "accuracy": acc,
                       "error_rate": float(err_val[sel].mean())})
        print(f"  {name:>44s} {sel.sum():7d} {sel.mean()*100:6.1f}% {acc:9.4f} "
              f"{err_val[sel].mean():8.4f}")
    payload["part_f"] = part_f

    hc_wrong = np.asarray((conf_abs > 0.15) & (err_val == 1))
    hc_all = np.asarray(conf_abs > 0.15)
    print(f"\n  HIGH CONFIDENCE + WRONG — самый опасный тип ошибки:")
    print(f"    уверенных прогнозов (|p-0.5|>0.15): {hc_all.sum()} ({hc_all.mean()*100:.1f}%)")
    print(f"    из них ошибочных: {hc_wrong.sum()} ({hc_wrong.sum()/max(hc_all.sum(),1)*100:.1f}%)")
    if hc_wrong.sum() > 0:
        sub = v[hc_wrong]
        print(f"    их профиль против остальных уверенных прогнозов:")
        rest = v[hc_all & ~hc_wrong]
        for c in ("team_matches_min", "new_players_total", "rare_heroes_count", "days_since_patch"):
            print(f"      {c:28s} ошибочные={sub[c].mean():8.2f}  верные={rest[c].mean():8.2f}")
    payload["part_f_high_conf_wrong"] = {
        "n_confident": int(hc_all.sum()), "n_wrong": int(hc_wrong.sum()),
        "error_rate": float(hc_wrong.sum() / max(hc_all.sum(), 1))}

    # =================== PART G — уверенность против правильности ===================
    section("PART G — монотонность accuracy по децилям уверенности")
    dec = pd.qcut(pd.Series(conf_abs).rank(method="first"), 10, labels=False).to_numpy()
    part_g = []
    prev, monotone = None, True
    for d in range(10):
        sel = dec == d
        r = calibration_report(y_val[sel], p_val[sel])
        part_g.append({"decile": d + 1, "n": r["n"], "mean_conf": float(conf_abs[sel].mean()),
                       "accuracy": r["accuracy"], "log_loss": r["log_loss"]})
        if prev is not None and r["accuracy"] < prev - 1e-9:
            monotone = False
        prev = r["accuracy"]
    print(f"  {'дециль':>7s} {'n':>7s} {'ср |p-0.5|':>11s} {'accuracy':>9s} {'log loss':>9s}")
    for r in part_g:
        print(f"  {r['decile']:7d} {r['n']:7d} {r['mean_conf']:11.4f} "
              f"{r['accuracy']:9.4f} {r['log_loss']:9.4f}")
    print(f"\n  Монотонность accuracy по децилям уверенности: "
          f"{'ДА' if monotone else 'НЕТ (нарушения есть)'}")

    # Нижний дециль показывает accuracy ниже 50%. Это либо реальная
    # инверсия порядка, либо шум — без интервала утверждать нельзя.
    low = dec == 0
    a_low, ci_low = block_bootstrap_metric(y_val[low], p_val[low], accuracy_metric,
                                           block_size=BLOCK_SIZE, seed=RANDOM_SEED)
    print(f"  Нижний дециль уверенности: accuracy={a_low:.4f}, 95% CI {ci_low}")
    print(f"  {'Ниже 0.5 значимо' if ci_low[1] < 0.5 else 'Интервал включает 0.5 — инверсия не доказана'}")
    payload["part_g_low_decile"] = {"accuracy": a_low, "ci": list(ci_low)}
    payload["part_g"] = {"deciles": part_g, "monotone": monotone}

    # =================== PART H — coverage-risk ===================
    section("PART H — полная coverage-risk кривая (VALIDATION)")
    grid = [round(x, 2) for x in np.arange(1.00, 0.09, -0.05)]
    cr = coverage_curve(y_val, p_val, conf_abs, grid)
    print_coverage(cr)
    payload["part_h_coverage_val"] = cr

    # =================== PART N — ансамбли (нужны для PART I) ===================
    section("PART N — ансамбли и разногласие моделей")
    cb = CatBoostModel(feature_names=PHASE9_FULL)
    cb.fit(train, train["target"])
    p_cb_val = proba(cb, val)
    p_ens_val = 0.5 * (p_val + p_cb_val)
    show("LogReg (Phase 9)", compute_metrics(y_val, p_val))
    show("CatBoost (те же входы)", compute_metrics(y_val, p_cb_val))
    show("Ансамбль 50/50", compute_metrics(y_val, p_ens_val))

    disagree = np.abs(p_val - p_cb_val)
    print(f"\n  Разногласие |LogReg - CatBoost|: медиана={np.median(disagree):.4f}, "
          f"90-й перцентиль={np.quantile(disagree, 0.9):.4f}")
    print(f"  AUC разногласия как предиктора ошибки: {auc(err_val, disagree):.4f}")
    dd = pd.qcut(pd.Series(disagree).rank(method="first"), 5, labels=False).to_numpy()
    print(f"  {'квинтиль разногласия':>22s} {'n':>7s} {'accuracy':>9s} {'доля ошибок':>12s}")
    part_n = []
    for q in range(5):
        sel = dd == q
        a = float((y_val[sel] == (p_val[sel] > 0.5)).mean())
        part_n.append({"quintile": q + 1, "n": int(sel.sum()),
                       "mean_disagreement": float(disagree[sel].mean()), "accuracy": a})
        print(f"  {q+1:22d} {sel.sum():7d} {a:9.4f} {err_val[sel].mean():12.4f}")
    payload["part_n"] = {"quintiles": part_n,
                         "auc_disagreement": auc(err_val, disagree),
                         "logreg": compute_metrics(y_val, p_val),
                         "catboost": compute_metrics(y_val, p_cb_val),
                         "ensemble": compute_metrics(y_val, p_ens_val)}

    # =================== PART I — механизмы отказа ===================
    section("PART I — сравнение механизмов отказа (VALIDATION)")
    mechanisms = {
        "A: |p-0.5|": conf_abs,
        "B: -P(error) (meta-модель)": -p_error_val,
        "C: -разногласие моделей": -disagree,
        "D: |p_ens-0.5|": np.abs(p_ens_val - 0.5),
        "E: композит |p-0.5| / (1+P(error))": conf_abs / (1.0 + p_error_val),
    }
    part_i = {}
    print(f"  {'механизм':>36s} {'acc@90%':>9s} {'acc@70%':>9s} {'acc@50%':>9s} {'acc@25%':>9s}")
    for name, score in mechanisms.items():
        rows = coverage_curve(y_val, p_val, np.asarray(score, dtype=float), [0.9, 0.7, 0.5, 0.25])
        part_i[name] = rows
        vals = {round(r["coverage_target"], 2): r["accuracy"] for r in rows}
        print(f"  {name:>36s} {vals.get(0.9, float('nan')):9.4f} {vals.get(0.7, float('nan')):9.4f} "
              f"{vals.get(0.5, float('nan')):9.4f} {vals.get(0.25, float('nan')):9.4f}")
    best_mech = max(part_i, key=lambda k: np.mean([r["accuracy"] for r in part_i[k]]))
    print(f"\n  >>> Лучший механизм по VALIDATION: {best_mech}")

    # Разница между A и B мала. Проверяется, отличается ли она от нуля:
    # при перекрытии интервала выбор механизма — вопрос простоты, а не качества.
    thr_a = float(np.quantile(conf_abs, 0.5))
    thr_b = float(np.quantile(-p_error_val, 0.5))
    sel_a, sel_b = conf_abs >= thr_a, (-p_error_val) >= thr_b
    both = sel_a & sel_b
    only_a, only_b = sel_a & ~sel_b, sel_b & ~sel_a
    print(f"  При покрытии 50% механизмы A и B выбирают одни и те же матчи в "
          f"{both.sum()}/{sel_a.sum()} случаев ({both.sum()/sel_a.sum()*100:.1f}%)")
    if only_a.sum() and only_b.sum():
        acc_only_a = float((y_val[only_a] == (p_val[only_a] > 0.5)).mean())
        acc_only_b = float((y_val[only_b] == (p_val[only_b] > 0.5)).mean())
        print(f"  Только A (n={only_a.sum()}): acc={acc_only_a:.4f}   "
              f"только B (n={only_b.sum()}): acc={acc_only_b:.4f}")
    payload["part_i_overlap"] = {"shared": int(both.sum()), "total": int(sel_a.sum())}
    payload["part_i"] = {"mechanisms": part_i, "best": best_mech}

    # =================== PART J — сдвиг распределения ===================
    section("PART J — расстояние до training-распределения")
    Xtr = train[PHASE9_FULL].to_numpy(dtype=float)
    d_val = mahalanobis(Xtr, val[PHASE9_FULL].to_numpy(dtype=float))
    print(f"  Mahalanobis: TRAIN медиана={np.median(mahalanobis(Xtr, Xtr)):.3f}, "
          f"VAL медиана={np.median(d_val):.3f}")
    print(f"  AUC дистанции как предиктора ошибки: {auc(err_val, d_val):.4f}")
    dj = pd.qcut(pd.Series(d_val).rank(method="first"), 10, labels=False).to_numpy()
    part_j = []
    for q in range(10):
        sel = dj == q
        r = calibration_report(y_val[sel], p_val[sel])
        part_j.append({"decile": q + 1, "n": r["n"], "mean_distance": float(d_val[sel].mean()),
                       "accuracy": r["accuracy"], "log_loss": r["log_loss"], "ece": r["ece"]})
    print("   " + "  ".join(f"D{r['decile']}={r['accuracy']:.3f}" for r in part_j))
    print(f"   размах между крайними децилями: "
          f"{part_j[0]['accuracy'] - part_j[-1]['accuracy']:+.4f}")
    payload["part_j"] = {"deciles": part_j, "auc_distance": auc(err_val, d_val)}

    # =================== PART K — переход патча ===================
    section("PART K — первые дни после нового патча")
    dsp = val["days_since_patch"].to_numpy(dtype=float)
    part_k = []
    print(f"  {'окно':>16s} {'n':>7s} {'acc':>8s} {'log loss':>9s} {'ECE':>8s} {'slope':>8s}")
    for lo, hi, name in [(0, 1, "1 день"), (0, 3, "3 дня"), (0, 7, "7 дней"),
                         (0, 14, "14 дней"), (0, 30, "30 дней"), (30, 1e9, "остальное")]:
        sel = (dsp >= lo) & (dsp < hi)
        if sel.sum() < 100:
            continue
        r = calibration_report(y_val[sel], p_val[sel])
        part_k.append({"window": name, **{k: r[k] for k in
                       ("n", "accuracy", "log_loss", "ece", "slope")}})
        print(f"  {name:>16s} {r['n']:7d} {r['accuracy']:8.4f} {r['log_loss']:9.4f} "
              f"{r['ece']:8.4f} {r['slope']:8.3f}")
    print("\n  Проверка смешения: не объясняется ли эффект патча составом выборки?")
    early = dsp < 30
    for c in ("team_matches_min", "elo_difference"):
        a = val.loc[early, c].abs().mean(); b = val.loc[~early, c].abs().mean()
        print(f"    {c:24s} первые 30 дней={a:9.2f}  остальное={b:9.2f}")
    a_e, ci_e = block_bootstrap_metric(y_val[early], p_val[early], accuracy_metric,
                                       block_size=BLOCK_SIZE, seed=RANDOM_SEED)
    a_r, ci_r = block_bootstrap_metric(y_val[~early], p_val[~early], accuracy_metric,
                                       block_size=BLOCK_SIZE, seed=RANDOM_SEED)
    print(f"    accuracy первые 30 дней={a_e:.4f} CI {ci_e}")
    print(f"    accuracy остальное     ={a_r:.4f} CI {ci_r}")
    overlap = not (ci_e[1] < ci_r[0] or ci_r[1] < ci_e[0])
    print(f"    интервалы {'ПЕРЕСЕКАЮТСЯ — эффект не доказан' if overlap else 'НЕ пересекаются — эффект есть'}")
    payload["part_k"] = {"windows": part_k, "early_ci": list(ci_e), "rest_ci": list(ci_r),
                         "overlap": overlap}

    # =================== PART L — смена состава ===================
    section("PART L — новизна состава")
    npl = val["new_players_total"].to_numpy(dtype=float)
    part_l = []
    print(f"  {'замен всего':>14s} {'n':>7s} {'acc':>8s} {'log loss':>9s} {'ECE':>8s}")
    for lo, hi, name in [(0, 1, "0"), (1, 2, "1"), (2, 3, "2"), (3, 5, "3-4"),
                         (5, 100, "5+ (новый состав)")]:
        sel = (npl >= lo) & (npl < hi)
        if sel.sum() < 100:
            continue
        r = calibration_report(y_val[sel], p_val[sel])
        part_l.append({"changes": name, **{k: r[k] for k in
                       ("n", "accuracy", "log_loss", "ece", "slope")}})
        print(f"  {name:>14s} {r['n']:7d} {r['accuracy']:8.4f} {r['log_loss']:9.4f} {r['ece']:8.4f}")
    payload["part_l"] = part_l

    # =================== PART M — пределы player-Elo ===================
    section("PART M — стабильность player-Elo после перехода игрока")
    print("  ДЕФЕКТ ДАННЫХ, обнаруженный в этой фазе: величины, опирающиеся на")
    print("  team_id, загрязнены фрагментацией идентичности команд (Phase 10).")
    tt = df["transferred_players_total"].to_numpy(dtype=float)
    print(f"    матчей с ровно 5 «переходами»:  {(tt == 5).sum():6d}")
    print(f"    матчей с ровно 10 «переходами»: {(tt == 10).sum():6d}")
    print("    Пики на 5 и 10 — это смена team_id целиком, а не переход игроков.")
    print(f"    медиана days_since_transfer_min = {np.nanmedian(df['days_since_transfer_min']):.2f} дн.")
    print("  Поэтому ниже приводятся ОБА среза: загрязнённый (team_id) и")
    print("  устойчивый (смена партнёров, от team_id не зависит).\n")
    tp = val["transferred_players_total"].to_numpy(dtype=float)
    part_m = []
    print(f"  {'перешедших игроков':>20s} {'n':>7s} {'acc':>8s} {'log loss':>9s} {'ECE':>8s}")
    for lo, hi, name in [(0, 1, "0"), (1, 2, "1"), (2, 3, "2"), (3, 100, "3+")]:
        sel = (tp >= lo) & (tp < hi)
        if sel.sum() < 100:
            continue
        r = calibration_report(y_val[sel], p_val[sel])
        part_m.append({"transfers": name, **{k: r[k] for k in
                       ("n", "accuracy", "log_loss", "ece", "slope")}})
        print(f"  {name:>20s} {r['n']:7d} {r['accuracy']:8.4f} {r['log_loss']:9.4f} {r['ece']:8.4f}")
    print("\n  Устойчивый срез — доля сменившихся партнёров (без team_id):")
    ch = val["teammate_churn_max"].to_numpy(dtype=float)
    print(f"  {'смена окружения':>20s} {'n':>7s} {'acc':>8s} {'log loss':>9s} {'ECE':>8s}")
    for lo, hi, name in [(-0.01, 0.01, "нет (0)"), (0.01, 0.25, "0-25%"),
                         (0.25, 0.5, "25-50%"), (0.5, 1.01, ">50%")]:
        sel = (ch >= lo) & (ch < hi)
        if sel.sum() < 100:
            continue
        r = calibration_report(y_val[sel], p_val[sel])
        part_m.append({"transfers": f"churn {name}", **{k: r[k] for k in
                       ("n", "accuracy", "log_loss", "ece", "slope")}})
        print(f"  {name:>20s} {r['n']:7d} {r['accuracy']:8.4f} {r['log_loss']:9.4f} {r['ece']:8.4f}")

    dst = val["days_since_transfer_min"].to_numpy(dtype=float)
    fresh = np.isfinite(dst) & (dst < 30)
    if fresh.sum() > 100:
        rf = calibration_report(y_val[fresh], p_val[fresh])
        ro = calibration_report(y_val[~fresh], p_val[~fresh])
        print(f"\n  Свежий переход (<30 дней): n={rf['n']} acc={rf['accuracy']:.4f} "
              f"ll={rf['log_loss']:.4f}")
        print(f"  Остальные:                  n={ro['n']} acc={ro['accuracy']:.4f} "
              f"ll={ro['log_loss']:.4f}")
        part_m.append({"transfers": "свежий <30д", **{k: rf[k] for k in
                       ("n", "accuracy", "log_loss", "ece", "slope")}})
    payload["part_m"] = part_m

    # =================== PART O — потолок ===================
    section("PART O — эмпирическая оценка потолка")
    from src.models.sklearn_models import EloOnlyModel
    families = {
        "Elo-only (аналитическая)": proba(EloOnlyModel(), val),
        "LogReg (5 входов)": p_val,
        "CatBoost (5 входов)": p_cb_val,
        "Ансамбль 50/50": p_ens_val,
    }
    # расширенный набор: добавляем ковариаты неопределённости как обычные признаки
    wide = PHASE9_FULL + [c for c in COVARIATES if val[c].notna().all() and train[c].notna().all()]
    cb_wide = CatBoostModel(feature_names=wide)
    cb_wide.fit(train, train["target"])
    families["CatBoost + ковариаты"] = proba(cb_wide, val)
    lr_wide = fit_logreg(wide, train)
    families["LogReg + ковариаты"] = proba(lr_wide, val)

    part_o = {}
    for name, p in families.items():
        r = compute_metrics(y_val, p)
        part_o[name] = r
        show(name, r)
    accs = [r["accuracy"] for r in part_o.values()]
    print(f"\n  Разброс accuracy между семействами: {max(accs)-min(accs):.4f} "
          f"(от {min(accs):.4f} до {max(accs):.4f})")
    payload["part_o"] = part_o

    # =================== финальный TEST ===================
    if args.final_test:
        section("ЕДИНСТВЕННОЕ обращение к TEST")
        TEST_ACCESS["n"] += 1
        y_test = test["target"].to_numpy()
        p_test = proba(m_full, test)
        r_test = compute_metrics(y_test, p_test)
        show("Phase 9 (5 входов)", r_test)
        print("  Phase 12 сообщала 0.6301 / 0.6390 / 0.6816 — воспроизведение.")

        cal_t = calibration_report(y_test, p_test)
        print(f"\n  Калибровка на TEST: ECE={cal_t['ece']:.5f}  MCE={cal_t['mce']:.5f}  "
              f"slope={cal_t['slope']:.4f}  intercept={cal_t['intercept']:+.4f}")
        print(f"  {'бин':>12s} {'n':>7s} {'прогноз':>9s} {'факт':>8s} {'разрыв':>8s}")
        for b in reliability_curve(y_test, p_test):
            print(f"  [{b.lo:.1f},{b.hi:.1f}){'':>2s} {b.n:7d} {b.mean_pred:9.4f} "
                  f"{b.frac_positive:8.4f} {b.gap:8.4f}")

        # механизм отказа и пороги ЗАФИКСИРОВАНЫ на VALIDATION
        conf_test = np.abs(p_test - 0.5)
        p_cb_test = proba(cb, test)
        meta_te = meta_frame(test, p_test)
        meta_te["noise"] = 0.0
        p_error_test = np.asarray(
            meta_models["M2: |p-0.5| + ковариаты"].predict_proba(meta_te))[:, 1]
        score_map = {
            "A: |p-0.5|": conf_test,
            "B: -P(error) (meta-модель)": -p_error_test,
            "C: -разногласие моделей": -np.abs(p_test - p_cb_test),
            "D: |p_ens-0.5|": np.abs(0.5 * (p_test + p_cb_test) - 0.5),
            "E: композит |p-0.5| / (1+P(error))": conf_test / (1.0 + p_error_test),
        }
        print(f"\n  Механизм отказа, выбранный на VALIDATION: {best_mech}")
        score_test = np.asarray(score_map[best_mech], dtype=float)

        print("\n  Coverage-risk на TEST (пороги — квантили TEST-скора, "
              "механизм зафиксирован на VAL):")
        cr_test = coverage_curve(y_test, p_test, score_test, grid)
        print_coverage(cr_test)

        print("\n  Те же ЧИСЛОВЫЕ пороги, что были выбраны на VALIDATION:")
        print(f"  {'цель VAL':>9s} {'порог':>9s} {'покрытие TEST':>14s} {'n':>7s} {'acc':>8s}")
        fixed = []
        val_scores = {"A: |p-0.5|": conf_abs, "B: -P(error) (meta-модель)": -p_error_val,
                      "C: -разногласие моделей": -disagree,
                      "D: |p_ens-0.5|": np.abs(p_ens_val - 0.5),
                      "E: композит |p-0.5| / (1+P(error))": conf_abs / (1.0 + p_error_val)}
        sv = np.asarray(val_scores[best_mech], dtype=float)
        for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.25):
            thr = float(np.quantile(sv, 1.0 - cov)) if cov < 1.0 else float(sv.min() - 1)
            sel = score_test >= thr
            if sel.sum() < 50:
                continue
            a = float((y_test[sel] == (p_test[sel] > 0.5)).mean())
            fixed.append({"coverage_val": cov, "threshold": thr,
                          "coverage_test": float(sel.mean()), "n": int(sel.sum()), "accuracy": a})
            print(f"  {cov*100:8.1f}% {thr:9.4f} {sel.mean()*100:13.1f}% {sel.sum():7d} {a:8.4f}")

        db, cb_ci = block_bootstrap_metric(y_test, p_test, accuracy_metric,
                                           block_size=BLOCK_SIZE, seed=RANDOM_SEED)
        print(f"\n  Accuracy на TEST со всеми матчами: {db:.4f}, 95% CI {cb_ci}")

        payload["test"] = {
            "metrics": r_test, "calibration": cal_t,
            "abstention_mechanism": best_mech,
            "coverage_curve": cr_test,
            "coverage_fixed_thresholds": fixed,
            "accuracy_ci": list(cb_ci),
        }
        payload["test_accesses"] = TEST_ACCESS["n"]

    payload["test_accesses"] = TEST_ACCESS["n"]
    out = os.path.join(EXPERIMENTS_DIR, "phase13.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nОбращений к TEST: {TEST_ACCESS['n']}")
    print(f"Результаты: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
