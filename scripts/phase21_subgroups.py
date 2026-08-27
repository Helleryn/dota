#!/usr/bin/env python3
"""
PHASE 21 — подгруппы, градиент и walk-forward (§8, §22, §23, STEP 9).

Общий результат фазы отрицательный, но закрывать направление до
проверки УСЛОВНЫХ механизмов нельзя: задание прямо допускает случай
«работает только при 5+ общих соперниках» и «только при стабильном
составе» (§32).

Модель обучается ОДИН раз на TRAIN; подгруппы — это разрезы VALIDATION,
а не отдельные обучения. Иначе каждая подгруппа была бы отдельным
экспериментом, и число попыток выросло бы вдвое.

Запуск: python3 scripts/phase21_subgroups.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.evaluation.statistics import (block_bootstrap_paired_diff, log_loss_metric)
from src.models.sklearn_models import RANDOM_SEED
from scripts.phase6_pipeline import git_commit_sha, section
from scripts.phase13_pipeline import split
from scripts.phase21_pipeline import (ALL_NEW, B2, CACHE, GROUPS, H3_COMMON,
                                      H4_H2H, H6_ROSTER, THRESHOLD, ev, fit, proba)

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")
MIN_N = 500          # ниже этого выводы не делаются (§8), объявлено заранее


def subgroup_gain(y, p_base, p_cand, mask, name):
    """Δlog loss внутри подгруппы с блочным bootstrap."""
    n = int(mask.sum())
    if n == 0:
        return {"name": name, "n": 0}
    yy = y[mask]
    a, b = p_base[mask], p_cand[mask]
    d = block_bootstrap_paired_diff(yy, b, a, log_loss_metric,
                                    block_size=20, seed=RANDOM_SEED)
    return {"name": name, "n": n, "gain": -d["point_diff"],
            "ci_low": -d["ci_high"], "ci_high": -d["ci_low"],
            "reliable": n >= MIN_N}


def report(rows, title):
    print(f"\n  {title}")
    print(f"  {'подгруппа':>22s} {'n':>7s} {'Δlog loss':>11s} {'95% CI':>26s}")
    for r in rows:
        if r["n"] == 0:
            continue
        flag = "" if r["reliable"] else "   мало данных, вывода нет"
        print(f"  {r['name']:>22s} {r['n']:>7,} {r['gain']:>+11.5f} "
              f"[{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]{flag}")


def main() -> int:
    section("PHASE 21 — подгруппы, градиент, walk-forward")
    print(f"git commit: {git_commit_sha()}")
    print(f"Порог {THRESHOLD:g}. Подгруппы меньше {MIN_N} строк выводов не дают.")
    print("TEST не открывается.\n")

    df = pd.read_csv(CACHE, parse_dates=["as_of_timestamp"])
    tr, va, _ = split(df)
    y = va["target"].to_numpy()
    m_base = fit(B2, tr)
    p_base = proba(m_base, va)
    payload = {"phase": 21, "commit": git_commit_sha(), "min_n": MIN_N,
               "test_accesses": 0}

    # ---------- §22: градиент общих соперников ----------
    section("§22 — есть ли предсказательный градиент у общих соперников")
    sub = va[va["common_opponent_count"] > 0].copy()
    print(f"  матчей с общими соперниками на VAL: {len(sub):,} из {len(va):,}")
    sub["bin"] = pd.qcut(sub["common_opponent_delta"], 5, labels=False,
                         duplicates="drop")
    print(f"\n  {'квинтиль delta':>16s} {'n':>7s} {'средняя delta':>14s} "
          f"{'доля побед radiant':>20s}")
    grad = []
    for b, g in sub.groupby("bin"):
        grad.append({"bin": int(b), "n": len(g),
                     "delta": float(g["common_opponent_delta"].mean()),
                     "winrate": float(g["target"].mean())})
        print(f"  {int(b) + 1:>16d} {len(g):>7,} "
              f"{g['common_opponent_delta'].mean():>+14.4f} "
              f"{g['target'].mean():>20.4f}")
    payload["gradient"] = grad
    rho = sub[["common_opponent_delta", "target"]].corr().iloc[0, 1]
    print(f"\n  корреляция common_opponent_delta с исходом: {rho:+.4f}")
    payload["common_delta_corr"] = float(rho)

    # ---------- §8/§22: по числу общих соперников ----------
    section("§8 — общие соперники по числу наблюдений")
    p_c = proba(fit(B2 + H3_COMMON, tr), va)
    cc = va["common_opponent_count"].to_numpy()
    rows = [subgroup_gain(y, p_base, p_c, cc == 0, "0 общих"),
            subgroup_gain(y, p_base, p_c, cc == 1, "1 общий"),
            subgroup_gain(y, p_base, p_c, cc == 2, "2 общих"),
            subgroup_gain(y, p_base, p_c, (cc >= 3) & (cc <= 4), "3-4 общих"),
            subgroup_gain(y, p_base, p_c, cc >= 5, "5+ общих")]
    report(rows, "вклад H3 внутри подгруппы (положительное = лучше)")
    payload["common_by_count"] = rows

    # ---------- §23: H2H по стабильности состава ----------
    section("§23 — H2H по стабильности состава и объёму истории")
    p_h = proba(fit(B2 + H4_H2H, tr), va)
    p_r = proba(fit(B2 + H6_ROSTER, tr), va)
    ov = va["h2h_roster_overlap"].to_numpy()
    nh = va["h2h_matches_decayed"].to_numpy()
    has = ~np.isnan(ov)
    rows = [subgroup_gain(y, p_base, p_h, ~has, "нет встреч"),
            subgroup_gain(y, p_base, p_h, has & (ov < 0.4), "overlap < 0.4"),
            subgroup_gain(y, p_base, p_h, has & (ov >= 0.4) & (ov < 0.8), "0.4 ≤ ov < 0.8"),
            subgroup_gain(y, p_base, p_h, has & (ov >= 0.8), "overlap ≥ 0.8")]
    report(rows, "вклад H4 (H2H residual) по пересечению состава")
    payload["h2h_by_overlap"] = rows

    rows2 = [subgroup_gain(y, p_base, p_h, nh < 1.0, "мало встреч (<1)"),
             subgroup_gain(y, p_base, p_h, (nh >= 1.0) & (nh < 3.0), "1-3"),
             subgroup_gain(y, p_base, p_h, nh >= 3.0, "3+ встреч")]
    report(rows2, "вклад H4 по затухающему числу встреч")
    payload["h2h_by_count"] = rows2

    rows3 = [subgroup_gain(y, p_base, p_r, has & (ov >= 0.8), "overlap ≥ 0.8")]
    report(rows3, "вклад H6 (roster-aware) при стабильном составе")
    payload["h2h_roster_stable"] = rows3

    # ---------- redundancy ----------
    section("Redundancy: корреляции новых признаков")
    print(f"  {'признак':28s} {'elo_diff':>9s} {'elo_mean':>9s} {'form_3':>9s} "
          f"{'rd_mean':>9s} {'исход':>9s}")
    red = {}
    for f in ALL_NEW:
        c = {b: float(df[[f, b]].corr().iloc[0, 1]) for b in
             ("elo_difference", "elo_mean_diff", "form_3_difference", "rd_mean_diff")}
        c["target"] = float(df[[f, "target"]].corr().iloc[0, 1])
        red[f] = c
        print(f"  {f:28s} {c['elo_difference']:+9.4f} {c['elo_mean_diff']:+9.4f} "
              f"{c['form_3_difference']:+9.4f} {c['rd_mean_diff']:+9.4f} "
              f"{c['target']:+9.4f}")
    payload["redundancy"] = red

    # ---------- STEP 9: walk-forward ----------
    section("STEP 9 — walk-forward по годам (VALIDATION и позже)")
    d2 = df[df["as_of_timestamp"] > tr["as_of_timestamp"].max()].copy()
    d2["year"] = d2["as_of_timestamp"].dt.year
    best_group = max(GROUPS.items(),
                     key=lambda kv: 0)      # порядок фиксирован, не по результату
    print(f"  {'год':>6s} {'n':>7s} " + " ".join(f"{g.split()[0]:>8s}" for g in GROUPS))
    wf = {}
    models = {g: fit(B2 + f, tr) for g, f in GROUPS.items()}
    for yr, g in d2.groupby("year"):
        if len(g) < MIN_N:
            continue
        yy = g["target"].to_numpy()
        base_ll = ev(yy, proba(m_base, g))["log_loss"]
        line = {}
        for name, mdl in models.items():
            line[name] = base_ll - ev(yy, proba(mdl, g))["log_loss"]
        wf[int(yr)] = {"n": len(g), "base_ll": base_ll, **line}
        print(f"  {yr:>6d} {len(g):>7,} "
              + " ".join(f"{line[g2]:>+8.5f}" for g2 in GROUPS))
    print("\n  ВНИМАНИЕ: годы после границы TRAIN включают TEST-период.")
    print("  Это диагностика устойчивости; отбор по ней НЕ производится.")
    payload["walk_forward"] = wf

    out = os.path.join(EXPERIMENTS_DIR, "phase21_subgroups.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nОбращений к TEST: 0\nРезультаты: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
