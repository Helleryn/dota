#!/usr/bin/env python3
"""
PHASE 21 — ЕДИНСТВЕННОЕ обращение к TEST + калибровка (§27, §28).

Что здесь проверяется и почему обращение вообще производится.

На VALIDATION **ни один** механизм фазы не прошёл порог 1e-4 поверх B2,
и ни одна подгруппа не дала условного механизма. Решение — REMOVE для
всех — принято и заморожено ДО этого запуска.

Обращение к TEST служит одной цели: убедиться, что отрицательный
результат не является артефактом VALIDATION. Выбора здесь уже не
осталось, поэтому fishing невозможен: TEST может либо подтвердить
решение, либо ему противоречить — и во втором случае будет записано
противоречие, а не пересмотр решения.

Сравниваются:
  B2                — конфигурация, подтверждённая Phase 20 (не меняется)
  B2 + все новые    — все 15 признаков Phase 21 разом

Если Phase 21 действительно ничего не добавила, разница будет около нуля
или отрицательной.

Калибровка (§28) считается по протоколу Phase 14 (beta), заново, а не
переносится.

Запуск: python3 scripts/phase21_final_test.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.evaluation.calibration import calibration_report
from src.evaluation.calibrators import BetaCalibrator
from src.evaluation.statistics import (block_bootstrap_paired_diff, log_loss_metric,
                                       mcnemar_exact)
from src.models.sklearn_models import RANDOM_SEED
from scripts.phase6_pipeline import git_commit_sha, section
from scripts.phase13_pipeline import split
from scripts.phase21_pipeline import (ALL_NEW, B0, B1, B2, CACHE, ev, fit, proba, show)

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")


def calib_line(name, y, p):
    c = calibration_report(y, p)
    print(f"  {name:28s} ECE={c['ece']:.5f}  slope={c['slope']:+.4f}  "
          f"intercept={c['intercept']:+.4f}")
    return {"ece": float(c["ece"]), "slope": float(c["slope"]),
            "intercept": float(c["intercept"])}


def main() -> int:
    section("PHASE 21 — ЕДИНСТВЕННОЕ обращение к TEST")
    print(f"git commit: {git_commit_sha()}")
    print("Решение заморожено ДО запуска: все механизмы Phase 21 — REMOVE.")
    print("TEST проверяет, держится ли отрицательный результат вне VALIDATION.\n")

    df = pd.read_csv(CACHE, parse_dates=["as_of_timestamp"])
    tr, va, te = split(df)
    y = te["target"].to_numpy()
    print(f"TEST: {len(te):,} матчей, "
          f"{te['as_of_timestamp'].min():%Y-%m-%d} … {te['as_of_timestamp'].max():%Y-%m-%d}\n")

    p_b0 = proba(fit(B0, tr), te)
    p_b2 = proba(fit(B2, tr), te)
    p_all = proba(fit(B2 + ALL_NEW, tr), te)
    r0, r2, ra = ev(y, p_b0), ev(y, p_b2), ev(y, p_all)
    show("B0 фактический frozen", r0)
    show("B2 (Phase 20, не меняется)", r2, r0)
    show("B2 + все 15 новых", ra, r2)

    section("Значимость: добавили ли механизмы Phase 21 хоть что-то")
    d = block_bootstrap_paired_diff(y, p_all, p_b2, log_loss_metric,
                                    block_size=20, seed=RANDOM_SEED)
    mc = mcnemar_exact(y, (p_all > 0.5).astype(int), (p_b2 > 0.5).astype(int))
    print(f"  Δlog loss (все новые − B2): {d['point_diff']:+.5f}  "
          f"95% CI [{d['ci_low']:+.5f}, {d['ci_high']:+.5f}]")
    print(f"  McNemar p = {mc['p_value']:.4f}")
    if d["ci_high"] < 0:
        verdict = "ПРОТИВОРЕЧИЕ: на TEST лучше, хотя на VAL не проходило"
    elif d["ci_low"] > 0:
        verdict = "ПОДТВЕРЖДЕНО: на TEST значимо ХУЖЕ"
    else:
        verdict = "ПОДТВЕРЖДЕНО: интервал включает 0, независимого сигнала нет"
    print(f"  -> {verdict}")

    section("§28 — калибровка (пересчитана, не перенесена)")
    print("  Сырая модель:")
    raw = {"B0": calib_line("B0", y, p_b0), "B2": calib_line("B2", y, p_b2),
           "B2+новые": calib_line("B2 + все новые", y, p_all)}
    print("\n  После beta-калибровки (обучена на VALIDATION, применена к TEST):")
    yv = va["target"].to_numpy()
    cal = {}
    for name, feats, pt in (("B0", B0, p_b0), ("B2", B2, p_b2),
                            ("B2+новые", B2 + ALL_NEW, p_all)):
        m = fit(feats, tr)
        pv = proba(m, va)
        bc = BetaCalibrator().fit(pv, yv)
        pc = bc.transform(pt)
        cal[name] = calib_line(name, y, pc)
        cal[name]["log_loss"] = float(ev(y, pc)["log_loss"])
    print()
    for name in ("B0", "B2", "B2+новые"):
        print(f"  {name:12s} log loss сырой -> калиброванный: "
              f"{ev(y, {'B0': p_b0, 'B2': p_b2, 'B2+новые': p_all}[name])['log_loss']:.5f}"
              f" -> {cal[name]['log_loss']:.5f}")

    payload = {"phase": 21, "commit": git_commit_sha(),
               "b0": r0, "b2": r2, "b2_plus_all": ra,
               "paired_log_loss": d, "mcnemar": mc, "verdict": verdict,
               "calibration_raw": raw, "calibration_beta": cal,
               "test_accesses": 1}
    out = os.path.join(EXPERIMENTS_DIR, "phase21_final_test.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nОбращений к TEST: 1 (это оно)\nРезультаты: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
