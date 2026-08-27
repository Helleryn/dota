#!/usr/bin/env python3
"""
PHASE 20 — ЕДИНСТВЕННОЕ обращение к TEST.

Конфигурация зафиксирована ДО этого запуска процедурой, объявленной в
плане: ablation (PART J) + redundancy audit (PART K).

  ablation к базе:        H1 +0.00417   H3 +0.00113
                          H4 +0.00012   H5 +0.00048   H2 +0.00007 (не прошла)
  incremental поверх H1:  H3 +0.00036   H4 −0.00028   H5 −0.00025
  leave-one-out:          H1 +0.00273   H3 +0.00032   H4 −0.00001  H5 +0.00003

Два объявленных правила разошлись на H4 и H5: по §6 плана они прошли
порог на VAL, по PART K у них нет incremental value. Разрешено в пользу
PART K — задание называет независимую incremental information главным
критерием фазы. Расхождение зафиксировано, а не сглажено.

    ФИНАЛЬНАЯ КОНФИГУРАЦИЯ:  M0 + H1 + H3

Основное сравнение — M0 против неё. M0+H1 показан справочно; решение по
TEST не пересматривается ни при каком его результате.

Запуск: python3 scripts/phase20_final_test.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.evaluation.statistics import (block_bootstrap_paired_diff, log_loss_metric,
                                       mcnemar_exact)
from src.models.sklearn_models import RANDOM_SEED
from scripts.phase6_pipeline import git_commit_sha, section
from scripts.phase13_pipeline import split
from scripts.phase20_pipeline import CACHE, H1, H3, M0, ev, fit, proba, show

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")
FINAL = M0 + H1 + H3


def main() -> int:
    section("PHASE 20 — ЕДИНСТВЕННОЕ обращение к TEST")
    print(f"git commit: {git_commit_sha()}")
    print(f"Конфигурация зафиксирована на VALIDATION: M0 + H1 + H3")
    print(f"  {FINAL}\n")

    df = pd.read_csv(CACHE, parse_dates=["as_of_timestamp"])
    tr, va, te = split(df)
    y = te["target"].to_numpy()
    print(f"TEST: {len(te):,} матчей, "
          f"{te['as_of_timestamp'].min():%Y-%m-%d} … {te['as_of_timestamp'].max():%Y-%m-%d}\n")

    p0 = proba(fit(M0, tr), te)
    p1 = proba(fit(M0 + H1, tr), te)
    pf = proba(fit(FINAL, tr), te)
    r0, r1, rf = ev(y, p0), ev(y, p1), ev(y, pf)
    show("M0 PREMATCH_BASELINE", r0)
    show("M0 + H1 (справочно)", r1, r0)
    show("M0 + H1 + H3 (финал)", rf, r0)

    section("Значимость")
    d = block_bootstrap_paired_diff(y, pf, p0, log_loss_metric,
                                    block_size=20, seed=RANDOM_SEED)
    mc = mcnemar_exact(y, (pf > 0.5).astype(int), (p0 > 0.5).astype(int))
    print(f"  Δlog loss (финал − M0): {d['point_diff']:+.5f}  "
          f"95% CI [{d['ci_low']:+.5f}, {d['ci_high']:+.5f}]")
    print(f"  McNemar p = {mc['p_value']:.4f}  "
          f"(discordant: {mc.get('n_discordant', '—')})")
    verdict = "KEEP" if d["ci_high"] < 0 else "RESEARCH FURTHER"
    print(f"  {'ЗНАЧИМО лучше' if d['ci_high'] < 0 else 'интервал включает 0'}"
          f"  ->  вердикт H1: {verdict}")

    print(f"\n  Δaccuracy: {rf['accuracy'] - r0['accuracy']:+.4f}")
    print(f"  ΔROC-AUC:  {rf['roc_auc'] - r0['roc_auc']:+.4f}")
    print(f"  ΔECE:      {rf['ece'] - r0['ece']:+.5f}")
    print(f"  ΔBrier:    {rf['brier'] - r0['brier']:+.5f}")

    payload = {"phase": 20, "commit": git_commit_sha(), "final_features": FINAL,
               "m0": r0, "m0_h1": r1, "final": rf,
               "paired_log_loss": d, "mcnemar": mc, "verdict": verdict,
               "test_accesses": 1}
    out = os.path.join(EXPERIMENTS_DIR, "phase20_final_test.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nОбращений к TEST: 1 (это оно)\nРезультаты: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
