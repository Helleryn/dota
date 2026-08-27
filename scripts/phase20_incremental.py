#!/usr/bin/env python3
"""
PHASE 20 — incremental value поверх победившего механизма (PART K).

Ablation дала: H1 в одиночку +0.00417, а все четыре выжившие вместе —
+0.00424. То есть H3/H4/H5 прошли порог ОТНОСИТЕЛЬНО БАЗЫ, но почти
ничего не добавляют ПОВЕРХ H1.

Задание PART K требует различать эти два случая прямо:

  «Если новый признак коррелирует с player-Elo, но не даёт incremental
   value: REMOVE. Если корреляция высокая, но incremental value есть:
   RESEARCH FURTHER.»

Этот срез может только ПОНИЗИТЬ вердикт, никогда не повысить: новых
кандидатов он не вводит и порогов не меняет. Поэтому он входит в
объявленный redundancy audit, а не является дополнительным отбором.

Запуск: python3 scripts/phase20_incremental.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from scripts.phase6_pipeline import git_commit_sha, section
from scripts.phase13_pipeline import split
from scripts.phase20_pipeline import (CACHE, GROUPS, H1, H3, H4, H5, M0,
                                      THRESHOLD, ev, fit, proba, show)

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")


def main() -> int:
    section("PHASE 20 — incremental value поверх H1 (VALIDATION)")
    print(f"git commit: {git_commit_sha()}")
    print(f"Порог тот же: Δlog loss > {THRESHOLD:g}. Новых кандидатов нет.\n")

    df = pd.read_csv(CACHE, parse_dates=["as_of_timestamp"])
    tr, va, _ = split(df)
    y = va["target"].to_numpy()

    r0 = ev(y, proba(fit(M0, tr), va))
    r1 = ev(y, proba(fit(M0 + H1, tr), va))
    show("M0 база", r0)
    show("M0 + H1", r1, r0)
    payload = {"phase": 20, "commit": git_commit_sha(), "M0": r0, "M1": r1,
               "on_top_of_H1": {}, "leave_one_out": {}}

    section("Каждый механизм ПОВЕРХ H1")
    for name, feats in (("H3 актуальность", H3), ("H4 сыгранность", H4),
                        ("H5 новизна", H5)):
        r = ev(y, proba(fit(M0 + H1 + feats, tr), va))
        gain = r1["log_loss"] - r["log_loss"]
        show(f"M0+H1+{name}", r, r1)
        payload["on_top_of_H1"][name] = {"gain_over_H1": gain,
                                         "passes": bool(gain > THRESHOLD), **r}
    print()
    for name, v in payload["on_top_of_H1"].items():
        print(f"  {name:20s} поверх H1: Δll={v['gain_over_H1']:+.5f}  "
              f"{'добавляет' if v['passes'] else 'НЕ добавляет'}")

    section("Leave-one-out из полного набора")
    full = H1 + H3 + H4 + H5
    rf = ev(y, proba(fit(M0 + full, tr), va))
    show("полный набор", rf, r0)
    print()
    for name, feats in (("H1 неопределённость", H1), ("H3 актуальность", H3),
                        ("H4 сыгранность", H4), ("H5 новизна", H5)):
        rest = [f for f in full if f not in feats]
        r = ev(y, proba(fit(M0 + rest, tr), va))
        loss = r["log_loss"] - rf["log_loss"]
        payload["leave_one_out"][name] = {"loss_when_removed": loss, **r}
        print(f"  без {name:22s} ll={r['log_loss']:.5f}  "
              f"ухудшение {loss:+.5f}  "
              f"{'НУЖЕН' if loss > THRESHOLD else 'не нужен'}")
    payload["full"] = rf

    section("Устойчивость H1 по годам")
    df2 = df.copy()
    df2["year"] = df2["as_of_timestamp"].dt.year
    print(f"  {'год':>6s} {'n':>7s} {'M0 ll':>9s} {'M0+H1 ll':>9s} {'Δ':>9s}")
    yearly = {}
    m0f, m1f = fit(M0, tr), fit(M0 + H1, tr)
    for yr, g in df2[df2["as_of_timestamp"] > tr["as_of_timestamp"].max()].groupby("year"):
        if len(g) < 200:
            continue
        yy = g["target"].to_numpy()
        a = ev(yy, proba(m0f, g))
        b = ev(yy, proba(m1f, g))
        d = a["log_loss"] - b["log_loss"]
        yearly[int(yr)] = {"n": len(g), "m0": a["log_loss"], "m1": b["log_loss"], "gain": d}
        print(f"  {yr:>6d} {len(g):>7,} {a['log_loss']:>9.4f} {b['log_loss']:>9.4f} {d:>+9.5f}")
    print("\n  ВНИМАНИЕ: годы после границы TRAIN включают TEST-период.")
    print("  Это диагностика устойчивости, отбор по ней НЕ производится;")
    print("  решение принято по VALIDATION выше.")
    payload["yearly"] = yearly

    out = os.path.join(EXPERIMENTS_DIR, "phase20_incremental.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nОбращений к TEST: 0\nРезультаты: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
