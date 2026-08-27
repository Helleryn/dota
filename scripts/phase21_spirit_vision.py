#!/usr/bin/env python3
"""
PHASE 21 — Spirit vs VISION: ретроспективная диагностика (§24).

Запускается ПОСЛЕ полного freeze: все вердикты фазы (REDUNDANT для всех
шести механизмов) приняты на VALIDATION, TEST уже открыт и закрыт.

Матч не участвует в отборе признаков, не входит в обучение и не влияет
на пороги. Это **retrospective case study, а не blind prospective
validation**: исход известен с Phase 19.

Драфт, результат и любые post-match данные не используются.

Запуск: python3 scripts/phase21_spirit_vision.py
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
from src.pit.opponent_context import build_opponent_context, load_context_matches
from scripts.phase6_pipeline import git_commit_sha, section
from scripts.phase13_pipeline import split
from scripts.phase20_pipeline import CACHE as P20_CACHE
from scripts.phase21_pipeline import (ALL_NEW, B0, B1, B2, GROUPS, HORIZON,
                                      ev, fit, proba)

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")
SPIRIT, VISION = 7119388, 9572001
SERIES = [8960577698, 8960655084, 8960762254, 8960882635, 8960991322]
SPIRIT_RADIANT = {8960577698: True, 8960655084: False, 8960762254: True,
                  8960882635: False, 8960991322: False}
SPIRIT_WON = {8960577698: True, 8960655084: False, 8960762254: True,
              8960882635: False, 8960991322: True}


def main() -> int:
    section("PHASE 21 — Spirit vs VISION: ретроспективная диагностика")
    print(f"git commit: {git_commit_sha()}")
    print("Все вердикты фазы заморожены ДО запуска. Матч не участвует")
    print("ни в обучении, ни в отборе. Драфт и результат не используются.\n")

    base = pd.read_csv(P20_CACHE, parse_dates=["as_of_timestamp"])
    eng = make_engine(load_settings())
    emit = set(base["match_id"]) | set(SERIES)
    print("проход: контекст соперников…", flush=True)
    rows = build_opponent_context(load_context_matches(eng), HORIZON, emit_only=emit)
    extra = pd.DataFrame([r.__dict__ for r in rows]).drop(
        columns=["prediction_at", "target"])
    df = base.merge(extra, on="match_id", how="left").sort_values(
        ["as_of_timestamp", "match_id"]).reset_index(drop=True)

    tr, _, _ = split(df)
    series = df[df["match_id"].isin(SERIES)].set_index("match_id")
    print(f"обучение: {len(tr):,} матчей до {tr['as_of_timestamp'].max():%Y-%m-%d}")
    print(f"серия найдена: {len(series)} из {len(SERIES)} игр\n")

    payload = {"phase": 21, "commit": git_commit_sha(), "games": {}}

    section("Контекст соперников до матча (знак: Spirit минус VISION)")
    for i, mid in enumerate(SERIES, 1):
        if mid not in series.index:
            continue
        row = series.loc[mid]
        sign = 1.0 if SPIRIT_RADIANT[mid] else -1.0
        rec = {}
        print(f"игра {i} ({mid}):")
        for f in ALL_NEW:
            v = row.get(f)
            if v is None or pd.isna(v):
                print(f"    {f:28s} {'—':>10s}")
                rec[f] = None
                continue
            # счётчики симметричны, знак к ним не применяется
            s = 1.0 if f.endswith(("_count", "_7", "_14", "_30", "_decayed", "_last",
                                   "_overlap")) and "diff" not in f and "residual" not in f \
                and "winrate" not in f else sign
            rec[f] = float(v) * s
            print(f"    {f:28s} {float(v) * s:+10.4f}")
        payload["games"][str(mid)] = rec
        print()

    section("Изменился бы прогноз")
    print("Каждый механизм добавляется к B2 ОТДЕЛЬНО. Вердикты не меняются.\n")
    models = {"B0 (frozen)": fit(B0, tr), "B1 = B0+H1": fit(B1, tr),
              "B2 (Phase 20)": fit(B2, tr)}
    for g, feats in GROUPS.items():
        models[f"B2 + {g}"] = fit(B2 + feats, tr)
    models["B2 + все 15 новых"] = fit(B2 + ALL_NEW, tr)

    print(f"  {'модель':32s} " + "  ".join(f"игра{i}" for i in range(1, 6)) + "   среднее")
    lines = {}
    for name, mdl in models.items():
        line = []
        for mid in SERIES:
            if mid in series.index:
                p_r = float(proba(mdl, series.loc[[mid]])[0])
                line.append(p_r if SPIRIT_RADIANT[mid] else 1.0 - p_r)
        lines[name] = line
        print(f"  {name:32s} " + "  ".join(f"{v:.3f}" for v in line)
              + f"   {np.mean(line):.4f}")
    payload["lines"] = lines

    section("Фактический результат (известен с Phase 19)")
    print("  Spirit 3 : 2 VISION (игры 1, 3, 5).")
    print("  Это ретроспектива, а не слепая проверка.\n")
    for name, line in lines.items():
        acc = sum(1 for mid, v in zip(SERIES, line)
                  if (v > 0.5) == SPIRIT_WON[mid]) / len(line)
        print(f"  {name:32s} угадано {acc:.0%} игр")
    payload["actual"] = {"spirit_series": "3:2"}

    out = os.path.join(EXPERIMENTS_DIR, "phase21_spirit_vision.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nРезультаты: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
