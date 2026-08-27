#!/usr/bin/env python3
"""
PHASE 20 — Spirit vs VISION как ДИАГНОСТИКА (PART H, PART P).

Матч не участвует в отборе признаков, не входит в обучение и не влияет
на пороги. Все решения фазы приняты на VALIDATION до этого запуска.

Вопрос PART P: существовал ли ДО матча механизм, по которому Spirit
можно было оценить выше? Ответ ищется только среди величин, доступных на
момент прогноза; объяснение задним числом не допускается.

Запуск: python3 scripts/phase20_spirit_vision.py
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
from src.models.sklearn_models import RANDOM_SEED, LogisticRegressionModel
from src.pit.engine import build_point_in_time_features
from src.pit.lineup_structure import build_lineup_features, load_lineup_matches
from src.pit.loader import load_pit_matches
from scripts.phase6_pipeline import git_commit_sha, section
from scripts.phase13_pipeline import load_common_set, split
from scripts.phase20_pipeline import GROUPS, HORIZON, M0, ev, fit, proba

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")
SPIRIT, VISION = 7119388, 9572001
SERIES = [8960577698, 8960655084, 8960762254, 8960882635, 8960991322]
# сторона Spirit по играм (аудит Phase 19)
SPIRIT_RADIANT = {8960577698: True, 8960655084: False, 8960762254: True,
                  8960882635: False, 8960991322: False}
# фактические исходы, из аудита Phase 19; используются ТОЛЬКО в конце
SPIRIT_WON = {8960577698: True, 8960655084: False, 8960762254: True,
              8960882635: False, 8960991322: True}


def main() -> int:
    section("PHASE 20 — Spirit vs VISION: диагностика, не отбор")
    print(f"git commit: {git_commit_sha()}")
    print("Матч не участвует в обучении и в отборе признаков.\n")

    ref = load_common_set()
    emit = set(ref["match_id"]) | set(SERIES)
    eng = make_engine(load_settings())

    print("проход 1/2: базовые признаки…", flush=True)
    matches, rp = load_pit_matches(eng)
    base = pd.DataFrame([{
        "match_id": r.match_id, "as_of_timestamp": r.start_time,
        "elo_difference": r.elo_difference, "form_3_difference": r.form_3_difference,
        "elo_mean_diff": r.elo_mean_diff, "five_vs_team_elo_diff": r.five_vs_team_elo_diff,
        "target": r.target}
        for r in build_point_in_time_features(matches, HORIZON, roster_provider=rp,
                                              include_draft=False, emit_only=emit)])
    print("проход 2/2: структура состава…", flush=True)
    rows = build_lineup_features(load_lineup_matches(eng), HORIZON, emit_only=emit)
    extra = pd.DataFrame([r.__dict__ for r in rows]).drop(
        columns=["prediction_at", "target"])
    df = base.merge(extra, on="match_id", how="left").sort_values(
        ["as_of_timestamp", "match_id"]).reset_index(drop=True)

    tr, _, _ = split(df)
    m0 = fit(M0, tr)
    series = df[df["match_id"].isin(SERIES)].set_index("match_id")
    print(f"\nобучение: {len(tr):,} матчей до {tr['as_of_timestamp'].max():%Y-%m-%d}")
    print(f"серия найдена: {len(series)} из {len(SERIES)} игр\n")

    payload = {"phase": 20, "commit": git_commit_sha(), "games": {}}

    # ---------- что говорили механизмы ----------
    section("Что каждый механизм говорил ДО матча")
    print("Знак признака дан в системе «Spirit минус VISION»: положительное")
    print("значение — в пользу Spirit.\n")
    cands = [f for feats in GROUPS.values() for f in feats]
    for mid in SERIES:
        if mid not in series.index:
            continue
        row = series.loc[mid]
        sr = SPIRIT_RADIANT[mid]
        sign = 1.0 if sr else -1.0
        p_r = float(proba(m0, series.loc[[mid]])[0])
        p_spirit = p_r if sr else 1.0 - p_r
        rec = {"p_spirit_M0": p_spirit, "spirit_radiant": sr, "features": {}}
        print(f"игра {SERIES.index(mid) + 1} ({mid}): P(Spirit) = {p_spirit:.4f}")
        for f in ("elo_difference", "elo_mean_diff", "five_vs_team_elo_diff"):
            v = row[f]
            rec["features"][f] = None if pd.isna(v) else float(v) * sign
            print(f"    {f:28s} {float(v) * sign:+10.3f}   [база]")
        for f in cands:
            v = row.get(f)
            if v is None or pd.isna(v):
                print(f"    {f:28s} {'—':>10s}")
                rec["features"][f] = None
                continue
            print(f"    {f:28s} {float(v) * sign:+10.3f}")
            rec["features"][f] = float(v) * sign
        payload["games"][str(mid)] = rec
        print()

    # ---------- изменился бы прогноз? ----------
    section("Изменился бы прогноз, если бы механизм был в модели")
    print("Каждая группа добавляется к M0 ОТДЕЛЬНО. Это диагностика:")
    print("решение о включении принято на VALIDATION и здесь не меняется.\n")
    print(f"  {'механизм':26s} " + "  ".join(f"игра{i}" for i in range(1, 6)))
    base_line = []
    for mid in SERIES:
        if mid in series.index:
            p_r = float(proba(m0, series.loc[[mid]])[0])
            base_line.append(p_r if SPIRIT_RADIANT[mid] else 1.0 - p_r)
    print(f"  {'M0 (база)':26s} " + "  ".join(f"{v:.3f}" for v in base_line))
    payload["m0_line"] = base_line

    per_group = {}
    for g, feats in GROUPS.items():
        mg = fit(M0 + feats, tr)
        line = []
        for mid in SERIES:
            if mid in series.index:
                p_r = float(proba(mg, series.loc[[mid]])[0])
                line.append(p_r if SPIRIT_RADIANT[mid] else 1.0 - p_r)
        per_group[g] = line
        delta = np.mean(line) - np.mean(base_line)
        print(f"  {g:26s} " + "  ".join(f"{v:.3f}" for v in line)
              + f"   среднее Δ={delta:+.4f}")
    payload["per_group"] = per_group

    # ---------- фактический результат ----------
    section("Фактический результат")
    print("  Spirit 3 : 2 VISION (игры 1, 3, 5).")
    won = sum(SPIRIT_WON.values())
    print(f"  Ни один механизм не выбирался по этому результату.\n")
    for g, line in per_group.items():
        acc = sum(1 for mid, v in zip(SERIES, line)
                  if (v > 0.5) == SPIRIT_WON[mid]) / len(line)
        print(f"  {g:26s} угадано {acc:.0%} игр")
    acc0 = sum(1 for mid, v in zip(SERIES, base_line)
               if (v > 0.5) == SPIRIT_WON[mid]) / len(base_line)
    print(f"  {'M0 (база)':26s} угадано {acc0:.0%} игр")
    payload["actual"] = {"spirit_wins": won, "m0_accuracy": acc0}

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    out = os.path.join(EXPERIMENTS_DIR, "phase20_spirit_vision.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nРезультаты: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
