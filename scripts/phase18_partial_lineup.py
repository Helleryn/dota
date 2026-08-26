#!/usr/bin/env python3
"""
PHASE 18 — сила состава, когда известны не все пятеро.

Исследование источников показало: среди 15 предстоящих матчей Dota нет
ни одного, где сопоставлены все десять игроков. Лучший случай — 9 из 10.
Значит рабочий режим — **частичный состав**, и вопрос не «mean или
median», а «что делать при k < 5».

Гипотезы (объявлены в docs/phase18-plan.md ДО эксперимента):

  L1  среднее по известным k лучше отказа от признака (T2)
  L2  усадка к нулю при малом k лучше простого среднего
  L3  качество монотонно растёт с k

Порог: Δlog loss > 1e-4 на VALIDATION. TEST не используется.

Запуск: python3 scripts/phase18_partial_lineup.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.config import load_settings
from src.db.engine import make_engine
from src.evaluation.calibration import calibration_report
from src.evaluation.metrics import compute_metrics
from src.models.sklearn_models import RANDOM_SEED, LogisticRegressionModel
from src.pit.engine import RosterProvider, build_point_in_time_features
from src.pit.loader import load_pit_matches
from scripts.phase6_pipeline import git_commit_sha, section
from scripts.phase13_pipeline import load_common_set, split

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")
HORIZON = timedelta(hours=24)
T1 = ["elo_difference", "form_3_difference", "elo_mean_diff", "five_vs_team_elo_diff"]
T2 = ["elo_difference", "form_3_difference"]
ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0]


def truncate(rp: RosterProvider, k: int) -> RosterProvider:
    """Оставляет k игроков на сторону.

    Отбор детерминированный (сортировка по account_id), а не случайный:
    иначе результат зависел бы от seed, и сравнение между k стало бы
    сравнением случайных подвыборок. Это НЕ моделирует, какие именно
    игроки становятся известны в реальности — реальный отбор смещён к
    более заметным игрокам. Ограничение названо прямо.
    """
    out = {}
    for mid, (r, d) in rp.rosters.items():
        out[mid] = (frozenset(sorted(r)[:k]), frozenset(sorted(d)[:k]))
    return RosterProvider(out)


def frame(matches, rp, emit) -> pd.DataFrame:
    rows = build_point_in_time_features(matches, HORIZON, roster_provider=rp,
                                        include_draft=False, emit_only=emit)
    return pd.DataFrame([{
        "match_id": r.match_id, "as_of_timestamp": r.start_time,
        "elo_difference": r.elo_difference, "form_3_difference": r.form_3_difference,
        "elo_mean_diff": r.elo_mean_diff, "five_vs_team_elo_diff": r.five_vs_team_elo_diff,
        "k_known": min(r.roster_known_radiant, r.roster_known_dire),
        "target": r.target,
    } for r in rows]).sort_values(["as_of_timestamp", "match_id"]).reset_index(drop=True)


def fit(feats, tr):
    m = LogisticRegressionModel(feature_names=feats, random_state=RANDOM_SEED)
    m.fit(tr, tr["target"])
    return m


def ev(y, p) -> Dict[str, float]:
    m = compute_metrics(y, p)
    c = calibration_report(y, p)
    return {"n": m["n"], "accuracy": m["accuracy"], "roc_auc": m["roc_auc"],
            "log_loss": m["log_loss"], "brier": m["brier_score"], "ece": c["ece"]}


def show(name, r, ref=None):
    d = f"  Δll={r['log_loss']-ref['log_loss']:+.5f}" if ref else ""
    print(f"{name:32s} acc={r['accuracy']:.4f} auc={r['roc_auc']:.4f} "
          f"ll={r['log_loss']:.4f} ECE={r['ece']:.5f}{d}", flush=True)


def main() -> int:
    section("PHASE 18 — сила состава при неполном знании (VALIDATION)")
    print(f"git commit: {git_commit_sha()}")
    print("Горизонт 24 ч. Порог отбора Δlog loss > 1e-4, объявлен заранее.\n")

    ref = load_common_set()
    emit = set(ref["match_id"])
    matches, rp_full = load_pit_matches(make_engine(load_settings()))

    payload = {"phase": 18, "hypotheses": {}}

    # --- база: T2, признаков состава нет вовсе ---
    f5 = frame(matches, rp_full, emit)
    tr, va, _ = split(f5)
    y = va["target"].to_numpy()
    r_t2 = ev(y, np.asarray(fit(T2, tr).predict_proba(va))[:, 1])
    show("T2 (без состава)", r_t2)

    # --- L1/L3: качество по числу известных игроков ---
    section("L1/L3 — среднее по известным k")
    results = {}
    frames = {}
    for k in (1, 2, 3, 4, 5):
        fk = frame(matches, truncate(rp_full, k), emit)
        frames[k] = fk
        trk, vak, _ = split(fk)
        rk = ev(vak["target"].to_numpy(),
                np.asarray(fit(T1, trk).predict_proba(vak))[:, 1])
        results[k] = rk
        show(f"  k={k} известных игроков", rk, r_t2)
    accs = [results[k]["accuracy"] for k in (1, 2, 3, 4, 5)]
    monotone = all(accs[i] <= accs[i + 1] + 1e-9 for i in range(len(accs) - 1))
    gains = {k: r_t2["log_loss"] - results[k]["log_loss"] for k in results}
    print(f"\n  Δlog loss к T2: " + "  ".join(f"k={k}:{g:+.5f}" for k, g in gains.items()))
    print(f"  L1 (k=1 лучше T2): {'ПОДТВЕРЖДЕНА' if gains[1] > 1e-4 else 'ОПРОВЕРГНУТА'}")
    print(f"  L3 (монотонность accuracy по k): {'ПОДТВЕРЖДЕНА' if monotone else 'ОПРОВЕРГНУТА'}")
    print(f"     accuracy по k: " + "  ".join(f"{k}:{a:.4f}" for k, a in zip((1,2,3,4,5), accs)))
    payload["hypotheses"]["L1"] = {"gain_k1": gains[1], "verdict":
                                   "KEEP" if gains[1] > 1e-4 else "REMOVE"}
    payload["hypotheses"]["L3"] = {"monotone": monotone, "accuracy_by_k": accs}
    payload["by_k"] = {str(k): results[k] for k in results}

    # --- L2: усадка при малом k ---
    section("L2 — усадка признаков состава при малом k")
    print("  Признаки состава умножаются на k/(k+alpha): при малом k величина")
    print("  тянется к нулю, то есть к «разница неизвестна», а не к «нулевая».\n")
    best = None
    l2 = []
    for alpha in ALPHAS:
        parts = []
        for k in (1, 2, 3, 4, 5):
            fk = frames[k].copy()
            w = k / (k + alpha) if (k + alpha) else 1.0
            fk["elo_mean_diff"] = fk["elo_mean_diff"].astype(float) * w
            fk["five_vs_team_elo_diff"] = fk["five_vs_team_elo_diff"].astype(float) * w
            fk["k_known"] = k
            parts.append(fk)
        mix = pd.concat(parts, ignore_index=True).sort_values(
            ["as_of_timestamp", "match_id", "k_known"]).reset_index(drop=True)
        trm, vam, _ = split(mix)
        r = ev(vam["target"].to_numpy(), np.asarray(fit(T1, trm).predict_proba(vam))[:, 1])
        l2.append({"alpha": alpha, **r})
        show(f"  alpha={alpha}", r, l2[0] if l2 else None)
        if best is None or r["log_loss"] < best[1]["log_loss"]:
            best = (alpha, r)
    gain2 = l2[0]["log_loss"] - best[1]["log_loss"]
    verdict2 = "KEEP" if (best[0] != 0.0 and gain2 > 1e-4) else "REMOVE"
    print(f"\n  лучшая alpha={best[0]}  Δlog loss к alpha=0: {gain2:+.5f}  ->  {verdict2}")
    payload["hypotheses"]["L2"] = {"best_alpha": best[0], "gain": gain2, "verdict": verdict2}
    payload["l2"] = l2

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    out = os.path.join(EXPERIMENTS_DIR, "phase18_partial_lineup.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nОбращений к TEST: 0\nРезультаты: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
