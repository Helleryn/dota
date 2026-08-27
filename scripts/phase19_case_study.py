#!/usr/bin/env python3
"""
PHASE 19 — исторический pre-match case study: Vision vs Team Spirit, TI15.

Конвейер «останавливает время» в момент T = start − Δ и выдаёт прогноз,
не имея права видеть ни результат матча, ни любой матч, начавшийся
позже T. Гарантия — движок Phase 17: события слиты в одну ленту, и при
равном времени PREDICT идёт строго раньше UPDATE.

Все решения зафиксированы в docs/phase19-plan.md ДО запуска. Здесь
ничего не подбирается: сценарий только считает.

Запуск: python3 scripts/phase19_case_study.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.config import load_settings
from src.db.engine import make_engine
from src.evaluation.metrics import compute_metrics
from src.evaluation.calibration import calibration_report
from src.models.sklearn_models import RANDOM_SEED, LogisticRegressionModel
from src.pit.engine import RosterProvider, build_point_in_time_features
from src.pit.loader import load_pit_matches
from scripts.phase6_pipeline import git_commit_sha, section
from scripts.phase13_pipeline import load_common_set, split

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")

# --- предмет исследования, установлен аудитом PART A ---
LEAGUE = 19719
SPIRIT, VISION = 7119388, 9572001
SERIES_ID = 1133004
SERIES = [8960577698, 8960655084, 8960762254, 8960882635, 8960991322]
# правило отбора subset объявлено в плане ДО того, как стал известен его размер
SUBSET_FROM = datetime(2026, 8, 21, tzinfo=timezone.utc)

# моменты прогноза (PART C)
DELTAS = [("T-72ч", timedelta(hours=72)), ("T-24ч", timedelta(hours=24)),
          ("T-6ч", timedelta(hours=6)), ("T-3ч", timedelta(hours=3)),
          ("T-1ч", timedelta(hours=1)), ("T-30м", timedelta(minutes=30))]

# PREMATCH_BASELINE (PART S): frozen модель требует драфта, здесь его нет
PREMATCH = ["elo_difference", "form_3_difference",
            "elo_mean_diff", "five_vs_team_elo_diff"]
TEAM_ONLY = ["elo_difference", "form_3_difference"]
POSTDRAFT = PREMATCH + ["hero_exp_decay_diff"]      # только диагностика DRAFT_T2
SHRINK_ALPHA = 32.0                                  # Phase 18, подтверждено на TEST


def frame(matches, rp, emit, horizon, include_draft=False) -> pd.DataFrame:
    rows = build_point_in_time_features(matches, horizon, roster_provider=rp,
                                        include_draft=include_draft, emit_only=emit)
    return pd.DataFrame([{
        "match_id": r.match_id, "as_of_timestamp": r.start_time,
        "prediction_at": r.prediction_at,
        "elo_difference": r.elo_difference, "form_3_difference": r.form_3_difference,
        "elo_mean_diff": r.elo_mean_diff, "five_vs_team_elo_diff": r.five_vs_team_elo_diff,
        "hero_exp_decay_diff": r.hero_exp_decay_diff,
        "pool_meta_diff": r.pool_meta_diff, "pool_size_min": r.pool_size_min,
        "k_r": r.roster_known_radiant, "k_d": r.roster_known_dire,
        "r_played": r.radiant_matches_before, "d_played": r.dire_matches_before,
        "target": r.target,
    } for r in rows]).sort_values(["as_of_timestamp", "match_id"]).reset_index(drop=True)


def shrink(df: pd.DataFrame, alpha: float = SHRINK_ALPHA) -> pd.DataFrame:
    """Вес Phase 18 w(k)=k/(k+alpha) по числу известных игроков.

    При полном составе k=5 вес постоянен, а StandardScaler поглощает
    общий множитель точно — то есть для этой серии операция тождественна.
    Объявлено в плане заранее.
    """
    out = df.copy()
    k = out[["k_r", "k_d"]].min(axis=1).astype(float)
    w = k / (k + alpha)
    for c in ("elo_mean_diff", "five_vs_team_elo_diff"):
        out[c] = out[c] * w
    return out


def fit(feats: List[str], tr: pd.DataFrame) -> LogisticRegressionModel:
    m = LogisticRegressionModel(feature_names=feats, random_state=RANDOM_SEED)
    m.fit(tr, tr["target"])
    return m


def proba(model, df: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(df))[:, 1]


def main(argv=None) -> int:
    section("PHASE 19 — case study: Vision vs Team Spirit, TI15, 23.08.2026")
    print(f"git commit: {git_commit_sha()}")
    print("Все решения зафиксированы в docs/phase19-plan.md ДО этого прогона.\n")

    engine = make_engine(load_settings())
    with engine.connect() as c:
        meta = {r.match_id: r for r in c.execute(text("""
            select match_id, start_time, radiant_team_id, dire_team_id, radiant_win, patch_id
            from matches where league_id=:l and start_time>=:t order by start_time, match_id
        """), {"l": LEAGUE, "t": SUBSET_FROM})}
    subset = sorted(meta)
    print(f"Subset по объявленному правилу (лига {LEAGUE}, start ≥ "
          f"{SUBSET_FROM:%Y-%m-%d}): N = {len(subset)}")
    if len(subset) < 10:
        print("  ВНИМАНИЕ: правило потребовало бы сдвига границы — см. план")

    ref = load_common_set()
    emit = set(ref["match_id"]) | set(SERIES) | set(subset)
    matches, rp = load_pit_matches(engine)
    print(f"Матчей в потоке обновлений: {len(matches):,}; в выдаче: {len(emit):,}\n")

    payload = {"phase": 19, "commit": git_commit_sha(), "series_id": SERIES_ID,
               "series": SERIES, "subset_n": len(subset), "predictions": {},
               "test_accesses": 0}

    # ---------- обучение: один раз, на TRAIN-доле, горизонт 24 ч ----------
    section("Обучение PREMATCH_BASELINE")
    base = shrink(frame(matches, rp, emit, timedelta(hours=24)))
    tr, va, te = split(base)
    print(f"TRAIN: {len(tr):,} матчей, {tr['as_of_timestamp'].min():%Y-%m-%d} … "
          f"{tr['as_of_timestamp'].max():%Y-%m-%d}")
    print(f"VAL:   {len(va):,} матчей, до {va['as_of_timestamp'].max():%Y-%m-%d}")
    print(f"TEST:  {len(te):,} матчей, до {te['as_of_timestamp'].max():%Y-%m-%d}")
    assert tr["as_of_timestamp"].max() < datetime(2026, 8, 23, tzinfo=timezone.utc), \
        "целевой матч не может лежать в обучении"
    print(f"\nГраница TRAIN на {(datetime(2026,8,23,tzinfo=timezone.utc) - tr['as_of_timestamp'].max()).days} "
          f"дней раньше целевого матча — попасть в обучение он не может.")
    payload["train"] = {"n": len(tr), "end": str(tr["as_of_timestamp"].max())}

    m_pre = fit(PREMATCH, tr)
    m_team = fit(TEAM_ONLY, tr)
    print(f"\nPREMATCH_BASELINE обучен на {len(tr):,} матчах, признаки: {PREMATCH}")
    print(f"TEAM_ONLY обучен на тех же матчах, признаки: {TEAM_ONLY}")

    # ---------- прогнозы на каждый момент T ----------
    section("Прогнозы (PART J)")
    print("p — вероятность победы RADIANT. Сторона Spirit указана отдельно.\n")
    per_delta = {}
    for label, d in DELTAS:
        f = shrink(frame(matches, rp, emit, d))
        f = f.set_index("match_id")
        rows = []
        for i, mid in enumerate(SERIES, 1):
            if mid not in f.index:
                rows.append({"game": i, "match_id": mid, "status": "ABSTAIN",
                             "reason": "признаки не построены"})
                continue
            r = f.loc[[mid]]
            p_r = float(proba(m_pre, r)[0])
            p_t = float(proba(m_team, r)[0])
            spirit_radiant = meta[mid].radiant_team_id == SPIRIT
            rows.append({
                "game": i, "match_id": mid, "status": "PARTIAL_DATA",
                "prediction_at": str(r["prediction_at"].iloc[0]),
                "spirit_radiant": bool(spirit_radiant),
                "p_radiant": p_r,
                "p_spirit": p_r if spirit_radiant else 1.0 - p_r,
                "p_spirit_team_only": p_t if spirit_radiant else 1.0 - p_t,
                "elo_difference": float(r["elo_difference"].iloc[0]),
                "form_3_difference": _f(r["form_3_difference"].iloc[0]),
                "elo_mean_diff": _f(r["elo_mean_diff"].iloc[0]),
                "five_vs_team_elo_diff": _f(r["five_vs_team_elo_diff"].iloc[0]),
                "pool_meta_diff": _f(r["pool_meta_diff"].iloc[0]),
                "k_r": int(r["k_r"].iloc[0]), "k_d": int(r["k_d"].iloc[0]),
            })
        per_delta[label] = rows
        print(f"{label}:")
        for r in rows:
            if r["status"] == "ABSTAIN":
                print(f"  игра {r['game']}  ABSTAIN — {r['reason']}")
            else:
                print(f"  игра {r['game']}  P(Spirit)={r['p_spirit']:.4f}   "
                      f"только команда={r['p_spirit_team_only']:.4f}   "
                      f"состав {r['k_r']}/{r['k_d']}   T={r['prediction_at'][:19]}")
        print()
    payload["predictions"] = per_delta

    # ---------- диагностика DRAFT_T2 ----------
    section("Диагностика DRAFT_T2 (пост-драфт, НЕ pre-match)")
    print("Замороженная модель Phase 9 целиком: пять входов, включая драфт.")
    print("Показано для сравнения; прогнозом фазы НЕ является.\n")
    fd = shrink(frame(matches, rp, emit, timedelta(0), include_draft=True))
    trd, _, _ = split(fd)
    m_full = fit(POSTDRAFT, trd)
    fdi = fd.set_index("match_id")
    draft_rows = []
    for i, mid in enumerate(SERIES, 1):
        if mid not in fdi.index:
            continue
        r = fdi.loc[[mid]]
        p_r = float(proba(m_full, r)[0])
        sr = meta[mid].radiant_team_id == SPIRIT
        ps = p_r if sr else 1.0 - p_r
        draft_rows.append({"game": i, "match_id": mid, "p_spirit": ps,
                           "hero_exp_decay_diff": _f(r["hero_exp_decay_diff"].iloc[0])})
        print(f"  игра {i}  P(Spirit)={ps:.4f}  hero_exp_decay_diff="
              f"{r['hero_exp_decay_diff'].iloc[0]:+.4f}")
    payload["draft_t2"] = draft_rows

    # ---------- subset (PART M / N) ----------
    section("Case-study subset (PART M) — exploratory")
    print(f"Правило объявлено заранее: лига {LEAGUE}, start ≥ {SUBSET_FROM:%Y-%m-%d}. "
          f"N = {len(subset)}\n")
    sub_res = {}
    for label, d in DELTAS:
        f = shrink(frame(matches, rp, emit, d))
        f = f[f["match_id"].isin(subset)]
        if f.empty:
            continue
        y = f["target"].to_numpy()
        res = {}
        for name, mdl, feats in (("PREMATCH_BASELINE", m_pre, PREMATCH),
                                 ("только команда", m_team, TEAM_ONLY)):
            p = proba(mdl, f)
            mm = compute_metrics(y, p)
            res[name] = {"n": int(mm["n"]), "accuracy": mm["accuracy"],
                         "roc_auc": mm["roc_auc"], "log_loss": mm["log_loss"],
                         "brier": mm["brier_score"]}
        sub_res[label] = res
        a, b = res["PREMATCH_BASELINE"], res["только команда"]
        print(f"{label}  n={a['n']:>3}  "
              f"состав: acc={a['accuracy']:.4f} ll={a['log_loss']:.4f} auc={a['roc_auc']:.4f}   |   "
              f"команда: acc={b['accuracy']:.4f} ll={b['log_loss']:.4f} auc={b['roc_auc']:.4f}")
    payload["subset"] = sub_res

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    out = os.path.join(EXPERIMENTS_DIR, "phase19_case_study.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\nОбращений к TEST-протоколу прошлых фаз: 0\nРезультаты: {out}")
    return 0


def _f(v) -> Optional[float]:
    return None if v is None or v != v else float(v)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
