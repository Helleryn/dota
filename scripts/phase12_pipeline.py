#!/usr/bin/env python3
"""
PHASE 12 — draft-aware прогноз: эксперименты по гипотезам H1-H8.

Протокол (как в исправленной Phase 9/10/11):
  * отбор признаков ТОЛЬКО по VALIDATION;
  * TEST — один прогон по заранее зафиксированному списку, только с флагом
    --final-test;
  * число обращений к TEST печатается для аудита.

Frozen baseline (Elo K=16 + Form3) и набор Phase 9 не меняются.

Гипотезы и критерии опровержения — reports/phase12-draft-feasibility.md.

Запуск:
    python3 scripts/phase12_pipeline.py                 # только VALIDATION
    python3 scripts/phase12_pipeline.py --final-test    # + единственный TEST
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.config import load_settings
from src.datasets.builder import _load_pro_matches
from src.datasets.draft_state import build_draft_state_features, classify_draft
from src.datasets.hero_strength_schemes import build_hero_scheme_features
from src.datasets.multi_window_features import build_multi_window_features
from src.datasets.roster_representation import build_roster_representation, to_feature_dict
from src.db.engine import make_engine
from src.evaluation.statistics import (
    accuracy_metric,
    block_bootstrap_paired_diff,
    log_loss_metric,
    mcnemar_exact,
)
from src.models.sklearn_models import RANDOM_SEED, LogisticRegressionModel
from scripts.phase6_pipeline import git_commit_sha, section

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")

ELO_K, FORM_WINDOW, BLOCK_SIZE = 16, 3, 20
PHASE9 = ["elo_difference", "form_3_difference", "elo_mean_diff",
          "five_vs_team_elo_diff", "hero_exp_decay_diff"]
# Порог отбора ужесточён после Phase 9 (там слишком мягкий порог впустил
# признак с нулевым эффектом) — см. reports/phase9-roster-draft.md, раздел 12.
MIN_LOGLOSS_GAIN = 1e-4
REVEAL_GRID = [0, 4, 8, 12, 16, 20, None]
TEST_ACCESS = {"n": 0}


@dataclass(frozen=True)
class Ac:
    ord: int
    is_pick: bool
    hero_id: Optional[int]
    team: int


@dataclass(frozen=True)
class Pl:
    account_id: int
    hero_id: int
    is_radiant: bool
    lane_role: Optional[int]


@dataclass(frozen=True)
class Mt:
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_roster: frozenset
    dire_roster: frozenset
    radiant_picks: tuple
    dire_picks: tuple
    radiant_bans: tuple
    dire_bans: tuple
    patch_id: Optional[int]
    radiant_win: bool
    players: Sequence[Pl]
    actions: Sequence[Ac]


def load_matches(engine) -> List[Mt]:
    with engine.connect() as conn:
        base = conn.execute(text("""
            WITH drafts AS (
              SELECT pb.match_id,
                     array_agg(pb.hero_id ORDER BY pb.ord) FILTER (WHERE pb.is_pick AND pb.team=0) AS r_picks,
                     array_agg(pb.hero_id ORDER BY pb.ord) FILTER (WHERE pb.is_pick AND pb.team=1) AS d_picks,
                     array_agg(pb.hero_id ORDER BY pb.ord) FILTER (WHERE NOT pb.is_pick AND pb.team=0) AS r_bans,
                     array_agg(pb.hero_id ORDER BY pb.ord) FILTER (WHERE NOT pb.is_pick AND pb.team=1) AS d_bans
              FROM picks_bans pb GROUP BY 1
            )
            SELECT m.match_id, m.start_time, m.patch_id, m.radiant_team_id, m.dire_team_id, m.radiant_win,
                   d.r_picks, d.d_picks, d.r_bans, d.d_bans
            FROM matches m
            JOIN leagues l ON l.league_id=m.league_id
            JOIN drafts d ON d.match_id=m.match_id
            WHERE l.tier IN ('professional','premium')
              AND m.radiant_team_id IS NOT NULL AND m.dire_team_id IS NOT NULL
              AND m.radiant_win IS NOT NULL
              AND array_length(d.r_picks,1)=5 AND array_length(d.d_picks,1)=5
            ORDER BY m.start_time, m.match_id
        """)).fetchall()
        players = conn.execute(text("""
            SELECT mp.match_id, mp.account_id, mp.hero_id, mp.is_radiant, mp.lane_role
            FROM match_players mp WHERE mp.account_id IS NOT NULL
        """)).fetchall()
        acts = conn.execute(text("""
            SELECT pb.match_id, pb.ord, pb.is_pick, pb.hero_id, pb.team
            FROM picks_bans pb ORDER BY pb.match_id, pb.ord
        """)).fetchall()

    by_match: Dict[int, list] = {}
    for p in players:
        by_match.setdefault(p.match_id, []).append(Pl(p.account_id, p.hero_id, p.is_radiant, p.lane_role))
    acts_by: Dict[int, list] = {}
    for a in acts:
        acts_by.setdefault(a.match_id, []).append(Ac(a.ord, a.is_pick, a.hero_id, a.team))

    out: List[Mt] = []
    for r in base:
        ps = by_match.get(r.match_id)
        aa = acts_by.get(r.match_id)
        if not ps or len(ps) != 10 or not aa:
            continue
        rr = frozenset(p.account_id for p in ps if p.is_radiant)
        dr = frozenset(p.account_id for p in ps if not p.is_radiant)
        if len(rr) != 5 or len(dr) != 5:
            continue
        st = r.start_time if r.start_time.tzinfo else r.start_time.replace(tzinfo=timezone.utc)
        out.append(Mt(r.match_id, st, r.radiant_team_id, r.dire_team_id, rr, dr,
                      tuple(r.r_picks), tuple(r.d_picks), tuple(r.r_bans or ()), tuple(r.d_bans or ()),
                      r.patch_id, r.radiant_win, ps, aa))
    return out


def build_base_dataset(engine, matches) -> pd.DataFrame:
    """Frozen baseline + KEEP-набор Phase 9. Не меняется в этой фазе."""
    ids = {m.match_id for m in matches}
    raw = _load_pro_matches(engine)
    b = build_multi_window_features(raw, k_factor=ELO_K)
    base = pd.DataFrame([{
        "match_id": r.match_id, "as_of_timestamp": r.as_of_timestamp,
        "elo_difference": r.elo_difference,
        "form_3_difference": r.recent_winrate_difference[FORM_WINDOW],
        "radiant_matches_played_before": r.radiant_matches_played_before,
        "dire_matches_played_before": r.dire_matches_played_before,
        "target": int(r.radiant_win),
    } for r in b])
    roster = pd.DataFrame([to_feature_dict(r) for r in build_roster_representation(matches, k_factor=ELO_K)])
    roster = roster[["match_id", "elo_mean_diff", "five_vs_team_elo_diff"]]
    hs = build_hero_scheme_features(matches)
    hero = pd.DataFrame([{"match_id": r.match_id,
                          "hero_exp_decay_diff": r.strength_diff["exp_decay"]} for r in hs])
    df = base[base["match_id"].isin(ids)].merge(roster, on="match_id").merge(hero, on="match_id")
    df = df[(df["radiant_matches_played_before"] > 0) & (df["dire_matches_played_before"] > 0)]
    return df.sort_values(["as_of_timestamp", "match_id"]).reset_index(drop=True)


DS_COLS = ["denied_comfort_diff", "denied_comfort_top_diff",
           "order_weighted_hero_strength_diff", "last_pick_counter_diff",
           "lane_balance_diff", "lane_prior_entropy_diff"]


CACHE_DIR = os.path.join(EXPERIMENTS_DIR, "cache")


def draft_frame(matches, reveal: Optional[int]) -> pd.DataFrame:
    """Кэш на диске: один walk-forward проход по 110k матчей стоит минуты,
    а информационная кривая H6 требует их семь. Кэш инвалидируется вручную
    при изменении draft_state.py (файл лежит в reports/experiments/cache)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"draft_state_reveal_{'full' if reveal is None else reveal}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    rows = build_draft_state_features(matches, reveal=reveal)
    df = pd.DataFrame([{
        "match_id": r.match_id,
        "draft_format": r.draft_format,
        **{c: getattr(r, c) for c in DS_COLS},
    } for r in rows])
    df.to_csv(path, index=False)
    return df


def split(df, tf=0.70, vf=0.15):
    n = len(df)
    a, b = int(n * tf), int(n * (tf + vf))
    tr, va, te = (df.iloc[:a].reset_index(drop=True),
                  df.iloc[a:b].reset_index(drop=True),
                  df.iloc[b:].reset_index(drop=True))
    assert tr["as_of_timestamp"].max() < va["as_of_timestamp"].min()
    assert va["as_of_timestamp"].max() < te["as_of_timestamp"].min()
    return tr, va, te


def fit(feats, train):
    m = LogisticRegressionModel(feature_names=feats, random_state=RANDOM_SEED)
    m.fit(train, train["target"])
    return m


def show(name, m, ref=None):
    d = ""
    if ref is not None:
        d = f"  Δacc={m['accuracy']-ref['accuracy']:+.4f}  Δll={m['log_loss']-ref['log_loss']:+.5f}"
    print(f"{name:48s} acc={m['accuracy']:.4f} ll={m['log_loss']:.4f} auc={m['roc_auc']:.4f}{d}", flush=True)


def calibration_table(y, p, bins=10):
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    out = []
    for b in range(bins):
        sel = idx == b
        if sel.sum() == 0:
            continue
        out.append((b / bins, (b + 1) / bins, int(sel.sum()), float(p[sel].mean()), float(y[sel].mean())))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--final-test", action="store_true",
                    help="единственный финальный прогон по TEST с зафиксированным набором")
    args = ap.parse_args(argv)

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    commit, run_ts = git_commit_sha(), datetime.now(timezone.utc).isoformat()
    section("PHASE 12 — draft-aware прогноз, состояние драфта и неопределённость")
    print(f"git commit: {commit}")
    print("Протокол: отбор по VALIDATION; TEST только с --final-test.")

    engine = make_engine(load_settings())
    matches = load_matches(engine)
    print(f"Матчей (10 игроков + полный драфт + действия): {len(matches)}", flush=True)

    fmts: Dict[str, int] = {}
    for m in matches:
        f = classify_draft(m.actions)
        fmts[f] = fmts.get(f, 0) + 1
    print("Форматы драфта:", ", ".join(f"{k}={v}" for k, v in sorted(fmts.items())))

    df0 = build_base_dataset(engine, matches)
    print(f"Строк после исключения team cold-start: {len(df0)}", flush=True)

    section("Полный драфт: признаки состояния (reveal=None)")
    full = draft_frame(matches, None)
    df = df0.merge(full, on="match_id")
    df = df[df["draft_format"] != "other"].reset_index(drop=True)
    print(f"Строк после отсева не-Captains-Mode: {len(df)}")
    train, val, test = split(df)
    print(f"TRAIN {len(train)} | VAL {len(val)} | TEST {len(test)}")
    print(f"VAL форматы: {dict(val['draft_format'].value_counts())}")
    print(f"TEST форматы: {dict(test['draft_format'].value_counts())}")

    section("Диагностика признаков (TRAIN+VAL, TEST не затрагивается)")
    diag = pd.concat([train, val], ignore_index=True)
    print(f"  {'признак':38s} {'std':>10s} {'доля!=0':>9s} {'corr с исходом':>15s}")
    for c in DS_COLS:
        v = diag[c].to_numpy(dtype=float)
        nz = float((np.abs(v) > 1e-12).mean())
        sd = float(v.std())
        corr = float(np.corrcoef(v, diag["target"].to_numpy(dtype=float))[0, 1]) if sd > 0 else float("nan")
        print(f"  {c:38s} {sd:10.5f} {nz:9.3f} {corr:+15.4f}")
    print("  (нулевая std или доля!=0 == 0 означала бы дефект признака, а не отсутствие сигнала)")

    section("Эксперименты на VALIDATION (H1-H4)")
    EXP: Dict[str, List[str]] = {
        "E0: Phase 9 (контроль)": PHASE9,
        "E1 H1: + denied_comfort": PHASE9 + ["denied_comfort_diff"],
        "E2 H1: + denied_comfort_top": PHASE9 + ["denied_comfort_top_diff"],
        "E3 H2: + order_weighted_hero": PHASE9 + ["order_weighted_hero_strength_diff"],
        "E4 H3: + last_pick_counter": PHASE9 + ["last_pick_counter_diff"],
        "E5 H4: + lane_composition": PHASE9 + ["lane_balance_diff", "lane_prior_entropy_diff"],
        "E6: все признаки Phase 12": PHASE9 + DS_COLS,
    }
    rng = np.random.default_rng(RANDOM_SEED)
    for d in (train, val, test):
        d["negative_control"] = rng.normal(size=len(d))
    EXP["E7: negative control (шум)"] = PHASE9 + ["negative_control"]

    val_res: Dict[str, dict] = {}
    models: Dict[str, object] = {}
    ref = None
    for name, feats in EXP.items():
        mdl = fit(feats, train)
        models[name] = mdl
        val_res[name] = mdl.evaluate(val, val["target"])
        if ref is None:
            ref = val_res[name]
        show(name, val_res[name], None if name.startswith("E0") else ref)

    section("Отбор по VALIDATION (порог Δlog loss > 1e-4)")
    keep: List[str] = []
    verdicts: Dict[str, str] = {}
    single = {"denied_comfort_diff": "E1 H1: + denied_comfort",
              "denied_comfort_top_diff": "E2 H1: + denied_comfort_top",
              "order_weighted_hero_strength_diff": "E3 H2: + order_weighted_hero",
              "last_pick_counter_diff": "E4 H3: + last_pick_counter"}
    for col, exp in single.items():
        gain = ref["log_loss"] - val_res[exp]["log_loss"]
        ok = gain > MIN_LOGLOSS_GAIN
        verdicts[col] = "KEEP" if ok else "REMOVE"
        print(f"  {col:38s} Δll={gain:+.5f}  ->  {verdicts[col]}")
        if ok:
            keep.append(col)
    gain5 = ref["log_loss"] - val_res["E5 H4: + lane_composition"]["log_loss"]
    ok5 = gain5 > MIN_LOGLOSS_GAIN
    verdicts["lane_composition"] = "KEEP" if ok5 else "REMOVE"
    print(f"  {'lane_composition (2 признака)':38s} Δll={gain5:+.5f}  ->  {verdicts['lane_composition']}")
    if ok5:
        keep += ["lane_balance_diff", "lane_prior_entropy_diff"]
    # denied_comfort: две формулировки одной гипотезы — оставляем лучшую
    if "denied_comfort_diff" in keep and "denied_comfort_top_diff" in keep:
        worse = ("denied_comfort_top_diff"
                 if val_res[single["denied_comfort_diff"]]["log_loss"]
                 <= val_res[single["denied_comfort_top_diff"]]["log_loss"]
                 else "denied_comfort_diff")
        keep.remove(worse)
        verdicts[worse] = "REMOVE (дубликат гипотезы H1)"
        print(f"  дубликат H1 удалён: {worse}")

    final_feats = PHASE9 + keep
    print(f"\n>>> Зафиксированный набор Phase 12: {final_feats}")
    if keep:
        mdl_final = fit(final_feats, train)
        val_final = mdl_final.evaluate(val, val["target"])
        show("ФИНАЛЬНЫЙ набор (VAL)", val_final, ref)
    else:
        mdl_final, val_final = models["E0: Phase 9 (контроль)"], ref
        print("Ни один признак Phase 12 не прошёл порог — финальный набор = Phase 9.")

    section("H6 — информационная кривая: сколько даёт частично раскрытый драфт")
    curve = []
    for k in REVEAL_GRID:
        if k is None:
            dk = df
        else:
            fk = draft_frame(matches, k)
            dk = df0.merge(fk, on="match_id")
            dk = dk[dk["draft_format"] != "other"].reset_index(drop=True)
        trk, vak, _ = split(dk)
        cols = keep if keep else DS_COLS
        mk = fit(PHASE9 + list(cols), trk)
        r = mk.evaluate(vak, vak["target"])
        curve.append({"reveal": k, **{x: r[x] for x in ("accuracy", "log_loss", "roc_auc")}})
        show(f"raскрыто действий: {k if k is not None else 'весь драфт'}", r, ref)

    section("H7 — калибровка (VALIDATION)")
    p_val = mdl_final.predict_proba(val)[:, 1]
    y_val = val["target"].to_numpy()
    print("  bin        n     ср.прогноз   факт")
    for lo, hi, n, mp, my in calibration_table(y_val, p_val):
        print(f"  [{lo:.1f},{hi:.1f})  {n:6d}    {mp:.4f}     {my:.4f}")
    ece = sum(n * abs(mp - my) for _, _, n, mp, my in calibration_table(y_val, p_val)) / len(y_val)
    print(f"  ECE (10 бинов) = {ece:.5f}")

    section("H8 — селективный прогноз: порог выбирается на VALIDATION")
    conf = np.abs(p_val - 0.5)
    sel_rows = []
    for cov in (1.0, 0.8, 0.6, 0.5, 0.3, 0.2):
        thr = np.quantile(conf, 1.0 - cov)
        sel = conf >= thr
        acc = float((y_val[sel] == (p_val[sel] > 0.5)).mean())
        sel_rows.append({"coverage": cov, "threshold": float(thr), "n": int(sel.sum()), "accuracy": acc})
        print(f"  покрытие {cov*100:5.1f}%  порог |p-0.5|>={thr:.4f}  n={sel.sum():6d}  acc={acc:.4f}")

    payload = {
        "phase": 12, "git_commit": commit, "run_timestamp": run_ts,
        "n_matches": len(matches), "n_rows": len(df),
        "draft_formats": fmts,
        "validation": {k: v for k, v in val_res.items()},
        "verdicts": verdicts,
        "final_features": final_feats,
        "reveal_curve": curve,
        "calibration_ece_val": ece,
        "selective_val": sel_rows,
        "test_accesses": TEST_ACCESS["n"],
    }

    if args.final_test:
        section("ЕДИНСТВЕННОЕ обращение к TEST")
        TEST_ACCESS["n"] += 1
        base_mdl = models["E0: Phase 9 (контроль)"]
        t_base = base_mdl.evaluate(test, test["target"])
        t_final = mdl_final.evaluate(test, test["target"])
        show("Phase 9 (контроль)", t_base)
        y = test["target"].to_numpy()
        pb = base_mdl.predict_proba(test)[:, 1]
        pf = mdl_final.predict_proba(test)[:, 1]
        payload["test"] = {"phase9": t_base, "phase12": t_final}

        if keep:
            show("Phase 12 финальный набор", t_final, t_base)
            da, ca = block_bootstrap_paired_diff(y, pb, pf, accuracy_metric, block_size=BLOCK_SIZE, seed=RANDOM_SEED)
            dl, cl = block_bootstrap_paired_diff(y, pb, pf, log_loss_metric, block_size=BLOCK_SIZE, seed=RANDOM_SEED)
            p_mc = mcnemar_exact(y, pb > 0.5, pf > 0.5)
            print(f"  Δacc={da:+.4f} CI={ca}   Δll={dl:+.5f} CI={cl}   McNemar p={p_mc:.4g}")
            payload["test"].update({"delta_accuracy": da, "ci_accuracy": list(ca),
                                    "delta_log_loss": dl, "ci_log_loss": list(cl),
                                    "mcnemar_p": p_mc})
        else:
            # Ни один признак Phase 12 не прошёл отбор, поэтому финальная модель
            # ТОЖДЕСТВЕННА Phase 9. Парное сравнение модели с самой собой
            # вырождено (Δ ровно 0, McNemar не определён) — печатать его как
            # результат было бы имитацией эксперимента.
            print("  Признаки Phase 12 отвергнуты на VALIDATION => финальная модель = Phase 9.")
            print("  Парное сравнение не проводится: модель сравнивалась бы сама с собой.")
            payload["test"]["paired_comparison"] = "не проводилось: набор признаков не изменился"

        section("H8 на TEST — пороги ЗАФИКСИРОВАНЫ на VALIDATION")
        confT = np.abs(pf - 0.5)
        sel_test = []
        for r in sel_rows:
            thr = r["threshold"]
            selT = confT >= thr
            n = int(selT.sum())
            accT = float((y[selT] == (pf[selT] > 0.5)).mean()) if n else float("nan")
            cov = float(selT.mean())
            sel_test.append({"coverage_val": r["coverage"], "threshold_from_val": thr,
                             "coverage_test": cov, "n": n, "accuracy": accT})
            print(f"  порог с VAL {thr:.4f} (VAL покрытие {r['coverage']*100:5.1f}%)"
                  f"  ->  TEST покрытие={cov*100:5.1f}%  n={n:6d}  acc={accT:.4f}")
        payload["test"]["selective"] = sel_test

        ece_t = (sum(n * abs(mp - my) for _, _, n, mp, my in calibration_table(y, pf))
                 / len(y))
        print(f"\n  ECE на TEST (10 бинов) = {ece_t:.5f}")
        payload["test"]["calibration_ece"] = ece_t
        payload["test_accesses"] = TEST_ACCESS["n"]

    out = os.path.join(EXPERIMENTS_DIR, "phase12.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nОбращений к TEST: {TEST_ACCESS['n']}")
    print(f"Результаты: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
