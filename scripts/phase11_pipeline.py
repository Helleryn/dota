#!/usr/bin/env python3
"""
PHASE 11 — Player x Hero x Role x Patch/Meta: эксперименты.

Протокол (как в исправленной Phase 9/10):
  * ВСЁ, что выбирается (half-life, состав финальной модели) — по VALIDATION;
  * TEST — один финальный прогон по заранее зафиксированному списку;
  * число обращений к TEST печатается для аудита.

Frozen baseline (Elo + Form3) и набор Phase 9 не меняются.

Запуск:
    python3 scripts/phase11_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.config import load_settings
from src.datasets.builder import _load_pro_matches
from src.datasets.hero_strength_schemes import build_hero_scheme_features
from src.datasets.multi_window_features import build_multi_window_features
from src.datasets.player_hero_features import build_player_hero_features
from src.datasets.roster_representation import build_roster_representation, to_feature_dict
from src.db.engine import make_engine
from src.evaluation.metrics import compute_metrics
from src.evaluation.statistics import (
    accuracy_metric,
    block_bootstrap_paired_diff,
    log_loss_metric,
    mcnemar_exact,
)
from src.models.sklearn_models import RANDOM_SEED, CatBoostModel, LogisticRegressionModel
from scripts.phase6_pipeline import git_commit_sha, section

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")

ELO_K, FORM_WINDOW, BLOCK_SIZE = 16, 3, 20
PHASE9 = ["elo_difference", "form_3_difference", "elo_mean_diff",
          "five_vs_team_elo_diff", "hero_exp_decay_diff"]
HALF_LIVES = [7, 14, 30, 60, 90, 180]
TEST_ACCESS = {"n": 0}


@dataclass(frozen=True)
class Pl:
    account_id: int
    hero_id: int
    is_radiant: bool
    lane_role: Optional[int]
    gold_per_min: Optional[int]


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


def load_matches(engine):
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
            SELECT mp.match_id, mp.account_id, mp.hero_id, mp.is_radiant,
                   mp.lane_role, mp.gold_per_min
            FROM match_players mp
            WHERE mp.account_id IS NOT NULL
        """)).fetchall()

    by_match = {}
    for p in players:
        by_match.setdefault(p.match_id, []).append(
            Pl(p.account_id, p.hero_id, p.is_radiant, p.lane_role, p.gold_per_min))

    out = []
    for r in base:
        ps = by_match.get(r.match_id)
        if not ps or len(ps) != 10:
            continue
        rr = frozenset(p.account_id for p in ps if p.is_radiant)
        dr = frozenset(p.account_id for p in ps if not p.is_radiant)
        if len(rr) != 5 or len(dr) != 5:
            continue
        st = r.start_time if r.start_time.tzinfo else r.start_time.replace(tzinfo=timezone.utc)
        out.append(Mt(r.match_id, st, r.radiant_team_id, r.dire_team_id, rr, dr,
                      tuple(r.r_picks), tuple(r.d_picks), tuple(r.r_bans or ()), tuple(r.d_bans or ()),
                      r.patch_id, r.radiant_win, ps))
    return out


def build_dataset(engine, matches):
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

    df = (base[base["match_id"].isin(ids)].merge(roster, on="match_id").merge(hero, on="match_id"))
    df = df[(df["radiant_matches_played_before"] > 0) & (df["dire_matches_played_before"] > 0)]
    return df.sort_values(["as_of_timestamp", "match_id"]).reset_index(drop=True)


def ph_frame(matches, hl, suffix=""):
    rows = build_player_hero_features(matches, half_life_days=hl)
    return pd.DataFrame([{
        "match_id": r.match_id,
        f"player_hero_strength_diff{suffix}": r.player_hero_strength_diff,
        f"player_role_strength_diff{suffix}": r.player_role_strength_diff,
        f"player_hero_role_strength_diff{suffix}": r.player_hero_role_strength_diff,
        f"role_entropy_diff{suffix}": r.role_entropy_diff,
        f"meta_relative_ph_diff{suffix}": r.meta_relative_ph_diff,
    } for r in rows])


def split(df, tf=0.70, vf=0.15):
    n = len(df); a, b = int(n*tf), int(n*(tf+vf))
    tr, va, te = df.iloc[:a].reset_index(drop=True), df.iloc[a:b].reset_index(drop=True), df.iloc[b:].reset_index(drop=True)
    assert tr["as_of_timestamp"].max() < va["as_of_timestamp"].min()
    assert va["as_of_timestamp"].max() < te["as_of_timestamp"].min()
    return tr, va, te


def fit(feats, train):
    m = LogisticRegressionModel(feature_names=feats, random_state=RANDOM_SEED)
    m.fit(train, train["target"]); return m


def show(name, m, ref=None):
    d = f"  Δacc={m['accuracy']-ref:+.4f}" if ref is not None else ""
    print(f"{name:44s} acc={m['accuracy']:.4f} ll={m['log_loss']:.4f} auc={m['roc_auc']:.4f}{d}")


def main() -> int:
    os.makedirs(FIGURES_DIR, exist_ok=True); os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    commit, run_ts = git_commit_sha(), datetime.now(timezone.utc).isoformat()
    section("PHASE 11 — Player x Hero x Role x Patch/Meta")
    print(f"git commit: {commit}")
    print("Протокол: отбор по VALIDATION, TEST один раз в конце.")

    engine = make_engine(load_settings())
    matches = load_matches(engine)
    print(f"Матчей (10 игроков + полный драфт): {len(matches)}")

    df = build_dataset(engine, matches)
    print(f"Строк после исключения team cold-start: {len(df)}")

    section("F2 — выбор half-life ТОЛЬКО по VALIDATION")
    hl_val = {}
    frames = {}
    for hl in HALF_LIVES:
        f = ph_frame(matches, hl)
        frames[hl] = f
        d = df.merge(f, on="match_id")
        tr, va, _ = split(d)
        m = fit(PHASE9 + ["player_hero_strength_diff"], tr)
        hl_val[hl] = m.evaluate(va, va["target"])
        show(f"half-life={hl}д", hl_val[hl])
    best_hl = min(hl_val, key=lambda k: hl_val[k]["log_loss"])
    print(f"\n>>> ПОБЕДИТЕЛЬ по VAL log_loss: half-life={best_hl} дней")

    df = df.merge(frames[best_hl], on="match_id")
    train, val, test = split(df)
    print(f"TRAIN {len(train)} | VAL {len(val)} | TEST {len(test)}")

    section("Эксперименты E0-E6 + negative control (VALIDATION)")
    EXP = {
        "E0: Phase 9 (контроль)": PHASE9,
        "E1: + player_hero": PHASE9 + ["player_hero_strength_diff"],
        "E2: + player_role": PHASE9 + ["player_role_strength_diff"],
        "E3: + player_hero_role": PHASE9 + ["player_hero_role_strength_diff"],
        "E4: + role_entropy": PHASE9 + ["role_entropy_diff"],
        "E5: + meta-relative ph": PHASE9 + ["meta_relative_ph_diff"],
    }
    val_res = {}
    for name, feats in EXP.items():
        val_res[name] = fit(feats, train).evaluate(val, val["target"])
        show(f"[VAL] {name}", val_res[name], val_res["E0: Phase 9 (контроль)"]["accuracy"])

    base_ll = val_res["E0: Phase 9 (контроль)"]["log_loss"]
    winners = []
    for name, feats in EXP.items():
        if name.startswith("E0"):
            continue
        # требуем ЗАМЕТНОЕ улучшение, а не любое (урок Phase 9: слишком мягкий порог
        # пропустил нулевой признак в финальный набор)
        if val_res[name]["log_loss"] < base_ll - 1e-4:
            winners.extend([f for f in feats if f not in PHASE9])
    winners = list(dict.fromkeys(winners))
    print(f"\nПрошли порог VAL (улучшение log_loss > 1e-4): {winners or 'НИЧЕГО'}")

    EXP["E6: всё, что прошло VAL"] = PHASE9 + winners if winners else PHASE9
    val_res["E6: всё, что прошло VAL"] = fit(EXP["E6: всё, что прошло VAL"], train).evaluate(val, val["target"])
    show("[VAL] E6: всё, что прошло VAL", val_res["E6: всё, что прошло VAL"],
         val_res["E0: Phase 9 (контроль)"]["accuracy"])

    section("ФИНАЛЬНАЯ ОЦЕНКА НА TEST")
    FINAL = {k: EXP[k] for k in ["E0: Phase 9 (контроль)", "E1: + player_hero",
                                  "E2: + player_role", "E3: + player_hero_role",
                                  "E4: + role_entropy", "E5: + meta-relative ph",
                                  "E6: всё, что прошло VAL"]}
    models, test_res = {}, {}
    for name, feats in FINAL.items():
        m = fit(feats, train); models[name] = m
        TEST_ACCESS["n"] += 1
        test_res[name] = m.evaluate(test, test["target"])
    ref = test_res["E0: Phase 9 (контроль)"]["accuracy"]
    for name in FINAL:
        show(name, test_res[name], ref)

    print()
    for name in ["E0: Phase 9 (контроль)", "E6: всё, что прошло VAL"]:
        cb = CatBoostModel(feature_names=FINAL[name]); cb.fit(train, train["target"])
        TEST_ACCESS["n"] += 1
        met = cb.evaluate(test, test["target"]); test_res[f"CatBoost {name}"] = met
        show(f"CatBoost {name}", met, ref)

    section("Статистика относительно E0")
    y = test["target"].to_numpy()
    p0 = models["E0: Phase 9 (контроль)"].predict_proba(test)[:, 1]
    stats = {}
    for name in FINAL:
        if name.startswith("E0"):
            continue
        p = models[name].predict_proba(test)[:, 1]
        da = block_bootstrap_paired_diff(y, p, p0, accuracy_metric, block_size=BLOCK_SIZE)
        dl = block_bootstrap_paired_diff(y, p, p0, log_loss_metric, block_size=BLOCK_SIZE)
        mc = mcnemar_exact(y, (p >= .5).astype(int), (p0 >= .5).astype(int))
        stats[name] = {"acc": da, "ll": dl, "mcnemar": mc}
        print(f"{name:34s} Δacc={da['point_diff']:+.4f} CI=[{da['ci_low']:+.4f},{da['ci_high']:+.4f}]  "
              f"Δll={dl['point_diff']:+.4f} CI=[{dl['ci_low']:+.4f},{dl['ci_high']:+.4f}]  p={mc['p_value']:.4f}")

    section("Walk-forward по годам")
    dy = df.copy(); dy["year"] = dy["as_of_timestamp"].dt.year
    wf = []
    for yv in sorted(dy["year"].unique())[1:]:
        tr, pe = dy[dy["year"] < yv], dy[dy["year"] == yv]
        if len(tr) < 3000 or len(pe) < 200:
            continue
        row = {"year": int(yv), "n": len(pe)}
        for name in ["E0: Phase 9 (контроль)", "E1: + player_hero", "E6: всё, что прошло VAL"]:
            row[name] = compute_metrics(pe["target"], fit(FINAL[name], tr).predict_proba(pe)[:, 1])
        wf.append(row)
        print(f"{yv}  n={row['n']:6d}  E0={row['E0: Phase 9 (контроль)']['accuracy']:.4f}  "
              f"E1={row['E1: + player_hero']['accuracy']:.4f}  E6={row['E6: всё, что прошло VAL']['accuracy']:.4f}")

    section("Покрытие: где player_hero реально имеет данные")
    cov = []
    t2 = test.copy()
    for label, mask in [("нет истории пары (0 игр)", t2["player_hero_games_min"] if False else None)]:
        pass
    # используем распределение самого признака как прокси покрытия
    nz = (test["player_hero_strength_diff"].abs() > 1e-9).mean()
    print(f"Доля TEST-строк с ненулевым player_hero сигналом: {nz*100:.1f}%")

    section("Коэффициенты E6")
    for k, v in sorted(models["E6: всё, что прошло VAL"].feature_importance().items(),
                       key=lambda x: -abs(x[1])):
        print(f"  {k:36s} {v:+.5f}")

    print(f"\nОбращений к TEST: {TEST_ACCESS['n']}")

    if wf:
        yrs = [r["year"] for r in wf]
        fig, ax = plt.subplots(figsize=(8, 4))
        for k, lbl in [("E0: Phase 9 (контроль)", "E0 Phase 9"),
                       ("E1: + player_hero", "E1 +player_hero"),
                       ("E6: всё, что прошло VAL", "E6 итог")]:
            ax.plot(yrs, [r[k]["accuracy"] for r in wf], marker="o", label=lbl)
        ax.set_title("Phase 11 — walk-forward"); ax.set_ylabel("accuracy"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "phase11_walk_forward.png"), dpi=110)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(HALF_LIVES, [hl_val[h]["log_loss"] for h in HALF_LIVES], marker="o")
    ax.set_xlabel("half-life, дней"); ax.set_ylabel("VAL log loss")
    ax.set_title("Phase 11 — выбор half-life по VALIDATION")
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "phase11_half_life.png"), dpi=110)
    plt.close(fig)

    entry = {
        "experiment_id": f"phase11_{run_ts}", "git_commit": commit, "timestamp": run_ts,
        "protocol": "selection on VALIDATION only; TEST touched once",
        "test_access_count": TEST_ACCESS["n"],
        "half_life_val": {str(k): v for k, v in hl_val.items()}, "best_half_life": best_hl,
        "n_total": len(df), "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "val_results": val_res, "test_results": test_res, "statistics": stats,
        "walk_forward": wf, "val_winners": winners,
        "player_hero_signal_coverage": float(nz),
        "coefficients_E6": models["E6: всё, что прошло VAL"].feature_importance(),
    }
    path = os.path.join(EXPERIMENTS_DIR, "phase11_player_hero.json")
    with open(path, "w") as f:
        json.dump(entry, f, indent=2, default=str)
    print(f"Записано: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
