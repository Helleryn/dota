#!/usr/bin/env python3
"""
PHASE 10 — canonical identity: влияние на Team Elo, five_vs_team_elo и Team x Hero.

Протокол — как в исправленной Phase 9: отбор по VALIDATION, TEST один раз.
Frozen baseline (Elo + Form3) и набор Phase 9 не меняются.

Варианты (все на ОДНОМ evaluation set):
    A  Phase 9 feature set
    B  Phase 9, но Team Elo на canonical_team_id
    C  B + five_vs_team_elo на canonical
    D  Phase 9 + Team x Hero residual
    E  C + Team x Hero residual

Запуск:
    python3 scripts/phase10_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Dict, FrozenSet, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.config import load_settings
from src.datasets.hero_strength_schemes import build_hero_scheme_features
from src.datasets.multi_window_features import build_multi_window_features
from src.datasets.roster_representation import build_roster_representation, to_feature_dict
from src.db.engine import make_engine
from src.evaluation.metrics import compute_metrics
from src.evaluation.statistics import (
    accuracy_metric,
    block_bootstrap_paired_diff,
    log_loss_metric,
    mcnemar_exact,
)
from src.identity.resolver import resolve_identities
from src.models.sklearn_models import RANDOM_SEED, CatBoostModel, LogisticRegressionModel
from scripts.phase6_pipeline import git_commit_sha, section

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")

ELO_K = 16
FORM_WINDOW = 3
BLOCK_SIZE = 20
BASELINE = ["elo_difference", "form_3_difference"]
PHASE9 = BASELINE + ["elo_mean_diff", "five_vs_team_elo_diff", "hero_exp_decay_diff"]

TEAM_HERO_HALF_LIFE = 180.0
TEAM_HERO_PRIOR = 20.0
TEST_ACCESS = {"n": 0}


@dataclass(frozen=True)
class FullMatch:
    match_id: int
    start_time: datetime
    patch_id: Optional[int]
    radiant_team_id: int
    dire_team_id: int
    radiant_roster: FrozenSet[int]
    dire_roster: FrozenSet[int]
    radiant_picks: Tuple[int, ...]
    dire_picks: Tuple[int, ...]
    radiant_bans: Tuple[int, ...]
    dire_bans: Tuple[int, ...]
    radiant_win: bool


def load_full_matches(engine):
    sql = text("""
        WITH rosters AS (
          SELECT mp.match_id,
                 array_agg(mp.account_id) FILTER (WHERE mp.is_radiant AND mp.account_id IS NOT NULL) AS r_roster,
                 array_agg(mp.account_id) FILTER (WHERE NOT mp.is_radiant AND mp.account_id IS NOT NULL) AS d_roster
          FROM match_players mp GROUP BY 1
        ), drafts AS (
          SELECT pb.match_id,
                 array_agg(pb.hero_id ORDER BY pb.ord) FILTER (WHERE pb.is_pick AND pb.team = 0) AS r_picks,
                 array_agg(pb.hero_id ORDER BY pb.ord) FILTER (WHERE pb.is_pick AND pb.team = 1) AS d_picks,
                 array_agg(pb.hero_id ORDER BY pb.ord) FILTER (WHERE NOT pb.is_pick AND pb.team = 0) AS r_bans,
                 array_agg(pb.hero_id ORDER BY pb.ord) FILTER (WHERE NOT pb.is_pick AND pb.team = 1) AS d_bans
          FROM picks_bans pb GROUP BY 1
        )
        SELECT m.match_id, m.start_time, m.patch_id, m.radiant_team_id, m.dire_team_id, m.radiant_win,
               r.r_roster, r.d_roster, d.r_picks, d.d_picks, d.r_bans, d.d_bans
        FROM matches m
        JOIN leagues l ON l.league_id = m.league_id
        JOIN rosters r ON r.match_id = m.match_id
        JOIN drafts  d ON d.match_id = m.match_id
        WHERE l.tier IN ('professional','premium')
          AND m.radiant_team_id IS NOT NULL AND m.dire_team_id IS NOT NULL
          AND m.radiant_win IS NOT NULL
          AND array_length(r.r_roster,1) = 5 AND array_length(r.d_roster,1) = 5
          AND array_length(d.r_picks,1) = 5 AND array_length(d.d_picks,1) = 5
        ORDER BY m.start_time ASC, m.match_id ASC
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
        names = {r.team_id: r.name for r in conn.execute(text("SELECT team_id, name FROM teams")).fetchall()}
    out = []
    for r in rows:
        st = r.start_time if r.start_time.tzinfo else r.start_time.replace(tzinfo=timezone.utc)
        out.append(FullMatch(
            match_id=r.match_id, start_time=st, patch_id=r.patch_id,
            radiant_team_id=r.radiant_team_id, dire_team_id=r.dire_team_id,
            radiant_roster=frozenset(r.r_roster), dire_roster=frozenset(r.d_roster),
            radiant_picks=tuple(r.r_picks), dire_picks=tuple(r.d_picks),
            radiant_bans=tuple(r.r_bans or ()), dire_bans=tuple(r.d_bans or ()),
            radiant_win=r.radiant_win,
        ))
    return out, names


def build_team_hero_features(matches, team_of):
    """
    PART M — team x hero RESIDUAL, а не raw winrate:

        residual = shrunk_winrate(team, hero) - strength(hero)

    То есть «играет ли команда на этом герое ЛУЧШЕ, чем герой играет в
    среднем». Затухание 180 дней (медленнее, чем у меты — специализация
    команды устойчивее), shrinkage prior=20 игр.

    Walk-forward: состояние обновляется строго ПОСЛЕ фиксации признаков.
    """
    th_w, th_g, th_t = defaultdict(float), defaultdict(float), {}
    h_w, h_g, h_t = defaultdict(float), defaultdict(float), {}

    def decay(key, store_w, store_g, store_t, ts, hl):
        last = store_t.get(key)
        if last is None:
            store_t[key] = ts
            return
        dt = (ts - last) / 86400.0
        if dt > 0:
            f = 0.5 ** (dt / hl)
            store_w[key] *= f
            store_g[key] *= f
        store_t[key] = ts

    rows = []
    for m in matches:
        ts = m.start_time.timestamp()
        r_team, d_team = team_of(m.radiant_team_id), team_of(m.dire_team_id)

        def residual(team_id, picks):
            vals = []
            for h in picks:
                decay((team_id, h), th_w, th_g, th_t, ts, TEAM_HERO_HALF_LIFE)
                decay(h, h_w, h_g, h_t, ts, TEAM_HERO_HALF_LIFE)
                tg, tw = th_g[(team_id, h)], th_w[(team_id, h)]
                hg, hw = h_g[h], h_w[h]
                team_wr = (tw + TEAM_HERO_PRIOR / 2.0) / (tg + TEAM_HERO_PRIOR)
                hero_wr = (hw + TEAM_HERO_PRIOR / 2.0) / (hg + TEAM_HERO_PRIOR)
                vals.append(team_wr - hero_wr)
            return sum(vals) / len(vals) if vals else 0.0

        r_res, d_res = residual(r_team, m.radiant_picks), residual(d_team, m.dire_picks)
        rows.append({"match_id": m.match_id, "team_hero_residual_diff": r_res - d_res})

        rw = m.radiant_win
        for team_id, picks, won in ((r_team, m.radiant_picks, rw), (d_team, m.dire_picks, not rw)):
            for h in picks:
                decay((team_id, h), th_w, th_g, th_t, ts, TEAM_HERO_HALF_LIFE)
                th_g[(team_id, h)] += 1.0
                th_w[(team_id, h)] += 1.0 if won else 0.0
        for h in m.radiant_picks:
            decay(h, h_w, h_g, h_t, ts, TEAM_HERO_HALF_LIFE)
            h_g[h] += 1.0; h_w[h] += 1.0 if rw else 0.0
        for h in m.dire_picks:
            decay(h, h_w, h_g, h_t, ts, TEAM_HERO_HALF_LIFE)
            h_g[h] += 1.0; h_w[h] += 1.0 if not rw else 0.0
    return pd.DataFrame(rows)


def features_frame(matches, suffix=""):
    """Baseline Elo/Form + roster-представление на переданных (возможно
    перемапленных) матчах."""
    base_rows = build_multi_window_features(matches, k_factor=ELO_K)
    base = pd.DataFrame([{
        "match_id": r.match_id, "as_of_timestamp": r.as_of_timestamp,
        f"elo_difference{suffix}": r.elo_difference,
        f"form_3_difference{suffix}": r.recent_winrate_difference[FORM_WINDOW],
        "radiant_matches_played_before": r.radiant_matches_played_before,
        "dire_matches_played_before": r.dire_matches_played_before,
        "target": int(r.radiant_win),
    } for r in base_rows])
    roster = pd.DataFrame([to_feature_dict(r) for r in build_roster_representation(matches, k_factor=ELO_K)])
    keep = ["match_id", "elo_mean_diff", "five_vs_team_elo_diff"]
    roster = roster[keep].rename(columns={
        "elo_mean_diff": f"elo_mean_diff{suffix}",
        "five_vs_team_elo_diff": f"five_vs_team_elo_diff{suffix}",
    })
    return base.merge(roster, on="match_id")


def split(df, tf=0.70, vf=0.15):
    n = len(df)
    a, b = int(n * tf), int(n * (tf + vf))
    tr, va, te = df.iloc[:a].reset_index(drop=True), df.iloc[a:b].reset_index(drop=True), df.iloc[b:].reset_index(drop=True)
    assert tr["as_of_timestamp"].max() < va["as_of_timestamp"].min()
    assert va["as_of_timestamp"].max() < te["as_of_timestamp"].min()
    return tr, va, te


def fit(features, train):
    m = LogisticRegressionModel(feature_names=features, random_state=RANDOM_SEED)
    m.fit(train, train["target"])
    return m


def show(name, m, ref=None):
    d = f"  Δacc={m['accuracy']-ref:+.4f}" if ref is not None else ""
    print(f"{name:40s} acc={m['accuracy']:.4f} ll={m['log_loss']:.4f} "
          f"brier={m['brier_score']:.4f} auc={m['roc_auc']:.4f}{d}")


def main() -> int:
    os.makedirs(FIGURES_DIR, exist_ok=True); os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    commit, run_ts = git_commit_sha(), datetime.now(timezone.utc).isoformat()

    section("PHASE 10 — canonical team identity")
    print(f"git commit: {commit}")
    engine = make_engine(load_settings())
    matches, names = load_full_matches(engine)
    print(f"Матчей (ростер 5+5, драфт 5+5): {len(matches)}")

    section("10.1 Walk-forward identity resolution")
    resolver, res = resolve_identities(matches, names=names)
    st = res.stats()
    for k, v in st.items():
        print(f"  {k}: {v}")
    print()
    print("Примеры установленных связей (первые 12 по времени):")
    for l in res.links[:12]:
        print(f"  {names.get(l.predecessor_team_id)!r}({l.predecessor_team_id}) -> "
              f"{names.get(l.source_team_id)!r}({l.source_team_id})  "
              f"{l.confidence} overlap={l.roster_overlap} gap={l.gap_days}д "
              f"valid_from={l.valid_from.date()}")

    # canonical-перемапленные матчи. Отображение построено walk-forward
    # (каждому team_id canonical назначен в момент ЕГО ПЕРВОГО матча и далее
    # не меняется), поэтому применение итогового словаря эквивалентно
    # применению его же по ходу времени — утечки нет (см. тест
    # test_identity_is_prefix_stable).
    cmap = resolver.canonical
    canon_matches = [replace(m, radiant_team_id=cmap(m.radiant_team_id),
                             dire_team_id=cmap(m.dire_team_id)) for m in matches]

    section("10.3 Построение признаков: source vs canonical")
    src = features_frame(matches, suffix="")
    can = features_frame(canon_matches, suffix="_canon")
    can = can.drop(columns=["as_of_timestamp", "target",
                            "radiant_matches_played_before", "dire_matches_played_before"])

    hs = build_hero_scheme_features(matches)
    hero = pd.DataFrame([{"match_id": r.match_id, "hero_exp_decay_diff": r.strength_diff["exp_decay"]} for r in hs])

    th_src = build_team_hero_features(matches, lambda t: t)
    th_can = build_team_hero_features(canon_matches, lambda t: t).rename(
        columns={"team_hero_residual_diff": "team_hero_residual_canon_diff"})

    df = (src.merge(can, on="match_id").merge(hero, on="match_id")
             .merge(th_src, on="match_id").merge(th_can, on="match_id"))
    df = df[(df["radiant_matches_played_before"] > 0) & (df["dire_matches_played_before"] > 0)]
    df = df.sort_values(["as_of_timestamp", "match_id"]).reset_index(drop=True)
    print(f"Строк после исключения team cold-start: {len(df)}")

    train, val, test = split(df)
    print(f"TRAIN {len(train)} | VAL {len(val)} | TEST {len(test)}")

    section("Отбор по VALIDATION (TEST не трогаем)")
    VARIANTS = {
        "A: Phase 9 (source id)": PHASE9,
        "B: + canonical Team Elo": ["elo_difference_canon", "form_3_difference", "elo_mean_diff",
                                     "five_vs_team_elo_diff", "hero_exp_decay_diff"],
        "C: + canonical Elo и five_vs_team": ["elo_difference_canon", "form_3_difference", "elo_mean_diff",
                                               "five_vs_team_elo_diff_canon", "hero_exp_decay_diff"],
        "D: Phase 9 + TeamxHero (source)": PHASE9 + ["team_hero_residual_diff"],
        "E: canonical + TeamxHero (canonical)": ["elo_difference_canon", "form_3_difference", "elo_mean_diff",
                                                  "five_vs_team_elo_diff_canon", "hero_exp_decay_diff",
                                                  "team_hero_residual_canon_diff"],
    }
    val_res = {}
    for name, feats in VARIANTS.items():
        m = fit(feats, train)
        val_res[name] = m.evaluate(val, val["target"])
        show(f"[VAL] {name}", val_res[name], val_res["A: Phase 9 (source id)"]["accuracy"])

    section("ФИНАЛЬНАЯ ОЦЕНКА НА TEST (состав зафиксирован)")
    models, test_res = {}, {}
    for name, feats in VARIANTS.items():
        m = fit(feats, train)
        models[name] = m
        TEST_ACCESS["n"] += 1
        test_res[name] = m.evaluate(test, test["target"])
    ref = test_res["A: Phase 9 (source id)"]["accuracy"]
    for name in VARIANTS:
        show(name, test_res[name], ref)

    print()
    for name in ["A: Phase 9 (source id)", "E: canonical + TeamxHero (canonical)"]:
        cb = CatBoostModel(feature_names=VARIANTS[name])
        cb.fit(train, train["target"])
        TEST_ACCESS["n"] += 1
        met = cb.evaluate(test, test["target"])
        test_res[f"CatBoost {name}"] = met
        show(f"CatBoost {name}", met, ref)

    section("Статистика относительно варианта A (Phase 9)")
    y = test["target"].to_numpy()
    p_a = models["A: Phase 9 (source id)"].predict_proba(test)[:, 1]
    stats = {}
    for name in ["B: + canonical Team Elo", "C: + canonical Elo и five_vs_team",
                 "D: Phase 9 + TeamxHero (source)", "E: canonical + TeamxHero (canonical)"]:
        p = models[name].predict_proba(test)[:, 1]
        da = block_bootstrap_paired_diff(y, p, p_a, accuracy_metric, block_size=BLOCK_SIZE)
        dl = block_bootstrap_paired_diff(y, p, p_a, log_loss_metric, block_size=BLOCK_SIZE)
        mc = mcnemar_exact(y, (p >= 0.5).astype(int), (p_a >= 0.5).astype(int))
        stats[name] = {"acc": da, "ll": dl, "mcnemar": mc}
        print(f"{name:38s} Δacc={da['point_diff']:+.4f} CI=[{da['ci_low']:+.4f},{da['ci_high']:+.4f}]  "
              f"Δll={dl['point_diff']:+.4f} CI=[{dl['ci_low']:+.4f},{dl['ci_high']:+.4f}]  p={mc['p_value']:.4f}")

    section("Walk-forward по годам")
    dy = df.copy(); dy["year"] = dy["as_of_timestamp"].dt.year
    wf = []
    for yv in sorted(dy["year"].unique())[1:]:
        tr, pe = dy[dy["year"] < yv], dy[dy["year"] == yv]
        if len(tr) < 3000 or len(pe) < 200:
            continue
        row = {"year": int(yv), "n": len(pe)}
        for name, feats in VARIANTS.items():
            row[name] = compute_metrics(pe["target"], fit(feats, tr).predict_proba(pe)[:, 1])
        wf.append(row)
        print(f"{yv}  n={row['n']:6d}  A={row['A: Phase 9 (source id)']['accuracy']:.4f}  "
              f"C={row['C: + canonical Elo и five_vs_team']['accuracy']:.4f}  "
              f"E={row['E: canonical + TeamxHero (canonical)']['accuracy']:.4f}")

    section("PART Q — robustness: где именно помогает canonical identity")
    merged_sources = {l.source_team_id for l in res.links} | {l.predecessor_team_id for l in res.links}
    t2 = test.copy()
    # признак «затронут ли матч слиянием» — по исходным id
    src_ids = df.loc[test.index, :] if False else None
    with engine.connect() as conn:
        mt = {r.match_id: (r.radiant_team_id, r.dire_team_id) for r in conn.execute(text(
            "SELECT match_id, radiant_team_id, dire_team_id FROM matches")).fetchall()}
    t2["touched"] = t2["match_id"].map(lambda mid: bool(set(mt.get(mid, ())) & merged_sources))
    robust = []
    for label, mask in [("матчи, затронутые слиянием", t2["touched"]),
                        ("не затронутые", ~t2["touched"])]:
        sub = t2[mask]
        if len(sub) < 100:
            print(f"{label}: n={len(sub)} — мало"); continue
        ma = compute_metrics(sub["target"], models["A: Phase 9 (source id)"].predict_proba(sub)[:, 1])
        mc_ = compute_metrics(sub["target"], models["C: + canonical Elo и five_vs_team"].predict_proba(sub)[:, 1])
        robust.append({"bucket": label, "n": len(sub), "A": ma["accuracy"], "C": mc_["accuracy"]})
        print(f"{label:30s} n={len(sub):6d}  A={ma['accuracy']:.4f}  C={mc_['accuracy']:.4f}  "
              f"Δ={mc_['accuracy']-ma['accuracy']:+.4f}")

    section("Коэффициенты варианта E")
    for k, v in sorted(models["E: canonical + TeamxHero (canonical)"].feature_importance().items(),
                       key=lambda x: -abs(x[1])):
        print(f"  {k:36s} {v:+.5f}")

    print(f"\nОбращений к TEST: {TEST_ACCESS['n']} (все после фиксации состава по VAL)")

    yrs = [r["year"] for r in wf]
    if wf:
        fig, ax = plt.subplots(figsize=(8, 4))
        for key, lbl in [("A: Phase 9 (source id)", "A: Phase 9"),
                         ("C: + canonical Elo и five_vs_team", "C: canonical"),
                         ("E: canonical + TeamxHero (canonical)", "E: canonical+TxH")]:
            ax.plot(yrs, [r[key]["accuracy"] for r in wf], marker="o", label=lbl)
        ax.set_title("Phase 10 — walk-forward"); ax.set_ylabel("accuracy"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "phase10_walk_forward.png"), dpi=110)
        plt.close(fig)

    entry = {
        "experiment_id": f"phase10_{run_ts}", "git_commit": commit, "timestamp": run_ts,
        "identity_stats": st,
        "links_sample": [{"pred": l.predecessor_team_id, "src": l.source_team_id,
                          "conf": l.confidence, "overlap": l.roster_overlap,
                          "gap_days": l.gap_days, "valid_from": str(l.valid_from),
                          "name_pred": names.get(l.predecessor_team_id),
                          "name_src": names.get(l.source_team_id)} for l in res.links[:200]],
        "n_total": len(df), "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "val_results": val_res, "test_results": test_res, "statistics": stats,
        "walk_forward": wf, "robustness": robust,
        "test_access_count": TEST_ACCESS["n"],
        "coefficients_E": models["E: canonical + TeamxHero (canonical)"].feature_importance(),
    }
    path = os.path.join(EXPERIMENTS_DIR, "phase10_team_identity.json")
    with open(path, "w") as f:
        json.dump(entry, f, indent=2, default=str)
    print(f"Записано: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
