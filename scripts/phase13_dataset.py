#!/usr/bin/env python3
"""
PHASE 13 — сборка и кэширование датасета.

Вынесено отдельно от анализа сознательно: walk-forward проход по 110k
матчей стоит минуты, а фаза требует множества разрезов одних и тех же
данных. Кэш лежит в reports/experiments/cache/ (не коммитится).

Состав кадра:
  * frozen baseline Phase 9 (5 входов модели);
  * ковариаты неопределённости (src/datasets/uncertainty_features.py);
  * календарные величины патча (дата релиза известна заранее — это не
    статистика будущего).

Запуск:
    python3 scripts/phase13_dataset.py [--rebuild]
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sqlalchemy import text

from src.config import load_settings
from src.datasets.builder import _load_pro_matches
from src.datasets.hero_strength_schemes import build_hero_scheme_features
from src.datasets.multi_window_features import build_multi_window_features
from src.datasets.roster_representation import build_roster_representation, to_feature_dict as roster_dict
from src.datasets.uncertainty_features import build_uncertainty_features, to_feature_dict as unc_dict
from src.db.engine import make_engine

CACHE = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments", "cache")
PATH = os.path.join(CACHE, "phase13_dataset.csv")

ELO_K, FORM_WINDOW = 16, 3

# Модель Phase 9/12 — ПЯТЬ входов. В задании Phase 13 перечислены только три
# добавки Phase 9; два признака frozen baseline там опущены. Расхождение
# разобрано в reports/phase13-plan.md, раздел 0.
PHASE9_FULL = ["elo_difference", "form_3_difference",
               "elo_mean_diff", "five_vs_team_elo_diff", "hero_exp_decay_diff"]
PHASE9_ADDITIONS_ONLY = ["elo_mean_diff", "five_vs_team_elo_diff", "hero_exp_decay_diff"]


@dataclass(frozen=True)
class Pl:
    account_id: int
    hero_id: int
    is_radiant: bool


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
            SELECT mp.match_id, mp.account_id, mp.hero_id, mp.is_radiant
            FROM match_players mp WHERE mp.account_id IS NOT NULL
        """)).fetchall()

    by_match: Dict[int, list] = {}
    for p in players:
        by_match.setdefault(p.match_id, []).append(Pl(p.account_id, p.hero_id, p.is_radiant))

    out: List[Mt] = []
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


def load_patches(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT patch_id, name, released_at FROM patches "
            "WHERE released_at IS NOT NULL ORDER BY released_at")).fetchall()
    return pd.DataFrame([{"patch_id": r.patch_id, "patch_name": r.name,
                          "released_at": r.released_at} for r in rows])


def build(engine) -> pd.DataFrame:
    matches = load_matches(engine)
    print(f"матчей: {len(matches)}", flush=True)

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
    print("baseline готов", flush=True)

    roster = pd.DataFrame([roster_dict(r) for r in build_roster_representation(matches, k_factor=ELO_K)])
    roster = roster[["match_id", "elo_mean_diff", "five_vs_team_elo_diff"]]
    print("roster готов", flush=True)

    hero = pd.DataFrame([{"match_id": r.match_id,
                          "hero_exp_decay_diff": r.strength_diff["exp_decay"]}
                         for r in build_hero_scheme_features(matches)])
    print("hero готов", flush=True)

    unc = pd.DataFrame([unc_dict(r) for r in build_uncertainty_features(matches)])
    print("uncertainty готов", flush=True)

    meta = pd.DataFrame([{"match_id": m.match_id, "patch_id": m.patch_id} for m in matches])

    df = (base.merge(roster, on="match_id").merge(hero, on="match_id")
              .merge(unc, on="match_id").merge(meta, on="match_id"))
    df = df[(df["radiant_matches_played_before"] > 0) & (df["dire_matches_played_before"] > 0)]

    patches = load_patches(engine)
    ts = pd.to_datetime(df["as_of_timestamp"], utc=True)
    rel = pd.to_datetime(patches["released_at"], utc=True)
    idx = rel.searchsorted(ts, side="right") - 1
    df["patch_name"] = [patches["patch_name"].iloc[i] if i >= 0 else None for i in idx]
    df["days_since_patch"] = [
        (t - rel.iloc[i]).total_seconds() / 86400.0 if i >= 0 else float("nan")
        for t, i in zip(ts, idx)
    ]
    df["year"] = ts.dt.year.to_numpy()

    return df.sort_values(["as_of_timestamp", "match_id"]).reset_index(drop=True)


def load_or_build(rebuild: bool = False) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    if os.path.exists(PATH) and not rebuild:
        return pd.read_csv(PATH, parse_dates=["as_of_timestamp"])
    df = build(make_engine(load_settings()))
    df.to_csv(PATH, index=False)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args(argv)
    df = load_or_build(rebuild=a.rebuild)
    print(f"\nстрок: {len(df)}  колонок: {len(df.columns)}")
    print(f"период: {df['as_of_timestamp'].min()} .. {df['as_of_timestamp'].max()}")
    print(f"кэш: {PATH}")
    print("\nколонки:", ", ".join(df.columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
