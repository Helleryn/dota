#!/usr/bin/env python3
"""
PHASE 17 — живой shadow-прогноз на настоящие будущие матчи (PART L/P).

Соединяет три готовых куска:
  Phase 16 — обнаружение будущих матчей (Valve DPC, официальный источник);
  Phase 17 — признаки на момент T (point-in-time);
  Phase 15 — неизменяемый снимок прогноза с проверкой среза данных.

Главное, что этот скрипт обязан уметь, — **честно отказаться**. Если
команды предстоящего матча не встречались в обучающей популяции, у них
нет ни Elo, ни формы, и любой прогноз был бы выдумкой. Статус в этом
случае — UNKNOWN_TEAMS / ABSTAIN, а не «50%».

Запуск:
    python3 scripts/phase17_live_shadow.py [--store]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sqlalchemy import text

from src.config import load_settings
from src.db.engine import make_engine
from src.pit.engine import PointInTimeState
from src.pit.loader import load_pit_matches
from src.shadow import repository as repo, states, versions
from src.shadow.snapshot import PredictionSnapshot, make_prediction_id
from src.sources.http import PolitClient, now_utc
from src.sources.valve_dpc import ValveDpcAdapter
from scripts.phase6_pipeline import section

# Минимум истории, ниже которого Elo команды — это база 1000, а не оценка.
MIN_TEAM_HISTORY = 5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", action="store_true", help="сохранить снимки в БД")
    args = ap.parse_args(argv)

    section("PHASE 17 — живой shadow-прогноз на будущие матчи")
    ua = f"dota2-predict/0.1 (contact: {os.environ.get('CONTACT_EMAIL','')})"
    now = now_utc()
    adapter = ValveDpcAdapter(PolitClient(ua))
    found = adapter.discover_upcoming(now=now, max_leagues=30)
    print(f"  источник: Valve DPC (официальный, ключ не нужен)")
    print(f"  статус={found.status.value}  будущих матчей={len(found.items)}")
    if not found.items:
        print("  Будущих матчей нет — это состояние источника, а не сбой.")
        return 0

    # состояние на СЕЙЧАС по нашей истории
    eng = make_engine(load_settings())
    matches, _ = load_pit_matches(eng)
    st = PointInTimeState()
    for m in sorted(matches, key=lambda x: (x.start_time, x.match_id)):
        if m.start_time <= now:
            st.apply(m)
    print(f"  состояние построено по {len(matches)} матчам нашей истории")

    with eng.connect() as c:
        latest = c.execute(text("SELECT max(start_time) FROM matches")).scalar()
    print(f"  свежесть нашей базы: {latest}")

    section("Прогнозы")
    print(f"  {'матч':<22} {'через':>8s} {'история A/B':>13s} {'статус':>16s} {'p':>7s}")
    snapshots = []
    counts = {"READY": 0, "UNKNOWN_TEAMS": 0, "LOW_HISTORY": 0}
    for mt in found.items:
        a, b = mt.team_a.valve_team_id, mt.team_b.valve_team_id
        na, nb = st.played.get(a, 0), st.played.get(b, 0)
        elo_diff = st.elo(a) - st.elo(b)
        fa, fb = st.recent_winrate(a), st.recent_winrate(b)

        if na == 0 or nb == 0:
            status, p = "UNKNOWN_TEAMS", None
        elif min(na, nb) < MIN_TEAM_HISTORY:
            status, p = "LOW_HISTORY", None
        else:
            status = "READY"
            p = float(1.0 / (1.0 + 10 ** (-elo_diff / 400.0)))
        counts[status] += 1

        feats = {"elo_difference": elo_diff,
                 "form_3_difference": (fa - fb) if (fa is not None and fb is not None) else None,
                 "elo_mean_diff": None, "five_vs_team_elo_diff": None,
                 "hero_exp_decay_diff": None}
        snapshots.append(PredictionSnapshot(
            prediction_id=make_prediction_id(mt.match_key, now, versions.PREDICTION_VERSION),
            match_key=mt.match_key, prediction_timestamp=now,
            match_start_time=mt.scheduled_start, features=feats,
            data_cutoff=now, feature_data_cutoff=latest,
            rating_state_timestamp=latest, roster_state_timestamp=latest,
            hero_meta_state_timestamp=latest,
            source="live_valve_dpc", state=states.PUBLISHED,
            radiant_team_id=a, dire_team_id=b, league_id=mt.league_id,
            tournament=mt.tournament,
            raw_probability=p, calibrated_probability=None,
            confidence=(abs(p - 0.5) if p is not None else None),
            decision=("PREDICT" if status == "READY" else "ABSTAIN"),
            invalid_reason=None,
            calibration_version=("none" if p is None else versions.CALIBRATION_VERSION)))
        print(f"  {mt.external_id:<22} {mt.lead_hours(now):7.1f}ч {na:6d}/{nb:<6d} "
              f"{status:>16s} {'—' if p is None else f'{p:.3f}'}")

    section("Итог")
    for k, v in counts.items():
        print(f"  {k:<16s} {v:3d}")
    print()
    if counts["READY"] == 0:
        print("  Ни одного пригодного прогноза. Причина названа прямо:")
        print("  команды этих матчей не встречались в обучающей популяции")
        print("  (только pro/premium). Источник расписания РАБОТАЕТ, но отдаёт")
        print("  матчи вне домена модели. Это несовпадение охвата, а не сбой")
        print("  API и не проблема сопоставления идентичностей — team_id у")
        print("  Valve и у нас одного пространства.")

    if args.store:
        with eng.begin() as conn:
            ok = dup = 0
            for s in snapshots:
                try:
                    repo.insert_snapshot(conn, s); ok += 1
                except repo.DuplicatePrediction:
                    dup += 1
        print(f"\n  сохранено снимков: {ok}, дубликатов отклонено: {dup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
