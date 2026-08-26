#!/usr/bin/env python3
"""
PHASE 15 — команды shadow-режима.

    python3 scripts/shadow_cli.py check-sources        # PART A: живая проверка API
    python3 scripts/shadow_cli.py predict-upcoming     # PART D: прогнозы на предстоящие
    python3 scripts/shadow_cli.py replay --since ... --until ...
    python3 scripts/shadow_cli.py resolve-finished     # PART F
    python3 scripts/shadow_cli.py evaluate-live        # PART I/J/K/N/O
    python3 scripts/shadow_cli.py calibration-status   # PART C + состояние слоя

Режим SHADOW: система только считает и сохраняет, никаких внешних
действий не совершается.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sqlalchemy import text

from src.config import load_settings
from src.datasources.http_client import HttpClientConfig, RateLimitedHttpClient
from src.db.engine import make_engine
from src.shadow import cutoff as cutoff_mod
from src.shadow import metrics as m
from src.shadow import pipeline, repository as repo, states, versions
from src.shadow.engine import FrozenEngine

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")


def _history() -> pd.DataFrame:
    from scripts.phase13_pipeline import load_common_set
    df = load_common_set()
    df["as_of_timestamp"] = pd.to_datetime(df["as_of_timestamp"], utc=True)
    return df


def _client(settings):
    return RateLimitedHttpClient(HttpClientConfig(
        base_url=settings.opendota_base_url, timeout_seconds=60,
        max_retries=2, requests_per_minute=settings.opendota_rate_limit_per_min))


def _section(t):
    print("=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


# ---------------------------------------------------------------- check-sources
def cmd_check_sources(args) -> int:
    s = load_settings()
    c = _client(s)
    _section("PART A — проверка источников (реальные ответы, а не допущения)")
    results = {}
    try:
        for path in ("/health", "/proMatches", "/live"):
            try:
                r = c.get_json(path, params={})
                n = len(r) if hasattr(r, "__len__") else 1
                results[path] = {"ok": True, "size": n}
                print(f"  OK   {path:20s} размер={n}")
            except Exception as e:
                results[path] = {"ok": False, "error": f"{type(e).__name__}"}
                print(f"  FAIL {path:20s} {type(e).__name__}")
        for path in ("/scheduledMatches", "/upcomingMatches", "/schedule", "/fixtures"):
            try:
                c.get_json(path, params={})
                results[path] = {"ok": True}
                print(f"  OK   {path:20s} (неожиданно — эндпоинт появился)")
            except Exception as e:
                results[path] = {"ok": False, "error": str(e)[:60]}
                print(f"  НЕТ  {path:20s} эндпоинта не существует")
        try:
            r = c.get_json("/explorer", params={"sql":
                "SELECT count(*) n FROM matches WHERE start_time > extract(epoch from now())"})
            n = r["rows"][0]["n"]
            results["future_matches_in_opendota"] = n
            print(f"\n  Матчей с будущим start_time в базе OpenDota: {n}")
            if n == 0:
                print("  => расписания будущих матчей не существует; режим fixture невозможен")
        except Exception as e:
            print(f"  /explorer недоступен: {type(e).__name__}")

        live = None
        try:
            live = c.get_json("/live", params={})
        except Exception:
            pass
        if isinstance(live, list):
            lg = [x for x in live if x.get("league_id")]
            pre = [x for x in lg if (x.get("game_time") or 0) <= 0]
            results["live_league_matches"] = len(lg)
            results["live_in_draft"] = len(pre)
            print(f"  В /live лиговых матчей: {len(lg)}, из них в фазе драфта: {len(pre)}")
    finally:
        c.close()
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    with open(os.path.join(EXPERIMENTS_DIR, "phase15_sources.json"), "w",
              encoding="utf-8") as f:
        json.dump({"checked_at": datetime.now(timezone.utc).isoformat(),
                   "results": results}, f, ensure_ascii=False, indent=2)
    return 0


# ------------------------------------------------------------- predict-upcoming
def cmd_predict_upcoming(args) -> int:
    s = load_settings()
    engine = make_engine(s)
    _section("PART D — прогнозы на предстоящие матчи (LIVE-DRAFT)")
    hist = _history()
    fe = FrozenEngine(hist)
    c = _client(s)
    try:
        found = pipeline.live_draft_discovery(c)
    finally:
        c.close()
    print(f"  обнаружено кандидатов: {len(found)}")
    if not found:
        print("  Про-матчей в фазе драфта сейчас нет. Это не ошибка: /live отдаёт")
        print("  топ-100 текущих игр, и про-матчи в них попадают лишь когда идут.")
        return 0
    snaps = [pipeline.build_snapshot(fe, d) for d in found]
    with engine.begin() as conn:
        st = pipeline.store_predictions(conn, snaps)
    print(f"  сохранено: {st}")
    for sn in snaps:
        print(f"    {sn.prediction_id[:12]} {sn.match_key:22s} state={sn.state} "
              f"reason={sn.invalid_reason}")
    return 0


# ------------------------------------------------------------------------ replay
def cmd_replay(args) -> int:
    s = load_settings()
    engine = make_engine(s)
    _section("REPLAY — прогнозы по недавнему окну истории тем же кодом")
    hist = _history()
    since = pd.Timestamp(args.since, tz="UTC").to_pydatetime()
    until = pd.Timestamp(args.until, tz="UTC").to_pydatetime()
    fe = FrozenEngine(hist)
    print(f"  frozen model обучена до {fe.train_cutoff}")
    print(f"  окно: {since:%Y-%m-%d} .. {until:%Y-%m-%d}")

    found = list(pipeline.replay_discovery(hist, since, until,
                                           lead_minutes=args.lead_minutes))
    print(f"  обнаружено матчей: {len(found)}", flush=True)
    if args.limit:
        found = found[: args.limit]
    snaps = [pipeline.build_snapshot(fe, d) for d in found]
    with engine.begin() as conn:
        st = pipeline.store_predictions(conn, snaps)
    print(f"  сохранено: {st}")
    inv = [x for x in snaps if x.state == states.INVALID]
    if inv:
        print(f"  непригодных: {len(inv)}, причины: "
              f"{sorted({x.invalid_reason for x in inv})}")
    return 0


# --------------------------------------------------------------- resolve-finished
def cmd_resolve_finished(args) -> int:
    s = load_settings()
    engine = make_engine(s)
    _section("PART F — разрешение завершённых матчей")
    with engine.connect() as conn:
        pending = repo.list_unresolved(conn, source=args.source)
    ids = [r.match_id for r in pending if r.match_id is not None]
    print(f"  неразрешённых снимков: {len(pending)} (с match_id: {len(ids)})")
    if not ids:
        return 0
    outcomes = {}
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT match_id, radiant_win, start_time FROM matches "
            "WHERE match_id = ANY(:ids) AND radiant_win IS NOT NULL"),
            {"ids": ids}).fetchall()
    for r in rows:
        st = r.start_time if r.start_time.tzinfo else r.start_time.replace(tzinfo=timezone.utc)
        outcomes[r.match_id] = (bool(r.radiant_win), st)
    print(f"  исходов найдено: {len(outcomes)}")
    with engine.begin() as conn:
        st = pipeline.resolve(conn, outcomes, source=args.source)
    print(f"  результат: {st}")
    return 0


# ----------------------------------------------------------------- evaluate-live
def cmd_evaluate_live(args) -> int:
    s = load_settings()
    engine = make_engine(s)
    _section("PART I/J/K/N/O — метрики shadow-потока")
    with engine.connect() as conn:
        rows = repo.list_resolved(conn, source=args.source)
    df = m.to_frame(rows)
    print(f"  разрешённых прогнозов: {len(df)}  источник: {args.source or 'все'}")
    if len(df) < 20:
        print("  Выборки недостаточно даже для описательной статистики.")
        return 0

    dm = m.dual_metrics(df)
    print(f"\n  {'вариант':>14s} {'n':>6s} {'acc':>8s} {'auc':>8s} {'ll':>8s} "
          f"{'brier':>8s} {'ECE':>8s} {'slope':>8s} {'int':>8s}")
    for k, v in dm.items():
        print(f"  {k:>14s} {v['n']:6d} {v['accuracy']:8.4f} {v['roc_auc']:8.4f} "
              f"{v['log_loss']:8.4f} {v['brier']:8.4f} {v['ece']:8.5f} "
              f"{v['slope']:8.3f} {v['intercept']:+8.3f}")
    if len(df) < m.MIN_SAMPLE_FOR_CLAIMS:
        print(f"\n  ВНИМАНИЕ: выборка меньше {m.MIN_SAMPLE_FOR_CLAIMS}. "
              f"Выводы по правилу PART O не делаются.")

    print("\n  По уровням уверенности (калиброванная):")
    for r in m.by_confidence(df):
        print(f"    {r['bucket']:>8s} n={r['n']:6d} acc={r['accuracy']:.4f} "
              f"ll={r['log_loss']:.4f} ECE={r['ece']:.5f}")

    print("\n  Селективный прогноз:")
    print(f"    {'покрытие':>9s} {'n':>7s} {'acc сырая':>10s} {'acc калибр':>11s} "
          f"{'ECE сырая':>10s} {'ECE калибр':>11s}")
    craw = {round(r["coverage"], 2): r for r in m.coverage_curve(df, "p_raw")}
    ccal = {round(r["coverage"], 2): r for r in m.coverage_curve(df, "p_cal")}
    for k in sorted(set(craw) | set(ccal), reverse=True):
        a, b = craw.get(k), ccal.get(k)
        print(f"    {k*100:8.1f}% {(a or b)['n']:7d} "
              f"{a['accuracy'] if a else float('nan'):10.4f} "
              f"{b['accuracy'] if b else float('nan'):11.4f} "
              f"{a['ece'] if a else float('nan'):10.5f} "
              f"{b['ece'] if b else float('nan'):11.5f}")

    print("\n  Накопление выборки (PART O):")
    print(f"    {'n':>7s} {'acc':>8s} {'CI ширина':>11s} {'log loss':>9s} {'CI ширина':>11s}")
    prog = m.sample_size_progression(df)
    for r in prog:
        print(f"    {r['n']:7d} {r['accuracy']:8.4f} {r['accuracy_ci_width']:11.4f} "
              f"{r['log_loss']:9.4f} {r['log_loss_ci_width']:11.4f}")

    roll = m.rolling(df, window=args.window, step=args.step)
    if roll:
        print(f"\n  Скользящий мониторинг (окно {args.window}, шаг {args.step}):")
        print(f"    {'до №':>7s} {'acc':>8s} {'auc':>8s} {'ECE':>8s} {'slope':>8s}")
        for r in roll:
            print(f"    {r['end_index']:7d} {r['accuracy']:8.4f} {r['roc_auc']:8.4f} "
                  f"{r['ece']:8.5f} {r['slope']:8.3f}")
        d = m.drift_split(roll)
        print(f"\n  PART K — разделение дрейфа:")
        print(f"    размах AUC={d['auc_range']:.4f}  ECE={d['ece_range']:.5f}  "
              f"slope={d['slope_range']:.3f}")
        print(f"    вердикт: {d['verdict']}")

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    with open(os.path.join(EXPERIMENTS_DIR, "phase15_live.json"), "w", encoding="utf-8") as f:
        json.dump({"source": args.source, "n": len(df), "dual": dm,
                   "by_confidence": m.by_confidence(df),
                   "coverage_raw": m.coverage_curve(df, "p_raw"),
                   "coverage_cal": m.coverage_curve(df, "p_cal"),
                   "sample_progression": prog, "rolling": roll,
                   "drift": m.drift_split(roll) if roll else None},
                  f, ensure_ascii=False, indent=2, default=str)
    return 0


# ------------------------------------------------------------ calibration-status
def cmd_calibration_status(args) -> int:
    s = load_settings()
    engine = make_engine(s)
    _section("PART C — срез данных и состояние слоя калибровки")
    print("  Замороженная спецификация:")
    for k, v in versions.frozen_spec().items():
        print(f"    {k:22s} {v}")
    with engine.connect() as conn:
        print(f"\n  Снимки по состояниям: {repo.counts_by_state(conn)}")
        rows = repo.list_snapshots(conn, source=args.source)
    audit = cutoff_mod.audit_rows(rows)
    print(f"\n  Проверка NO DATA AFTER prediction_timestamp:")
    print(f"    всего снимков: {audit['total']}")
    print(f"    чистых:        {audit['clean']}")
    print(f"    с нарушением:  {audit['violated']}")
    for d in audit["details"][:10]:
        print(f"      {d['prediction_id'][:12]} {d['reason']} {d['violations']}")
    if audit["violated"] == 0 and audit["total"]:
        print("    => ни один прогноз не использовал данные позже своего времени")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-sources").set_defaults(fn=cmd_check_sources)
    sub.add_parser("predict-upcoming").set_defaults(fn=cmd_predict_upcoming)

    p = sub.add_parser("replay")
    p.add_argument("--since", required=True)
    p.add_argument("--until", required=True)
    p.add_argument("--lead-minutes", type=int, default=pipeline.DEFAULT_LEAD_MINUTES)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("resolve-finished")
    p.add_argument("--source", default=None)
    p.set_defaults(fn=cmd_resolve_finished)

    p = sub.add_parser("evaluate-live")
    p.add_argument("--source", default=None)
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--step", type=int, default=250)
    p.set_defaults(fn=cmd_evaluate_live)

    p = sub.add_parser("calibration-status")
    p.add_argument("--source", default=None)
    p.set_defaults(fn=cmd_calibration_status)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
