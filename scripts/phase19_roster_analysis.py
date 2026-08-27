#!/usr/bin/env python3
"""
PHASE 19 — состояние мира на момент T: игроки, пятёрки, мета.

Отвечает на PART D (составы), PART E (команда против пятёрки),
PART H (снимок меты) и PART P (неопределённость).

Проигрывает ленту матчей до момента T и останавливается. Ни один матч со
start_time >= T не применяется — это то же правило, что в движке Phase 17,
воспроизведённое здесь явно, чтобы отчёт можно было проверить руками.

Запуск: python3 scripts/phase19_roster_analysis.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_settings
from src.db.engine import make_engine
from src.pit.engine import PointInTimeState, PLAYER_ELO_BASE, TEAM_ELO_BASE
from src.pit.loader import load_pit_matches
from scripts.phase6_pipeline import git_commit_sha, section

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")

SPIRIT_ID, VISION_ID = 7119388, 9572001
GAME1_START = datetime(2026, 8, 23, 6, 15, 57, tzinfo=timezone.utc)
PATCH_RELEASED = datetime(2026, 3, 24, 0, 50, 59, tzinfo=timezone.utc)

SPIRIT = {321580662: "Yatoro", 106305042: "Larl", 302214028: "Collapse",
          218231587: "not_me", 847565596: "rue"}
VISION = {1044002267: "Satanic", 106573901: "No[o]ne-", 195108598: "Noticed",
          164199202: "9Class", 73401082: "Dukalis"}

DELTAS = [("T-72ч", timedelta(hours=72)), ("T-24ч", timedelta(hours=24)),
          ("T-6ч", timedelta(hours=6)), ("T-3ч", timedelta(hours=3)),
          ("T-1ч", timedelta(hours=1)), ("T-30м", timedelta(minutes=30))]


def replay_until(matches, t: datetime):
    """Состояние на момент t плюс история рейтингов каждого игрока.

    Строгое `<`: матч, стартующий ровно в t, к моменту прогноза исхода
    ещё не имеет. То же правило, что в движке.
    """
    st = PointInTimeState()
    hist: Dict[int, List[float]] = {}
    seen: Dict[int, int] = {}
    last_seen: Dict[int, datetime] = {}
    ts = t.timestamp()
    for m in matches:
        if m.start_time.timestamp() >= ts:
            break
        st.apply(m)
        for pid in list(m.radiant_roster) + list(m.dire_roster):
            hist.setdefault(pid, []).append(st.p_elo(pid))
            seen[pid] = seen.get(pid, 0) + 1
            last_seen[pid] = m.start_time
    return st, hist, seen, last_seen


def volatility(series: List[float], window: int = 20):
    """Эмпирическая волатильность рейтинга: sd приращений за последние N игр.

    Это НЕ апостериорная дисперсия: Elo не является байесовской оценкой,
    и делать вид, что у него есть posterior, было бы выдумыванием. Здесь
    измеряется ровно то, что измеримо — насколько рейтинг игрока реально
    ходил в его последних матчах.
    """
    if len(series) < 3:
        return None
    d = [series[i + 1] - series[i] for i in range(len(series) - 1)][-window:]
    return statistics.pstdev(d) if len(d) >= 2 else None


def confidence(n: int) -> str:
    if n >= 200:
        return "высокая"
    if n >= 50:
        return "средняя"
    if n >= 10:
        return "низкая"
    return "очень низкая"


def main() -> int:
    section("PHASE 19 — состояние мира до матча")
    print(f"git commit: {git_commit_sha()}")
    matches, _ = load_pit_matches(make_engine(load_settings()))
    print(f"матчей в ленте: {len(matches):,}\n")

    payload = {"phase": 19, "commit": git_commit_sha(), "moments": {}}

    for label, d in DELTAS:
        t = GAME1_START - d
        st, hist, seen, last_seen = replay_until(matches, t)
        section(f"{label}  —  T = {t:%Y-%m-%d %H:%M:%S} UTC")

        # ---------- PART G: патч ----------
        age = (t - PATCH_RELEASED).days
        print(f"патч 7.41, выпущен {PATCH_RELEASED:%Y-%m-%d}, возраст на T: {age} дней")
        print(f"матчей применено до T: {sum(1 for m in matches if m.start_time < t):,}\n")

        rec = {"t": str(t), "patch": "7.41", "patch_age_days": age, "teams": {}}

        for tname, tid, roster in (("Team Spirit", SPIRIT_ID, SPIRIT),
                                   ("TEAM VISION", VISION_ID, VISION)):
            elos, rows = [], []
            for pid, nick in roster.items():
                e = st.p_elo(pid)
                n = seen.get(pid, 0)
                vol = volatility(hist.get(pid, []))
                ls = last_seen.get(pid)
                rows.append({
                    "nick": nick, "account_id": pid, "player_elo": e,
                    "matches_before_T": n, "confidence": confidence(n),
                    "elo_volatility_20": vol,
                    "days_since_last_match": None if ls is None else round((t - ls).days + (t - ls).seconds / 86400, 2),
                    "hero_pool_size": len(st.pool.get(pid, {})),
                    "role": "UNKNOWN",          # pre-match источника роли не существует
                    "role_confidence": "UNKNOWN_ROLE",
                })
                elos.append(e)

            team_elo = st.elo(tid)
            p_team = st.p_team_elo(tid)
            mean_e, med_e = sum(elos) / len(elos), statistics.median(elos)
            rec["teams"][tname] = {
                "team_id": tid, "team_elo": team_elo,
                "player_team_elo": p_team,
                "mean_player_elo": mean_e, "median_player_elo": med_e,
                "five_vs_team_elo": mean_e - p_team,
                "known_players": len(rows), "roster_completeness": len(rows) / 5,
                "lineup_spread": max(elos) - min(elos),
                "min_matches_before_T": min(r["matches_before_T"] for r in rows),
                "team_matches_before_T": st.played.get(tid, 0),
                "players": rows,
            }
            print(f"{tname} ({tid})")
            print(f"  team Elo            {team_elo:9.2f}")
            print(f"  player-team Elo     {p_team:9.2f}")
            print(f"  mean player Elo     {mean_e:9.2f}")
            print(f"  median player Elo   {med_e:9.2f}")
            print(f"  five_vs_team_elo    {mean_e - p_team:+9.2f}")
            print(f"  разброс в пятёрке   {max(elos) - min(elos):9.2f}")
            print(f"  матчей команды до T {st.played.get(tid, 0):9,}")
            for r in sorted(rows, key=lambda x: -x["player_elo"]):
                v = "—" if r["elo_volatility_20"] is None else f"{r['elo_volatility_20']:.2f}"
                print(f"    {r['nick']:<10} Elo={r['player_elo']:8.2f}  матчей={r['matches_before_T']:>5}  "
                      f"довер.={r['confidence']:<12} волат.={v:>6}  пул={r['hero_pool_size']:>3}  "
                      f"посл. матч {r['days_since_last_match']} дн назад")
            print()

        a = rec["teams"]["Team Spirit"]
        b = rec["teams"]["TEAM VISION"]
        print(f"  РАЗНИЦЫ Spirit − VISION:")
        print(f"    team Elo          {a['team_elo'] - b['team_elo']:+9.2f}")
        print(f"    mean player Elo   {a['mean_player_elo'] - b['mean_player_elo']:+9.2f}")
        print(f"    median player Elo {a['median_player_elo'] - b['median_player_elo']:+9.2f}")
        print(f"    five_vs_team_elo  {a['five_vs_team_elo'] - b['five_vs_team_elo']:+9.2f}")
        rec["diff"] = {
            "team_elo": a["team_elo"] - b["team_elo"],
            "mean_player_elo": a["mean_player_elo"] - b["mean_player_elo"],
            "median_player_elo": a["median_player_elo"] - b["median_player_elo"],
            "five_vs_team_elo": a["five_vs_team_elo"] - b["five_vs_team_elo"],
        }

        # ---------- PART H: снимок меты ----------
        if label == "T-24ч":
            section("PART H — снимок меты на T-24ч")
            print("Сила героя — экспоненциально затухающий winrate с усадкой к 0.5.")
            print("Считается ТОЛЬКО по матчам со start_time < T.\n")
            hs = [(h, st.hero.strength(h, t.timestamp()),
                   st.hero._g.get(h, 0.0)) for h in sorted(st.hero._g)]
            hs = [x for x in hs if x[2] >= 20.0]      # пул героя должен существовать
            hs.sort(key=lambda x: -x[1])
            print(f"  героев с затухающим объёмом ≥ 20 игр: {len(hs)}")
            print("\n  сильнейшие 10:")
            for h, s, g in hs[:10]:
                print(f"    hero_id={h:>4}  сила={s:+.4f}  затух. игр={g:.1f}")
            print("\n  слабейшие 10:")
            for h, s, g in hs[-10:]:
                print(f"    hero_id={h:>4}  сила={s:+.4f}  затух. игр={g:.1f}")
            rec["meta"] = {"n_heroes": len(hs),
                           "top": [{"hero_id": h, "strength": s, "games": g} for h, s, g in hs[:10]],
                           "bottom": [{"hero_id": h, "strength": s, "games": g} for h, s, g in hs[-10:]]}
            print()

        payload["moments"][label] = rec

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    out = os.path.join(EXPERIMENTS_DIR, "phase19_roster_analysis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"Результаты: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
