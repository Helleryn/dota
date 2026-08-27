#!/usr/bin/env python3
"""
PHASE 19 — live shadow с явной тройкой статусов (PART Q).

Отличие от Phase 18 одно, но существенное: там статус был один и смешивал
три разных вопроса. Здесь их три, и каждый отвечает на свой:

    roster_status      FULL | PARTIAL | UNKNOWN
        сколько игроков вообще сопоставлено

    lineup_status      CONFIRMED | PREDICTED | UNKNOWN
        откуда взят состав. CONFIRMED требует источника, объявившего
        состав НА ЭТОТ МАТЧ. Такого источника не существует (Phase 16/18:
        role и заявка не публикуются), поэтому CONFIRMED недостижим, и
        это видно в отчёте, а не спрятано.

    prediction_status  READY | PARTIAL_DATA | ABSTAIN
        что система реально готова выдать

Смешивать их нельзя: полный состав, взятый из реестра «на сегодня», —
это FULL + PREDICTED, а не «подтверждённый состав». Phase 18 такую
разницу не выражала.

Запуск: python3 scripts/phase19_live_shadow.py [--store]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import unicodedata
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text

from src.config import load_settings
from src.db.engine import make_engine
from src.pit.engine import PointInTimeState
from src.pit.loader import load_pit_matches
from src.shadow import repository as repo, states, versions
from src.shadow.snapshot import PredictionSnapshot, make_prediction_id
from scripts.phase6_pipeline import section

BO3 = "https://api.bo3.gg/api/v1"
DOTA_DISCIPLINE = 4          # проверено запросом /disciplines
MIN_TEAM_HISTORY = 5
MIN_PLAYERS_PER_SIDE = 3     # ниже этого состав слишком неполон


def _get(url: str, ua: str, timeout: int = 60):
    h = {"User-Agent": ua, "Accept": "application/json", "Accept-Encoding": "gzip"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
        b = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            b = gzip.decompress(b)
        return json.loads(b)


def _norm(s) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def build_player_index(ua: str) -> Dict[str, Set[int]]:
    """Ник -> множество account_id. Множество, а не одно значение:
    неоднозначные ники обязаны остаться неоднозначными, а не разрешаться
    произвольным выбором."""
    pro = _get("https://api.opendota.com/api/proPlayers", ua, timeout=120)
    idx: Dict[str, Set[int]] = {}
    for p in pro:
        for key in (p.get("name"), p.get("personaname")):
            n = _norm(key)
            if n:
                idx.setdefault(n, set()).add(p["account_id"])
    return idx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", action="store_true")
    args = ap.parse_args(argv)
    ua = f"dota2-predict/0.1 (contact: {os.environ.get('CONTACT_EMAIL','')})"
    now = datetime.now(timezone.utc)

    section("PHASE 19 — live shadow: тройка статусов")
    matches_raw = _get(
        f"{BO3}/matches?filter[matches.status][eq]=upcoming"
        f"&filter[matches.discipline_id][eq]={DOTA_DISCIPLINE}"
        f"&sort=start_date&page[limit]=100&with=players", ua).get("results") or []
    print(f"  источник: bo3.gg, дисциплина {DOTA_DISCIPLINE} (Dota 2)")
    print(f"  предстоящих матчей: {len(matches_raw)}")
    if not matches_raw:
        print("  Предстоящих матчей нет — состояние источника, не сбой.")
        return 0

    idx = build_player_index(ua)
    print(f"  реестр про-игроков OpenDota: {len(idx)} различимых ников")

    eng = make_engine(load_settings())
    pit, _ = load_pit_matches(eng)
    st = PointInTimeState()
    for m in sorted(pit, key=lambda x: (x.start_time, x.match_id)):
        if m.start_time <= now:
            st.apply(m)
    with eng.connect() as c:
        latest = c.execute(text("SELECT max(start_time) FROM matches")).scalar()
        in_db = {r[0] for r in c.execute(text("SELECT DISTINCT account_id FROM match_players"))}
    print(f"  состояние построено по {len(pit)} матчам; свежесть базы {latest}")

    section("Прогнозы")
    print(f"  {'матч':>8s} {'через':>7s} {'A/B':>8s} {'roster':>8s} {'lineup':>10s} "
          f"{'prediction':>13s} {'p_raw':>7s}  причина")
    snapshots: List[PredictionSnapshot] = []
    counts: Dict[str, int] = {}
    rs_counts: Dict[str, int] = {}
    ls_counts: Dict[str, int] = {}
    for m in matches_raw:
        mid = str(m.get("id"))
        sd = m.get("start_date")
        start = (datetime.fromisoformat(sd.replace("Z", "+00:00")) if sd else None)
        lead = (start - now).total_seconds() / 3600 if start else float("nan")
        t1, t2 = m.get("team1_id"), m.get("team2_id")

        sides: Dict[str, List[int]] = {"a": [], "b": []}
        ambiguous = 0
        for p in m.get("players") or []:
            cands = idx.get(_norm(p.get("nickname")), set())
            if len(cands) != 1:
                ambiguous += 1
                continue
            aid = next(iter(cands))
            if aid not in in_db:
                continue
            key = "a" if str(p.get("team_id")) == str(t1) else "b"
            sides[key].append(aid)
        na, nb = len(sides["a"]), len(sides["b"])

        # --- три независимых статуса (PART Q) ---
        if min(na, nb) >= 5:
            roster_status = "FULL"
        elif min(na, nb) >= MIN_PLAYERS_PER_SIDE:
            roster_status = "PARTIAL"
        else:
            roster_status = "UNKNOWN"

        # CONFIRMED требует источника, объявившего состав на ЭТОТ матч.
        # Такого источника нет; сопоставление по реестру даёт в лучшем
        # случае PREDICTED. Ветка CONFIRMED оставлена явной, чтобы было
        # видно: она недостижима не по недосмотру.
        if roster_status == "UNKNOWN":
            lineup_status = "UNKNOWN"
        else:
            lineup_status = "PREDICTED"

        reason = None
        if start is None:
            status, reason = "INVALID", "no_start_time"
        elif roster_status == "UNKNOWN":
            status, reason = "ABSTAIN", "roster_too_incomplete"
        else:
            status = "READY" if roster_status == "FULL" else "PARTIAL_DATA"

        p_raw = None
        if status in ("READY", "PARTIAL_DATA"):
            ra = sum(st.p_elo(x) for x in sides["a"]) / na
            rb = sum(st.p_elo(x) for x in sides["b"]) / nb
            p_raw = float(1.0 / (1.0 + 10 ** (-(ra - rb) / 400.0)))
        counts[status] = counts.get(status, 0) + 1

        feats = {"elo_mean_diff": (None if p_raw is None else
                                   sum(st.p_elo(x) for x in sides["a"]) / na
                                   - sum(st.p_elo(x) for x in sides["b"]) / nb),
                 "players_known_a": na, "players_known_b": nb,
                 "ambiguous_nicknames": ambiguous,
                 "roster_status": roster_status,
                 "lineup_status": lineup_status,
                 "prediction_status": status}
        snapshots.append(PredictionSnapshot(
            prediction_id=make_prediction_id(f"bo3:{mid}", now, versions.PREDICTION_VERSION),
            match_key=f"bo3:{mid}", prediction_timestamp=now,
            match_start_time=start, features=feats,
            data_cutoff=now, feature_data_cutoff=latest,
            rating_state_timestamp=latest, roster_state_timestamp=now,
            hero_meta_state_timestamp=latest,
            source="live_bo3gg_dota", state=states.PUBLISHED,
            raw_probability=p_raw, calibrated_probability=None,
            confidence=(abs(p_raw - 0.5) if p_raw is not None else None),
            decision=("PREDICT" if status in ("READY", "PARTIAL_DATA") else "ABSTAIN"),
            invalid_reason=reason, calibration_version="none"))
        rs_counts[roster_status] = rs_counts.get(roster_status, 0) + 1
        ls_counts[lineup_status] = ls_counts.get(lineup_status, 0) + 1
        print(f"  {mid:>8s} {lead:6.1f}ч {na:>3d}/{nb:<4d} {roster_status:>8s} "
              f"{lineup_status:>10s} {status:>13s} "
              f"{'—' if p_raw is None else f'{p_raw:.3f}':>7s}  {reason or ''}")

    section("Итог")
    for title, d in (("roster_status", rs_counts), ("lineup_status", ls_counts),
                     ("prediction_status", counts)):
        print(f"  {title}:")
        for k in sorted(d):
            print(f"    {k:<16s} {d[k]:3d}")
    if not ls_counts.get("CONFIRMED"):
        print("\n  CONFIRMED не достигнут ни разу: источника, объявляющего")
        print("  состав на конкретный матч, не существует (Phase 16/18).")
    predicted = counts.get("READY", 0) + counts.get("PARTIAL_DATA", 0)
    print(f"\n  прогнозов выдано: {predicted} из {len(matches_raw)}")
    print(f"  отказов: {len(matches_raw) - predicted}")

    if args.store:
        with eng.begin() as conn:
            ok = dup = 0
            for s in snapshots:
                try:
                    repo.insert_snapshot(conn, s); ok += 1
                except repo.DuplicatePrediction:
                    dup += 1
        print(f"  сохранено снимков: {ok}, дубликатов отклонено: {dup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
