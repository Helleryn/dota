#!/usr/bin/env python3
"""
PHASE 10.0 — IDENTITY AUDIT.

Только аудит: НИКАКОГО автоматического merge. Задача — измерить масштаб и
структуру проблемы идентичности `team_id` и найти реальные кейсы.

Запуск:
    python3 scripts/phase10_identity_audit.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text

from src.config import load_settings
from src.db.engine import make_engine
from scripts.phase6_pipeline import section

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "experiments")

MIN_SHARED_PLAYERS = 3   # порог кандидата: 3 из 5 общих игроков
AUDIT_LOG: list = []      # audit log нормализации имён (PART D)


def normalize_name(name: str) -> str:
    """
    PART D. Консервативная нормализация: регистр, unicode, пробелы,
    разделители. НЕ удаляем слова вроде 'Team'/'Academy' — 'Entity' и
    'Entity Academy' могут быть РАЗНЫМИ сущностями (прямое требование
    задания). Каждое преобразование логируется.
    """
    if name is None:
        return ""
    original = name
    s = unicodedata.normalize("NFKC", name)
    s = s.casefold().strip()
    s = re.sub(r"[_\-–—.]+", " ", s)   # разделители -> пробел
    s = re.sub(r"\s+", " ", s).strip()
    if s != original:
        AUDIT_LOG.append({"original": original, "normalized": s})
    return s


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> int:
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    engine = make_engine(load_settings())

    section("PHASE 10.0 — IDENTITY AUDIT (без автоматического merge)")

    # ---------- базовая статистика ----------
    with engine.connect() as c:
        teams = c.execute(text("""
            SELECT t.team_id, t.name, t.tag, t.first_seen_at, t.last_seen_at
            FROM teams t
        """)).fetchall()

        # ростеры каждой команды по матчам (pro/premium)
        roster_rows = c.execute(text("""
            SELECT m.match_id, m.start_time,
                   CASE WHEN mp.is_radiant THEN m.radiant_team_id ELSE m.dire_team_id END AS team_id,
                   array_agg(mp.account_id ORDER BY mp.account_id) AS roster
            FROM match_players mp
            JOIN matches m ON m.match_id = mp.match_id
            JOIN leagues l ON l.league_id = m.league_id
            WHERE l.tier IN ('professional','premium') AND mp.account_id IS NOT NULL
              AND m.radiant_team_id IS NOT NULL AND m.dire_team_id IS NOT NULL
            GROUP BY m.match_id, m.start_time, 3
        """)).fetchall()

    section("1. Базовая статистика")
    name_by_id = {t.team_id: t.name for t in teams}
    print(f"Уникальных team_id в teams: {len(teams)}")
    named = [t for t in teams if t.name]
    print(f"  из них с непустым именем: {len(named)}")
    norm_names = {t.team_id: normalize_name(t.name) for t in named}
    print(f"Уникальных нормализованных имён: {len(set(norm_names.values()))}")
    print(f"Записей в audit log нормализации: {len(AUDIT_LOG)}")

    # team_id, реально игравшие pro/premium
    team_matches = defaultdict(list)
    for r in roster_rows:
        st = r.start_time if r.start_time.tzinfo else r.start_time.replace(tzinfo=timezone.utc)
        team_matches[r.team_id].append((st, frozenset(r.roster)))
    for tid in team_matches:
        team_matches[tid].sort(key=lambda x: x[0])
    print(f"team_id с матчами в pro/premium: {len(team_matches)}")
    sizes = sorted((len(v) for v in team_matches.values()))
    print(f"  матчей на team_id: медиана={sizes[len(sizes)//2]}, "
          f"p90={sizes[int(len(sizes)*0.9)]}, max={sizes[-1]}")
    print(f"  team_id с 1-2 матчами: {sum(1 for s in sizes if s <= 2)} "
          f"({sum(1 for s in sizes if s <= 2)/len(sizes)*100:.1f}%)")

    # ---------- PART A.1 — одинаковые имена, разные team_id ----------
    section("2. Одинаковые нормализованные имена под разными team_id (PART A.1)")
    by_norm = defaultdict(list)
    for tid, nn in norm_names.items():
        if nn and tid in team_matches:
            by_norm[nn].append(tid)
    dup_name_groups = {n: ids for n, ids in by_norm.items() if len(ids) > 1}
    dup_pairs = sum(len(ids) * (len(ids) - 1) // 2 for ids in dup_name_groups.values())
    print(f"Групп с одинаковым именем: {len(dup_name_groups)}")
    print(f"Затронуто team_id: {sum(len(v) for v in dup_name_groups.values())}")
    print(f"Пар с одинаковым именем: {dup_pairs}")
    top_dups = sorted(dup_name_groups.items(), key=lambda kv: -len(kv[1]))[:10]
    print("Топ по числу team_id с одним именем:")
    for n, ids in top_dups:
        print(f"  {n!r}: {len(ids)} team_id -> {ids[:6]}{'...' if len(ids) > 6 else ''}")

    # ---------- PART A / C3 — кандидаты по ростеру ----------
    section("3. Кандидаты по roster continuity (PART C3) — главный сигнал")
    player_teams = defaultdict(set)
    for tid, hist in team_matches.items():
        for _, roster in hist:
            for p in roster:
                player_teams[p].add(tid)

    shared = Counter()
    for p, tids in player_teams.items():
        if len(tids) < 2 or len(tids) > 40:   # игроки в 40+ team_id — мусор низовых лиг
            continue
        tl = sorted(tids)
        for i in range(len(tl)):
            for j in range(i + 1, len(tl)):
                shared[(tl[i], tl[j])] += 1

    candidates = {pair: n for pair, n in shared.items() if n >= MIN_SHARED_PLAYERS}
    print(f"Пар team_id, деливших >= {MIN_SHARED_PLAYERS} игроков: {len(candidates)}")

    # ---------- оценка каждого кандидата ----------
    section("4. Оценка кандидатов: roster overlap + временная связь (PART C3/C5)")
    scored = []
    for (a, b), n_shared in candidates.items():
        ha, hb = team_matches[a], team_matches[b]
        a_first, a_last = ha[0][0], ha[-1][0]
        b_first, b_last = hb[0][0], hb[-1][0]

        # пересечение "последний состав A" vs "первый состав B" (и наоборот)
        ov_ab = jaccard(ha[-1][1], hb[0][1])
        ov_ba = jaccard(hb[-1][1], ha[0][1])
        best_overlap = max(ov_ab, ov_ba)

        # временное соотношение: последовательные или параллельные?
        overlap_days = (min(a_last, b_last) - max(a_first, b_first)).total_seconds() / 86400.0
        if overlap_days > 0:
            relation = "parallel"          # существовали одновременно
            gap_days = 0.0
        else:
            relation = "sequential"
            gap_days = -overlap_days

        same_name = norm_names.get(a, "") == norm_names.get(b, "") and norm_names.get(a, "") != ""
        scored.append({
            "team_a": a, "team_b": b, "n_shared_players": n_shared,
            "name_a": name_by_id.get(a), "name_b": name_by_id.get(b),
            "same_normalized_name": same_name,
            "best_roster_overlap": round(best_overlap, 3),
            "relation": relation, "gap_days": round(gap_days, 1),
            "a_matches": len(ha), "b_matches": len(hb),
        })

    n_parallel = sum(1 for s in scored if s["relation"] == "parallel")
    n_seq = len(scored) - n_parallel
    print(f"Кандидатов всего: {len(scored)}")
    print(f"  ПАРАЛЛЕЛЬНЫЕ (жили одновременно) -> merge ЗАПРЕЩЁН: {n_parallel} ({n_parallel/len(scored)*100:.1f}%)")
    print(f"  ПОСЛЕДОВАТЕЛЬНЫЕ (кандидаты на преемственность): {n_seq}")

    print()
    print("КРИТИЧЕСКОЕ НАБЛЮДЕНИЕ: параллельное существование — прямое")
    print("опровержение гипотезы 'это одна сущность'. Две команды, игравшие")
    print("в один и тот же период, не могут быть одной командой, сколько бы")
    print("игроков они ни делили (это shared-игроки/стеки, не преемственность).")

    seq = [s for s in scored if s["relation"] == "sequential"]
    print()
    print("Распределение последовательных кандидатов по roster overlap:")
    for lo, hi in [(0.8, 1.01), (0.6, 0.8), (0.4, 0.6), (0.0, 0.4)]:
        sub = [s for s in seq if lo <= s["best_roster_overlap"] < hi]
        near = [s for s in sub if s["gap_days"] <= 90]
        print(f"  overlap [{lo:.1f},{hi:.1f}): {len(sub):5d}  из них разрыв <=90 дней: {len(near)}")

    # ---------- уровни доверия ----------
    section("5. Предварительные уровни доверия (PART E) — БЕЗ применения merge")

    def confidence(s):
        if s["relation"] == "parallel":
            return "REJECTED_PARALLEL"
        strong_roster = s["best_roster_overlap"] >= 0.6
        near_time = s["gap_days"] <= 90
        if strong_roster and near_time and s["same_normalized_name"]:
            return "HIGH"
        if strong_roster and near_time:
            return "MEDIUM"
        if s["same_normalized_name"] and s["best_roster_overlap"] >= 0.4:
            return "MEDIUM"
        if s["same_normalized_name"] or s["best_roster_overlap"] >= 0.4:
            return "LOW"
        return "UNRESOLVED"

    for s in scored:
        s["confidence"] = confidence(s)
    counts = Counter(s["confidence"] for s in scored)
    for lvl in ["HIGH", "MEDIUM", "LOW", "UNRESOLVED", "REJECTED_PARALLEL"]:
        print(f"  {lvl:20s}: {counts.get(lvl, 0)}")

    print()
    print("Примеры HIGH (сильный roster overlap + близко по времени + имя):")
    for s in sorted([x for x in scored if x["confidence"] == "HIGH"],
                    key=lambda x: -x["best_roster_overlap"])[:8]:
        print(f"  {s['name_a']!r}({s['team_a']}) -> {s['name_b']!r}({s['team_b']}) "
              f"overlap={s['best_roster_overlap']} gap={s['gap_days']}д shared={s['n_shared_players']}")

    print()
    print("Примеры MEDIUM без совпадения имени (вероятный ребрендинг):")
    med = [x for x in scored if x["confidence"] == "MEDIUM" and not x["same_normalized_name"]]
    for s in sorted(med, key=lambda x: -x["best_roster_overlap"])[:8]:
        print(f"  {s['name_a']!r}({s['team_a']}) -> {s['name_b']!r}({s['team_b']}) "
              f"overlap={s['best_roster_overlap']} gap={s['gap_days']}д")

    print()
    print("Примеры ОДИНАКОВОЕ ИМЯ, но ПАРАЛЛЕЛЬНЫЕ (merge был бы ошибкой):")
    par_same = [x for x in scored if x["relation"] == "parallel" and x["same_normalized_name"]]
    for s in par_same[:8]:
        print(f"  {s['name_a']!r}: {s['team_a']} и {s['team_b']} играли одновременно, "
              f"overlap={s['best_roster_overlap']}")
    print(f"  (всего таких пар: {len(par_same)})")

    # ---------- сколько матчей реально затронуто ----------
    section("6. Практический масштаб: сколько матчей затронуто")
    high_med = [s for s in scored if s["confidence"] in ("HIGH", "MEDIUM")]
    affected_ids = {s["team_a"] for s in high_med} | {s["team_b"] for s in high_med}
    affected_matches = sum(len(team_matches[t]) for t in affected_ids if t in team_matches)
    total_team_matches = sum(len(v) for v in team_matches.values())
    print(f"team_id в HIGH/MEDIUM парах: {len(affected_ids)}")
    print(f"Их team-matches: {affected_matches} из {total_team_matches} "
          f"({affected_matches/total_team_matches*100:.1f}%)")

    out = {
        "n_team_ids": len(teams),
        "n_team_ids_with_matches": len(team_matches),
        "n_unique_normalized_names": len(set(norm_names.values())),
        "duplicate_name_groups": len(dup_name_groups),
        "duplicate_name_pairs": dup_pairs,
        "n_candidates": len(scored),
        "n_parallel_rejected": n_parallel,
        "n_sequential": n_seq,
        "confidence_counts": dict(counts),
        "affected_team_ids": len(affected_ids),
        "affected_team_matches": affected_matches,
        "total_team_matches": total_team_matches,
        "candidates": sorted(scored, key=lambda s: (-s["best_roster_overlap"], s["gap_days"]))[:400],
        "name_normalization_audit_sample": AUDIT_LOG[:50],
    }
    path = os.path.join(EXPERIMENTS_DIR, "phase10_identity_audit.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nЗаписано: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
