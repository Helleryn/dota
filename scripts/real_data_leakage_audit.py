#!/usr/bin/env python3
"""
Adversarial leakage audit НА РЕАЛЬНЫХ данных (не fixtures) — задание,
раздел 22-23. Дополняет (не заменяет) tests/leakage/ — там те же свойства
проверены на синтетике как часть pytest suite, здесь — на реальном датасете
из живого OpenDota (src/datasets/builder._load_pro_matches).

Тесты:
  1. Меняем radiant_win историческому матчу из середины истории ->
     признаки ВСЕХ более ранних матчей не должны измениться.
  2. Добавляем будущий матч (позже всех реальных) -> признаки всех
     существующих матчей не должны измениться.
  3. Меняем результат будущего матча (после добавления) -> признаки более
     ранних матчей (prediction) не должны измениться.
  4. Меняем данные СТРОГО ПОСЛЕ prediction_timestamp целевого матча ->
     признаки ЭТОГО матча не должны измениться (и предыдущих тоже).

Ничего не пишет в БД — работает над списком строк, скопированным из реальной
выборки (docs/data-leakage.md).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_settings
from src.datasets.builder import _load_pro_matches
from src.datasets.feature_set_0 import build_feature_set_0
from src.db.engine import make_engine


@dataclass
class M:
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_win: bool


def to_list(rows):
    return [M(r.match_id, r.start_time, r.radiant_team_id, r.dire_team_id, r.radiant_win) for r in rows]


def feature_snapshot(rows):
    """dict match_id -> кортеж всех pre-match полей (без radiant_win — это target, не признак)."""
    out = {}
    for r in rows:
        out[r.match_id] = (
            round(r.radiant_elo, 6), round(r.dire_elo, 6), round(r.elo_difference, 6),
            r.radiant_recent_winrate, r.dire_recent_winrate, r.recent_winrate_difference,
            r.radiant_days_since_last_match, r.dire_days_since_last_match,
            r.radiant_matches_played_before, r.dire_matches_played_before,
        )
    return out


def main() -> int:
    settings = load_settings()
    engine = make_engine(settings)
    raw = _load_pro_matches(engine)
    if len(raw) < 10:
        print(f"Недостаточно pro-матчей для аудита ({len(raw)}) — нужно >=10")
        return 1

    baseline_matches = to_list(raw)
    baseline_rows = build_feature_set_0(baseline_matches)
    baseline_snapshot = feature_snapshot(baseline_rows)
    n = len(baseline_matches)
    mid = n // 2

    failures = []

    # --- Test 1: изменить исход матча из середины истории ---
    mutated = [M(m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id, m.radiant_win) for m in baseline_matches]
    mutated[mid].radiant_win = not mutated[mid].radiant_win
    mutated_rows = build_feature_set_0(mutated)
    mutated_snapshot = feature_snapshot(mutated_rows)

    earlier_ids = [m.match_id for m in baseline_matches[:mid]]
    changed_before = [mid_ for mid_ in earlier_ids if baseline_snapshot[mid_] != mutated_snapshot[mid_]]
    test1_ok = len(changed_before) == 0
    if not test1_ok:
        failures.append(f"Test 1 FAILED: смена исхода матча #{mid} изменила признаки {len(changed_before)} БОЛЕЕ РАННИХ матчей: {changed_before[:5]}")

    # --- Test 2: добавить будущий матч ---
    last_time = baseline_matches[-1].start_time
    future_match = M(
        match_id=999_999_999_001,
        start_time=last_time + timedelta(days=1),
        radiant_team_id=baseline_matches[0].radiant_team_id,
        dire_team_id=baseline_matches[0].dire_team_id,
        radiant_win=True,
    )
    with_future = baseline_matches + [future_match]
    with_future_rows = build_feature_set_0(with_future)
    with_future_snapshot = feature_snapshot(with_future_rows)

    all_original_ids = [m.match_id for m in baseline_matches]
    changed_by_future = [mid_ for mid_ in all_original_ids if baseline_snapshot[mid_] != with_future_snapshot[mid_]]
    test2_ok = len(changed_by_future) == 0
    if not test2_ok:
        failures.append(f"Test 2 FAILED: добавление будущего матча изменило признаки {len(changed_by_future)} существующих матчей: {changed_by_future[:5]}")

    # --- Test 3: изменить результат будущего матча ---
    future_match_2 = M(
        match_id=999_999_999_001,
        start_time=last_time + timedelta(days=1),
        radiant_team_id=baseline_matches[0].radiant_team_id,
        dire_team_id=baseline_matches[0].dire_team_id,
        radiant_win=False,  # изменили относительно future_match
    )
    with_future_2 = baseline_matches + [future_match_2]
    with_future_2_rows = build_feature_set_0(with_future_2)
    with_future_2_snapshot = feature_snapshot(with_future_2_rows)

    changed_by_future_result = [mid_ for mid_ in all_original_ids if with_future_snapshot[mid_] != with_future_2_snapshot[mid_]]
    test3_ok = len(changed_by_future_result) == 0
    if not test3_ok:
        failures.append(f"Test 3 FAILED: смена результата будущего матча изменила признаки {len(changed_by_future_result)} прошлых предсказаний: {changed_by_future_result[:5]}")

    # --- Test 4: изменить данные СТРОГО ПОСЛЕ prediction_timestamp целевого матча ---
    target_idx = mid - 1 if mid > 0 else 0
    target_id = baseline_matches[target_idx].match_id
    later_idx = mid + 1 if mid + 1 < n else n - 1
    mutated_later = [M(m.match_id, m.start_time, m.radiant_team_id, m.dire_team_id, m.radiant_win) for m in baseline_matches]
    if later_idx > target_idx:
        mutated_later[later_idx].radiant_win = not mutated_later[later_idx].radiant_win
        mutated_later_rows = build_feature_set_0(mutated_later)
        mutated_later_snapshot = feature_snapshot(mutated_later_rows)
        test4_ok = baseline_snapshot[target_id] == mutated_later_snapshot[target_id]
        if not test4_ok:
            failures.append(f"Test 4 FAILED: изменение матча #{later_idx} (после prediction_timestamp матча #{target_idx}) изменило признаки матча #{target_idx} (match_id={target_id})")
    else:
        test4_ok = True  # нечего мутировать, целевой матч — последний

    print(f"Матчей в реальной pro-выборке: {n}")
    print(f"Test 1 (смена исхода прошлого матча не влияет на более ранние признаки): {'PASS' if test1_ok else 'FAIL'}")
    print(f"Test 2 (добавление будущего матча не влияет на признаки существующих): {'PASS' if test2_ok else 'FAIL'}")
    print(f"Test 3 (смена результата будущего матча не влияет на прошлые предсказания): {'PASS' if test3_ok else 'FAIL'}")
    print(f"Test 4 (изменение данных после prediction_timestamp не влияет на признаки цели): {'PASS' if test4_ok else 'FAIL'}")

    if failures:
        print("\n--- ОШИБКИ ---")
        for f in failures:
            print(f)
        return 1

    print("\nВсе 4 адверсариальных теста пройдены на РЕАЛЬНЫХ данных.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
