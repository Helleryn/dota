#!/usr/bin/env python3
"""
Прототип leakage-safe walk-forward Elo (Phase 3, раздел ELO).

Цель — не продовый код, а проверка гипотезы: можем ли мы посчитать рейтинг
команды на момент КАЖДОГО матча так, чтобы он гарантированно зависел только
от матчей, сыгранных строго раньше. Формула Elo (K=32, база 1000) взята из
verified-источника — реализации team_rating в odota/core (см.
docs/data-sources.md).

Работает на синтетических данных (без сети, без реального датасета) —
этого достаточно, чтобы проверить корректность самого механизма, что и
требуется на исследовательской фазе.

Запуск: python3 scripts/elo_prototype.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SyntheticMatch:
    match_id: int
    start_time: int  # unix-подобная возрастающая метка времени
    team_a: str
    team_b: str
    a_win: bool


def compute_walk_forward_elo(
    matches: List[SyntheticMatch],
    k: float = 32.0,
    base_rating: float = 1000.0,
) -> Tuple[Dict[int, Tuple[float, float]], Dict[str, float]]:
    """
    Идёт по матчам строго в хронологическом порядке. Для каждого матча
    сначала ЧИТАЕТ текущий (то есть накопленный только по прошлым матчам)
    рейтинг обеих команд и только ПОТОМ обновляет его результатом этого матча.

    Возвращает:
      pre_match_ratings: {match_id: (team_a_rating_до_матча, team_b_rating_до_матча)}
      final_ratings: {team_name: рейтинг после последнего матча в списке}
    """
    ratings: Dict[str, float] = {}
    pre_match_ratings: Dict[int, Tuple[float, float]] = {}

    for m in sorted(matches, key=lambda x: x.start_time):
        ra = ratings.get(m.team_a, base_rating)
        rb = ratings.get(m.team_b, base_rating)
        pre_match_ratings[m.match_id] = (ra, rb)

        expected_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        expected_b = 1.0 - expected_a
        score_a = 1.0 if m.a_win else 0.0
        score_b = 1.0 - score_a

        ratings[m.team_a] = ra + k * (score_a - expected_a)
        ratings[m.team_b] = rb + k * (score_b - expected_b)

    return pre_match_ratings, ratings


def assert_leakage_safe(matches: List[SyntheticMatch]) -> None:
    """
    Проверка: pre-match рейтинг команды в матче m должен ТОЧНО совпадать
    с рейтингом, независимо пересчитанным только по матчам со
    start_time < m.start_time. Если это не так — значит, реализация Elo
    где-то заглядывает в будущее (типичная ошибка: пересчёт по всей
    таблице разом без сортировки, или использование среза "as of today"
    вместо "as of match date").
    """
    sorted_matches = sorted(matches, key=lambda x: x.start_time)
    full_pre_ratings, _ = compute_walk_forward_elo(sorted_matches)

    violations = []
    for i, m in enumerate(sorted_matches):
        earlier_matches = sorted_matches[:i]
        _, independent_final_ratings = compute_walk_forward_elo(earlier_matches)

        independent_a = independent_final_ratings.get(m.team_a, 1000.0)
        independent_b = independent_final_ratings.get(m.team_b, 1000.0)
        pipeline_a, pipeline_b = full_pre_ratings[m.match_id]

        if abs(independent_a - pipeline_a) > 1e-9 or abs(independent_b - pipeline_b) > 1e-9:
            violations.append(
                (m.match_id, (independent_a, independent_b), (pipeline_a, pipeline_b))
            )

    if violations:
        raise AssertionError(f"Обнаружена утечка данных в Elo-пайплайне: {violations}")

    print(f"OK: {len(sorted_matches)} матчей, утечки не обнаружено (pre-match rating == "
          f"независимый пересчёт только по более ранним матчам)")


def _demo() -> None:
    """Небольшой синтетический пример: 4 команды, 12 матчей."""
    matches = [
        SyntheticMatch(1, 100, "A", "B", True),
        SyntheticMatch(2, 110, "C", "D", False),
        SyntheticMatch(3, 120, "A", "C", True),
        SyntheticMatch(4, 130, "B", "D", True),
        SyntheticMatch(5, 140, "A", "D", True),
        SyntheticMatch(6, 150, "B", "C", False),
        SyntheticMatch(7, 160, "A", "B", False),
        SyntheticMatch(8, 170, "C", "D", True),
        SyntheticMatch(9, 180, "A", "C", False),
        SyntheticMatch(10, 190, "B", "D", False),
        SyntheticMatch(11, 200, "A", "D", False),
        SyntheticMatch(12, 210, "B", "C", True),
    ]

    pre_match, final = compute_walk_forward_elo(matches)
    print("Pre-match рейтинги (первые 3 матча):")
    for m in matches[:3]:
        ra, rb = pre_match[m.match_id]
        print(f"  match {m.match_id}: {m.team_a}={ra:.1f} vs {m.team_b}={rb:.1f}")
    print("Итоговые рейтинги:", {k: round(v, 1) for k, v in final.items()})

    assert_leakage_safe(matches)


if __name__ == "__main__":
    _demo()
