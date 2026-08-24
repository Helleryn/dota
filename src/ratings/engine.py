"""
RatingEngine — компонент walk-forward Elo (ADR-003, Phase 4 раздел 12).

Формализует логику, уже проверенную на утечку в scripts/elo_prototype.py,
как переиспользуемый класс. Критично: ЭТОТ ЖЕ класс обязан использоваться
и при построении обучающего датасета, и в BacktestEngine, и в
PredictionService (см. docs/architecture.md, "единый код для as_of-
зависимых вычислений") — иначе возникает training/serving skew.

Именование методов сделано так, чтобы разработчик не мог случайно передать
пост-матчевое значение туда, где нужно pre-match: process_match() явно
возвращает RatingSnapshot с раздельными pre/post полями.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol


class MatchLike(Protocol):
    """Минимальный контракт матча, нужный RatingEngine. Совпадает по
    смыслу с полями таблицы `matches` (docs/database-design.md)."""

    match_id: int
    radiant_team_id: int
    dire_team_id: int
    radiant_win: bool


@dataclass(frozen=True)
class RatingSnapshot:
    match_id: int
    radiant_pre: float
    dire_pre: float
    radiant_post: float
    dire_post: float


class RatingEngine:
    """
    Walk-forward Elo. Матчи обязаны подаваться в process_match() строго в
    хронологическом порядке (по start_time) — сам класс это не проверяет
    (не имеет доступа к start_time через MatchLike), ответственность за
    порядок — на вызывающем коде (Dataset Builder / BacktestEngine /
    ingestion pipeline, все уже читают матчи из БД с ORDER BY start_time).
    """

    def __init__(self, k_factor: float = 32.0, base_rating: float = 1000.0):
        self.k_factor = k_factor
        self.base_rating = base_rating
        self._ratings: Dict[int, float] = {}

    def current_rating(self, team_id: int) -> float:
        """
        Текущий (= рейтинг ПОСЛЕ последнего обработанного матча этой
        команды) рейтинг. Используется PredictionService для live-прогноза
        ПОСЛЕ того, как движок "прогнан" по всей истории до настоящего
        момента — на этот вызов не действует утечка, т.к. "будущих"
        матчей относительно текущего момента по определению не существует.
        """
        return self._ratings.get(team_id, self.base_rating)

    def process_match(self, match: MatchLike) -> RatingSnapshot:
        """
        Единственный метод, изменяющий состояние. Сначала ЧИТАЕТ текущие
        (накопленные только по прошлым матчам) рейтинги — это и есть
        pre-match значения, которые должны использоваться как признаки для
        ПРОГНОЗА этого матча — и только потом обновляет состояние
        результатом. Вызывающий код обязан использовать *_pre поля для
        обучения/прогноза и НЕ обращаться к current_rating() до вызова
        process_match() для текущего матча (иначе получит то же самое
        pre-match значение, но лишний вызов проще перепутать местами).
        """
        radiant_pre = self._ratings.get(match.radiant_team_id, self.base_rating)
        dire_pre = self._ratings.get(match.dire_team_id, self.base_rating)

        expected_radiant = 1.0 / (1.0 + 10 ** ((dire_pre - radiant_pre) / 400.0))
        expected_dire = 1.0 - expected_radiant
        score_radiant = 1.0 if match.radiant_win else 0.0
        score_dire = 1.0 - score_radiant

        radiant_post = radiant_pre + self.k_factor * (score_radiant - expected_radiant)
        dire_post = dire_pre + self.k_factor * (score_dire - expected_dire)

        self._ratings[match.radiant_team_id] = radiant_post
        self._ratings[match.dire_team_id] = dire_post

        return RatingSnapshot(
            match_id=match.match_id,
            radiant_pre=radiant_pre,
            dire_pre=dire_pre,
            radiant_post=radiant_post,
            dire_post=dire_post,
        )


def _self_test() -> None:
    """
    Тот же leakage-safety тест, что в scripts/elo_prototype.py, но против
    класса RatingEngine — подтверждает, что оборачивание в класс не
    сломало гарантию отсутствия утечки.
    """
    from dataclasses import dataclass as _dc

    @_dc(frozen=True)
    class M:
        match_id: int
        radiant_team_id: int
        dire_team_id: int
        radiant_win: bool

    matches = [
        M(1, 100, 200, True),
        M(2, 300, 400, False),
        M(3, 100, 300, True),
        M(4, 200, 400, True),
    ]

    engine = RatingEngine()
    snapshots = [engine.process_match(m) for m in matches]

    # Независимый пересчёт: pre-match рейтинг команды в матче i должен
    # совпадать с post-match рейтингом её последнего предыдущего матча
    # (или base_rating, если это первый матч команды) — то есть НЕ должен
    # зависеть от match i или чего-либо после него.
    independent = RatingEngine()
    last_known: Dict[int, float] = {}
    for i, m in enumerate(matches):
        expected_radiant_pre = last_known.get(m.radiant_team_id, independent.base_rating)
        expected_dire_pre = last_known.get(m.dire_team_id, independent.base_rating)
        assert abs(snapshots[i].radiant_pre - expected_radiant_pre) < 1e-9, "УТЕЧКА: radiant_pre не совпал"
        assert abs(snapshots[i].dire_pre - expected_dire_pre) < 1e-9, "УТЕЧКА: dire_pre не совпал"
        last_known[m.radiant_team_id] = snapshots[i].radiant_post
        last_known[m.dire_team_id] = snapshots[i].dire_post

    print(f"OK: {len(matches)} матчей обработано, RatingEngine.process_match() leakage-safe")
    for s in snapshots:
        print(f"  match {s.match_id}: radiant {s.radiant_pre:.1f}->{s.radiant_post:.1f}, "
              f"dire {s.dire_pre:.1f}->{s.dire_post:.1f}")


if __name__ == "__main__":
    _self_test()
