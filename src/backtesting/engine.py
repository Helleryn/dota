"""
BacktestEngine — walk-forward backtesting (Phase 4, раздел 17;
docs/backtesting.md).

Скелет + рабочая интеграционная демонстрация на синтетических данных,
объединяющая RatingEngine (src/ratings/engine.py) и BasePredictionModel
(src/models/base.py) в один цикл:

    for match in chronological_matches:
        features = ... (используя состояние ДО этого матча)
        predict
        record
        update state (ПОСЛЕ predict)

Реальная интеграция с БД (docs/database-design.md) и полный набор метрик —
Phase 8. Здесь — контракт и доказательство, что порядок операций
(predict строго до update) реализуем и не путается местами.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Protocol

from src.models.base import BasePredictionModel
from src.ratings.engine import MatchLike, RatingEngine


@dataclass
class BacktestRecord:
    match_id: int
    predicted_radiant_proba: float
    actual_radiant_win: bool


@dataclass
class BacktestReport:
    records: List[BacktestRecord] = field(default_factory=list)

    def accuracy(self) -> float:
        if not self.records:
            return 0.0
        correct = sum(
            1 for r in self.records
            if (r.predicted_radiant_proba >= 0.5) == r.actual_radiant_win
        )
        return correct / len(self.records)

    def log_loss(self) -> float:
        import math

        if not self.records:
            return 0.0
        eps = 1e-15
        total = 0.0
        for r in self.records:
            p = min(max(r.predicted_radiant_proba, eps), 1 - eps)
            y = 1.0 if r.actual_radiant_win else 0.0
            total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        return total / len(self.records)

    def brier_score(self) -> float:
        if not self.records:
            return 0.0
        total = sum(
            (r.predicted_radiant_proba - (1.0 if r.actual_radiant_win else 0.0)) ** 2
            for r in self.records
        )
        return total / len(self.records)


class BacktestEngine:
    """
    Намеренно НЕ зависит от конкретной формы признаков — принимает
    RatingEngine напрямую (для MVP-скоупа, где Elo — единственный
    stateful-признак, см. docs/research-summary.md, Feature Set 0). Полная
    версия (Phase 8) обобщит это до произвольного списка feature-
    калькуляторов (docs/backtesting.md), интерфейс здесь останется
    совместимым — метод run() не изменит сигнатуру.
    """

    def __init__(self, model: BasePredictionModel, rating_engine: RatingEngine):
        self.model = model
        self.rating_engine = rating_engine

    def run(self, matches: Iterable[MatchLike]) -> BacktestReport:
        report = BacktestReport()

        for match in matches:
            # 1. Признаки — читаем состояние ДО этого матча.
            radiant_pre = self.rating_engine.current_rating(match.radiant_team_id)
            dire_pre = self.rating_engine.current_rating(match.dire_team_id)

            # 2. Прогноз — строго до обновления состояния.
            proba = self.model.predict_proba([(radiant_pre, dire_pre)])[0][1]

            # 3. Фиксация результата для метрик.
            report.records.append(
                BacktestRecord(
                    match_id=match.match_id,
                    predicted_radiant_proba=proba,
                    actual_radiant_win=match.radiant_win,
                )
            )

            # 4. Обновление состояния — ПОСЛЕДНИЙ шаг, после predict.
            self.rating_engine.process_match(match)

        return report


def _self_test() -> None:
    from dataclasses import dataclass as _dc

    from src.models.base import EloRuleModel

    @_dc(frozen=True)
    class M:
        match_id: int
        radiant_team_id: int
        dire_team_id: int
        radiant_win: bool

    # Синтетика: команда 1 стабильно сильнее команды 2 (выигрывает 4 из 5
    # очных встреч) — ожидаем, что backtest покажет accuracy > random (0.5)
    # и разумный log loss, а НЕ идеальные 100%/0.0 (что было бы подозрительно
    # и указывало бы на утечку).
    matches = [
        M(1, 1, 2, True),
        M(2, 2, 1, False),  # team 1 побеждает (radiant=team2 проигрывает)
        M(3, 1, 2, True),
        M(4, 2, 1, False),
        M(5, 1, 2, False),  # один проигрыш для реализма
        M(6, 1, 2, True),
        M(7, 2, 1, False),
        M(8, 1, 2, True),
    ]

    engine = BacktestEngine(model=EloRuleModel(), rating_engine=RatingEngine())
    engine.model.fit(None, None)  # EloRuleModel.fit — no-op, но вызывается для единообразия

    report = engine.run(matches)

    print(f"OK: backtest прогнан на {len(report.records)} матчах")
    print(f"  accuracy={report.accuracy():.3f} log_loss={report.log_loss():.3f} "
          f"brier={report.brier_score():.3f}")

    # Первый матч ОБЯЗАН быть предсказан как ровно 0.5 (обе команды стартуют
    # с базовым рейтингом, история отсутствует) — это и есть проверка, что
    # предсказание не подглядывает в исход этого же матча.
    assert abs(report.records[0].predicted_radiant_proba - 0.5) < 1e-9, (
        "УТЕЧКА: первый матч команды не может быть предсказан иначе, чем 0.5 "
        "(нет предшествующей истории)"
    )
    print("OK: первый матч предсказан как 0.5 (нет истории) — утечки в начальном состоянии нет")


if __name__ == "__main__":
    _self_test()
