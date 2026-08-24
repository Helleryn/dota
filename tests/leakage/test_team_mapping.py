"""
test_team_mapping_independent_of_winner (Phase 5, раздел 10 и 20) —
один из самых важных тестов проекта.

Правило: team_a/team_b (в нашей реализации — radiant/dire, см.
docs/database-design.md, "team_a/team_b маппинг не должен зависеть от
исхода") определяются ДО того, как известен результат. Тест проверяет это
адверсариально: меняем radiant_win на fixture с ОДИНАКОВЫМИ остальными
данными и убеждаемся, что radiant_team_id/dire_team_id НЕ меняются местами
и не пересчитываются в зависимости от того, кто победил.
"""

from datetime import datetime, timezone

import pytest

from src.datasources.base import RawMatch
from src.normalization.normalize import normalize_match


def _raw_match(radiant_win: bool) -> RawMatch:
    return RawMatch(
        match_id=1,
        start_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
        duration_seconds=1800,
        radiant_team_id=100,
        dire_team_id=200,
        radiant_team_name="Team Alpha",
        dire_team_name="Team Beta",
        radiant_win=radiant_win,
        league_id=5000,
        league_tier="premium",
        series_id=1,
        series_type=1,
        patch=None,
        source="opendota",
        fetched_at=datetime.now(timezone.utc),
    )


def test_team_mapping_independent_of_winner():
    """
    Адверсариальный тест из Phase 5, раздел 27: "change match winner" не
    должно менять pre-match признаки — здесь конкретно team_id mapping.
    """
    # ingested_at фиксирован явно — иначе datetime.now() по умолчанию даёт
    # микросекундный шум между двумя вызовами, не имеющий отношения к
    # утечке данных (это технический артефакт теста, не свойство кода).
    fixed_ingested_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    match_radiant_wins = normalize_match(_raw_match(radiant_win=True), ingested_at=fixed_ingested_at)
    match_dire_wins = normalize_match(_raw_match(radiant_win=False), ingested_at=fixed_ingested_at)

    # radiant_team_id/dire_team_id идентичны в ОБОИХ случаях — единственное
    # различие между двумя normalized-объектами обязано быть в radiant_win,
    # и НИГДЕ БОЛЬШЕ.
    assert match_radiant_wins.radiant_team_id == match_dire_wins.radiant_team_id == 100
    assert match_radiant_wins.dire_team_id == match_dire_wins.dire_team_id == 200

    fields_that_may_differ = {"radiant_win"}
    for field in match_radiant_wins.__dataclass_fields__:
        if field in fields_that_may_differ:
            continue
        v1 = getattr(match_radiant_wins, field)
        v2 = getattr(match_dire_wins, field)
        assert v1 == v2, (
            f"Поле '{field}' отличается между исходами матча ({v1!r} vs {v2!r}), "
            f"хотя единственная легитимная разница — radiant_win. Это означает "
            f"УТЕЧКУ: какое-то поле неявно зависит от результата."
        )


def test_team_mapping_is_not_winner_loser():
    """
    Явная проверка запрещённого паттерна из задания: team_a != победитель,
    team_b != проигравший. Мы используем radiant/dire (реальная,
    известная до матча игровая асимметрия), не "winner"/"loser".
    """
    lost = normalize_match(_raw_match(radiant_win=False))
    # radiant_team_id остаётся "радиантом" даже когда radiant ПРОИГРАЛ —
    # если бы маппинг был по принципу winner/loser, radiant_team_id здесь
    # оказался бы равен 200 (команде-победителю), а не 100.
    assert lost.radiant_team_id == 100, (
        "radiant_team_id обязан оставаться игровой ролью, не "
        "победителем — иначе колонка сама кодирует целевую переменную"
    )


@pytest.mark.parametrize("radiant_win", [True, False])
def test_normalize_match_deterministic_regardless_of_outcome(radiant_win):
    """Повторный вызов normalize_match с теми же входными данными должен
    давать идентичный результат — детерминированность (docs/ml-architecture.md)."""
    raw = _raw_match(radiant_win=radiant_win)
    result1 = normalize_match(raw, ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    result2 = normalize_match(raw, ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert result1 == result2
