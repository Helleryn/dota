"""
Leakage test для полного Feature Set 0 (Phase 5, раздел 27) — не только Elo
(уже покрыт scripts/elo_prototype.py и src/ratings/engine.py), а ВСЕГО
набора признаков (recent_winrate, days_since_last_match и т.д.).

Два теста:
1. "Если удалить всю информацию после prediction_timestamp, dataset должен
   остаться идентичным" — пересчёт признаков только по префиксу истории
   должен совпасть построчно с полным прогоном.
2. Adversarial: изменение исхода БУДУЩЕГО матча не должно менять признаки
   более РАННИХ матчей (a fortiori — признаки самого этого матча, помимо
   целевой переменной).
"""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from src.datasets.feature_set_0 import build_feature_set_0


@dataclass(frozen=True)
class SyntheticMatch:
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_win: bool


def _matches():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    pairings = [(100, 200), (300, 400), (100, 300), (200, 400), (100, 400), (200, 300)]
    return [
        SyntheticMatch(
            match_id=1000 + i,
            start_time=base + timedelta(days=3 * i),
            radiant_team_id=pairings[i % len(pairings)][0],
            dire_team_id=pairings[i % len(pairings)][1],
            radiant_win=(i % 3 != 0),
        )
        for i in range(10)
    ]


def test_truncating_future_does_not_change_past_rows():
    """
    Phase 5, раздел 27: "Если удалить всю информацию после
    prediction_timestamp, dataset должен остаться идентичным."
    """
    matches = _matches()
    full_rows = build_feature_set_0(matches)

    for cutoff in range(1, len(matches) + 1):
        truncated_rows = build_feature_set_0(matches[:cutoff])
        # Последняя строка truncated-прогона обязана побитово совпадать с
        # соответствующей строкой полного прогона — она не могла "увидеть"
        # ничего, что появилось после cutoff, в обоих случаях.
        assert truncated_rows[-1] == full_rows[cutoff - 1], (
            f"Строка для матча на позиции {cutoff - 1} отличается между полным "
            f"прогоном и прогоном, обрезанным по этому же матчу — это означает "
            f"УТЕЧКУ данных из будущего."
        )


def test_changing_future_winner_does_not_affect_earlier_rows():
    """Adversarial-тест: подменяем исход ПОСЛЕДНЕГО матча — все более ранние
    строки обязаны остаться абсолютно неизменными (byte-for-byte)."""
    matches = _matches()
    original_rows = build_feature_set_0(matches)

    flipped = list(matches)
    flipped[-1] = replace(flipped[-1], radiant_win=not flipped[-1].radiant_win)
    flipped_rows = build_feature_set_0(flipped)

    for i in range(len(matches) - 1):
        assert original_rows[i] == flipped_rows[i], (
            f"Строка {i} изменилась из-за подмены исхода ПОСЛЕДНЕГО матча — "
            f"признаки более раннего матча не могут зависеть от будущего результата."
        )


def test_changing_future_winner_does_not_affect_that_matchs_own_pre_match_features():
    """
    Более тонкий вариант: меняем исход матча k, признаки самого матча k
    (elo/winrate/days_since_last_match ДО его начала) не должны измениться
    — они уже зафиксированы к моменту, когда становится известен исход.
    Меняется только radiant_win и всё, что идёт ПОСЛЕ матча k.
    """
    matches = _matches()
    k = 5
    original_rows = build_feature_set_0(matches)

    flipped = list(matches)
    flipped[k] = replace(flipped[k], radiant_win=not flipped[k].radiant_win)
    flipped_rows = build_feature_set_0(flipped)

    row_original = original_rows[k]
    row_flipped = flipped_rows[k]

    assert row_original.radiant_elo == row_flipped.radiant_elo
    assert row_original.dire_elo == row_flipped.dire_elo
    assert row_original.radiant_recent_winrate == row_flipped.radiant_recent_winrate
    assert row_original.dire_recent_winrate == row_flipped.dire_recent_winrate
    assert row_original.radiant_days_since_last_match == row_flipped.radiant_days_since_last_match
    assert row_original.dire_days_since_last_match == row_flipped.dire_days_since_last_match
    # Единственное легитимное отличие — сама целевая переменная.
    assert row_original.radiant_win != row_flipped.radiant_win

    # А вот строки ПОСЛЕ k — обязаны отличаться (иначе признак вообще не
    # зависит от исхода матчей, что означало бы, что Elo/форма сломаны и
    # ничего не считают, а не то, что утечки нет).
    changed_after = any(
        original_rows[j] != flipped_rows[j] for j in range(k + 1, len(matches))
    )
    assert changed_after, (
        "Ни одна из строк ПОСЛЕ матча k не изменилась при смене его исхода — "
        "это означает, что признаки вообще не реагируют на историю (баг в "
        "другую сторону, не leakage, но тоже неправильно)."
    )
