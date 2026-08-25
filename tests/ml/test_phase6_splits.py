"""
Phase 6, раздел 2 и 6 — обязательные тесты:
  - team_a/team_b не должны быть закодированы через winner/loser;
  - chronological split не должен допускать temporal overlap.

Работают на синтетических pandas DataFrame, без БД — быстрые unit-тесты,
не integration (в отличие от tests/integration/test_database.py).
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from scripts.phase6_pipeline import audit_data_cutoff, audit_team_a_team_b_not_winner_loser, chronological_split


def _make_df(n: int, radiant_win_pattern) -> pd.DataFrame:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        rows.append({
            "match_id": i,
            "as_of_timestamp": base + timedelta(hours=i),
            "radiant_team_id": 100 + (i % 5),
            "dire_team_id": 200 + (i % 7),
            "target": int(radiant_win_pattern(i)),
        })
    return pd.DataFrame(rows)


def test_team_a_team_b_not_winner_loser_passes_on_balanced_target():
    df = _make_df(100, lambda i: i % 2 == 0)  # ровно 50/50, не тождественно winner
    audit_team_a_team_b_not_winner_loser(df)  # не должно поднять исключение


def test_team_a_team_b_check_catches_winner_coded_target():
    """Если target тождественно True (т.е. 'radiant' на самом деле означает
    'winner') — audit ОБЯЗАН это поймать, не пропустить молча."""
    df = _make_df(100, lambda i: True)
    with pytest.raises(AssertionError):
        audit_team_a_team_b_not_winner_loser(df)


def test_chronological_split_no_temporal_overlap():
    df = _make_df(100, lambda i: i % 2 == 0)
    train, val, test, periods = chronological_split(df, train_frac=0.6, val_frac=0.2)

    assert periods["train_end"] < periods["validation_start"]
    assert periods["validation_end"] < periods["test_start"]
    assert len(train) + len(val) + len(test) == len(df)

    # Дополнительная проверка на уровне самих строк, не только агрегатов periods:
    assert train["as_of_timestamp"].max() < val["as_of_timestamp"].min()
    assert val["as_of_timestamp"].max() < test["as_of_timestamp"].min()


def test_data_cutoff_audit_detects_inconsistent_matches_played_before():
    df = _make_df(20, lambda i: i % 2 == 0)
    df["radiant_matches_played_before"] = 0
    df["dire_matches_played_before"] = 0
    # Валидно: 0 матчей у каждой команды до начала (данные синтетические, каждая
    # команда впервые встречается в своей первой строке по построению _make_df
    # не гарантированно, поэтому тест проверяет ИМЕННО детекцию несоответствия).
    with pytest.raises(AssertionError):
        audit_data_cutoff(df)  # т.к. реальный подсчёт даст >0 для повторных встреч команд
