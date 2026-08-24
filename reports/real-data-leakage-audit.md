# REAL DATA LEAKAGE AUDIT

Два независимых слоя проверки. Оба используют РЕАЛЬНЫЙ код пайплайна
(`build_feature_set_0`, `RatingEngine`, `DatasetBuilder._load_pro_matches`),
не изолированные прототипы.

## Слой 1: pytest suite (fixture-based, регрессионный)

```
tests/leakage/test_feature_set_0_leakage.py::test_truncating_future_does_not_change_past_rows PASSED
tests/leakage/test_feature_set_0_leakage.py::test_changing_future_winner_does_not_affect_earlier_rows PASSED
tests/leakage/test_feature_set_0_leakage.py::test_changing_future_winner_does_not_affect_that_matchs_own_pre_match_features PASSED
tests/leakage/test_team_mapping.py::test_team_mapping_independent_of_winner PASSED
tests/leakage/test_team_mapping.py::test_team_mapping_is_not_winner_loser PASSED
tests/leakage/test_team_mapping.py::test_normalize_match_deterministic_regardless_of_outcome[True] PASSED
tests/leakage/test_team_mapping.py::test_normalize_match_deterministic_regardless_of_outcome[False] PASSED

7 passed
```

Это регрессионные тесты на синтетических (детерминированных) данных —
проверяют инвариант структурно, на управляемых edge cases. Без изменений
в этой сессии, унаследованы из Phase 5.

## Слой 2: adversarial audit на РЕАЛЬНЫХ данных (новое в этой сессии)

`scripts/real_data_leakage_audit.py` — берёт реальную pro-выборку из
PostgreSQL через `DatasetBuilder._load_pro_matches` (тот же запрос, что
используется для построения датасета — не отдельная копия логики) и
проводит 4 намеренные атаки, требуемые заданием (раздел 23):

| # | Атака | Проверяемое свойство | Результат |
|---|---|---|---|
| 1 | Меняем `radiant_win` матчу из середины реальной истории | Признаки ВСЕХ более ранних матчей не меняются | **PASS** |
| 2 | Добавляем синтетический будущий матч (после всех реальных) | Признаки всех существующих матчей не меняются | **PASS** |
| 3 | Меняем результат этого будущего матча | Признаки/прогнозы прошлых матчей не меняются | **PASS** |
| 4 | Меняем данные строго ПОСЛЕ `prediction_timestamp` целевого матча | Признаки целевого матча не меняются | **PASS** |

```
Матчей в реальной pro-выборке: 178
Test 1 (смена исхода прошлого матча не влияет на более ранние признаки): PASS
Test 2 (добавление будущего матча не влияет на признаки существующих): PASS
Test 3 (смена результата будущего матча не влияет на прошлые предсказания): PASS
Test 4 (изменение данных после prediction_timestamp не влияет на признаки цели): PASS

Все 4 адверсариальных теста пройдены на РЕАЛЬНЫХ данных.
```

Каждый тест сравнивает ПОЛНЫЙ снимок pre-match признаков (elo обеих
команд, elo_difference, recent_winrate обеих команд и разница,
days_since_last_match обеих команд, matches_played_before обеих команд) —
не только одно поле, чтобы исключить частичную утечку через побочные
признаки.

## Temporal audit (раздел 22 задания)

`build_feature_set_0` — чистая функция от хронологически отсортированного
потока матчей: `RatingEngine.process_match()` возвращает pre-match snapshot
(`radiant_pre`/`dire_pre`) ДО обновления состояния, `_TeamFormTracker`
читает `recent_winrate`/`days_since_last_match` ДО вызова `record()` для
текущего матча (см. код, `src/datasets/feature_set_0.py`, строки
"1. ЧТЕНИЕ состояния ДО" / "2. ОБНОВЛЕНИЕ состояния — строго ПОСЛЕ").
Инвариант "все feature source records < prediction_timestamp" следует из
самой структуры кода (нет обратных ссылок, нет двухпроходной агрегации),
что и подтверждают адверсариальные тесты выше — они атакуют именно эту
гарантию с реальными данными, а не проверяют её "по построению" на веру.

## Итог

Leakage-safety подтверждена на 178 реальных pro-tier матчах (май-август
2026, `PRO_LEAGUE_TIERS`-отфильтрованная выборка) — не только на fixtures.
Ни один из 11 тестов (7 pytest + 4 adversarial) не выявил утечки.
