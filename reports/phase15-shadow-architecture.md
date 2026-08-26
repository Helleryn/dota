# PHASE 15 — архитектура shadow-конвейера

> Модель заморожена. Новых признаков нет. Production API не переделан —
> построен domain/backend слой.

---

## 1. Компоненты

| Модуль | Ответственность |
|---|---|
| `src/shadow/versions.py` | замороженная спецификация и четыре версии |
| `src/shadow/states.py` | состояния прогноза и причины отказа |
| `src/shadow/snapshot.py` | неизменяемый снимок и запись разрешения |
| `src/shadow/cutoff.py` | проверка **NO DATA AFTER prediction_timestamp** |
| `src/shadow/repository.py` | хранение (только вставка и движение состояния) |
| `src/shadow/engine.py` | замороженная модель + скользящий слой калибровки |
| `src/shadow/pipeline.py` | обнаружение, сборка снимка, разрешение |
| `src/shadow/metrics.py` | метрики потока |
| `scripts/shadow_cli.py` | команды |
| миграции `a1b2c3d4e5f6`, `b2c3d4e5f6a7` | таблицы и триггер неизменяемости |

---

## 2. Что именно заморожено (PART: FROZEN MODEL / FROZEN CALIBRATION)

```
features:    elo_difference, form_3_difference, elo_mean_diff,
             five_vs_team_elo_diff, hero_exp_decay_diff        # ПЯТЬ, не три
elo_k:       16
form_window: 3
random_seed: 42
calibration: beta, rolling, window=5000, min_history=500
```

Версии в каждом снимке:

```
model_version        phase9-logreg-5feat-k16-form3-seed42
feature_version      phase9-frozen-v1
calibration_version  phase14-beta-rolling5000-v1
prediction_version   phase15-shadow-v1
```

Отдельно зафиксировано решение Phase 14: **патч-локальная составляющая
гибрида в shadow не берётся.** На VALIDATION она давала лучший ECE в
переходных окнах, на TEST в трёх окнах из четырёх сделала хуже. В прод
идёт чистое скользящее окно.

Тест `test_model_version_and_spec_are_reproducible` проверяет в том числе,
что признаков ровно пять — чтобы сокращение до трёх не повторилось.

---

## 3. Признаки нового матча без переписывания замороженного кода

Модули признаков Phase 9 читают состояние ДО матча и обновляют строго
ПОСЛЕ; это доказано тестами утечки каждого модуля.

Отсюда приём: к истории добавляется целевой матч **с заглушкой исхода**,
walk-forward проход выполняется по всей последовательности, берётся
последняя строка. Её признаки по построению зависят только от матчей
раньше неё.

Приём был бы утечкой, если бы заглушка попадала в собственные признаки,
поэтому он проверяется напрямую:
`test_placeholder_outcome_cannot_reach_its_own_features` переворачивает
заглушку и требует побитового совпадения признаков.

---

## 4. Неизменяемость — инвариант СУБД, а не соглашение кода

Требование PART E: опубликованный прогноз нельзя изменить задним числом,
и результат матча не должен иметь такой возможности в принципе.

Если бы это обеспечивалось дисциплиной кода, любая будущая правка
репозитория молча сломала бы гарантию. Поэтому запрет живёт в триггере:

```sql
IF (to_jsonb(NEW) - 'state') IS DISTINCT FROM (to_jsonb(OLD) - 'state') THEN
    RAISE EXCEPTION 'опубликованный снимок % неизменяем: ...';
END IF;
```

### Как эта формулировка появилась

Первая версия триггера перечисляла защищённые колонки поимённо. Тест
`test_patch_change_does_not_mutate_historical_prediction` нашёл дыру:
`patch_name` в перечне отсутствовал, и обновить его у опубликованного
снимка удавалось. Перечисление колонок ненадёжно в принципе — при
добавлении колонки о нём забудут.

Логика инвертирована (миграция `b2c3d4e5f6a7`): после публикации
отличаться может **только** `state`, всё остальное сравнивается целиком.
Новые колонки защищены автоматически.

Удаление запрещено полностью. Триггер построчный, поэтому `TRUNCATE`
остаётся доступен для административного сброса — именно так очищается
тестовая база. Первый прогон тестов уткнулся в собственный триггер на
`DELETE`, что и есть подтверждение работы защиты.

---

## 5. Состояния (PART E)

```
DISCOVERED → FEATURES_READY → PREDICTED → CALIBRATED → PUBLISHED
                                                          ↓
                             MATCH_STARTED → MATCH_FINISHED → RESOLVED
                                                          ↓
                                                       INVALID
```

Движение только вперёд; `INVALID` достижим из любого состояния — это
отметка «прогноз непригоден», а не шаг конвейера. Порядок продублирован
в триггере: нарушение в коде всё равно не пройдёт в базу.

---

## 6. Разделение прогноза и исхода (PART F/G)

Две таблицы, а не одна:

* `prediction_snapshots` — что система знала и что предсказала;
* `prediction_resolutions` — что произошло и как это оценивается.

Запись исхода физически не может изменить прогноз.
`test_resolution_does_not_alter_prediction` сверяет содержимое снимка до
и после записи разрешения.

**Двойной вывод (PART G):** в снимке хранятся обе вероятности —
`raw_probability` и `calibrated_probability`. Это единственный способ
проверить, работает ли результат Phase 14 на потоке, а не только на
историческом TEST.

---

## 7. Обработка отказов (PART R) — тихого fallback нет

| Ситуация | Статус | Как проверено |
|---|---|---|
| источник недоступен | `INVALID: source_unavailable` | `test_unavailable_source_yields_explicit_status_not_empty_list` |
| состав неизвестен | `INVALID: lineup_unknown` | `test_missing_features_are_refused_not_imputed` |
| матч отменён | `INVALID: match_cancelled` | `test_cancelled_match_gets_explicit_status` |
| матч перенесён | новый прогноз с новым временем | `test_postponed_match_creates_new_prediction_not_overwrite` |
| матч уже начался | `INVALID: match_already_started` | `test_prediction_after_match_start_is_invalid` |
| истории для калибровки мало | `calibration_version = "none"` | `test_short_calibration_history_is_reported_not_silently_skipped` |
| данные позже прогноза | `INVALID: future_data_detected` | `test_cutoff_check_rejects_state_later_than_prediction` |
| **модель обучена не раньше прогноза** | `INVALID: model_trained_after_prediction` | `test_prediction_before_train_cutoff_is_invalid` |

Последняя строка — вид утечки, найденный в ходе фазы и не покрываемый
проверкой отметок среза: если прогноз сделан раньше, чем заканчивается
обучающая выборка, модель видела исходы матчей не раньше собственного
прогноза. Отметки среза при этом чисты, а прогноз всё равно непригоден.

---

## 8. Дубликаты (PART U-8/9)

Идентификатор прогноза детерминированный:
`sha256(match_key | prediction_timestamp | prediction_version)` с
округлением времени до секунды. Повторный запуск с теми же входами даёт
тот же идентификатор, дубликат отсекается первичным ключом и возбуждает
`DuplicatePrediction` — а не создаётся молча. То же для разрешений.

Округление до секунды существенно: без него два запуска в одну секунду
дали бы разные идентификаторы из-за микросекунд, и защита не сработала бы.

---

## 9. Команды (PART T)

```
python3 scripts/shadow_cli.py check-sources        # PART A
python3 scripts/shadow_cli.py predict-upcoming     # PART D, режим live_draft
python3 scripts/shadow_cli.py replay --since ... --until ...
python3 scripts/shadow_cli.py resolve-finished --source replay
python3 scripts/shadow_cli.py evaluate-live --source replay
python3 scripts/shadow_cli.py calibration-status
```

Разрешение идёт партиями до исчерпания. Первый прогон разрешил 5 000 из
14 944 и остановился на лимите чтения — расхождение чисел было заметно
только при сверке, поэтому цикл добавлен явно.

---

## 10. Структура для дашборда (PART S)

`src/shadow/metrics.to_frame()` отдаёт плоский датафрейм с полями:
прогноз, обе вероятности, уверенность, ведро уверенности, решение,
источник, патч, лига, исход, и признаки модели. Этого достаточно для
дашборда; API при этом не переделан — построен только domain-слой.
