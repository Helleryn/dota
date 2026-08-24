# PHASE 4 — Data Pipeline

## Стадии

```text
Extract  →  Raw  →  Validate  →  Normalize  →  Deduplicate  →  Enrich  →  Feature Engineering  →  Dataset
```

| Стадия | Вход | Выход | Идемпотентна? |
|---|---|---|---|
| **Extract** | HTTP-запрос к `DataSource`-адаптеру | сырой JSON-ответ | Да (сам запрос не имеет побочных эффектов) |
| **Raw** | сырой JSON-ответ | строка в `raw_responses` | Да — запись всегда `INSERT` (append-only лог), повторный fetch = новая строка, не перезапись |
| **Validate** | сырой JSON | тот же JSON + результат схема-проверки (pass/fail) | Да (чистая функция) |
| **Normalize** | валидный сырой JSON | доменные объекты (`RawMatch`/`RawPickBan`/`RawPlayerMatch`, см. `src/datasources/base.py`) | Да |
| **Deduplicate** | доменные объекты | те же объекты, без дублей по natural key | Да (по построению — идемпотентная операция) |
| **Enrich** | доменные объекты | + `organization_id` (team identity resolution), + `patch_id` (по `start_time`) | Да |
| **Feature Engineering** | обогащённые объекты + история из БД | `match_features` строки | Да, при условии walk-forward обработки в фиксированном хронологическом порядке (см. `docs/data-leakage.md`) |
| **Dataset** | `match_features` + `matches.radiant_win` | ML-датасет (pandas DataFrame / parquet) | Да (чистая проекция из БД, без побочных эффектов) |

Идемпотентность на каждой стадии — не факультативное свойство, а требование:
пайплайн должен уметь безопасно перезапускаться после сбоя на любой стадии
без дублирования данных и без необходимости вручную определять, что уже
обработано.

## Идемпотентность в normalized-слое

Все таблицы normalized-слоя (`matches`, `match_players`, `picks_bans`,
`teams`, `players`) заполняются через `INSERT ... ON CONFLICT (natural_key)
DO UPDATE` (upsert), не через `DO NOTHING` — источник может присылать
уточнённые данные для уже известного `match_id` (например, `radiant_score`
проставляется не сразу). `raw_responses` — единственное исключение
(чистый append-only, см. `docs/database-design.md`).

## Retry и rate-limit handling

- **Транзиентные ошибки** (5xx, timeout, connection reset) → retry с
  экспоненциальным backoff (например, 1s/2s/4s/8s, до 4 попыток), после
  чего — записать в `ingestion_runs.error` и остановить run, не пропускать
  молча.
- **Постоянные ошибки** (4xx кроме 429) → НЕ retry, залогировать
  `http_status` в `raw_responses`, продолжить со следующим элементом (один
  сломанный `match_id` не должен останавливать весь backfill).
- **429 (rate limit exceeded)** → не считается ни транзиентной, ни
  постоянной ошибкой отдельно — respect `Retry-After`, если есть, иначе
  backoff.
- **Проактивный rate limiting** (не дожидаясь 429): token bucket на стороне
  адаптера, настроенный ниже подтверждённого лимита OpenDota (60/мин без
  ключа, 300/мин с ключом — `VERIFIED`, см. `docs/data-sources.md`) —
  например, 55/мин без ключа как safety margin, конфигурируется через
  `OPENDOTA_RATE_LIMIT_PER_MIN`.

## Checkpoints и инкрементальные обновления (data freshness)

`ingestion_runs.checkpoint` (JSONB) хранит состояние прогресса, специфичное
для источника. Для OpenDota — `{"max_match_id_seen": ..., "backfill_complete_before": ...}`.

Различаются два режима:

- **Backfill (первичная загрузка истории)**: пагинация `/proMatches` НАЗАД
  по времени через параметр `less_than_match_id` (подтверждено чтением
  исходников — эндпоинт отдаёт до 100 записей за вызов, `ORDER BY match_id
  DESC`). Продолжается, пока `start_time` матчей в странице не станет
  меньше `DATA_START_DATE` (см. Configuration ниже) или пока API не начнёт
  возвращать пустые страницы.
- **Incremental sync (регулярное обновление)**: запрос `/proMatches` БЕЗ
  параметра (== новейшие 100 матчей), сравнение с `checkpoint.max_match_id_seen`
  — если в странице есть `match_id > checkpoint`, эти записи — новые,
  обрабатываются; если старые матчи в границах страницы ещё не пересекли
  checkpoint, запрашивается следующая страница (`less_than_match_id` = min
  match_id текущей страницы), пока не будет достигнут checkpoint. После
  успешного прохода `checkpoint.max_match_id_seen` обновляется на
  максимальный увиденный `match_id`.

**"Какие данные уже загружены?"** отвечается одним запросом:
`SELECT checkpoint FROM ingestion_runs WHERE source = 'opendota' AND status
= 'succeeded' ORDER BY finished_at DESC LIMIT 1` — не требует пересчёта по
всей таблице `matches`.

Запуск пайплайна (по расписанию) — на MVP простой `cron`/scheduled-скрипт,
не Airflow/Celery — задание прямо предупреждает не добавлять инфраструктуру
раньше необходимости (раздел 25 исходного задания: "не добавляй
Prometheus/Grafana, пока не нужны" — тот же принцип применяется к
оркестраторам).

## Data validation

Три уровня, выполняются в указанном порядке (fail fast):

1. **Schema validation** (стадия Validate) — Pydantic-модели, зеркалящие
   `RawMatch`/`RawPickBan`/`RawPlayerMatch` (`src/datasources/base.py`):
   обязательные поля присутствуют, типы корректны. Провал → запись помечена
   `http_status`/статус в `raw_responses`, не идёт дальше по пайплайну,
   не блокирует остальной batch.
2. **Business rule validation** (стадия Normalize/Enrich): `duration_seconds
   > 0`, `start_time` в разумном диапазоне (не в будущем относительно
   `fetched_at`, не раньше 2011 года — первый год Dota 2), `radiant_win`
   булево (не `NULL` для завершённого матча), `league.tier IN ('premium',
   'professional')` — иначе матч не попадает в основной pro-датасет (но
   может сохраняться в raw-слое для возможного будущего использования).
3. **Дубликаты**: по natural key (`match_id`, составные ключи детей) — upsert
   гарантирует отсутствие дублей на уровне БД (constraint), но пайплайн
   логирует случаи "тот же `match_id` получен повторно с ОТЛИЧАЮЩИМИСЯ
   значениями" отдельно (data-quality warning — источник данных
   "передумал"/исправил историю, стоит знать об этом, не просто молча
   перезаписать).

## OpenDota Adapter (дизайн, не полная реализация в Phase 4)

Реализует интерфейс `DataSource` (`src/datasources/base.py`, введён в
Phase 2). Конкретика:

| Метод интерфейса | Endpoint(ы) OpenDota | Особенности |
|---|---|---|
| `fetch_matches(since, until)` | `/explorer` с агрегирующим SQL (для получения списка `match_id` + базовых полей за диапазон — один-два запроса вместо сотен постраничных) **или** `/proMatches` с пагинацией (для incremental sync, см. выше) | Выбор между `/explorer` и `/proMatches` зависит от размера диапазона: большой исторический backfill эффективнее через `/explorer` (при условии, что размер ответа не упрётся в лимит — `REQUIRES LIVE VERIFICATION`), инкрементальный sync — через `/proMatches` (естественно подходит для "top N new records") |
| `fetch_picks_bans(match_id)` | `/matches/{match_id}` (поле `picks_bans` в ответе) | Один вызов на матч — самая "дорогая" по квоте часть пайплайна, вызывается только для матчей, ещё не имеющих детального ответа в `raw_responses` |
| `fetch_player_matches(match_id)` | `/matches/{match_id}` (поле `players`) | Тот же вызов, что и picks_bans — оба извлекаются из одного детального ответа за один HTTP-запрос, не два отдельных |
| Справочники (heroes, patches) | **НЕ через OpenDota API** | Синхронизируются из пакета `odota/dotaconstants` (см. `docs/data-sources.md`) — экономит квоту API на данных, которые и так статичны и версионируются отдельно |
| Лиги (`leagues`, для `tier`) | `/leagues` | Один вызов возвращает всю таблицу лиг целиком (подтверждено чтением исходников — без пагинации) — синхронизируется целиком при каждом запуске, дёшево |

**Нормализация в адаптере:** ответ `/matches/{match_id}` мэппится в
`RawMatch` + список `RawPickBan` + список `RawPlayerMatch` (унифицированные
структуры интерфейса `DataSource`) — вся OpenDota-специфичная форма ответа
(имена полей, `player_slot` кодировка radiant/dire через `< 128`) остаётся
внутри адаптера и не просачивается в normalized-слой БД или выше.

**Ошибки, специфичные для OpenDota:** матч без `picks_bans` (не Captain's
Mode или не записан) — не ошибка, а валидное отсутствие данных (`NULL` в
БД, не retry).

## Enrichment: patch и organization

- **`patch_id`**: вычисляется детерминированно из `start_time` матча через
  таблицу `patches` (`released_at <= start_time`, взять максимальный) — не
  требует внешнего запроса, чистая функция над уже загрученным справочником.
- **`organization_id`**: применяется правило по умолчанию (1:1 с `team_id`)
  плюс проверка `organization_aliases` (см. `docs/team-identity.md`) — тоже
  чистая функция от уже нормализованных данных, не требует сетевых запросов.

## Логирование

Структурированные логи (JSON lines) на каждой стадии, обязательные поля:
`stage`, `source`, `match_id` (если применимо), `ingestion_run_id`,
`timestamp`, `level`, `message`. Соответствует принципу Observability
(раздел 25 исходного задания — structured logging достаточно для MVP, без
Prometheus/Grafana).

## Configuration

Границы данных и train/val/test — **параметры, не константы в коде**
(прямое требование этой фазы, п.1):

```text
DATA_START_DATE       — с какой даты грузить историю (backfill lower bound)
DATA_END_DATE         — по какую дату грузить (обычно "сейчас", но
                         конфигурируемо для воспроизводимых экспериментов)
MIN_MATCH_DATE        — нижняя граница включения матча в ML-датасет
                         (может отличаться от DATA_START_DATE — например,
                         данные грузятся с 2015, но в датасет для конкретного
                         эксперимента берутся только с 2018)
TRAIN_START / VALIDATION_START / TEST_START
                       — границы time-based split (ADR-005), не хардкодятся
```

Детали реализации — `src/config.py` (см. итог Phase 4, раздел Configuration
Management). Когда `scripts/verify_data_source.py` вернёт реальные цифры
исторического покрытия, значения этих параметров обновляются в `.env`/конфиге
без изменения кода пайплайна.
