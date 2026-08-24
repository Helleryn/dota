# PHASE 4 — Database Design

PostgreSQL. Схема спроектирована вокруг двух принципов, зафиксированных в
Phase 2-3 и ADR-002/ADR-003: (1) слоистость Raw → Normalized → Feature →
Dataset (физически — разные группы таблиц в одной БД, не разные СУБД), и
(2) **никакая изменяющаяся во времени сущность не хранится как единственный
мутируемый снэпшот** — только там, где это оправдано (см. таблицу ниже),
используется append-only лог или явный temporal-интервал `valid_from/valid_to`.

Список таблиц из исходного задания использован как отправная точка, но не
скопирован автоматически — ниже для каждой таблицы явно обосновано, зачем
она нужна именно в этом виде.

## ER-диаграмма

```mermaid
erDiagram
    ORGANIZATIONS ||--o{ TEAMS : "resolves to"
    TEAMS ||--o{ TEAM_ROSTER_PERIODS : "has roster periods"
    PLAYERS ||--o{ TEAM_ROSTER_PERIODS : "member during period"
    TEAMS ||--o{ MATCHES : "radiant_team_id"
    TEAMS ||--o{ MATCHES : "dire_team_id"
    LEAGUES ||--o{ MATCHES : "hosts"
    PATCHES ||--o{ MATCHES : "played on"
    MATCHES ||--o{ MATCH_PLAYERS : "has"
    PLAYERS ||--o{ MATCH_PLAYERS : "plays in"
    HEROES ||--o{ MATCH_PLAYERS : "played as"
    MATCHES ||--o{ PICKS_BANS : "has draft"
    HEROES ||--o{ PICKS_BANS : "picked/banned"
    TEAMS ||--o{ TEAM_RATINGS : "rating event log"
    MATCHES ||--o{ TEAM_RATINGS : "triggers update"
    FEATURE_SETS ||--o{ MATCH_FEATURES : "version"
    MATCHES ||--o{ MATCH_FEATURES : "features computed for"
    TEAMS ||--o{ MATCH_FEATURES : "features about"
    FEATURE_SETS ||--o{ MODELS : "trained on"
    MODELS ||--o{ PREDICTIONS : "produced by"
    MATCHES ||--o{ PREDICTIONS : "prediction for"

    ORGANIZATIONS {
        bigint organization_id PK
        text canonical_name
        timestamptz created_at
    }
    TEAMS {
        bigint team_id PK "natural key = OpenDota team_id"
        bigint organization_id FK "nullable, до resolution"
        text name
        text tag
        timestamptz first_seen_at
        timestamptz last_seen_at
    }
    PLAYERS {
        bigint account_id PK "natural key = Steam account_id"
        text name "nullable, если не notable_player"
        timestamptz first_seen_at
        timestamptz last_seen_at
    }
    TEAM_ROSTER_PERIODS {
        bigint id PK
        bigint team_id FK
        bigint account_id FK
        timestamptz valid_from
        timestamptz valid_to "nullable = по настоящий момент"
        text role "nullable"
        text source "reconstructed_from_matches | liquipedia"
        real confidence
    }
    LEAGUES {
        bigint league_id PK
        text name
        text tier "premium | professional | ..."
    }
    PATCHES {
        int patch_id PK
        text name UK "напр. 7.41"
        timestamptz released_at
    }
    HEROES {
        int hero_id PK
        text name
        text localized_name
        text primary_attr
        text attack_type
    }
    MATCHES {
        bigint match_id PK
        timestamptz start_time
        int duration_seconds
        bigint radiant_team_id FK
        bigint dire_team_id FK
        boolean radiant_win
        bigint league_id FK
        int patch_id FK
        bigint series_id
        smallint series_type "INFERRED кодировка, см. ADR/data-feasibility"
        text source
        timestamptz ingested_at
    }
    MATCH_PLAYERS {
        bigint match_id PK_FK
        smallint player_slot PK
        bigint account_id FK "nullable"
        boolean is_radiant
        int hero_id FK
        int kills
        int deaths
        int assists
        int gold_per_min
        int xp_per_min
    }
    PICKS_BANS {
        bigint match_id PK_FK
        smallint ord PK
        boolean is_pick
        int hero_id FK
        smallint team "0=radiant, 1=dire"
    }
    TEAM_RATINGS {
        bigint team_id PK_FK
        bigint match_id PK_FK
        real rating_before
        real rating_after
        real k_factor
        text rating_engine_version
        timestamptz computed_at
    }
    FEATURE_SETS {
        text feature_set_version PK
        text description
        text code_git_sha
        jsonb config
        timestamptz created_at
    }
    MATCH_FEATURES {
        bigint match_id PK_FK
        bigint team_id PK_FK
        text feature_set_version PK_FK
        timestamptz as_of_timestamp
        timestamptz calculated_at
        jsonb features
    }
    MODELS {
        bigint model_id PK
        text name
        text version
        text algorithm
        text feature_set_version FK
        timestamptz train_start
        timestamptz train_end
        timestamptz val_start
        timestamptz val_end
        timestamptz test_start
        timestamptz test_end
        jsonb config
        jsonb metrics
        text artifact_path
        boolean is_active
        timestamptz trained_at
    }
    PREDICTIONS {
        bigint prediction_id PK
        bigint match_id FK
        bigint model_id FK
        timestamptz predicted_at
        timestamptz data_cutoff
        bigint team_a_id FK
        bigint team_b_id FK
        real team_a_probability
        jsonb explanation
        timestamptz created_at
    }
    RAW_RESPONSES {
        bigint id PK
        text source
        text endpoint
        jsonb request_params
        timestamptz fetched_at
        int http_status
        jsonb response_body
    }
    INGESTION_RUNS {
        bigint id PK
        text source
        timestamptz started_at
        timestamptz finished_at
        text status
        int records_fetched
        jsonb checkpoint
        text error
    }
```

`RAW_RESPONSES` и `INGESTION_RUNS` не связаны FK с остальной схемой намеренно
(см. таблицу ниже) — они не показаны на диаграмме отношений выше, чтобы не
загромождать её, но входят в общую схему.

## Таблица за таблицей

### `organizations`

- **Зачем:** единственная сущность, которую мы считаем "непрерывной
  историей команды" через переименования/ребрендинги — то, что человек
  интуитивно называет "Team Liquid" независимо от того, как менялся
  `team_id`/имя в источниках. Подробная стратегия резолюции —
  `docs/team-identity.md`.
- **Grain:** одна строка = одна конкурентная организация за всю историю.
- **PK:** `organization_id` (surrogate, наш собственный).
- **FK:** нет (корень идентичности).
- **Индексы:** нет специальных сверх PK на MVP.
- **Timestamps:** `created_at`.
- **Historical validity:** сама таблица не temporal — она *агрегирует*
  историю через `teams.organization_id`, а не описывает интервалы сама.

### `teams`

- **Зачем:** сырая идентичность команды **как она приходит из источника**
  (natural key = OpenDota `team_id`) — это НЕ то же самое, что
  `organization_id` (см. `team-identity.md`: одна organization может
  соответствовать нескольким team_id при смене регистрации, и наоборот
  редко, но бывает, слияние).
- **Grain:** одна строка на `team_id` источника.
- **PK:** `team_id` (natural key, не surrogate — стабильный ID из OpenDota).
- **FK:** `organization_id → organizations` (**nullable** — пока не пройдена
  identity resolution, легитимно `NULL`, не блокирует ingestion).
- **Индексы:** `idx_teams_organization_id`.
- **Unique constraints:** нет доп. (team_id уже уникален как PK).
- **Timestamps:** `first_seen_at`, `last_seen_at` (обновляются при каждом
  новом матче с участием команды — дёшево вычисляется в ETL).
- **Historical validity:** имя/tag команды может меняться со временем;
  в MVP хранится только последнее известное значение (не temporal) — это
  осознанное упрощение: имя команды используется только для отображения в
  UI, не как признак модели, поэтому temporal-точность здесь не критична
  (в отличие от ростера, где это критично).

### `players`

- **Зачем:** канонический игрок. **Важное отличие от команд:** у игрока
  `account_id` (Steam ID) — по-настоящему стабильный, глобально уникальный
  natural key. Проблема identity resolution, острая для команд, здесь
  практически отсутствует.
- **Grain:** одна строка на `account_id`.
- **PK:** `account_id` (natural key).
- **FK:** нет.
- **Индексы:** нет доп.
- **Timestamps:** `first_seen_at`, `last_seen_at`.
- **Historical validity:** не требуется — сам игрок не temporal-сущность
  (temporal — его *принадлежность к команде*, см. `team_roster_periods`).

### `team_roster_periods`

- **Зачем:** ядро temporal-моделирования состава — прямой ответ на вопрос
  задания "кто входил в Team X на 15 марта 2025". Данные наполняются двумя
  способами (поле `source`): реконструкцией из фактических составов матчей
  (`reconstructed_from_matches`, наш собственный ETL-процесс) и/или
  подтверждением из Liquipedia (`liquipedia`) — см. `docs/team-identity.md`.
- **Grain:** один непрерывный период членства одного игрока в одной команде.
- **PK:** `id` (surrogate bigserial) — составной естественный ключ
  `(team_id, account_id, valid_from)` тоже был бы валиден, но surrogate
  проще для FK из будущих таблиц, если понадобятся.
- **FK:** `team_id → teams`, `account_id → players`.
- **Индексы:** `idx_roster_team_account_valid (team_id, account_id, valid_from)`,
  `idx_roster_valid_range (team_id, valid_from, valid_to)` — критично для
  запроса "состав команды на дату X" (`WHERE team_id = ? AND valid_from <= X
  AND (valid_to IS NULL OR valid_to > X)`).
- **Unique constraints (data-quality invariant, не всегда БД-constraint):**
  периоды одного игрока в одной команде не должны пересекаться. На MVP это
  проверяется в ETL-валидации (`docs/data-pipeline.md`); упомянутый как
  путь эволюции — Postgres `EXCLUDE USING gist` с `tsrange(valid_from,
  valid_to)` и расширением `btree_gist` даёт БД-уровневую гарантию, не
  включено в MVP-миграцию, чтобы не требовать дополнительного расширения
  Postgres на старте.
- **Historical validity:** это САМА temporal-таблица (`valid_from`/`valid_to`,
  `valid_to = NULL` означает "действует по настоящий момент").

### `leagues`

- **Зачем:** прямое отражение `VERIFIED`-фильтра качества OpenDota
  (`tier IN (premium, professional)`) — наш основной quality-gate при отборе
  матчей в датасет.
- **Grain:** одна строка на `league_id` источника.
- **PK:** `league_id` (natural key).
- **FK:** нет.
- **Индексы:** `idx_leagues_tier`.
- **Historical validity:** не требуется, `tier` присваивается один раз при
  создании турнира.

### `patches`

- **Зачем:** справочник версий игры с точными датами начала действия
  (`VERIFIED`, источник — `odota/dotaconstants`).
- **Grain:** одна строка на патч.
- **PK:** `patch_id` (surrogate int, соответствует индексу в dotaconstants).
- **Unique constraints:** `name` уникально.
- **Historical validity:** сама таблица append-only (новый патч = новая
  строка), не требует temporal-интервалов сверх `released_at`.

### `heroes`

- **Зачем:** справочник героев (id → имя/атрибуты) из `dotaconstants`.
- **Grain:** одна строка на героя.
- **PK:** `hero_id` (natural key).
- **Historical validity:** не версионируется по патчам в MVP (базовые
  атрибуты героя технически могут меняться по патчам, но это не нужно для
  MVP-признаков — используется только для draft-aware Feature Set 4,
  отложено).

### `matches`

- **Зачем:** ядро датасета — факт матча и результат.
- **Grain:** одна строка на `match_id`.
- **PK:** `match_id` (natural key источника).
- **FK:** `radiant_team_id → teams`, `dire_team_id → teams` (оба nullable —
  бывают матчи без привязки команды, хотя для tier-фильтрованных про-матчей
  редкость), `league_id → leagues`, `patch_id → patches`.
- **Индексы:** `idx_matches_start_time` (**критичен** — почти все запросы
  temporal-признаков и backtesting идут по диапазону времени),
  `idx_matches_league_id`, `idx_matches_radiant_team_id`,
  `idx_matches_dire_team_id`.
- **Unique constraints:** нет доп. (match_id уже PK).
- **Timestamps:** `ingested_at` (когда наш pipeline впервые записал эту
  строку — отдельно от `start_time`, игрового времени события; см.
  `docs/data-pipeline.md` про разницу "момент события" vs "момент появления
  в системе").
- **Historical validity:** сам факт матча неизменяем после того, как сыгран
  — не temporal-таблица.
- **Дизайн-решение:** `radiant_team_id`/`dire_team_id` хранятся
  денормализованно прямо в `matches` (как в источнике), а не через
  отдельную join-таблицу `match_teams` — это естественная форма данных
  OpenDota, и почти все запросы всё равно идут "по матчу". Симметричный
  team-centric доступ (для `RatingEngine`/`match_features`) реализуется VIEW:

```sql
CREATE VIEW match_teams AS
  SELECT match_id, radiant_team_id AS team_id, TRUE AS is_radiant, radiant_win AS won FROM matches
  UNION ALL
  SELECT match_id, dire_team_id AS team_id, FALSE AS is_radiant, NOT radiant_win AS won FROM matches;
```

### `match_players`

- **Зачем:** посекундная/поматчевая статистика игрока — источник для
  player-level признаков (Feature Set 2+) и для реконструкции ростера.
- **Grain:** один игрок в одном матче (`player_slot` — 0-9, соответствует
  позиции в источнике).
- **PK:** составной `(match_id, player_slot)`.
- **FK:** `match_id → matches`, `account_id → players` (**nullable** —
  анонимные аккаунты бывают), `hero_id → heroes`.
- **Индексы:** `idx_match_players_account_id` (критичен для реконструкции
  ростера и player-level агрегатов — "все матчи этого игрока"),
  `idx_match_players_hero_id`.
- **Historical validity:** факт, не temporal.

### `picks_bans`

- **Зачем:** структура драфта, ядро draft-aware режима (Feature Set 4).
- **Grain:** одно действие драфта (пик или бан) в одном матче.
- **PK:** составной `(match_id, ord)` (`ord` = порядок действия, уникален
  внутри матча по построению источника).
- **FK:** `match_id → matches`, `hero_id → heroes`.
- **Индексы:** `idx_picks_bans_hero_id` (для агрегатов hero winrate).
- **Historical validity:** факт, не temporal. **Архитектурное правило:**
  эта таблица физически не участвует в построении pre-match датасета (см.
  `docs/features.md`, H.1 vs H.2) — Dataset Builder для pre-match режима
  просто не делает join к ней.

### `team_ratings`

- **Зачем:** append-only лог Elo-обновлений от `RatingEngine` (ADR-003) —
  сознательно НЕ мутируемый "текущий рейтинг", а событийный журнал, что и
  даёт point-in-time доступ без дополнительных temporal-интервалов (сама
  последовательность матчей уже задаёт порядок).
- **Grain:** одно обновление рейтинга одной команды в результате одного
  матча, в котором она участвовала.
- **PK:** составной `(team_id, match_id)`.
- **FK:** `team_id → teams`, `match_id → matches`.
- **Индексы:** `idx_team_ratings_team_match (team_id, match_id)` — для
  запроса "последний rating_after команды до даты X" эффективнее всего join
  с `matches.start_time` и `ORDER BY start_time DESC LIMIT 1`, поэтому этот
  композитный индекс должен покрывать связку с `matches` через `match_id`.
- **Historical validity:** сама природа append-only лога — это и есть
  temporal-механизм (в отличие от `valid_from/valid_to` у ростера, здесь
  дискретные точки-события подходят лучше, т.к. рейтинг обновляется именно
  дискретными событиями-матчами, а не непрерывными периодами).

### `feature_sets`

- **Зачем:** версионирование конфигурации признаков — прямое требование
  задания (Phase 3, Feature Set 0-4) и общего принципа reproducibility
  (раздел 17 исходного задания).
- **Grain:** одна строка на версию набора признаков.
- **PK:** `feature_set_version` (текстовый natural key, напр. `"v0_baseline"`,
  `"v1_patch_context"` — человекочитаемый, в отличие от суррогатного ID,
  что упрощает дебаг и ссылки в коде/логах).
- **Timestamps:** `created_at`.
- **Historical validity:** append-only, версии не изменяются задним числом
  (новая логика расчёта = новая версия, не апдейт старой).

### `match_features`

- **Зачем:** прямая реализация требования раздела 13 задания
  ("PostgreSQL + versioned feature tables" достаточно для MVP), с
  обязательным `as_of_timestamp`.
- **Grain:** признаки ОДНОЙ команды в ОДНОМ матче для ОДНОЙ версии feature
  set (не пара "team_a vs team_b" одной строкой — см. дизайн-решение ниже).
- **PK:** составной `(match_id, team_id, feature_set_version)`.
- **FK:** `match_id → matches`, `team_id → teams`,
  `feature_set_version → feature_sets`.
- **Индексы:** `idx_match_features_asof (as_of_timestamp)` — вспомогательный,
  для аудита "какие признаки были посчитаны на дату X".
- **Схема признаков:** `features JSONB`, а не широкая типизированная таблица
  "одна колонка на признак". **Обоснование:** задание явно разрешает не
  строить отдельную промышленную Feature Store и обойтись
  "PostgreSQL + versioned feature tables"; при этом набор признаков растёт
  поэтапно (Feature Set 0→4, `docs/features.md`) — широкая таблица требовала
  бы миграции при каждом добавлении признака, что создаёт трение именно на
  той фазе (Phase 7), когда мы должны быстро экспериментировать. JSONB с GIN
  индексом (`CREATE INDEX ... USING GIN (features)`) даёт достаточную
  запросную гибкость для MVP-объёма без цены постоянных ALTER TABLE. Явно
  зафиксированный компромисс: типобезопасность на уровне БД теряется
  (компенсируется Pydantic-валидацией на границе приложения, см.
  `docs/architecture.md`).
- **Timestamps:** `as_of_timestamp` (момент, НА КОТОРЫЙ признаки
  актуальны — обычно `start_time` матча минус эпсилон), `calculated_at`
  (когда наш pipeline физически выполнил расчёт — может быть сильно позже
  `as_of_timestamp` при историческом backfill).
- **Historical validity:** append-only по построению (новая версия feature
  set — новые строки, не перезапись).
- **Дизайн-решение "по команде, не по паре":** хранение per-team (не
  `team_a_elo`/`team_b_elo` в одной строке матча) отделяет вопрос "какая
  команда radiant/dire" от вопроса "какая команда team_a/team_b для модели"
  — второе решается на уровне Dataset Builder (ML pipeline), не схемы БД.
  **Критически важное правило, зафиксированное здесь явно:** маппинг
  "team_a"/"team_b" НИКОГДА не должен зависеть от исхода матча (например,
  "team_a = победитель") — это была бы прямая утечка целевой переменной в
  признаки через сам порядок столбцов. Используется нейтральный
  детерминированный маппинг: `radiant`/`dire` (реальная, известная до матча
  игровая асимметрия, не связанная с исходом).

### `models`

- **Зачем:** минимальный Model Registry (ADR-004: не MLflow, но
  архитектурно совместимо по смыслу полей).
- **Grain:** одна обученная версия модели.
- **PK:** `model_id` (surrogate).
- **FK:** `feature_set_version → feature_sets`.
- **Индексы:** `idx_models_is_active` (частичный: `WHERE is_active`) — для
  быстрого поиска "текущей продовой модели".
- **Unique constraints:** `(name, version)` уникальны вместе.
- **Timestamps:** `trained_at`; плюс явные границы train/val/test
  (`train_start`...`test_end`) — обязательны для аудита "на каких данных
  обучена и провалидирована эта модель" (прямое требование reproducibility).
- **Historical validity:** append-only, модели не перезаписываются —
  переобучение создаёт новую строку.

### `predictions`

- **Зачем:** лог прогнозов с обязательными `predicted_at`/`data_cutoff`
  (раздел 19 задания) — доказуемость, что прогноз не использовал будущее.
- **Grain:** один прогноз одной модели для одного матча (возможно несколько
  строк на матч — переpredict при обновлении данных, или два режима
  pre-draft/draft-aware, различаемые через `explanation`/отдельное поле
  `mode`, см. `docs/api.md`).
- **PK:** `prediction_id` (surrogate).
- **FK:** `match_id → matches`, `model_id → models`, `team_a_id → teams`,
  `team_b_id → teams`.
- **Индексы:** `idx_predictions_match_id` (последний прогноз на матч — частый
  API-запрос), `idx_predictions_predicted_at`.
- **Timestamps:** `predicted_at` (когда сделан запрос на прогноз),
  `data_cutoff` (граница данных, использованных для расчёта признаков —
  может быть раньше `predicted_at` на время вычисления), `created_at`.
- **Historical validity:** append-only — прогнозы не перезаписываются, даже
  повторный прогноз того же матча создаёт новую строку (это и есть аудиторский
  след, требуемый заданием).

### `raw_responses` (Raw Data layer, ADR-002)

- **Зачем:** точная копия сырого ответа источника + метаданные запроса —
  основа воспроизводимости и аудита (Phase 2.3).
- **Grain:** один HTTP-ответ одного запроса.
- **PK:** `id` (surrogate bigserial).
- **FK:** намеренно нет (raw-слой не должен зависеть от того, успешно ли
  прошла нормализация — иначе неудачный парсинг мог бы заблокировать саму
  запись сырых данных).
- **Индексы:** `idx_raw_responses_source_endpoint_fetched
  (source, endpoint, fetched_at)`.
- **Timestamps:** `fetched_at`.
- **Historical validity:** append-only, каждый повторный fetch — новая
  строка (позволяет увидеть, как менялся ответ источника со временем, если
  вообще менялся).

### `ingestion_runs`

- **Зачем:** идемпотентность и инкрементальные обновления (раздел 9
  задания: "какие данные уже загружены, не скачивать повторно").
- **Grain:** один запуск ingestion-процесса для одного источника.
- **PK:** `id` (surrogate).
- **Индексы:** `idx_ingestion_runs_source_status (source, status,
  started_at)` — для запроса "последний успешный run по источнику X".
- **`checkpoint` (JSONB):** источник-специфичный указатель прогресса
  (например, для OpenDota — `{"last_less_than_match_id": ...}`, т.к.
  пагинация `/proMatches` идёт через этот параметр — см. `docs/data-pipeline.md`).
- **Historical validity:** append-only, каждый запуск — новая строка (не
  перезапись предыдущего run).

## Что сознательно НЕ включено в MVP-схему

- **`tournament_stages`/`bracket`** (group/playoff/upper/lower) — зависит от
  интеграции Liquipedia (Feature Set 3, не MVP); таблица появится в
  соответствующей миграции, не раньше.
- **Отдельная `hero_stats`/`matchups` таблица** — для MVP не нужна (нет
  draft-aware); при необходимости Feature Set 4 будет либо материализовывать
  агрегаты OpenDota `/heroStats` в отдельную таблицу, либо считать на лету —
  решение отложено до Phase 4.2/Phase 7.
- **Партиционирование `matches`/`raw_responses` по дате** — не нужно на
  MVP-объёме (десятки тысяч строк, не миллионы); путь эволюции очевиден
  (native Postgres partitioning by range on `start_time`), не проектируется
  заранее без данных о реальном объёме.
