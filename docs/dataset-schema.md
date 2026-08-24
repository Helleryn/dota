# Dataset Schema — Feature Set 0 (`v0_baseline`)

Первый реальный (не спроектированный, а построенный и проверенный на
загруженных данных — Phase 5) ML-датасет. Одна строка = один
профессиональный матч. Реализация — `src/datasets/feature_set_0.py` +
`src/datasets/builder.py`. Каждый признак вычислен walk-forward через
`RatingEngine` (ADR-003) — pre-match значение, состояние обновляется
СТРОГО после того, как признаки для текущего матча уже зафиксированы
(см. `docs/data-leakage.md`).

Соглашение об именовании: `radiant`/`dire`, не `team_a`/`team_b` —
осознанное решение (`docs/database-design.md`, раздел `match_features`):
маппинг стороны не должен зависеть от исхода матча, а radiant/dire — это
реальная, известная до начала матча игровая роль.

## Поля

| Поле | Тип | Значение | Формула | Available at | Leakage risk | Missing-value стратегия |
|---|---|---|---|---|---|---|
| `match_id` | int | Идентификатор матча | — | — | — | Не бывает |
| `as_of_timestamp` | datetime | Момент, на который признаки актуальны | `= start_time` матча | — | — | Не бывает |
| `radiant_team_id` / `dire_team_id` | int | Команды | Из `matches` | До матча (расписание известно заранее) | Нет | Строки без обеих команд исключены на этапе загрузки (`_load_pro_matches`) |
| `radiant_elo` / `dire_elo` | float | Walk-forward Elo рейтинг ДО этого матча | `RatingEngine.process_match(...).{radiant,dire}_pre`, K=32, база 1000 | Мгновенно перед `as_of_timestamp` (по построению) | Низкий — `RatingEngine` проверен `scripts/elo_prototype.py` и `src/ratings/engine.py::_self_test` на независимый пересчёт | Отсутствует — новая команда получает базовое значение 1000 (не NaN, т.к. это осмысленный нейтральный старт для Elo по определению алгоритма) |
| `elo_difference` | float | `radiant_elo - dire_elo` | Разность | Мгновенно перед матчем | Низкий (производная от уже безопасных полей) | Не бывает (0.0 при равных стартовых рейтингах) |
| `radiant_recent_winrate` / `dire_recent_winrate` | float или `None` | Win rate за последние ≤5 матчей команды до этого | `sum(wins) / min(5, сыграно_матчей)` | Мгновенно перед матчем | Низкий — `_TeamFormTracker.record()` вызывается строго после того, как признаки текущего матча уже прочитаны | `None`, если у команды 0 предыдущих матчей в выборке — НЕ подменяется на 0.5 (задание, раздел 9: "если неизвестно — NULL, не выдуманное значение") |
| `recent_winrate_difference` | float или `None` | `radiant_recent_winrate - dire_recent_winrate` | Разность | Мгновенно перед матчем | Низкий | `None`, если хотя бы одно из двух полей `None` |
| `radiant_days_since_last_match` / `dire_days_since_last_match` | float или `None` | Дней с прошлого матча команды | `(as_of_timestamp - last_match_time) / 86400` | Мгновенно перед матчем | Низкий | `None` для первого матча команды в выборке |
| `radiant_matches_played_before` / `dire_matches_played_before` | int | Сколько матчей команда сыграла до этого в выборке | Счётчик `_TeamFormTracker` | Мгновенно перед матчем | Низкий | `0` — валидное значение, не пропуск |
| `radiant_win` | bool | **Целевая переменная** | Из `matches.radiant_win` | Известно только ПОСЛЕ матча | Специально не входит ни в один признак — используется только как target | Не бывает (обязательное поле) |

## Что сознательно НЕ включено в Feature Set 0 (Phase 5, раздел 25)

- **Patch winrate** — на MVP-объёме (первые 12-1000 матчей) число матчей одной
  команды на одном патче слишком мало для устойчивой статистики; отложено до
  накопления большего объёма (Feature Set 1, `docs/features.md`).
- **Head-to-head** — та же причина, задание прямо предупреждает не
  переоценивать этот признак при малом числе очных встреч.
- **Roster/player-level** — требует либо реконструкции ростера (следующая
  фаза), либо Liquipedia; не в MVP по архитектурному решению Phase 2
  (`docs/data-feasibility.md`, MVP dataset).

## Персистентность

Каждая строка сохраняется как ДВЕ записи в `match_features`
(`docs/database-design.md`) — одна на команду (`team_id`), не на пару.
`feature_set_version = "v0_baseline"`, зарегистрирован в `feature_sets` с
конфигурацией `{"rating_engine": "elo_k32_base1000", "recent_form_window": 5}`.
ML-датасет (одна строка на матч, `radiant`/`dire` рядом) собирается поверх
этого через `DatasetBuilder`/`_rows_to_dataframe`, join не требуется —
`build_feature_set_0` уже отдаёт готовые пары.

## Проверено на реальных загруженных данных

12 из 14 загруженных в Phase 5 матчей (2 amateur-tier отфильтрованы
`PRO_LEAGUE_TIERS`). Первый матч каждой команды в выборке — `elo=1000.0`,
`recent_winrate=None`, `days_since_last_match=None` — холодный старт
подтверждён не только unit-тестом на синтетике, но и на реально
прошедшем через весь pipeline (`OpenDota → Raw → Normalized → PostgreSQL
→ RatingEngine → Feature Set 0`) наборе данных.
