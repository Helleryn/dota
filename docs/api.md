# PHASE 4 — Prediction API (дизайн)

FastAPI, реализация — Phase 10 (не Phase 4/5). Здесь фиксируется контракт.

## Эндпоинты

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | Liveness/readiness (включая проверку соединения с БД и наличие активной модели) |
| GET | `/matches/upcoming` | Список предстоящих матчей (`start_time > now()`), с пагинацией и фильтром по `league_tier` |
| GET | `/matches/{match_id}` | Детали конкретного матча (команды, турнир, patch, статус — предстоящий/завершён) |
| GET | `/predictions/{match_id}` | Прогноз для матча (см. схему ниже). Параметр `mode=pre_draft\|draft_aware` (по умолчанию `pre_draft`) |
| GET | `/teams/{team_id}` | Карточка команды (текущий рейтинг, текущий состав из `team_roster_periods`) |
| GET | `/teams/{team_id}/form` | История формы команды (recent_winrate/EMA временной ряд — данные для графика) |
| GET | `/models/current` | Метаданные активной модели (версия, `feature_set_version`, метрики на test, дата обучения) |

Опциональный query-параметр `data_cutoff` на `GET /predictions/{match_id}`
(ISO-timestamp, только для уже прошедших матчей) — позволяет **воспроизвести**,
что система предсказала бы, если бы запрос пришёл в указанный момент, а не
сейчас. Это прямой аудиторский инструмент, доказывающий на живых данных
(не только в backtest), что прогноз не использует будущее относительно
`data_cutoff` — тот же принцип, что `docs/backtesting.md`, но доступный через
API, а не только offline.

## Response schema: `GET /predictions/{match_id}`

```json
{
  "match_id": "8123456789",
  "mode": "pre_draft",
  "model_version": "catboost-v3-feature_set_1",
  "predicted_at": "2026-08-24T12:00:03Z",
  "data_cutoff": "2026-08-24T11:59:58Z",
  "team_a": {
    "id": "36",
    "name": "Team Spirit",
    "probability": 0.637
  },
  "team_b": {
    "id": "8599101",
    "name": "Gaimin Gladiators",
    "probability": 0.363
  },
  "confidence": "medium",
  "explanation": [
    { "feature": "team_elo", "impact": 0.08, "direction": "positive", "for_team": "team_a" },
    { "feature": "recent_winrate_10", "impact": 0.05, "direction": "positive", "for_team": "team_a" },
    { "feature": "team_matches_on_current_patch", "impact": -0.03, "direction": "negative", "for_team": "team_a" },
    { "feature": "h2h_winrate_alltime", "impact": 0.02, "direction": "positive", "for_team": "team_a", "note": "низкая достоверность: h2h_matches_count=2" }
  ]
}
```

Отличия от примера в исходном задании: добавлены `mode` (готовность к V4,
`docs/architecture.md` раздел 12) и `data_cutoff` (раздел 19 задания —
доказуемость отсутствия утечки). Поле `for_team` в `explanation` явно
указывает, в чью пользу фактор (в исходном примере задания подразумевалось,
но не было явным полем — при нескольких факторах в обе стороны без этого
поля ответ неоднозначен).

`team_a`/`team_b` в API-ответе — это НЕ `radiant`/`dire` из БД напрямую:
API-слой может переупорядочивать для читаемости (например, `team_a` =
запрошенная в URL команда, если есть такой контекст), но это ответственность
API-слоя, не `match_features`/Dataset Builder (см. `docs/ml-architecture.md`)
— то есть переупорядочивание происходит на выходе, после того как модель уже
предсказала, не влияет на обучение.

## `confidence` — эвристика MVP (требует калибровки в Phase 8-9)

```text
|probability - 0.5| < 0.07   → "low"
|probability - 0.5| < 0.20   → "medium"
иначе                         → "high"
```

Это **placeholder-эвристика**, не откалиброванный статистический показатель
— пороги произвольны и подлежат пересмотру после Phase 9 (calibration curve
может показать, что, например, "high confidence" при `p=0.85` в
действительности сбывается лишь в 70% случаев, если модель не откалибрована
идеально). Явно помечено, чтобы не создать ложное впечатление точности там,
где её ещё предстоит проверить.

## Response schema: `GET /teams/{team_id}/form`

```json
{
  "team_id": "36",
  "series": [
    { "match_id": "...", "start_time": "...", "elo_after": 1542.3, "recent_winrate_10": 0.7 }
  ]
}
```

Прямая проекция из `team_ratings` + `match_features` этой команды,
отсортированная по времени — данные для графика на странице команды
(`docs/architecture.md`, раздел 11).

## Ошибки

Стандартные HTTP-коды: `404` — матч/команда не найдены, `409` — прогноз
запрошен для `mode=draft_aware`, но драфт ещё не завершён (данных нет, это
не "500 внутренняя ошибка", а ожидаемое состояние), `503` — нет активной
модели в `models` (`is_active`) — сервис технически поднят, но не готов
отвечать прогнозами (отражается и в `/health`).

## Версионирование API

Не проектируется отдельно в Phase 4 — на MVP один неверсионированный `/v1`
префикс достаточен (одна команда-потребитель — собственный frontend).
Пересмотреть при появлении внешних потребителей API.
