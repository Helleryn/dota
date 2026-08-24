# Fixtures OpenDota API

**Важно:** это НЕ захваченные вживую ответы (сеть к api.opendota.com
недоступна из среды разработки, см. `docs/environment-constraints.md`).
Это синтетические данные, сконструированные строго по схеме, проверенной
чтением исходного кода `odota/core` (см. `docs/data-sources.md`,
`svc/api/spec.ts`, `svc/api/responses/MatchResponse.ts`, реальный пример
`picks_bans` из `odota/core/json/details_api_pro.json`). Имена команд,
match_id и даты вымышлены.

Используются для unit/integration-тестов `OpenDotaSource`, normalization,
validation — без обращения к сети. Когда live-доступ станет возможен,
эти fixtures стоит заменить/дополнить реальными захваченными ответами
(не меняя тестов, если реальная схема совпадёт).

Файлы:
- `leagues.json` — ответ `GET /leagues` (полная таблица лиг)
- `pro_matches_page1.json` / `pro_matches_page2.json` — страницы `GET /proMatches`
  (для проверки пагинации через `less_than_match_id`)
- `match_detail_*.json` — ответы `GET /matches/{id}` (с `picks_bans`/`players`)
