# Вендоренные справочные данные (`dotaconstants`)

`patches.json`, `heroes.json` — точные копии соответствующих файлов из
открытого пакета [`odota/dotaconstants`](https://github.com/odota/dotaconstants)
(npm-пакет; здесь используется как обычные статические JSON-файлы, т.к.
проект на Python — устанавливать npm-зависимость ради двух JSON-файлов
избыточно).

Источники (загружены 2026-08-24):
- https://raw.githubusercontent.com/odota/dotaconstants/master/build/patch.json
- https://raw.githubusercontent.com/odota/dotaconstants/master/build/heroes.json

`heroes.json` обрезан до полей, реально нужных схеме `heroes`
(`docs/database-design.md`): `id`, `name`, `localized_name`, `primary_attr`,
`attack_type`, `roles` — исходный файл dotaconstants содержит на порядок
больше игровых атрибутов (base_health, attack_range и т.д.), не используемых
в MVP-признаках.

Обновление: перезапустить `curl`-запросы к URL выше, для `heroes.json` —
повторить обрезку полей. Патчи выходят раз в несколько месяцев, обновление
не требует изменений кода (`src/normalization/enrich.py` читает файл целиком
при старте).
