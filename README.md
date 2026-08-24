# Dota 2 Match Prediction — исследовательско-инженерный проект

Сервис прогнозирования вероятности победы команды в профессиональном матче Dota 2,
на основе исторических данных и объяснимого ML.

## Статус проекта

**PHASE 0-5 завершены.** Реальный (не только спроектированный) pipeline
`OpenDota → Raw → Normalized → PostgreSQL → RatingEngine → Feature Set 0 →
Dataset` работает и проверен сквозным прогоном — см.
[`reports/phase5-summary.md`](reports/phase5-summary.md). Сеть к
`api.opendota.com` недоступна из текущей среды разработки
([`docs/environment-constraints.md`](docs/environment-constraints.md)), поэтому
Phase 5 выполнена и проверена в offline fixture mode — код идентичен тому,
что выполнится против живого API, но фактические цифры (объём, историческая
глубина) не подтверждены. Итоговый отчёт по Phase 2+3:
[`docs/research-summary.md`](docs/research-summary.md).

## Быстрый старт (воспроизвести весь pipeline)

```bash
pip install -r requirements.txt
cp .env.example .env   # заполнить DATABASE_URL реальными данными локальной PostgreSQL

alembic upgrade head                                     # схема БД (docs/database-design.md)
python3 -m src.ingestion.run_opendota --source fixtures --since 2024-01-01 --until 2025-01-01 --limit 100
python3 scripts/build_dataset.py                          # Feature Set 0 (docs/dataset-schema.md)
python3 scripts/sanity_check_dataset.py                   # shape/missing/class balance/sanity-fit
python3 -m pytest tests/                                  # 45 тестов, включая leakage-аудит
```

Замените `--source fixtures` на `--source live` в среде с доступом к
`api.opendota.com` — остальные шаги не меняются.

## Документация

### Исследование (Phase 1-3)

- [`docs/data-sources.md`](docs/data-sources.md) — сравнительная таблица источников данных Dota 2 esports.
- [`docs/environment-constraints.md`](docs/environment-constraints.md) — ограничение текущей среды
  выполнения (сетевой egress) и его влияние на архитектуру сбора данных.
- [`docs/data-feasibility.md`](docs/data-feasibility.md) — какой dataset реально можем построить,
  обоснование MVP, слоистая схема Raw → Normalized → Feature → Dataset.
- [`docs/features.md`](docs/features.md) — полный feature catalog (категории A-H, Feature Set 0-4).
- [`docs/data-leakage.md`](docs/data-leakage.md) — аудит утечек данных по каждой группе признаков.
- [`docs/research-summary.md`](docs/research-summary.md) — итоговый отчёт Phase 2+3.

### Архитектура (Phase 4)

- [`docs/architecture.md`](docs/architecture.md) — системная архитектура, диаграммы, project structure,
  технологический стек, эволюция V1→V5.
- [`docs/database-design.md`](docs/database-design.md) — PostgreSQL-схема с ER-диаграммой (Mermaid),
  обоснование каждой таблицы.
- [`docs/team-identity.md`](docs/team-identity.md) — стратегия identity resolution
  (organization/team/roster/match participant).
- [`docs/data-pipeline.md`](docs/data-pipeline.md) — extract/validate/normalize/enrich, идемпотентность,
  retry, rate limiting, checkpoints, OpenDota adapter.
- [`docs/ml-architecture.md`](docs/ml-architecture.md) — dataset builder, модели, калибровка, model registry.
- [`docs/backtesting.md`](docs/backtesting.md) — walk-forward backtesting engine.
- [`docs/api.md`](docs/api.md) — дизайн REST API и схема ответа прогноза.
- [`docs/costs.md`](docs/costs.md) — оценка стоимости MVP/production.
- [`docs/decisions/`](docs/decisions/) — Architecture Decision Records (ADR-001..005).

### Реализация (Phase 5)

- [`reports/phase5-summary.md`](reports/phase5-summary.md) — что реально загружено, какие проблемы
  обнаружены реальным прогоном, ответы на все контрольные вопросы фазы.
- [`reports/data-quality-report.md`](reports/data-quality-report.md) — количественный data quality
  отчёт (дубли, пропуски, coverage by year/patch/tournament).
- [`reports/team-identity-issues.md`](reports/team-identity-issues.md) — что реально проверено/не
  проверено по identity resolution на этой фазе.
- [`docs/dataset-schema.md`](docs/dataset-schema.md) — схема первого реального ML-датасета (Feature Set 0).
- [`src/datasources/opendota.py`](src/datasources/opendota.py) — `OpenDotaSource`: HTTP-клиент с
  retry/backoff/rate-limit ([`src/datasources/http_client.py`](src/datasources/http_client.py)),
  пагинация, извлечение draft/player-level данных.
- [`src/datasources/opendota_fixtures.py`](src/datasources/opendota_fixtures.py) — offline fixture mode
  (сеть к OpenDota недоступна из этой среды, см. `docs/environment-constraints.md`).
- [`src/normalization/`](src/normalization/) — normalize/validate/enrich (patch resolution из
  вендоренного `dotaconstants`).
- [`src/repositories/`](src/repositories/) — идемпотентный upsert-слой (raw/match/ingestion_run).
- [`src/ingestion/run_opendota.py`](src/ingestion/run_opendota.py) — CLI ingestion pipeline с
  checkpoint-based incremental sync.
- [`src/datasets/`](src/datasets/) — `DatasetBuilder` + Feature Set 0 (walk-forward Elo/recent form
  через `RatingEngine`, `src/ratings/engine.py`).
- [`migrations/`](migrations/) — Alembic-миграции, единственный источник схемы —
  [`src/db/schema.py`](src/db/schema.py) (SQLAlchemy Core).
- [`tests/`](tests/) — 45 тестов (unit/integration/leakage), включая
  [`tests/leakage/`](tests/leakage/) — адверсариальные проверки отсутствия утечки данных на
  РЕАЛЬНОМ коде пайплайна, не только на изолированных прототипах.

Прототипы предыдущих фаз ([`scripts/verify_data_source.py`](scripts/verify_data_source.py),
[`scripts/elo_prototype.py`](scripts/elo_prototype.py), [`src/models/base.py`](src/models/base.py),
[`src/backtesting/engine.py`](src/backtesting/engine.py)) остаются актуальными — Phase 5 их не заменила,
а построила поверх них ([`RatingEngine`](src/ratings/engine.py) используется и в
[`src/datasets/feature_set_0.py`](src/datasets/feature_set_0.py), и в `BacktestEngine`).

## Принцип разработки

> Сначала исследование и доказательство того, какие данные нам доступны. Потом архитектура.
> Потом реализация. Не наоборот.

Работа ведётся по фазам (Phase 0 — Repository inspection, Phase 1 — Research, Phase 2 — Data
feasibility, ... Phase 12 — Production). После каждой фазы фиксируются решения в `docs/` и
делается отдельный коммит, прежде чем переходить дальше.
