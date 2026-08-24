# Dota 2 Match Prediction — исследовательско-инженерный проект

Сервис прогнозирования вероятности победы команды в профессиональном матче Dota 2,
на основе исторических данных и объяснимого ML.

## Статус проекта

**PHASE 0-4 завершены** (repository inspection, research, data feasibility, feature catalog,
architecture). Проект находится в исследовательско-архитектурной стадии: до реализации
production-пайплайна зафиксированы источники данных, риски утечки данных, схема БД и полная
архитектура — см. `docs/`. Итоговый отчёт по Phase 2+3: [`docs/research-summary.md`](docs/research-summary.md).

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

Вспомогательный код (прототипы/скелеты, не полная production-реализация):

- [`scripts/verify_data_source.py`](scripts/verify_data_source.py) — автономный скрипт для запуска
  вне этой среды: проверяет реальную доступность OpenDota API, формат ответа, rate limits и
  историческую глубину данных.
- [`scripts/elo_prototype.py`](scripts/elo_prototype.py) — прототип leakage-safe walk-forward Elo.
- [`src/config.py`](src/config.py) — конфигурация (`DATA_START_DATE`, `TRAIN_START` и т.д.),
  runnable, с валидацией порядка time-based split.
- [`src/datasources/base.py`](src/datasources/base.py) — абстрактный интерфейс `DataSource`.
- [`src/ratings/engine.py`](src/ratings/engine.py) — `RatingEngine`, тот же walk-forward Elo как
  переиспользуемый компонент (training/backtesting/serving — один код).
- [`src/models/base.py`](src/models/base.py) — `BasePredictionModel` + референсная реализация `EloRuleModel`.
- [`src/backtesting/engine.py`](src/backtesting/engine.py) — `BacktestEngine`, интеграционная демонстрация
  полного цикла `RatingEngine` + модель на синтетических данных.

Все `.py`-файлы выше запускаются напрямую (`python3 <path>` или
`python3 -m <module>`) и содержат self-test/демонстрацию — не только код,
но и проверку заявленных гарантий (в первую очередь — отсутствия утечки данных).

## Принцип разработки

> Сначала исследование и доказательство того, какие данные нам доступны. Потом архитектура.
> Потом реализация. Не наоборот.

Работа ведётся по фазам (Phase 0 — Repository inspection, Phase 1 — Research, Phase 2 — Data
feasibility, ... Phase 12 — Production). После каждой фазы фиксируются решения в `docs/` и
делается отдельный коммит, прежде чем переходить дальше.
