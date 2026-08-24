# Dota 2 Match Prediction — исследовательско-инженерный проект

Сервис прогнозирования вероятности победы команды в профессиональном матче Dota 2,
на основе исторических данных и объяснимого ML.

## Статус проекта

**PHASE 0-3 завершены** (repository inspection, research, data feasibility, feature catalog).
Проект находится в исследовательской стадии: до крупной реализации фиксируются источники
данных, их реальная доступность, риски утечки данных (data leakage) и архитектура — см. `docs/`.
Итоговый отчёт по Phase 2+3: [`docs/research-summary.md`](docs/research-summary.md).

## Документация

- [`docs/data-sources.md`](docs/data-sources.md) — сравнительная таблица источников данных Dota 2 esports
  (OpenDota, STRATZ, Steam Web API, Liquipedia, Dotabuff, datdota, dotaconstants, Kaggle) с проверенными фактами.
- [`docs/environment-constraints.md`](docs/environment-constraints.md) — важное ограничение текущей
  среды выполнения (сетевой egress) и его влияние на архитектуру сбора данных.
- [`docs/data-feasibility.md`](docs/data-feasibility.md) — какой dataset реально можем построить,
  обоснование MVP, слоистая схема Raw → Normalized → Feature → Dataset, концепция DataSource abstraction.
- [`docs/features.md`](docs/features.md) — полный feature catalog (категории A-H, Feature Set 0-4).
- [`docs/data-leakage.md`](docs/data-leakage.md) — аудит утечек данных по каждой группе признаков,
  обоснование собственного walk-forward Elo вместо снэпшота OpenDota.
- [`docs/research-summary.md`](docs/research-summary.md) — итоговый отчёт Phase 2+3, план Phase 4.

Вспомогательный код (не production, а исследовательские прототипы):

- [`scripts/verify_data_source.py`](scripts/verify_data_source.py) — автономный скрипт для запуска
  вне этой среды (см. `environment-constraints.md`): проверяет реальную доступность OpenDota API,
  формат ответа, rate limits и историческую глубину данных.
- [`scripts/elo_prototype.py`](scripts/elo_prototype.py) — прототип leakage-safe walk-forward Elo
  с автоматической проверкой отсутствия утечки на синтетических данных.
- [`src/datasources/base.py`](src/datasources/base.py) — минимальный абстрактный интерфейс
  `DataSource` (концепция независимости архитектуры от конкретного источника).

Остальные документы (`database-design.md`, `architecture.md`, `ml-models.md`, `evaluation.md`,
`backtesting.md`, `api.md`) будут добавляться по мере прохождения следующих фаз (см. PHASE 4+).

## Принцип разработки

> Сначала исследование и доказательство того, какие данные нам доступны. Потом архитектура.
> Потом реализация. Не наоборот.

Работа ведётся по фазам (Phase 0 — Repository inspection, Phase 1 — Research, Phase 2 — Data
feasibility, ... Phase 12 — Production). После каждой фазы фиксируются решения в `docs/` и
делается отдельный коммит, прежде чем переходить дальше.
