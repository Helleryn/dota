# Dota 2 Match Prediction — исследовательско-инженерный проект

Сервис прогнозирования вероятности победы команды в профессиональном матче Dota 2,
на основе исторических данных и объяснимого ML.

## Статус проекта

**PHASE 1 (исследование источников данных) завершена.** Проект находится в исследовательской
стадии: до крупной реализации фиксируются источники данных, их реальная доступность,
риски утечки данных (data leakage) и архитектура — см. `docs/`.

## Документация

- [`docs/data-sources.md`](docs/data-sources.md) — сравнительная таблица источников данных Dota 2 esports
  (OpenDota, STRATZ, Steam Web API, Liquipedia, Dotabuff, datdota, dotaconstants, Kaggle) с проверенными фактами.
- [`docs/environment-constraints.md`](docs/environment-constraints.md) — важное ограничение текущей
  среды выполнения (сетевой egress) и его влияние на архитектуру сбора данных.

Остальные документы (`data-feasibility.md`, `features.md`, `data-leakage.md`, `database-design.md`,
`architecture.md`, `ml-models.md`, `evaluation.md`, `backtesting.md`, `api.md`) будут добавляться
по мере прохождения следующих фаз (см. PHASE 2+).

## Принцип разработки

> Сначала исследование и доказательство того, какие данные нам доступны. Потом архитектура.
> Потом реализация. Не наоборот.

Работа ведётся по фазам (Phase 0 — Repository inspection, Phase 1 — Research, Phase 2 — Data
feasibility, ... Phase 12 — Production). После каждой фазы фиксируются решения в `docs/` и
делается отдельный коммит, прежде чем переходить дальше.
