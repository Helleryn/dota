"""
PHASE 15 — версии замороженных компонентов.

Смысл модуля — сделать невозможной ситуацию «прогноз есть, а чем он
получен, восстановить нельзя». Каждый снимок несёт четыре версии, и
любое изменение соответствующего компонента ОБЯЗАНО менять свою версию.
Правило PART P: менять модель после начала prospective validation можно
только через новую версию, старые прогнозы при этом не трогаются и не
удаляются.
"""

from __future__ import annotations

from typing import List

# Замороженный набор признаков Phase 9. Пять входов, а не три: в части
# документации набор ошибочно сокращали до трёх добавок KEEP, опуская два
# входа frozen baseline. Именно этой пятёркой получены все опубликованные
# величины Phase 9/12/13/14.
FROZEN_FEATURES: List[str] = [
    "elo_difference",
    "form_3_difference",
    "elo_mean_diff",
    "five_vs_team_elo_diff",
    "hero_exp_decay_diff",
]

ELO_K = 16
FORM_WINDOW = 3
RANDOM_SEED = 42

# Калибровка — результат Phase 14. Патч-локальная составляющая гибрида
# СОЗНАТЕЛЬНО не берётся: на VALIDATION она помогала, на TEST в трёх окнах
# из четырёх сделала хуже (reports/phase14-patch-drift.md, раздел 4).
CALIBRATION_METHOD = "beta"
CALIBRATION_MODE = "rolling"
CALIBRATION_WINDOW = 5000
CALIBRATION_REFIT_EVERY = 10
CALIBRATION_MIN_HISTORY = 500

MODEL_VERSION = "phase9-logreg-5feat-k16-form3-seed42"
FEATURE_VERSION = "phase9-frozen-v1"
CALIBRATION_VERSION = "phase14-beta-rolling5000-v1"
PREDICTION_VERSION = "phase15-shadow-v1"
RESOLUTION_VERSION = "phase15-resolution-v1"


def version_block() -> dict:
    return {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "prediction_version": PREDICTION_VERSION,
    }


def frozen_spec() -> dict:
    """Полная спецификация замороженного конвейера — уходит в отчёт и в
    тест воспроизводимости."""
    return {
        "features": list(FROZEN_FEATURES),
        "elo_k": ELO_K,
        "form_window": FORM_WINDOW,
        "random_seed": RANDOM_SEED,
        "calibration": {
            "method": CALIBRATION_METHOD,
            "mode": CALIBRATION_MODE,
            "window": CALIBRATION_WINDOW,
            "refit_every": CALIBRATION_REFIT_EVERY,
            "min_history": CALIBRATION_MIN_HISTORY,
        },
        **version_block(),
    }
