"""
PHASE 15 — состояния прогноза (PART E).

Состояние двигается только вперёд. INVALID достижим из любого состояния —
это не шаг конвейера, а отметка «прогноз непригоден». Тот же порядок
продублирован триггером в СУБД: если правило будет нарушено в коде,
запись всё равно не пройдёт.
"""

from __future__ import annotations

from typing import List

DISCOVERED = "DISCOVERED"
FEATURES_READY = "FEATURES_READY"
PREDICTED = "PREDICTED"
CALIBRATED = "CALIBRATED"
PUBLISHED = "PUBLISHED"
MATCH_STARTED = "MATCH_STARTED"
MATCH_FINISHED = "MATCH_FINISHED"
RESOLVED = "RESOLVED"
INVALID = "INVALID"

ORDER: List[str] = [
    DISCOVERED, FEATURES_READY, PREDICTED, CALIBRATED, PUBLISHED,
    MATCH_STARTED, MATCH_FINISHED, RESOLVED,
]
ALL = ORDER + [INVALID]

# После публикации содержимое снимка неизменяемо.
PUBLISHED_RANK = ORDER.index(PUBLISHED)


def rank(state: str) -> int:
    if state == INVALID:
        return -1
    if state not in ORDER:
        raise ValueError(f"неизвестное состояние: {state}")
    return ORDER.index(state)


def can_transition(old: str, new: str) -> bool:
    if new == INVALID:
        return True
    if old == INVALID:
        return False          # непригодный прогноз не «чинится», делается новый
    return rank(new) >= rank(old)


def is_published(state: str) -> bool:
    return state in ORDER and rank(state) >= PUBLISHED_RANK


# Причины отказа (PART R). Тихого fallback нет ни в одном случае.
INVALID_REASONS = {
    "source_unavailable": "OpenDota недоступен на момент прогноза",
    "lineup_unknown": "состав неизвестен",
    "match_cancelled": "матч отменён",
    "ambiguous_team_identity": "идентичность команды неоднозначна",
    "patch_unknown": "патч на момент матча неизвестен",
    "inconsistent_data": "данные противоречивы",
    "future_data_detected": "обнаружены данные позже времени прогноза",
    "match_already_started": "матч уже начался к моменту прогноза",
    "model_trained_after_prediction": "модель обучалась на матчах не раньше прогноза",
}
