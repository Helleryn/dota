"""
PHASE 16 — временное хранилище фактов (PART D/I).

Отвечает на единственный вопрос, ради которого фаза и затевалась:

    «Какой состав был ИЗВЕСТЕН СИСТЕМЕ на момент T?»

Обратите внимание на слово «известен». Два разных условия должны
выполняться одновременно, и путать их — самая тонкая утечка фазы:

| Условие | Метка | Смысл |
|---|---|---|
| факт **действовал** в момент T | `valid_from` / `valid_to` | игрок числился в составе |
| факт **был известен** в момент T | `provenance.observed_at` | мы уже видели эту запись |

Состав, выгруженный сегодня, не говорит ничего о том, кто играл месяц
назад: он действовал, но известен не был. Запрос по одному лишь интервалу
действия молча протащил бы в прошлое сегодняшнее знание.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from src.sources.base import SourceConfidence
from src.sources.models import ExternalTeamRef, PatchInfo, RosterMembership

# Порядок доверия к источнику: используется при разрешении конфликтов.
CONFIDENCE_RANK = {
    SourceConfidence.HIGH: 3,
    SourceConfidence.MEDIUM: 2,
    SourceConfidence.LOW: 1,
    SourceConfidence.UNKNOWN: 0,
}


def _team_key(ref: ExternalTeamRef) -> Tuple[str, str]:
    """Ключ команды — пара (источник, внешний id).

    Намеренно НЕ имя: Phase 10 нашла 461 пару `team_id` с одинаковым
    именем, существовавших параллельно. Слияние идентичностей между
    источниками — отдельное решение, а не побочный эффект хранения.
    """
    return (ref.source, ref.external_id)


@dataclass
class TemporalRosterStore:
    """Хранилище членств с запросами «на момент T»."""

    _by_team: Dict[Tuple[str, str], List[RosterMembership]] = None

    def __post_init__(self):
        if self._by_team is None:
            self._by_team = defaultdict(list)

    def add(self, m: RosterMembership) -> None:
        self._by_team[_team_key(m.team_ref)].append(m)

    def add_all(self, ms: Sequence[RosterMembership]) -> None:
        for m in ms:
            self.add(m)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_team.values())

    # ------------------------------------------------------------------
    def roster_as_of(self, ref: ExternalTeamRef, t: datetime,
                     require_known: bool = True) -> List[RosterMembership]:
        """Состав команды на момент t.

        `require_known=True` (по умолчанию) — только факты, уже
        наблюдавшиеся к моменту t. Отключать его допустимо лишь для
        исторического анализа, где вопрос стоит иначе: «кто фактически
        числился», а не «что мы знали».
        """
        out = []
        for m in self._by_team.get(_team_key(ref), []):
            if not m.active_at(t):
                continue
            if require_known and not m.known_at(t):
                continue
            out.append(m)
        return out

    def teams(self) -> List[Tuple[str, str]]:
        return list(self._by_team)


def patch_as_of(patches: Sequence[PatchInfo], t: datetime,
                require_known: bool = True) -> Optional[PatchInfo]:
    """Патч, действующий на момент t (PART G).

    Берётся последний релиз не позже t. Будущий патч использовать нельзя
    даже если он уже анонсирован: на момент прогноза игра идёт на текущем.
    """
    best: Optional[PatchInfo] = None
    for p in patches:
        if not p.active_at(t):
            continue
        if require_known and p.provenance.observed_at > t:
            continue
        if best is None or p.released_at > best.released_at:
            best = p
    return best


def roster_data_confidence(members: Sequence[RosterMembership]) -> SourceConfidence:
    """Общее доверие к составу = доверие САМОГО СЛАБОГО факта в нём.

    Не среднее: состав, где четверо подтверждены официально, а пятый
    угадан, не является «в основном надёжным» — прогноз строится на всех
    пяти сразу.
    """
    if not members:
        return SourceConfidence.UNKNOWN
    worst = min(CONFIDENCE_RANK[m.provenance.confidence] for m in members)
    for k, v in CONFIDENCE_RANK.items():
        if v == worst:
            return k
    return SourceConfidence.UNKNOWN
