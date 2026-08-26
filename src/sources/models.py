"""
PHASE 16 — нормализованные объекты предметной области.

Источники отдают разные идентификаторы: у Valve свои `team_id`, у bo3.gg
свои, у Liquipedia — названия страниц. Нормализация **не склеивает их
автоматически**: Phase 10 нашла 461 пару `team_id` с одинаковым именем,
существовавших параллельно, поэтому имя команды не является
идентичностью. Внешние идентификаторы хранятся раздельно, а связывание —
отдельное решение с собственным уровнем доверия.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from src.sources.base import Provenance, SourceConfidence


class RoleConfidence(str, Enum):
    """PART E: уровни знания о роли. Подменять один другим запрещено —
    именно это Phase 11 показала на `lane_role` из реплея."""
    CONFIRMED = "CONFIRMED_ROLE"   # заявлено источником на конкретный матч
    PREDICTED = "PREDICTED_ROLE"   # исторический приор игрока
    UNKNOWN = "UNKNOWN_ROLE"


@dataclass(frozen=True)
class ExternalTeamRef:
    """Ссылка на команду в терминах КОНКРЕТНОГО источника."""
    source: str
    external_id: str
    name: Optional[str] = None
    valve_team_id: Optional[int] = None   # заполняется только при доказанном соответствии


@dataclass(frozen=True)
class UpcomingMatch:
    """Нормализованный предстоящий матч."""
    source: str
    external_id: str
    scheduled_start: datetime
    team_a: ExternalTeamRef
    team_b: ExternalTeamRef
    provenance: Provenance
    tournament: Optional[str] = None
    tournament_external_id: Optional[str] = None
    stage: Optional[str] = None
    best_of: Optional[int] = None
    status: str = "SCHEDULED"          # SCHEDULED | POSTPONED | CANCELLED | STARTED
    league_id: Optional[int] = None
    stream_count: int = 0

    @property
    def match_key(self) -> str:
        """Ключ, стабильный внутри источника. Межисточниковое слияние —
        отдельная задача и здесь не выполняется."""
        return f"{self.source}:{self.external_id}"

    def lead_hours(self, now: datetime) -> float:
        return (self.scheduled_start - now).total_seconds() / 3600.0


@dataclass(frozen=True)
class RosterMembership:
    """Членство игрока в команде на ИНТЕРВАЛЕ времени (PART D).

    Именно интервал, а не «текущий состав»: без `valid_from` / `valid_to`
    невозможно ответить на вопрос «какой состав был известен на момент T»,
    а ответ на него — условие корректности всей фазы.
    """
    team_ref: ExternalTeamRef
    player_id: str
    player_name: Optional[str]
    valid_from: Optional[datetime]
    valid_to: Optional[datetime]
    provenance: Provenance
    position: Optional[int] = None                     # 1-5
    role_confidence: RoleConfidence = RoleConfidence.UNKNOWN
    announcement_timestamp: Optional[datetime] = None

    def active_at(self, t: datetime) -> bool:
        """Активен ли на момент t.

        `valid_from is None` трактуется как «неизвестно, когда пришёл» и
        считается активным до `valid_to` — иначе игрок с неизвестной датой
        прихода выпадал бы из всех запросов. Это осознанное допущение,
        и оно понижает доверие: такой факт не может иметь HIGH.
        """
        if self.valid_from is not None and t < self.valid_from:
            return False
        if self.valid_to is not None and t >= self.valid_to:
            return False
        return True

    def known_at(self, t: datetime) -> bool:
        """Была ли эта запись ИЗВЕСТНА системе на момент t.

        Отдельно от `active_at`: факт может действовать с прошлого месяца,
        но стать известным нам только вчера. Прогноз, сделанный до
        `observed_at`, использовать его не имеет права.
        """
        return self.provenance.observed_at <= t


@dataclass(frozen=True)
class PatchInfo:
    name: str
    released_at: datetime
    provenance: Provenance
    patch_id: Optional[int] = None

    def active_at(self, t: datetime) -> bool:
        return t >= self.released_at


@dataclass(frozen=True)
class PreMatchSnapshotDraft:
    """PART L — проект pre-match снимка.

    Отличается от `PredictionSnapshot` Phase 15 тем, что несёт
    происхождение КАЖДОГО внешнего факта, а не только отметки среза.
    Модель, признаки и калибровка при этом те же самые и не меняются.
    """
    match: UpcomingMatch
    prediction_time: datetime
    roster_a: List[RosterMembership] = field(default_factory=list)
    roster_b: List[RosterMembership] = field(default_factory=list)
    patch: Optional[PatchInfo] = None
    features: Dict[str, float] = field(default_factory=dict)
    raw_probability: Optional[float] = None
    calibrated_probability: Optional[float] = None
    confidence: Optional[float] = None
    data_confidence: SourceConfidence = SourceConfidence.UNKNOWN
    provenance: Dict[str, Provenance] = field(default_factory=dict)
    unresolved_conflicts: List[str] = field(default_factory=list)
    model_version: Optional[str] = None
    feature_version: Optional[str] = None
    calibration_version: Optional[str] = None

    def roster_complete(self) -> bool:
        return len(self.roster_a) == 5 and len(self.roster_b) == 5

    def all_roles_confirmed(self) -> bool:
        rs = self.roster_a + self.roster_b
        return bool(rs) and all(r.role_confidence == RoleConfidence.CONFIRMED for r in rs)
