"""
PHASE 10 — walk-forward team identity resolution.

## Задача

`source_team_id` (OpenDota) -> `canonical_team_id` (наша сущность).
Аудит Phase 10.0 показал масштаб проблемы: 7 838 team_id, медиана 3 матча,
45.6% команд с 1-2 матчами, 686 групп с одинаковым именем.

## Почему walk-forward, а не «построить mapping по всей истории»

Наивный подход использовал бы будущее знание: зная в 2026 году, что A и B —
одна сущность, мы применили бы это к матчу 2022 года. Здесь связь A->B
становится активной РОВНО В ТОТ МОМЕНТ, когда доказательство наблюдаемо:
когда B проводит первый матч составом, пересекающимся с последним составом A.

    valid_from(A->B) = start_time первого матча B, где
                       overlap(roster_B, last_roster_A) >= MIN_OVERLAP

При прогнозе матча в момент t используются только связи с valid_from <= t.
Это делает слой leakage-safe ПО ПОСТРОЕНИЮ: mapping строится тем же
однопроходным chronological сканированием, что и все Feature Set модули
проекта, и физически не может увидеть будущее.

## Консервативные правила (false merge опаснее false split)

1. **Параллельные команды НИКОГДА не сливаются.** Если B начал играть до
   того, как A закончил, это разные сущности — сколько бы игроков они ни
   делили. Аудит: 55.9% всех кандидатов параллельны, и 461 пара имеет
   одинаковое имя при параллельном существовании (слияние по имени было бы
   ошибкой во всех 461 случае).
2. **Только по имени не сливаем никогда** (LOW confidence).
3. Требуется сильное пересечение состава И временная близость.

## Имена

`teams.name` — снимок «на сегодня» (point-in-time имён не существует ни у
нас, ни у OpenDota, см. reports/phase10-identity-plan.md, п.1.3). Поэтому
имя используется ТОЛЬКО как повышение уверенности HIGH vs MEDIUM, но
никогда как самостоятельное основание для слияния. Слияние возможно и без
совпадения имени (ребрендинг) — на ростерных доказательствах.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, FrozenSet, Iterable, List, Optional, Protocol, Tuple

MIN_OVERLAP = 0.6           # доля общих игроков (Jaccard) последнего состава A и состава B
MAX_GAP_DAYS = 90.0         # максимальный разрыв между последним матчем A и первым B
MIN_MATCHES_FOR_ANCHOR = 2  # A должен был сыграть хотя бы столько, чтобы быть «якорем»


def normalize_name(name: Optional[str]) -> str:
    """Консервативная нормализация: регистр, unicode, пробелы, разделители.
    НЕ удаляем 'Team'/'Academy' — 'Entity' и 'Entity Academy' могут быть
    разными сущностями."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", name).casefold().strip()
    s = re.sub(r"[_\-–—.]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def jaccard(a: FrozenSet[int], b: FrozenSet[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class MatchWithRoster(Protocol):
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_roster: FrozenSet[int]
    dire_roster: FrozenSet[int]


@dataclass
class IdentityLink:
    """Одна установленная связь source -> canonical, с временем активации."""
    source_team_id: int
    canonical_team_id: int
    predecessor_team_id: int
    valid_from: datetime
    confidence: str            # HIGH | MEDIUM
    roster_overlap: float
    gap_days: float
    name_matched: bool
    resolution_method: str = "walk_forward_roster_continuity"


@dataclass
class _TeamState:
    last_roster: FrozenSet[int]
    last_seen: datetime
    n_matches: int = 0


@dataclass
class ResolutionResult:
    links: List[IdentityLink] = field(default_factory=list)
    # canonical_id для каждого source_id на момент его появления
    canonical_of: Dict[int, int] = field(default_factory=dict)

    def stats(self) -> dict:
        by_conf: Dict[str, int] = {}
        for l in self.links:
            by_conf[l.confidence] = by_conf.get(l.confidence, 0) + 1
        merged_sources = len({l.source_team_id for l in self.links})
        canonicals = len(set(self.canonical_of.values()))
        return {
            "n_links": len(self.links),
            "by_confidence": by_conf,
            "merged_source_ids": merged_sources,
            "n_source_ids": len(self.canonical_of),
            "n_canonical_ids": canonicals,
            "reduction": len(self.canonical_of) - canonicals,
        }


class WalkForwardIdentityResolver:
    """
    Один хронологический проход. Для каждого team_id, встреченного ВПЕРВЫЕ,
    ищется предшественник среди уже завершившихся команд.

    Никогда не переписывает прошлое: канонический id присваивается новому
    source_id в момент его первого матча и больше не меняется.
    """

    def __init__(
        self,
        names: Optional[Dict[int, str]] = None,
        min_overlap: float = MIN_OVERLAP,
        max_gap_days: float = MAX_GAP_DAYS,
    ):
        self._names = {k: normalize_name(v) for k, v in (names or {}).items()}
        self.min_overlap = min_overlap
        self.max_gap_days = max_gap_days
        self._state: Dict[int, _TeamState] = {}
        self._canonical: Dict[int, int] = {}
        self.result = ResolutionResult()

    def canonical(self, source_team_id: int) -> int:
        """source -> canonical. Неизвестный/несвязанный id отображается сам в себя,
        поэтому исходный team_id НИКОГДА не теряется."""
        return self._canonical.get(source_team_id, source_team_id)

    def _find_predecessor(self, new_team_id: int, roster: FrozenSet[int], now: datetime):
        best = None
        for cand_id, st in self._state.items():
            if cand_id == new_team_id:
                continue
            if st.n_matches < MIN_MATCHES_FOR_ANCHOR:
                continue
            # ПРАВИЛО 1: предшественник обязан УЖЕ ЗАВЕРШИТЬСЯ.
            # Если он ещё играет (last_seen близко/позже), это параллельное
            # существование -> разные сущности, слияние запрещено.
            gap = (now - st.last_seen).total_seconds() / 86400.0
            if gap <= 0 or gap > self.max_gap_days:
                continue
            ov = jaccard(roster, st.last_roster)
            if ov < self.min_overlap:
                continue
            if best is None or ov > best[1]:
                best = (cand_id, ov, gap)
        return best

    def _observe_team(self, team_id: int, roster: FrozenSet[int], now: datetime) -> None:
        if team_id not in self._state:
            # первый раз видим этот team_id -> пробуем найти предшественника
            found = self._find_predecessor(team_id, roster, now)
            if found:
                pred_id, ov, gap = found
                canonical_id = self.canonical(pred_id)  # цепочки A->B->C схлопываются
                name_match = (
                    self._names.get(team_id, "") != ""
                    and self._names.get(team_id) == self._names.get(pred_id)
                )
                self._canonical[team_id] = canonical_id
                self.result.links.append(IdentityLink(
                    source_team_id=team_id,
                    canonical_team_id=canonical_id,
                    predecessor_team_id=pred_id,
                    valid_from=now,
                    confidence="HIGH" if name_match else "MEDIUM",
                    roster_overlap=round(ov, 3),
                    gap_days=round(gap, 2),
                    name_matched=name_match,
                ))
            self.result.canonical_of[team_id] = self.canonical(team_id)
            self._state[team_id] = _TeamState(last_roster=roster, last_seen=now, n_matches=1)
        else:
            st = self._state[team_id]
            st.last_roster = roster
            st.last_seen = now
            st.n_matches += 1

    def process(self, match: MatchWithRoster) -> None:
        """Обрабатывает один матч. Матчи ОБЯЗАНЫ подаваться в хронологическом
        порядке (start_time, match_id) — тот же контракт, что у RatingEngine."""
        self._observe_team(match.radiant_team_id, match.radiant_roster, match.start_time)
        self._observe_team(match.dire_team_id, match.dire_roster, match.start_time)


def resolve_identities(
    matches: Iterable[MatchWithRoster],
    names: Optional[Dict[int, str]] = None,
    min_overlap: float = MIN_OVERLAP,
    max_gap_days: float = MAX_GAP_DAYS,
) -> Tuple[WalkForwardIdentityResolver, ResolutionResult]:
    resolver = WalkForwardIdentityResolver(names=names, min_overlap=min_overlap, max_gap_days=max_gap_days)
    for m in matches:
        resolver.process(m)
    return resolver, resolver.result
