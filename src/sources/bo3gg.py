"""
PHASE 16 — адаптер bo3.gg.

## Что установлено измерением

| Проверка | Результат |
|---|---|
| `/api/v1/matches` | **200**, 79 370 матчей Dota всего |
| фильтр `status=upcoming` | **378** предстоящих матчей |
| запас по времени | медиана **94.8 ч**, максимум 580 ч |
| известны за ≥24 ч | 311 из 378 (**82.3%**) |
| известны за ≥7 суток | 115 (30.4%) |
| `with=players` на предстоящем матче | **составы есть**, но полны редко |
| матчей ровно с 10 игроками | 38 из 378 (**10.1%**) |
| матчей с нулём игроков | 293 (**77.5%**) |
| полнота у матчей < 3 ч | 42.9% против 5.5% у матчей ≥ 24 ч |

Пагинация — `page[offset]` и `page[limit]`; форма `offset`/`limit` молча
игнорируется и возвращает первые 10 записей. Это стоит помнить: неверный
параметр не даёт ошибки, он даёт неполные данные.

Идентификаторы команд и игроков — **собственные**, не совпадают с Valve.
Соответствие не выводится автоматически (Phase 10: имя не есть
идентичность).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.sources.base import (
    Provenance,
    SourceAdapter,
    SourceConfidence,
    SourceResult,
    SourceStatus,
)
from src.sources.http import PolitClient, now_utc
from src.sources.models import (
    ExternalTeamRef,
    RoleConfidence,
    RosterMembership,
    UpcomingMatch,
)

BASE = "https://api.bo3.gg/api/v1"
DOTA_DISCIPLINE = 1


def _parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


class Bo3ggAdapter(SourceAdapter):
    name = "bo3gg"
    # Сообщество со структурированным API: не официальный источник, но и
    # не догадка. Ростеры отсюда не могут получить HIGH.
    confidence = SourceConfidence.MEDIUM
    requires_key = False

    def __init__(self, client: PolitClient):
        self.client = client

    def _page(self, path: str, offset: int, limit: int) -> Any:
        sep = "&" if "?" in path else "?"
        url = f"{BASE}{path}{sep}page[offset]={offset}&page[limit]={limit}"
        return self.client.get_json(url)

    def upcoming(self, now: Optional[datetime] = None, max_items: int = 400,
                 with_players: bool = True) -> SourceResult[UpcomingMatch]:
        now = now or now_utc()
        q = (f"/matches?filter[matches.status][eq]=upcoming"
             f"&filter[matches.discipline_id][eq]={DOTA_DISCIPLINE}&sort=start_date")
        if with_players:
            q += "&with=players"
        prov = Provenance(self.name, now, self.confidence, url=BASE + q)

        items: List[UpcomingMatch] = []
        self._players_by_match: Dict[str, List[Dict[str, Any]]] = {}
        offset = 0
        while offset < max_items:
            st, body, _ = self._page(q, offset, 100)
            if st == 429:
                return SourceResult(SourceStatus.RATE_LIMITED, items=items, error="429")
            if st != 200 or not isinstance(body, dict):
                return SourceResult(SourceStatus.UNAVAILABLE, items=items,
                                    error=f"http {st}")
            rows = body.get("results") or []
            for m in rows:
                start = _parse_ts(m.get("start_date"))
                t1, t2 = m.get("team1_id"), m.get("team2_id")
                if start is None or not t1 or not t2:
                    continue          # без времени или участников матч бесполезен
                if start <= now:
                    continue
                ext = str(m.get("id"))
                items.append(UpcomingMatch(
                    source=self.name, external_id=ext, scheduled_start=start,
                    team_a=ExternalTeamRef(self.name, str(t1)),
                    team_b=ExternalTeamRef(self.name, str(t2)),
                    provenance=prov,
                    tournament_external_id=(str(m["tournament_id"])
                                            if m.get("tournament_id") else None),
                    best_of=m.get("bo_type"),
                    status="SCHEDULED",
                ))
                if with_players:
                    self._players_by_match[ext] = m.get("players") or []
            if len(rows) < 100:
                break
            offset += 100
        return SourceResult(SourceStatus.OK if items else SourceStatus.EMPTY,
                            items=items, provenance=prov)

    def roster_for_match(self, match: UpcomingMatch,
                         now: Optional[datetime] = None) -> SourceResult[RosterMembership]:
        """Состав на матч — если источник его знает.

        Измерение: полный состав из десяти игроков есть у 10.1% предстоящих
        матчей, и почти всегда — у ближайших. Поэтому пустой ответ здесь
        нормален и обязан возвращаться как EMPTY, а не как ошибка.

        Позиции источник не отдаёт, поэтому роль остаётся UNKNOWN_ROLE.
        Подставлять сюда историческую роль нельзя — это разные уровни
        знания (PART E).
        """
        now = now or now_utc()
        cache = getattr(self, "_players_by_match", {})
        raw = cache.get(match.external_id)
        if raw is None:
            return SourceResult(SourceStatus.EMPTY,
                                error="состав не запрашивался вместе с матчем")
        prov = Provenance(self.name, now, self.confidence,
                          url=f"{BASE}/matches?filter[matches.id][eq]={match.external_id}",
                          raw_id=match.external_id)
        out: List[RosterMembership] = []
        for p in raw:
            pid = p.get("id")
            if pid is None:
                continue
            tid = p.get("team_id")
            ref = (match.team_a if str(tid) == match.team_a.external_id
                   else match.team_b if str(tid) == match.team_b.external_id
                   else ExternalTeamRef(self.name, str(tid) if tid else "?"))
            out.append(RosterMembership(
                team_ref=ref, player_id=str(pid),
                player_name=p.get("nickname") or p.get("first_name"),
                valid_from=None, valid_to=None, provenance=prov,
                position=None, role_confidence=RoleConfidence.UNKNOWN))
        return SourceResult(SourceStatus.OK if out else SourceStatus.EMPTY,
                            items=out, provenance=prov)
