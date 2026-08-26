"""
PHASE 16 — адаптер официальных эндпоинтов Valve (`www.dota2.com/webapi`).

## Что установлено измерением (не по документации)

| Эндпоинт | Ответ | Что даёт |
|---|---|---|
| `GetLeagueInfoList` | **200**, 9 867 лиг | список лиг, tier, даты |
| `GetLeagueData?league_id=` | **200**, до 160 КБ | сетка турнира: `node_groups` -> `nodes` |
| `GetLiveLeagueGames` | 200, пустой | ничего |
| `api.steampowered.com/.../GetScheduledLeagueGames` | **404** | эндпоинта нет |
| `api.steampowered.com/.../GetLiveLeagueGames` | **403** | нужен ключ |

Узел сетки содержит `scheduled_time`, `team_id_1`, `team_id_2`,
`series_id`, `has_started`, `is_completed`, `stream_ids`.

## Главное ограничение, найденное сканом 106 активных лиг

* 1 142 узла, `scheduled_time` заполнено у **219 (19.2%)**;
* будущих матчей — **38**, и все они из **ОДНОЙ** лиги;
* `registered_players` пуст **во всех 106 лигах** — составов здесь нет.

То есть источник официальный и бесплатный, но заполнение расписания
зависит от организатора турнира, и полагаться на него как на
единственный нельзя.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from src.sources.base import (
    Provenance,
    SourceAdapter,
    SourceConfidence,
    SourceResult,
    SourceStatus,
)
from src.sources.http import PolitClient, now_utc
from src.sources.models import ExternalTeamRef, UpcomingMatch

BASE = "https://www.dota2.com/webapi/IDOTA2DPC"


def _flatten_groups(groups: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for g in groups or []:
        out.append(g)
        out.extend(_flatten_groups(g.get("node_groups") or []))
    return out


class ValveDpcAdapter(SourceAdapter):
    name = "valve_dpc"
    confidence = SourceConfidence.HIGH      # официальный источник Valve
    requires_key = False

    def __init__(self, client: PolitClient):
        self.client = client

    # ------------------------------------------------------------------
    def active_leagues(self, now: Optional[datetime] = None,
                       within_days: int = 30) -> SourceResult[Dict[str, Any]]:
        now = now or now_utc()
        st, body, _ = self.client.get_json(f"{BASE}/GetLeagueInfoList/v001/?start_timestamp=0")
        if st == 429:
            return SourceResult(SourceStatus.RATE_LIMITED, error="429")
        if st != 200 or not isinstance(body, dict):
            return SourceResult(SourceStatus.UNAVAILABLE, error=f"http {st}")
        infos = body.get("infos") or []
        cut = now.timestamp() - within_days * 86400
        act = [x for x in infos if (x.get("most_recent_activity") or 0) > cut]
        act.sort(key=lambda x: -(x.get("most_recent_activity") or 0))
        return SourceResult(
            SourceStatus.OK if act else SourceStatus.EMPTY, items=act,
            provenance=Provenance(self.name, now, self.confidence,
                                  url=f"{BASE}/GetLeagueInfoList/v001/"))

    # ------------------------------------------------------------------
    def upcoming_for_league(self, league_id: int, now: Optional[datetime] = None,
                            league_name: Optional[str] = None) -> SourceResult[UpcomingMatch]:
        now = now or now_utc()
        url = f"{BASE}/GetLeagueData/v001/?league_id={league_id}"
        st, body, _ = self.client.get_json(url)
        if st == 429:
            return SourceResult(SourceStatus.RATE_LIMITED, error="429")
        if st != 200 or not isinstance(body, dict):
            return SourceResult(SourceStatus.UNAVAILABLE, error=f"http {st}")

        prov = Provenance(self.name, now, self.confidence, url=url, raw_id=str(league_id))
        out: List[UpcomingMatch] = []
        for g in _flatten_groups(body.get("node_groups") or []):
            for n in g.get("nodes") or []:
                ts = n.get("scheduled_time") or 0
                if ts <= 0:
                    continue                      # организатор не заполнил время
                start = datetime.fromtimestamp(ts, tz=timezone.utc)
                if start <= now:
                    continue                      # матч уже начался или прошёл
                if n.get("has_started") or n.get("is_completed"):
                    continue
                t1, t2 = n.get("team_id_1"), n.get("team_id_2")
                if not t1 or not t2:
                    continue                      # участники ещё не определены сеткой
                out.append(UpcomingMatch(
                    source=self.name,
                    external_id=f"{league_id}:{n.get('node_id')}",
                    scheduled_start=start,
                    team_a=ExternalTeamRef(self.name, str(t1), valve_team_id=int(t1)),
                    team_b=ExternalTeamRef(self.name, str(t2), valve_team_id=int(t2)),
                    provenance=prov,
                    tournament=league_name,
                    tournament_external_id=str(league_id),
                    stage=g.get("name") or n.get("name"),
                    league_id=league_id,
                    stream_count=len(n.get("stream_ids") or []),
                ))
        return SourceResult(SourceStatus.OK if out else SourceStatus.EMPTY,
                            items=out, provenance=prov)

    # ------------------------------------------------------------------
    def discover_upcoming(self, now: Optional[datetime] = None, max_leagues: int = 40,
                          within_days: int = 30) -> SourceResult[UpcomingMatch]:
        now = now or now_utc()
        lg = self.active_leagues(now, within_days)
        if not lg.ok:
            return SourceResult(lg.status, error=lg.error)
        found: List[UpcomingMatch] = []
        for l in lg.items[:max_leagues]:
            r = self.upcoming_for_league(int(l["league_id"]), now, l.get("name"))
            if r.status == SourceStatus.RATE_LIMITED:
                # Лимит — результат, а не повод давить дальше.
                return SourceResult(SourceStatus.RATE_LIMITED, items=found,
                                    error="429 при обходе лиг")
            found.extend(r.items)
        found.sort(key=lambda m: m.scheduled_start)
        return SourceResult(SourceStatus.OK if found else SourceStatus.EMPTY, items=found,
                            provenance=Provenance(self.name, now, self.confidence))
