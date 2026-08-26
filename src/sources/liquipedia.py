"""
PHASE 16 — адаптер Liquipedia (ростеры и роли).

## Что установлено измерением

| Проверка | Результат |
|---|---|
| `api.php` без gzip | **406**: «Gzip encoding is required for API requests» |
| `api.php` с gzip и описательным User-Agent | **200** |
| `action=parse` страницы команды | **200**, wikitext ~51 КБ |
| в wikitext | **`joindate` 91, `leavedate` 41, `position` 88** |
| `action=cargoquery` (интервал 35 с) | **429** |
| `api.liquipedia.net/api/v3` (LPDB) | **429** без ключа |
| три `parse`-запроса с интервалом 3 с | **1 успех, 2 отказа 429** |

Выводы, которые из этого следуют:

1. Единственный работающий путь без ключа — `action=parse` с разбором
   wikitext. Структурированные интерфейсы (cargoquery, LPDB v3) закрыты.
2. Пропускная способность крайне низкая и **нестабильная**: даже интервал
   3 секунды даёт 429. Источник пригоден только с агрессивным кэшем и
   никогда — как жёсткая зависимость.
3. Зато это **единственный найденный источник ролей 1–5 и интервалов
   членства** — того, чего нет ни у Valve, ни у bo3.gg, ни у OpenDota.

Gzip и содержательный User-Agent здесь — выполнение требований ToS,
а не обход. Лимит не обходится: при 429 интервал только увеличивается.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.sources.base import (
    Provenance,
    SourceAdapter,
    SourceConfidence,
    SourceResult,
    SourceStatus,
)
from src.sources.http import PolitClient, now_utc
from src.sources.models import ExternalTeamRef, RoleConfidence, RosterMembership

API = "https://liquipedia.net/dota2/api.php"
_PERSON = re.compile(r"\{\{Person\|([^}]*)\}\}")
_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    m = _DATE.match(s.strip())
    if not m:
        return None
    return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)


def parse_roster_wikitext(wikitext: str) -> List[Dict[str, Optional[str]]]:
    """Разбор шаблонов `{{Person|...}}`.

    Вынесен отдельной функцией без обращений к сети, чтобы разбор можно
    было проверять тестами на фиксированном тексте — сетевой источник для
    этого непригоден.
    """
    out: List[Dict[str, Optional[str]]] = []
    for m in _PERSON.finditer(wikitext or ""):
        fields: Dict[str, str] = {}
        for part in m.group(1).split("|"):
            if "=" in part:
                k, v = part.split("=", 1)
                fields[k.strip().lower()] = v.strip()
        if not fields.get("id"):
            continue
        out.append({
            "id": fields.get("id"),
            "name": fields.get("name"),
            "position": fields.get("position"),
            "joindate": fields.get("joindate"),
            "leavedate": fields.get("leavedate"),
        })
    return out


class LiquipediaAdapter(SourceAdapter):
    name = "liquipedia"
    # Данные вычитываются из вики: структурные, но редактируемые
    # сообществом. Официальным анонсом это не является.
    confidence = SourceConfidence.MEDIUM
    requires_key = False

    def __init__(self, client: PolitClient, cache: Optional[Dict[str, str]] = None):
        self.client = client
        # Кэш обязателен, а не желателен: без него источник упирается в 429
        # на втором же запросе.
        self.cache: Dict[str, str] = cache if cache is not None else {}

    def fetch_team_wikitext(self, page: str) -> SourceResult[str]:
        if page in self.cache:
            return SourceResult(SourceStatus.OK, items=[self.cache[page]])
        url = (f"{API}?action=parse&page={page}&prop=wikitext&format=json&formatversion=2")
        st, body, hdrs = self.client.get_json(url)
        if st == 429:
            ra = hdrs.get("retry-after")
            return SourceResult(SourceStatus.RATE_LIMITED, error="429",
                                retry_after_seconds=float(ra) if (ra or "").isdigit() else None)
        if st == 406:
            return SourceResult(SourceStatus.UNAVAILABLE,
                                error="источник требует gzip (см. api-terms-of-use)")
        if st != 200 or not isinstance(body, dict):
            return SourceResult(SourceStatus.UNAVAILABLE, error=f"http {st}")
        wt = (body.get("parse") or {}).get("wikitext")
        if isinstance(wt, dict):
            wt = wt.get("*", "")
        if not wt:
            return SourceResult(SourceStatus.EMPTY, error="пустой wikitext")
        self.cache[page] = wt
        return SourceResult(SourceStatus.OK, items=[wt])

    def roster(self, page: str, team_ref: Optional[ExternalTeamRef] = None,
               now: Optional[datetime] = None) -> SourceResult[RosterMembership]:
        now = now or now_utc()
        r = self.fetch_team_wikitext(page)
        if not r.ok:
            return SourceResult(r.status, error=r.error,
                                retry_after_seconds=r.retry_after_seconds)
        ref = team_ref or ExternalTeamRef(self.name, page, name=page.replace("_", " "))
        prov = Provenance(self.name, now, self.confidence,
                          url=f"https://liquipedia.net/dota2/{page}", raw_id=page)
        out: List[RosterMembership] = []
        for p in parse_roster_wikitext(r.items[0]):
            pos = p.get("position")
            pos_i = int(pos) if pos and pos.isdigit() and 1 <= int(pos) <= 5 else None
            out.append(RosterMembership(
                team_ref=ref, player_id=p["id"], player_name=p.get("name"),
                valid_from=_parse_date(p.get("joindate")),
                valid_to=_parse_date(p.get("leavedate")),
                provenance=prov, position=pos_i,
                # Роль из вики — исторически заявленная позиция игрока в
                # составе, а НЕ подтверждение роли на конкретный матч.
                # Повышать её до CONFIRMED нельзя (PART E).
                role_confidence=(RoleConfidence.PREDICTED if pos_i
                                 else RoleConfidence.UNKNOWN)))
        return SourceResult(SourceStatus.OK if out else SourceStatus.EMPTY,
                            items=out, provenance=prov)
