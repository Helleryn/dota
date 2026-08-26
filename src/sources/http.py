"""
PHASE 16 — минимальный HTTP-клиент для внешних источников.

Отдельно от `src/datasources/http_client.py`: тот заточен под OpenDota и
его политику. Здесь нужны три вещи, которых там нет: обязательный gzip
(требование ToS Liquipedia), пер-хостовые интервалы и превращение 429 в
результат, а не в исключение.

**429 не обходится.** Он возвращается вызывающему как статус
`RATE_LIMITED` вместе с `Retry-After`. Обход лимитов прямо запрещён
заданием, и технически здесь его нет: интервал только увеличивается.
"""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


@dataclass
class HostPolicy:
    """Минимальный интервал между запросами к хосту, в секундах."""
    min_interval: float = 0.0
    timeout: float = 40.0


DEFAULT_POLICIES: Dict[str, HostPolicy] = {
    # Liquipedia: их ToS требуют описательный User-Agent, gzip и низкую
    # частоту. Замер Phase 16: даже при интервале 3 с два запроса из трёх
    # получили 429, поэтому здесь взят заведомо больший интервал.
    "liquipedia.net": HostPolicy(min_interval=30.0, timeout=45.0),
    "api.liquipedia.net": HostPolicy(min_interval=30.0, timeout=45.0),
    "www.dota2.com": HostPolicy(min_interval=0.5, timeout=60.0),
    "api.bo3.gg": HostPolicy(min_interval=0.3, timeout=40.0),
    "api.opendota.com": HostPolicy(min_interval=1.1, timeout=60.0),
}


class PolitClient:
    def __init__(self, user_agent: str, policies: Optional[Dict[str, HostPolicy]] = None,
                 sleeper=time.sleep, clock=time.monotonic):
        self.user_agent = user_agent
        self.policies = dict(policies or DEFAULT_POLICIES)
        self._last: Dict[str, float] = {}
        self._sleep = sleeper
        self._clock = clock

    def _host(self, url: str) -> str:
        return url.split("://", 1)[-1].split("/", 1)[0]

    def _wait(self, host: str) -> None:
        pol = self.policies.get(host, HostPolicy())
        last = self._last.get(host)
        if last is not None and pol.min_interval > 0:
            gap = self._clock() - last
            if gap < pol.min_interval:
                self._sleep(pol.min_interval - gap)
        self._last[host] = self._clock()

    def get_json(self, url: str) -> Tuple[int, Optional[Any], Dict[str, str]]:
        """(http_status, распарсенное тело или None, заголовки).

        Исключений не бросает: сетевая ошибка приходит как status=0 —
        вызывающий обязан различать «источник ответил пусто» и «источник
        не ответил».
        """
        host = self._host(url)
        self._wait(host)
        pol = self.policies.get(host, HostPolicy())
        req = urllib.request.Request(url, headers={
            "User-Agent": self.user_agent,
            "Accept": "application/json",
            "Accept-Encoding": "gzip",     # требование ToS Liquipedia
        })
        try:
            with urllib.request.urlopen(req, timeout=pol.timeout) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                hdrs = {k.lower(): v for k, v in r.headers.items()}
                try:
                    return r.status, json.loads(raw), hdrs
                except json.JSONDecodeError:
                    return r.status, None, hdrs
        except urllib.error.HTTPError as e:
            hdrs = {k.lower(): v for k, v in (e.headers or {}).items()}
            if e.code == 429:
                # Интервал для хоста УВЕЛИЧИВАЕТСЯ. Обхода лимита нет.
                ra = hdrs.get("retry-after")
                bump = float(ra) if (ra or "").isdigit() else pol.min_interval * 2 + 1
                self.policies[host] = HostPolicy(min_interval=max(pol.min_interval, bump),
                                                 timeout=pol.timeout)
            return e.code, None, hdrs
        except Exception:
            return 0, None, {}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
