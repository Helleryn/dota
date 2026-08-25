"""
OpenDotaExplorerSource — bulk-исторический адаптер OpenDota через `/explorer`
(прямой read-only SQL к БД OpenDota), для Phase 6 (baseline ML требует
многолетний датасет — `reports/phase5-live-summary.md` рекомендует
2021-01-01 → today).

Отличие от `OpenDotaSource` (Phase 5, живой incremental/backfill адаптер
через `/proMatches`): `/proMatches` постранично идёт НАЗАД по `match_id` от
самых свежих матчей — чтобы дойти до 2021 года, пришлось бы пройти
постранично ВЕСЬ объём (~250к матчей всех tier), что при лимите 60
запросов/мин заняло бы часы ради данных, 90% из которых (excluded-tier,
Phase 5 live-находка) всё равно будут отброшены `PRO_LEAGUE_TIERS`-фильтром.

`docs/data-pipeline.md` (Phase 4) уже предвидел эту проблему: "большой
исторический backfill эффективнее через `/explorer` ... инкрементальный
sync — через `/proMatches`" — этот файл реализует ту часть дизайна, которая
раньше была только предположением ("REQUIRES LIVE VERIFICATION").

Намеренное отличие от `OpenDotaSource.fetch_matches` (который НЕ фильтрует
по tier — это принцип адаптера, см. opendota.py): здесь `tier IN
('professional', 'premium')` фильтруется НА УРОВНЕ SQL, не после
загрузки. Это осознанное исключение из общего принципа "адаптер не решает
бизнес-правила", а не случайная деградация: единственная цель этого
источника — эффективная выборка ML-тренировочных данных на многолетнем
диапазоне, где сплошная загрузка (как у `OpenDotaSource`) физически
неосуществима за разумное время. Каждый вызов `fetch_matches()` — ОДИН
HTTP-запрос к `/explorer`, не пагинация.

`fetch_picks_bans`/`fetch_player_matches` возвращают пустые итераторы —
draft/player-level данные для этого bulk-backfill не собираются (Phase 6
прямо запрещает draft-признаки на этом этапе, раздел 32 задания), это не
ограничение источника, а осознанный выбор объёма работы.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Iterable, List, Optional

from src.datasources.base import DataSource, RawMatch, RawPickBan, RawPlayerMatch, RawResponseRecord
from src.datasources.opendota import _content_hash
from src.datasources.http_client import HttpClientConfig, RateLimitedHttpClient

logger = logging.getLogger("datasources.opendota_explorer")

SOURCE_NAME = "opendota_explorer"


class OpenDotaExplorerSource(DataSource):
    name = SOURCE_NAME

    def __init__(
        self,
        client: RateLimitedHttpClient,
        on_raw_response: Optional[Callable[[RawResponseRecord], None]] = None,
    ):
        self._client = client
        self.on_raw_response = on_raw_response

    @classmethod
    def from_config(
        cls,
        base_url: str,
        timeout_seconds: float,
        max_retries: int,
        rate_limit_per_min: int,
        on_raw_response: Optional[Callable[[RawResponseRecord], None]] = None,
    ) -> "OpenDotaExplorerSource":
        http_config = HttpClientConfig(
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            requests_per_minute=rate_limit_per_min,
        )
        return cls(client=RateLimitedHttpClient(http_config), on_raw_response=on_raw_response)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenDotaExplorerSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def fetch_matches(self, since: datetime, until: datetime) -> Iterable[RawMatch]:
        since_epoch = int(since.timestamp())
        until_epoch = int(until.timestamp())
        sql = (
            "SELECT m.match_id, m.start_time, m.duration, m.radiant_team_id, m.dire_team_id, "
            "m.radiant_team_name, m.dire_team_name, m.radiant_win, m.leagueid, "
            "m.series_id, m.series_type, l.tier "
            "FROM matches m JOIN leagues l ON l.leagueid = m.leagueid "
            "WHERE l.tier IN ('professional', 'premium') "
            "AND m.radiant_team_id IS NOT NULL AND m.dire_team_id IS NOT NULL "
            "AND m.radiant_win IS NOT NULL AND m.duration IS NOT NULL "
            f"AND m.start_time >= {since_epoch} AND m.start_time < {until_epoch} "
            "ORDER BY m.start_time ASC"
        )
        fetched_at = datetime.now(timezone.utc)
        payload = self._client.get_json("/explorer", params={"sql": sql})

        if payload.get("err"):
            raise ValueError(f"/explorer вернул ошибку: {payload['err']}")

        if self.on_raw_response:
            self.on_raw_response(
                RawResponseRecord(
                    source=SOURCE_NAME,
                    endpoint="/explorer",
                    request_params={"sql": sql},
                    fetched_at=fetched_at,
                    http_status=200,
                    response_body=payload,
                    content_hash=_content_hash(payload),
                )
            )

        for row in payload.get("rows", []):
            yield RawMatch(
                match_id=row["match_id"],
                start_time=datetime.fromtimestamp(row["start_time"], tz=timezone.utc),
                duration_seconds=row["duration"],
                radiant_team_id=row.get("radiant_team_id"),
                dire_team_id=row.get("dire_team_id"),
                radiant_team_name=row.get("radiant_team_name"),
                dire_team_name=row.get("dire_team_name"),
                radiant_win=row["radiant_win"],
                league_id=row.get("leagueid"),
                league_tier=row.get("tier"),
                series_id=row.get("series_id"),
                series_type=row.get("series_type"),
                patch=None,  # проставляется в enrichment-слое, как и у OpenDotaSource
                source=SOURCE_NAME,
                fetched_at=fetched_at,
            )

    def fetch_picks_bans(self, match_id: int) -> Iterable[RawPickBan]:
        return []

    def fetch_player_matches(self, match_id: int) -> Iterable[RawPlayerMatch]:
        return []
