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

По умолчанию (`fetch_draft_and_players=False`, поведение Phase 6/6.5, НЕ
изменено) `fetch_picks_bans`/`fetch_player_matches` возвращают пустые
итераторы — сохраняет воспроизводимость `reports/phase6-summary.md` и
`reports/phase6_5-summary.md` (тот же код, тот же результат при повторном
запуске тех скриптов).

Phase 7 добавляет `fetch_draft_and_players=True`: OpenDota хранит
`picks_bans` прямо КОЛОНКОЙ на `matches` (не отдельными вызовами
`/matches/{id}`, как предполагала Phase 5) и даёт отдельную explorer-таблицу
`player_matches` (match_id/account_id/player_slot/hero_id + игровая
статистика, из которой берутся только identity-поля, НЕ KDA/GPM/урон —
раздел 8 задания Phase 7: "не начинай сразу с KDA/GPM/damage"). Оба
получаются bulk-запросом `/explorer` ВНУТРИ `fetch_matches()` (те же 1-2
HTTP-вызова на диапазон, не по вызову на матч) и кешируются в памяти —
`fetch_picks_bans`/`fetch_player_matches` читают из кеша, не делают
дополнительных HTTP-запросов. Раздел 25 задания Phase 5 ("не используй
`/teams/{id}/players` как источник истины для исторического ростера") —
`player_matches` даёт РЕАЛЬНЫЙ point-in-time состав конкретного матча,
не all-time агрегат, соответствует тому же принципу.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional

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
        fetch_draft_and_players: bool = False,
    ):
        self._client = client
        self.on_raw_response = on_raw_response
        self._fetch_draft_and_players = fetch_draft_and_players
        self._picks_bans_cache: Dict[int, List[RawPickBan]] = {}
        self._player_matches_cache: Dict[int, List[RawPlayerMatch]] = {}

    @classmethod
    def from_config(
        cls,
        base_url: str,
        timeout_seconds: float,
        max_retries: int,
        rate_limit_per_min: int,
        on_raw_response: Optional[Callable[[RawResponseRecord], None]] = None,
        fetch_draft_and_players: bool = False,
    ) -> "OpenDotaExplorerSource":
        http_config = HttpClientConfig(
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            requests_per_minute=rate_limit_per_min,
        )
        return cls(
            client=RateLimitedHttpClient(http_config),
            on_raw_response=on_raw_response,
            fetch_draft_and_players=fetch_draft_and_players,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenDotaExplorerSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def fetch_matches(self, since: datetime, until: datetime) -> Iterable[RawMatch]:
        since_epoch = int(since.timestamp())
        until_epoch = int(until.timestamp())
        picks_bans_column = ", m.picks_bans" if self._fetch_draft_and_players else ""
        sql = (
            "SELECT m.match_id, m.start_time, m.duration, m.radiant_team_id, m.dire_team_id, "
            "m.radiant_team_name, m.dire_team_name, m.radiant_win, m.leagueid, "
            f"m.series_id, m.series_type, l.tier{picks_bans_column} "
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

        team_by_match: Dict[int, tuple] = {}
        for row in payload.get("rows", []):
            match_id = row["match_id"]
            radiant_team_id = row.get("radiant_team_id")
            dire_team_id = row.get("dire_team_id")
            team_by_match[match_id] = (radiant_team_id, dire_team_id)

            if self._fetch_draft_and_players:
                self._picks_bans_cache[match_id] = [
                    RawPickBan(
                        match_id=match_id,
                        is_pick=pb["is_pick"],
                        hero_id=pb["hero_id"],
                        team=pb["team"],
                        order=pb["order"],
                    )
                    for pb in (row.get("picks_bans") or [])
                ]

            yield RawMatch(
                match_id=match_id,
                start_time=datetime.fromtimestamp(row["start_time"], tz=timezone.utc),
                duration_seconds=row["duration"],
                radiant_team_id=radiant_team_id,
                dire_team_id=dire_team_id,
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

        if self._fetch_draft_and_players and team_by_match:
            self._fetch_player_matches_bulk(since_epoch, until_epoch, team_by_match)

    def _fetch_player_matches_bulk(self, since_epoch: int, until_epoch: int, team_by_match: Dict[int, tuple]) -> None:
        """
        Один bulk-запрос `player_matches` на весь диапазон вместо запроса
        на матч (Phase 7). Берутся ТОЛЬКО identity-поля (match_id,
        account_id, player_slot, hero_id) + минимальные kills/deaths/assists/
        gold_per_min/xp_per_min — они уже требуются структурой RawPlayerMatch
        (Phase 5), но НЕ используются как признаки в Phase 7 (раздел 8
        задания: без KDA/GPM/damage на этом этапе) — только сохраняются в
        normalized-слое как есть, как и у OpenDotaSource.
        """
        sql = (
            "SELECT pm.match_id, pm.account_id, pm.player_slot, pm.hero_id, "
            "pm.kills, pm.deaths, pm.assists, pm.gold_per_min, pm.xp_per_min "
            "FROM player_matches pm JOIN matches m ON m.match_id = pm.match_id "
            f"WHERE m.start_time >= {since_epoch} AND m.start_time < {until_epoch}"
        )
        fetched_at = datetime.now(timezone.utc)
        payload = self._client.get_json("/explorer", params={"sql": sql})

        if payload.get("err"):
            raise ValueError(f"/explorer вернул ошибку (player_matches): {payload['err']}")

        if self.on_raw_response:
            self.on_raw_response(
                RawResponseRecord(
                    source=SOURCE_NAME,
                    endpoint="/explorer",
                    request_params={"sql": sql},
                    fetched_at=fetched_at,
                    http_status=200,
                    response_body={"rowCount": payload.get("rowCount")},  # тело не дублируется целиком в raw (десятки МБ)
                    content_hash=_content_hash({"sql": sql, "rowCount": payload.get("rowCount")}),
                )
            )

        for row in payload.get("rows", []):
            match_id = row["match_id"]
            teams = team_by_match.get(match_id)
            if teams is None:
                continue
            radiant_team_id, dire_team_id = teams
            is_radiant = row["player_slot"] < 128
            self._player_matches_cache.setdefault(match_id, []).append(
                RawPlayerMatch(
                    match_id=match_id,
                    account_id=row.get("account_id"),
                    team_id=radiant_team_id if is_radiant else dire_team_id,
                    hero_id=row["hero_id"],
                    is_radiant=is_radiant,
                    kills=row.get("kills") or 0,
                    deaths=row.get("deaths") or 0,
                    assists=row.get("assists") or 0,
                    gold_per_min=row.get("gold_per_min") or 0,
                    xp_per_min=row.get("xp_per_min") or 0,
                )
            )

    def fetch_picks_bans(self, match_id: int) -> Iterable[RawPickBan]:
        return self._picks_bans_cache.get(match_id, [])

    def fetch_player_matches(self, match_id: int) -> Iterable[RawPlayerMatch]:
        return self._player_matches_cache.get(match_id, [])
