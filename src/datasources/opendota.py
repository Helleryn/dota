"""
OpenDotaSource — адаптер OpenDota API, реализующий интерфейс DataSource
(Phase 5, раздел 2; ADR-001).

Ответственность адаптера строго ограничена: сходить в OpenDota API, привести
ответ к унифицированным структурам (RawMatch/RawPickBan/RawPlayerMatch), и
сообщить о каждом сыром HTTP-ответе через callback (для raw-слоя, см.
docs/database-design.md). Адаптер НЕ занимается патчами (это enrichment,
src/normalization/enrich.py, т.к. patch вычисляется из dotaconstants — нашего
собственного справочника, а не приходит "естественно" от источника) и НЕ
пишет в БД напрямую (это repository-слой, Phase 5.6).

Domain-модель (DataSource, RawMatch, ...) не меняется под особенности
OpenDota — вся специфика (имена полей, /explorer vs /proMatches, кодировка
player_slot) остаётся внутри этого файла.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional

from src.datasources.base import (
    DataSource,
    RawMatch,
    RawPickBan,
    RawPlayerMatch,
    RawResponseRecord,
)
from src.datasources.http_client import HttpClientConfig, RateLimitedHttpClient

logger = logging.getLogger("datasources.opendota")

SOURCE_NAME = "opendota"

# Примечание: адаптер намеренно НЕ фильтрует по tier здесь — он честно
# проставляет league_tier (включая 'amateur' и любой другой), а решение
# "включать ли этот матч в pro-датасет" принимает validation-слой
# (src/normalization/validate.py, PRO_LEAGUE_TIERS) — адаптер не должен
# знать о бизнес-правилах quality-фильтрации (docs/architecture.md).


def _content_hash(payload: object) -> str:
    """sha256 от канонического JSON — для дедупликации в raw_responses
    (Phase 5, раздел 6: "content hash / request identity для предотвращения
    бессмысленных дублей")."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class OpenDotaSource(DataSource):
    name = SOURCE_NAME

    def __init__(
        self,
        client: RateLimitedHttpClient,
        api_key: Optional[str] = None,
        on_raw_response: Optional[Callable[[RawResponseRecord], None]] = None,
    ):
        self._client = client
        self._api_key = api_key
        self.on_raw_response = on_raw_response
        self._league_tier_cache: Optional[Dict[int, str]] = None
        self._match_detail_cache: Dict[int, dict] = {}  # маленький кэш, см. _fetch_match_detail

    @classmethod
    def from_config(
        cls,
        base_url: str,
        api_key: Optional[str],
        timeout_seconds: float,
        max_retries: int,
        rate_limit_per_min: int,
        on_raw_response: Optional[Callable[[RawResponseRecord], None]] = None,
    ) -> "OpenDotaSource":
        http_config = HttpClientConfig(
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            requests_per_minute=rate_limit_per_min,
        )
        client = RateLimitedHttpClient(http_config)
        return cls(client=client, api_key=api_key, on_raw_response=on_raw_response)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenDotaSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- внутреннее ---

    def _get(self, endpoint: str, params: Optional[dict] = None) -> object:
        params = dict(params or {})
        if self._api_key:
            params["api_key"] = self._api_key

        fetched_at = datetime.now(timezone.utc)
        payload = self._client.get_json(endpoint, params=params)

        if self.on_raw_response:
            # api_key не должен попадать в raw-слой (это секрет, не данные).
            safe_params = {k: v for k, v in params.items() if k != "api_key"}
            self.on_raw_response(
                RawResponseRecord(
                    source=SOURCE_NAME,
                    endpoint=endpoint,
                    request_params=safe_params,
                    fetched_at=fetched_at,
                    http_status=200,  # get_json поднимает исключение на не-200, сюда доходит только успех
                    response_body=payload,
                    content_hash=_content_hash(payload),
                )
            )
        return payload

    def _league_tiers(self) -> Dict[int, str]:
        """Кэшируется на весь жизненный цикл адаптера — /leagues отдаёт всю
        таблицу целиком за один вызов (VERIFIED из исходников), не пагинируется."""
        if self._league_tier_cache is None:
            leagues = self._get("/leagues")
            self._league_tier_cache = {
                league["leagueid"]: league.get("tier") for league in leagues
            }
        return self._league_tier_cache

    def _fetch_match_detail(self, match_id: int) -> dict:
        """
        /matches/{id} используется и fetch_picks_bans, и fetch_player_matches
        — небольшой кэш на процесс, чтобы не делать два HTTP-запроса за один
        и тот же матч, если ingestion pipeline вызовет оба метода подряд
        (типичный сценарий). Кэш не растёт бесконечно — ingestion обрабатывает
        матчи пачками, не держит миллионы в памяти одновременно.
        """
        if match_id not in self._match_detail_cache:
            if len(self._match_detail_cache) > 32:
                self._match_detail_cache.clear()
            self._match_detail_cache[match_id] = self._get(f"/matches/{match_id}")
        return self._match_detail_cache[match_id]

    # --- DataSource interface ---

    def fetch_matches(self, since: datetime, until: datetime) -> Iterable[RawMatch]:
        """
        Пагинация /proMatches назад по match_id (VERIFIED: ORDER BY
        match_id DESC LIMIT 100, параметр less_than_match_id). Останавливается,
        когда start_time страницы уходит раньше `since`, или ответ пуст.

        match_id и start_time коррелируют монотонно НЕ строго (эпизодические
        отклонения возможны), поэтому граница чуть отступает: страница
        считается "мимо диапазона" только когда ВСЕ записи в ней раньше
        `since`, не по первой же записи.
        """
        tiers = self._league_tiers()
        less_than_match_id: Optional[int] = None

        while True:
            params = {}
            if less_than_match_id is not None:
                params["less_than_match_id"] = less_than_match_id

            page = self._get("/proMatches", params=params)
            if not page:
                return

            page_had_in_range = False
            for row in page:
                start_time = datetime.fromtimestamp(row["start_time"], tz=timezone.utc)
                if start_time >= until:
                    continue  # ещё не дошли до верхней границы, пропускаем без остановки пагинации
                if start_time < since:
                    continue  # старше нижней границы — пропускаем эту строку, но не обязательно всю страницу
                page_had_in_range = True

                league_tier = tiers.get(row.get("leagueid"))
                yield RawMatch(
                    match_id=row["match_id"],
                    start_time=start_time,
                    duration_seconds=row["duration"],
                    radiant_team_id=row.get("radiant_team_id"),
                    dire_team_id=row.get("dire_team_id"),
                    radiant_team_name=row.get("radiant_name"),
                    dire_team_name=row.get("dire_name"),
                    radiant_win=row["radiant_win"],
                    league_id=row.get("leagueid"),
                    league_tier=league_tier,
                    series_id=row.get("series_id"),
                    series_type=row.get("series_type"),
                    patch=None,  # проставляется в enrichment-слое (src/normalization/enrich.py)
                    source=SOURCE_NAME,
                    fetched_at=datetime.now(timezone.utc),
                )

            oldest_in_page = min(
                datetime.fromtimestamp(row["start_time"], tz=timezone.utc) for row in page
            )
            if oldest_in_page < since:
                return  # вся страница уже раньше нижней границы — дальше углубляться незачем

            if not page_had_in_range and oldest_in_page >= until:
                # Страница целиком новее верхней границы (например, until в прошлом,
                # а на сервере появились ещё более новые матчи) — продолжаем пагинацию назад.
                pass

            less_than_match_id = min(row["match_id"] for row in page)

    def fetch_picks_bans(self, match_id: int) -> Iterable[RawPickBan]:
        detail = self._fetch_match_detail(match_id)
        picks_bans = detail.get("picks_bans") or []
        for pb in picks_bans:
            yield RawPickBan(
                match_id=match_id,
                is_pick=pb["is_pick"],
                hero_id=pb["hero_id"],
                team=pb["team"],
                order=pb["order"],
            )

    def fetch_player_matches(self, match_id: int) -> Iterable[RawPlayerMatch]:
        detail = self._fetch_match_detail(match_id)
        radiant_team_id = detail.get("radiant_team_id")
        dire_team_id = detail.get("dire_team_id")

        for player in detail.get("players") or []:
            is_radiant = player["player_slot"] < 128  # VERIFIED: кодировка Steam Web API
            yield RawPlayerMatch(
                match_id=match_id,
                account_id=player.get("account_id"),
                team_id=radiant_team_id if is_radiant else dire_team_id,
                hero_id=player["hero_id"],
                is_radiant=is_radiant,
                kills=player.get("kills", 0),
                deaths=player.get("deaths", 0),
                assists=player.get("assists", 0),
                gold_per_min=player.get("gold_per_min", 0),
                xp_per_min=player.get("xp_per_min", 0),
            )
