"""
Offline fixture mode для OpenDotaSource (Phase 5, раздел 1: "если OpenDota
API недоступен из текущего окружения — НЕ пытайся обходить сетевые
ограничения, вместо этого реализуй offline fixture mode").

Строит OpenDotaSource поверх httpx.MockTransport, роутящего запросы на
JSON-файлы в tests/fixtures/opendota/ (см. README там же — это
сконструированные по проверенной схеме данные, не захваченные вживую).
Используется и тестами (tests/integration/test_opendota_source.py), и CLI
ingestion pipeline (src/ingestion/run_opendota.py --source=fixtures) — один
и тот же код, не дублируется.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import httpx

from src.datasources.base import RawResponseRecord
from src.datasources.http_client import HttpClientConfig, RateLimitedHttpClient
from src.datasources.opendota import OpenDotaSource

DEFAULT_FIXTURES_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "opendota"


def build_fixture_opendota_source(
    fixtures_dir: Path = DEFAULT_FIXTURES_DIR,
    on_raw_response: Optional[Callable[[RawResponseRecord], None]] = None,
) -> OpenDotaSource:
    def _load(name: str):
        return json.loads((fixtures_dir / name).read_text())

    leagues = _load("leagues.json")
    page1 = _load("pro_matches_page1.json")
    page2 = _load("pro_matches_page2.json")

    match_detail_files = {
        int(p.stem.replace("match_detail_", "")): p
        for p in fixtures_dir.glob("match_detail_*.json")
    }
    match_details = {mid: json.loads(p.read_text()) for mid, p in match_detail_files.items()}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/leagues":
            return httpx.Response(200, json=leagues)
        if path == "/proMatches":
            less_than = request.url.params.get("less_than_match_id")
            if less_than is None:
                return httpx.Response(200, json=page1)
            if int(less_than) == min(m["match_id"] for m in page1):
                return httpx.Response(200, json=page2)
            return httpx.Response(200, json=[])
        if path.startswith("/matches/"):
            match_id = int(path.rsplit("/", 1)[-1])
            if match_id in match_details:
                return httpx.Response(200, json=match_details[match_id])
            return httpx.Response(404, json={"error": "not found in fixtures"})
        return httpx.Response(404)

    config = HttpClientConfig(
        base_url="https://fixtures.local", max_retries=1, backoff_base_seconds=0.0, requests_per_minute=6000
    )
    client = RateLimitedHttpClient(config, transport=httpx.MockTransport(handler))
    return OpenDotaSource(client=client, api_key=None, on_raw_response=on_raw_response)
