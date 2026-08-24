"""test_http_errors, test_retry, test_rate_limit (Phase 5, раздел 20)."""

import time

import httpx
import pytest

from src.datasources.http_client import (
    HttpClientConfig,
    PermanentHttpError,
    RateLimitedHttpClient,
    RateLimiter,
    TransientHttpError,
)


def _client(handler, max_retries=3, backoff=0.01, rpm=6000):
    config = HttpClientConfig(base_url="https://fake.test", max_retries=max_retries,
                               backoff_base_seconds=backoff, requests_per_minute=rpm)
    return RateLimitedHttpClient(config, transport=httpx.MockTransport(handler))


def test_success_returns_parsed_json():
    def handler(request):
        return httpx.Response(200, json={"a": 1})

    with _client(handler) as client:
        assert client.get_json("/x") == {"a": 1}


def test_http_errors_permanent_no_retry():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404, json={"error": "nope"})

    with _client(handler) as client:
        with pytest.raises(PermanentHttpError) as exc_info:
            client.get_json("/missing")
        assert exc_info.value.status_code == 404
    assert calls["n"] == 1, "4xx (кроме 429) не должен повторяться"


def test_retry_recovers_after_transient_failures():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json={"ok": True})

    with _client(handler, max_retries=5) as client:
        result = client.get_json("/flaky")
    assert result == {"ok": True}
    assert calls["n"] == 3


def test_retry_exhausted_raises_transient_error():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500)

    with _client(handler, max_retries=3) as client:
        with pytest.raises(TransientHttpError):
            client.get_json("/always-down")
    assert calls["n"] == 3, "не более max_retries попыток — НЕ бесконечный retry"


def test_invalid_json_raises_value_error():
    def handler(request):
        return httpx.Response(200, content=b"not json{{{")

    with _client(handler) as client:
        with pytest.raises(ValueError):
            client.get_json("/bad-json")


def test_rate_limit_enforces_minimum_interval():
    """test_rate_limit — центральный rate limiter реально ограничивает скорость запросов."""
    limiter = RateLimiter(requests_per_minute=600)  # 0.1s между запросами
    t0 = time.monotonic()
    for _ in range(3):
        limiter.wait()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.2, f"3 запроса по 0.1s между ними должны занять >= 0.2s, заняли {elapsed:.3f}s"


def test_rate_limit_respects_retry_after_on_429():
    calls = {"n": 0}
    sleep_calls = []

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0.05"}, json={"error": "slow down"})
        return httpx.Response(200, json={"ok": True})

    with _client(handler, max_retries=3) as client:
        result = client.get_json("/limited")
    assert result == {"ok": True}
    assert calls["n"] == 2
