"""
Надёжный HTTP-слой (Phase 5, раздел 3) — переиспользуемый, не привязан к
OpenDota. Конкретные адаптеры (src/datasources/opendota.py) конфигурируют
его под свой base_url/лимиты, не переопределяют логику retry/rate-limit.

Поддерживает: timeout, retries с экспоненциальным backoff, централизованный
rate limiting (token-bucket по времени), различение транзиентных и
постоянных HTTP-ошибок (docs/data-pipeline.md), JSON-валидацию, structured
logging.

Намеренно НЕ делает бесконечный retry — max_retries конечен, после чего
поднимается исключение, а не тихо возвращается None (вызывающий код,
ingestion pipeline, обязан явно решить, что делать с окончательным сбоем).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

logger = logging.getLogger("datasources.http_client")


class TransientHttpError(Exception):
    """5xx, timeout, connection error — стоит повторить (docs/data-pipeline.md)."""


class PermanentHttpError(Exception):
    """4xx (кроме 429) — повторять бессмысленно, запрос некорректен по сути."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class RateLimiter:
    """
    Простой rate limiter с равномерным интервалом между запросами
    (min_interval = 60 / requests_per_minute). Достаточно для
    однопоточного последовательного ingestion — не token bucket с burst,
    т.к. нам не нужны всплески, наоборот, важна предсказуемая,
    равномерная нагрузка на чужой бесплатный сервис.
    """

    def __init__(self, requests_per_minute: int):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute должен быть положительным")
        self.min_interval = 60.0 / requests_per_minute
        self._lock = threading.Lock()
        self._last_request_at: Optional[float] = None

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_request_at is not None:
                elapsed = now - self._last_request_at
                remaining = self.min_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            self._last_request_at = time.monotonic()


@dataclass(frozen=True)
class HttpClientConfig:
    base_url: str
    timeout_seconds: float = 20.0
    max_retries: int = 4
    backoff_base_seconds: float = 1.0  # 1s, 2s, 4s, 8s
    requests_per_minute: int = 55
    user_agent: str = "dota2-predict/0.1"


class RateLimitedHttpClient:
    """
    Обёртка над httpx.Client с retry/backoff/rate-limit/логированием.
    Используется адаптерами DataSource (opendota.py и т.д.), сама не знает
    ничего о доменной модели Dota 2.
    """

    def __init__(self, config: HttpClientConfig, transport: Optional[httpx.BaseTransport] = None):
        self.config = config
        self._rate_limiter = RateLimiter(config.requests_per_minute)
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            headers={"User-Agent": config.user_agent},
            transport=transport,  # позволяет подменить транспорт в тестах (httpx.MockTransport)
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "RateLimitedHttpClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get_json(self, path: str, params: Optional[dict] = None) -> Any:
        """
        GET-запрос с retry/backoff/rate-limit, возвращает распарсенный JSON.
        Поднимает PermanentHttpError на 4xx (кроме 429), TransientHttpError —
        если все попытки исчерпаны на 5xx/timeout/сетевых ошибках.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            self._rate_limiter.wait()
            request_log = {"path": path, "params": params, "attempt": attempt}

            try:
                response = self._client.get(path, params=params)
            except httpx.TimeoutException as e:
                last_error = TransientHttpError(f"timeout: {e}")
                logger.warning("http_timeout", extra=request_log)
                self._sleep_backoff(attempt)
                continue
            except httpx.HTTPError as e:
                last_error = TransientHttpError(f"connection error: {e}")
                logger.warning("http_connection_error", extra={**request_log, "error": str(e)})
                self._sleep_backoff(attempt)
                continue

            status = response.status_code

            if status == 200:
                try:
                    payload = response.json()
                except json.JSONDecodeError as e:
                    # Формат ответа неожиданный — это структурная проблема
                    # (не транзиентная сетевая), но и не "клиент виноват" —
                    # логируем как отдельный класс проблемы, не путаем ни с
                    # TransientHttpError, ни с PermanentHttpError.
                    raise ValueError(f"невалидный JSON в ответе {path}: {e}") from e
                logger.info("http_success", extra={**request_log, "status": status})
                return payload

            if status == 429:
                # Rate limit — не постоянная, не совсем "обычная" транзиентная
                # ошибка: уважаем Retry-After, если сервер его прислал.
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else self._backoff_delay(attempt)
                logger.warning("http_429_rate_limited", extra={**request_log, "retry_after": delay})
                last_error = TransientHttpError("rate limited (429)")
                time.sleep(delay)
                continue

            if 500 <= status < 600:
                last_error = TransientHttpError(f"HTTP {status}")
                logger.warning("http_5xx", extra={**request_log, "status": status})
                self._sleep_backoff(attempt)
                continue

            # Любой другой 4xx — постоянная ошибка, retry бессмыслен.
            logger.error("http_4xx_permanent", extra={**request_log, "status": status})
            raise PermanentHttpError(f"HTTP {status} для {path}", status_code=status)

        assert last_error is not None
        raise last_error

    def _backoff_delay(self, attempt: int) -> float:
        return self.config.backoff_base_seconds * (2 ** (attempt - 1))

    def _sleep_backoff(self, attempt: int) -> None:
        if attempt < self.config.max_retries:
            time.sleep(self._backoff_delay(attempt))
