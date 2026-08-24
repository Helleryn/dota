#!/usr/bin/env python3
"""
Скрипт проверки реальной доступности источников данных (Phase 2.1).

ВАЖНО: этот скрипт НЕЛЬЗЯ запустить из текущей среды разработки — сетевой
egress-прокси блокирует api.opendota.com и api.stratz.com (см.
docs/environment-constraints.md). Запустите его на своей машине или в среде
с обычным доступом в интернет:

    python3 -m pip install requests
    python3 scripts/verify_data_source.py

Результат — JSON-отчёт (stdout) с полями, которые в docs/data-feasibility.md
и docs/research-summary.md помечены как REQUIRES LIVE VERIFICATION.
Отчёт не хранит и не отправляет никаких персональных данных.

Что проверяется:
  1. Доступность API (status code, latency).
  2. Аутентификация (без ключа / с ключом, если задан OPENDOTA_API_KEY).
  3. Формат ответа (наличие ожидаемых полей).
  4. Rate limit (заголовки ответа, если сервис их отдаёт).
  5. Минимальный объём и историческая глубина данных — через один
     агрегирующий SQL-запрос к /explorer, а не постраничным перебором
     (экономит квоту запросов).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

OPENDOTA_BASE = "https://api.opendota.com/api"
USER_AGENT = "dota2-predict-research/0.1 (verification script; contact: set CONTACT_EMAIL env var before running)"

# Liquipedia ToS требует кастомный User-Agent с контактной информацией.
# Placeholder намеренно не содержит реального email пользователя —
# впишите свой контакт перед запуском, если планируете обращаться к Liquipedia.
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "REPLACE_ME@example.com")


@dataclass
class CheckResult:
    name: str
    status: str  # "VERIFIED" | "FAILED" | "SKIPPED"
    detail: str
    data: Optional[dict[str, Any]] = None


def _get(url: str, headers: dict[str, str], timeout: float = 20.0) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, dict(resp.headers), resp.read()


def check_opendota_status(api_key: Optional[str]) -> CheckResult:
    """Проверка 1: доступность API + задержка."""
    url = f"{OPENDOTA_BASE}/status"
    if api_key:
        url += f"?api_key={urllib.parse.quote(api_key)}"
    headers = {"User-Agent": USER_AGENT}
    t0 = time.monotonic()
    try:
        status, resp_headers, body = _get(url, headers)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        return CheckResult(
            name="opendota_status",
            status="VERIFIED" if status == 200 else "FAILED",
            detail=f"HTTP {status}, latency {latency_ms}ms",
            data={"status_code": status, "latency_ms": latency_ms},
        )
    except urllib.error.HTTPError as e:
        return CheckResult("opendota_status", "FAILED", f"HTTP {e.code}: {e.reason}")
    except Exception as e:  # noqa: BLE001 — верификационный скрипт, нужен максимально широкий catch для отчёта
        return CheckResult("opendota_status", "FAILED", f"{type(e).__name__}: {e}")


def check_promatches_format(api_key: Optional[str]) -> CheckResult:
    """Проверка 2+3: доступность и формат ответа /proMatches."""
    url = f"{OPENDOTA_BASE}/proMatches"
    if api_key:
        url += f"?api_key={urllib.parse.quote(api_key)}"
    headers = {"User-Agent": USER_AGENT}
    try:
        status, resp_headers, body = _get(url, headers)
        if status != 200:
            return CheckResult("opendota_promatches_format", "FAILED", f"HTTP {status}")
        payload = json.loads(body)
        if not isinstance(payload, list) or not payload:
            return CheckResult("opendota_promatches_format", "FAILED", "Пустой или неожиданный формат ответа")
        expected_fields = {
            "match_id", "duration", "start_time", "radiant_team_id", "radiant_name",
            "dire_team_id", "dire_name", "leagueid", "league_name", "radiant_win",
        }
        actual_fields = set(payload[0].keys())
        missing = expected_fields - actual_fields
        rate_limit_headers = {
            k: v for k, v in resp_headers.items() if "rate" in k.lower() or "limit" in k.lower()
        }
        return CheckResult(
            name="opendota_promatches_format",
            status="VERIFIED" if not missing else "FAILED",
            detail=(
                "Схема соответствует ожиданиям" if not missing
                else f"Отсутствуют ожидаемые поля: {sorted(missing)}"
            ),
            data={
                "sample_count": len(payload),
                "fields_seen": sorted(actual_fields),
                "rate_limit_headers": rate_limit_headers,
                "oldest_match_id_in_sample": min(m["match_id"] for m in payload),
                "newest_match_id_in_sample": max(m["match_id"] for m in payload),
            },
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult("opendota_promatches_format", "FAILED", f"{type(e).__name__}: {e}")


def check_historical_coverage(api_key: Optional[str]) -> CheckResult:
    """
    Проверка 5: историческая глубина через ОДИН агрегирующий запрос к
    /explorer, а не постраничным перебором /proMatches (последнее стоило
    бы сотни вызовов и было бы медленным и дорогим по квоте).
    """
    sql = """
        SELECT
          date_trunc('year', to_timestamp(matches.start_time)) AS year,
          count(*) AS match_count
        FROM matches
        JOIN leagues USING(leagueid)
        WHERE leagues.tier IN ('premium', 'professional')
        GROUP BY 1
        ORDER BY 1
    """.strip()
    url = f"{OPENDOTA_BASE}/explorer?sql={urllib.parse.quote(sql)}"
    if api_key:
        url += f"&api_key={urllib.parse.quote(api_key)}"
    headers = {"User-Agent": USER_AGENT}
    try:
        status, resp_headers, body = _get(url, headers, timeout=60.0)
        if status != 200:
            return CheckResult("opendota_historical_coverage", "FAILED", f"HTTP {status}")
        payload = json.loads(body)
        rows = payload.get("rows") or []
        if payload.get("err"):
            return CheckResult("opendota_historical_coverage", "FAILED", f"SQL error: {payload['err']}")
        return CheckResult(
            name="opendota_historical_coverage",
            status="VERIFIED" if rows else "FAILED",
            detail=f"Получено {len(rows)} годовых срезов" if rows else "Пустой результат",
            data={"rows_by_year": rows},
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult("opendota_historical_coverage", "FAILED", f"{type(e).__name__}: {e}")


def check_picks_bans_sample(api_key: Optional[str], sample_match_id: Optional[int]) -> CheckResult:
    """Проверка структуры драфта на одном реальном матче."""
    if sample_match_id is None:
        return CheckResult("opendota_picks_bans_sample", "SKIPPED", "match_id не передан")
    url = f"{OPENDOTA_BASE}/matches/{sample_match_id}"
    if api_key:
        url += f"?api_key={urllib.parse.quote(api_key)}"
    headers = {"User-Agent": USER_AGENT}
    try:
        status, resp_headers, body = _get(url, headers, timeout=30.0)
        if status != 200:
            return CheckResult("opendota_picks_bans_sample", "FAILED", f"HTTP {status}")
        payload = json.loads(body)
        pb = payload.get("picks_bans")
        if not pb:
            return CheckResult(
                "opendota_picks_bans_sample", "FAILED",
                "У этого матча нет picks_bans — выберите другой match_id",
            )
        expected_keys = {"is_pick", "hero_id", "team", "order"}
        actual_keys = set(pb[0].keys())
        return CheckResult(
            name="opendota_picks_bans_sample",
            status="VERIFIED" if expected_keys.issubset(actual_keys) else "FAILED",
            detail="Структура picks_bans подтверждена" if expected_keys.issubset(actual_keys)
                   else f"Неожиданная структура: {sorted(actual_keys)}",
            data={"picks_bans_count": len(pb), "example": pb[0]},
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult("opendota_picks_bans_sample", "FAILED", f"{type(e).__name__}: {e}")


def rate_limit_sleep(has_key: bool) -> None:
    """
    Пауза между запросами согласно проверенным лимитам OpenDota
    (60/мин без ключа, 300/мин с ключом — см. docs/data-sources.md).
    Скрипт делает всего ~4 запроса, но пауза оставлена для безопасности,
    если вы добавите свои проверки поверх.
    """
    time.sleep(1.1 if not has_key else 0.25)


def main() -> int:
    api_key = os.environ.get("OPENDOTA_API_KEY")
    sample_match_id_env = os.environ.get("SAMPLE_MATCH_ID")
    sample_match_id = int(sample_match_id_env) if sample_match_id_env else None

    if CONTACT_EMAIL == "REPLACE_ME@example.com":
        print(
            "ВНИМАНИЕ: CONTACT_EMAIL не задан — это ок для OpenDota, но обязательно "
            "для Liquipedia API (см. api-terms-of-use). Проверки Liquipedia в этой "
            "версии скрипта не реализованы намеренно (нужен явный контакт в UA).",
            file=sys.stderr,
        )

    checks = []
    checks.append(check_opendota_status(api_key))
    rate_limit_sleep(bool(api_key))
    checks.append(check_promatches_format(api_key))
    rate_limit_sleep(bool(api_key))
    checks.append(check_historical_coverage(api_key))
    rate_limit_sleep(bool(api_key))
    checks.append(check_picks_bans_sample(api_key, sample_match_id))

    report = {
        "generated_by": "scripts/verify_data_source.py",
        "opendota_api_key_used": bool(api_key),
        "checks": [asdict(c) for c in checks],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    failed = [c for c in checks if c.status == "FAILED"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
