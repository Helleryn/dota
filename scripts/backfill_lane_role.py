#!/usr/bin/env python3
"""
PHASE 11 — заполнение `match_players.lane_role` из OpenDota.

## Что это и чем это НЕ является

`lane_role` — линия, определённая OpenDota при разборе реплея:
1=safelane, 2=mid, 3=offlane, 4=jungle.

**Это POST-MATCH величина.** Она физически не существует до конца игры.
Поэтому:

* использовать `lane_role` ТЕКУЩЕГО матча как признак — прямая утечка,
  запрещено;
* использовать `lane_role` матчей СТРОГО РАНЬШЕ прогнозируемого, чтобы
  охарактеризовать игрока («на каких линиях он играл до сих пор») —
  законно, эта информация доступна к моменту прогноза.

Кроме того, это **линия, а не позиция 1-5**: в safelane выходят и керри
(поз.1), и хард-саппорт (поз.5); в offlane — оффлейнер (поз.3) и поз.4.
Позиция выводится дополнительно рангом GPM внутри пары одной линии
(см. src/datasets/role_features.py).

## Почему понадобилась загрузка

Аудит Phase 11.0 измерил: ранг GPM внутри пятёрки (данные уже были локально)
воспроизводит точную позицию лишь на **56.5%**, хотя ядро/саппорт разделяет
на 96-98%. При этом герой определяет ядро/саппорт на 94%, но точную позицию —
лишь на 71.5%. То есть точная позиция несёт информацию, которой нет ни в
GPM-ранге, ни в самом герое — её нельзя получить без `lane_role`.

Запуск:
    python3 scripts/backfill_lane_role.py --since 2021 --until 2027
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text

from src.config import load_settings
from src.datasources.http_client import HttpClientConfig, RateLimitedHttpClient
from src.db.engine import make_engine


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--since", type=int, default=2021)
    p.add_argument("--until", type=int, default=2027)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    engine = make_engine(settings)
    client = RateLimitedHttpClient(HttpClientConfig(
        base_url=settings.opendota_base_url,
        timeout_seconds=max(settings.request_timeout_seconds, 180.0),
        max_retries=settings.max_retries,
        requests_per_minute=settings.opendota_rate_limit_per_min,
    ))

    total = 0
    try:
        for year in range(args.since, args.until):
            sql = (
                "SELECT pm.match_id, pm.player_slot, pm.lane_role "
                "FROM player_matches pm JOIN matches m ON m.match_id = pm.match_id "
                "JOIN leagues l ON l.leagueid = m.leagueid "
                "WHERE l.tier IN ('professional','premium') "
                "AND pm.lane_role IS NOT NULL "
                f"AND m.start_time >= extract(epoch FROM date '{year}-01-01') "
                f"AND m.start_time <  extract(epoch FROM date '{year + 1}-01-01')"
            )
            payload = client.get_json("/explorer", params={"sql": sql})
            if payload.get("err"):
                raise ValueError(f"/explorer error ({year}): {payload['err']}")
            rows = payload.get("rows", [])
            print(f"{year}: получено {len(rows)} строк", flush=True)

            batch = [{"mid": r["match_id"], "slot": r["player_slot"], "lr": r["lane_role"]} for r in rows]
            with engine.begin() as conn:
                for i in range(0, len(batch), 5000):
                    conn.execute(
                        text("UPDATE match_players SET lane_role = :lr "
                             "WHERE match_id = :mid AND player_slot = :slot"),
                        batch[i:i + 5000],
                    )
            total += len(batch)
    finally:
        client.close()

    with engine.connect() as conn:
        filled = conn.execute(text("SELECT count(*) FROM match_players WHERE lane_role IS NOT NULL")).scalar()
        allrows = conn.execute(text("SELECT count(*) FROM match_players")).scalar()
    print(f"\nОбработано {total}; lane_role заполнен у {filled} из {allrows} ({filled/allrows*100:.1f}%)")
    print("НАПОМИНАНИЕ: lane_role — post-match. Только для истории (< t), никогда для текущего матча.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
