"""
PHASE 16 — разрешение противоречий между источниками (PART N).

Правило фазы: **не выбирать молча.** Если источники расходятся, выбор
делается по объявленной политике, а неразрешённый конфликт становится
видимой причиной понизить доверие к данным — вплоть до отказа от
прогноза.

Порядок правил (сверху вниз):

1. **официальный источник** — HIGH побеждает MEDIUM и LOW;
2. **подтверждение несколькими источниками** — при равном доверии
   выигрывает вариант, который назвали больше независимых источников;
3. **свежесть наблюдения** — при прочих равных выигрывает более позднее
   `observed_at`;
4. **явный отказ** — если и это не разводит варианты, возвращается
   `UNKNOWN` вместе с описанием конфликта.

Четвёртый пункт существеннее первых трёх: он превращает «система молча
выбрала одно из двух» в «система знает, что не знает».
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from src.sources.base import SourceConfidence
from src.sources.models import RosterMembership
from src.sources.temporal import CONFIDENCE_RANK


@dataclass(frozen=True)
class Resolution:
    resolved: Optional[List[RosterMembership]]
    rule: str
    conflict: Optional[str] = None

    @property
    def is_unknown(self) -> bool:
        return self.resolved is None


def _signature(ms: Sequence[RosterMembership]) -> Tuple[str, ...]:
    return tuple(sorted(m.player_id for m in ms))


def resolve_roster(candidates: Dict[str, List[RosterMembership]]) -> Resolution:
    """candidates: источник -> предложенный им состав.

    Возвращает выбранный состав и правило, по которому он выбран, либо
    `UNKNOWN` с описанием конфликта.
    """
    live = {s: ms for s, ms in candidates.items() if ms}
    if not live:
        return Resolution(None, "нет данных", "ни один источник не дал состава")
    if len(live) == 1:
        s, ms = next(iter(live.items()))
        return Resolution(ms, f"единственный источник: {s}")

    by_sig: Dict[Tuple[str, ...], List[Tuple[str, List[RosterMembership]]]] = defaultdict(list)
    for s, ms in live.items():
        by_sig[_signature(ms)].append((s, ms))

    if len(by_sig) == 1:
        s, ms = next(iter(live.items()))
        return Resolution(ms, f"источники согласны ({len(live)})")

    # 1. официальный источник
    best_conf = max(CONFIDENCE_RANK[m[1][0].provenance.confidence]
                    for m in ((s, ms) for s, ms in live.items()))
    top = {s: ms for s, ms in live.items()
           if CONFIDENCE_RANK[ms[0].provenance.confidence] == best_conf}
    if len({_signature(ms) for ms in top.values()}) == 1:
        s, ms = next(iter(top.items()))
        return Resolution(ms, f"высшее доверие источника: {s}")

    # 2. подтверждение несколькими источниками
    counts = {sig: len(v) for sig, v in by_sig.items()}
    mx = max(counts.values())
    winners = [sig for sig, c in counts.items() if c == mx]
    if mx > 1 and len(winners) == 1:
        s, ms = by_sig[winners[0]][0]
        return Resolution(ms, f"подтверждено источниками: {mx}")

    # 3. свежесть наблюдения
    def observed(ms: List[RosterMembership]) -> datetime:
        return max(m.provenance.observed_at for m in ms)
    freshest = sorted(top.items() or live.items(), key=lambda kv: observed(kv[1]))[-1]
    times = {observed(ms) for ms in (top or live).values()}
    if len(times) > 1:
        return Resolution(freshest[1], f"самое свежее наблюдение: {freshest[0]}")

    # 4. явный отказ
    desc = "; ".join(f"{s}={'/'.join(sorted(m.player_id for m in ms))}"
                     for s, ms in sorted(live.items()))
    return Resolution(None, "не разрешено", f"источники расходятся: {desc}")
