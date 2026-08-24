"""
Enrichment: patch_id по start_time (Phase 5, раздел 11; ADR-003/database-design.md).

Патч — наш собственный справочник (odota/dotaconstants, вендорено в
src/normalization/data/), не приходит естественно от источника (OpenDota
отдаёт свой internal patch index, но мы используем собственный расчёт для
полного контроля, см. docs/data-pipeline.md, раздел Enrichment).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, TypedDict

_DATA_DIR = Path(__file__).parent / "data"


class PatchInfo(TypedDict):
    id: int
    name: str
    date: str


@lru_cache(maxsize=1)
def _load_patches() -> List[PatchInfo]:
    data = json.loads((_DATA_DIR / "patches.json").read_text())
    return sorted(data, key=lambda p: p["date"])


def resolve_patch_id(start_time: datetime) -> Optional[int]:
    """
    Патч, действовавший на момент start_time. None, если start_time раньше
    самого первого известного патча (некорректная/очень старая дата — не
    подменяется выдуманным значением, Phase 5 раздел 9: "если неизвестен —
    NULL, а не fake value").
    """
    candidate: Optional[int] = None
    for patch in _load_patches():
        released_at = datetime.fromisoformat(patch["date"].replace("Z", "+00:00"))
        if released_at.tzinfo is None:
            released_at = released_at.replace(tzinfo=timezone.utc)
        if released_at <= start_time:
            candidate = patch["id"]
        else:
            break
    return candidate


def load_heroes_reference() -> dict:
    return json.loads((_DATA_DIR / "heroes.json").read_text())


def load_patches_reference() -> List[PatchInfo]:
    return _load_patches()
