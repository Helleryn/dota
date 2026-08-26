"""
PHASE 15 — shadow-конвейер: обнаружение, прогноз, разрешение.

Режим работы — SHADOW (PART V): система получает данные, строит прогноз,
сохраняет его, позже получает исход и считает метрики. Никаких внешних
действий не совершается.

## Три источника обнаружения и что с ними на самом деле

| Источник | Состояние | Почему |
|---|---|---|
| `fixture` | **невозможен** | у OpenDota нет расписания: 404 на всех эндпоинтах фикстур, ноль матчей с будущим `start_time` в её базе |
| `live_draft` | реализован | матч, пойманный в `/live` во время драфта; в момент проверки про-матчей в выдаче не было вовсе |
| `replay` | реализован | тот же код по недавнему окну истории с искусственным «сейчас» |

Каждый прогноз несёт своё `source`, и результаты разных источников
никогда не складываются в одну метрику.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.shadow import cutoff as cutoff_mod
from src.shadow import repository as repo
from src.shadow import states, versions
from src.shadow.engine import FrozenEngine
from src.shadow.snapshot import (
    PredictionSnapshot,
    make_prediction_id,
    score_resolution,
)

DEFAULT_LEAD_MINUTES = 30


@dataclass(frozen=True)
class DiscoveredMatch:
    match_key: str
    prediction_timestamp: datetime
    match_start_time: Optional[datetime]
    features: Dict[str, float]
    source: str
    match_id: Optional[int] = None
    radiant_team_id: Optional[int] = None
    dire_team_id: Optional[int] = None
    radiant_team_name: Optional[str] = None
    dire_team_name: Optional[str] = None
    patch_id: Optional[int] = None
    patch_name: Optional[str] = None
    league_id: Optional[int] = None
    tournament: Optional[str] = None
    feature_data_cutoff: Optional[datetime] = None
    invalid_reason: Optional[str] = None


def replay_discovery(history: pd.DataFrame, since: datetime, until: datetime,
                     lead_minutes: int = DEFAULT_LEAD_MINUTES) -> Iterator[DiscoveredMatch]:
    """Обнаружение в режиме REPLAY.

    Прогноз выставляется на `lead_minutes` РАНЬШЕ начала матча, а
    `feature_data_cutoff` — время последнего матча строго раньше этого
    момента. Так воспроизводится ровно та ситуация, в которой окажется
    боевая система: признаки известны, исход — нет.
    """
    h = history.sort_values(["as_of_timestamp", "match_id"]).reset_index(drop=True)
    ts = pd.to_datetime(h["as_of_timestamp"], utc=True)
    sel = h[(ts >= pd.Timestamp(since)) & (ts < pd.Timestamp(until))]
    # numpy datetime64 несовместим с tz-aware сравнением, поэтому
    # хранится массив секунд от эпохи — величина, не зависящая от
    # разрешения столбца (та же ловушка, что сломала затухание в Phase 14).
    all_epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    all_ts_py = ts.dt.to_pydatetime()

    for _, row in sel.iterrows():
        start = pd.to_datetime(row["as_of_timestamp"], utc=True).to_pydatetime()
        pred_ts = start - timedelta(minutes=lead_minutes)
        # последний матч строго раньше момента прогноза
        pred_epoch = (pred_ts - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()
        idx = int(np.searchsorted(all_epoch, pred_epoch, side="left")) - 1
        fdc = all_ts_py[idx] if idx >= 0 else None
        feats = {k: float(row[k]) for k in versions.FROZEN_FEATURES}
        yield DiscoveredMatch(
            match_key=f"replay:{int(row['match_id'])}",
            prediction_timestamp=pred_ts,
            match_start_time=start,
            features=feats,
            source="replay",
            match_id=int(row["match_id"]),
            patch_id=(int(row["patch_id"]) if pd.notna(row.get("patch_id")) else None),
            patch_name=(row.get("patch_name") if pd.notna(row.get("patch_name")) else None),
            league_id=(int(row["league_id"]) if pd.notna(row.get("league_id")) else None),
            feature_data_cutoff=fdc,
        )


def live_draft_discovery(client, now: Optional[datetime] = None) -> List[DiscoveredMatch]:
    """Обнаружение в режиме LIVE-DRAFT.

    Берутся только матчи с `league_id` и `game_time <= 0` — то есть идёт
    драфт, игра ещё не началась. Матч, у которого игра уже пошла,
    сознательно пропускается: прогноз обязан существовать до начала
    (PART W).

    Возвращает список; при недоступности источника — запись с
    `invalid_reason='source_unavailable'`, а не пустой список, чтобы
    отличать «нет матчей» от «источник не ответил».
    """
    now = now or datetime.now(timezone.utc)
    try:
        live = client.get_json("/live", params={})
    except Exception:
        return [DiscoveredMatch(
            match_key=f"live:unavailable:{now.isoformat()}",
            prediction_timestamp=now, match_start_time=None, features={},
            source="live_draft", invalid_reason="source_unavailable")]

    out: List[DiscoveredMatch] = []
    for m in live or []:
        if not m.get("league_id"):
            continue
        if (m.get("game_time") or 0) > 0:
            continue          # игра уже началась
        out.append(DiscoveredMatch(
            match_key=f"live:{m.get('match_id') or m.get('lobby_id')}",
            prediction_timestamp=now,
            match_start_time=None,
            features={},
            source="live_draft",
            match_id=m.get("match_id"),
            radiant_team_id=m.get("team_id_radiant"),
            dire_team_id=m.get("team_id_dire"),
            radiant_team_name=m.get("team_name_radiant"),
            dire_team_name=m.get("team_name_dire"),
            league_id=m.get("league_id"),
            invalid_reason=None,
        ))
    return out


def build_snapshot(engine: FrozenEngine, d: DiscoveredMatch,
                   abstain_threshold: float = 0.0) -> PredictionSnapshot:
    """Собирает снимок. Все отказы получают явную причину — тихого
    fallback нет ни в одном случае (PART R)."""
    pid = make_prediction_id(d.match_key, d.prediction_timestamp,
                             versions.PREDICTION_VERSION)
    base = dict(
        prediction_id=pid, match_key=d.match_key,
        prediction_timestamp=d.prediction_timestamp,
        match_start_time=d.match_start_time, match_id=d.match_id,
        radiant_team_id=d.radiant_team_id, dire_team_id=d.dire_team_id,
        radiant_team_name=d.radiant_team_name, dire_team_name=d.dire_team_name,
        patch_id=d.patch_id, patch_name=d.patch_name,
        league_id=d.league_id, tournament=d.tournament,
        source=d.source, data_cutoff=d.prediction_timestamp,
        feature_data_cutoff=d.feature_data_cutoff,
        rating_state_timestamp=d.feature_data_cutoff,
        roster_state_timestamp=d.feature_data_cutoff,
        hero_meta_state_timestamp=d.feature_data_cutoff,
    )

    if d.invalid_reason:
        return PredictionSnapshot(features={}, state=states.INVALID,
                                  invalid_reason=d.invalid_reason, **base)
    # Отдельный вид утечки, который не ловится отметками среза: если
    # прогноз сделан раньше, чем заканчивается обучающая выборка модели,
    # то модель видела исходы матчей не раньше собственного прогноза.
    train_cut = getattr(engine, "train_cutoff", None)
    if train_cut is not None:
        tc = train_cut.to_pydatetime() if hasattr(train_cut, "to_pydatetime") else train_cut
        if d.prediction_timestamp <= tc:
            return PredictionSnapshot(features=dict(d.features), state=states.INVALID,
                                      invalid_reason="model_trained_after_prediction",
                                      **base)

    missing = [f for f in versions.FROZEN_FEATURES if f not in d.features]
    if missing:
        return PredictionSnapshot(features=dict(d.features), state=states.INVALID,
                                  invalid_reason="lineup_unknown", **base)

    res = engine.predict(d.features, d.prediction_timestamp,
                         abstain_threshold=abstain_threshold,
                         state_timestamps={
                             "feature_data_cutoff": d.feature_data_cutoff,
                             "rating_state_timestamp": d.feature_data_cutoff,
                             "roster_state_timestamp": d.feature_data_cutoff,
                             "hero_meta_state_timestamp": d.feature_data_cutoff})

    snap = PredictionSnapshot(
        features=dict(d.features), state=states.PUBLISHED,
        raw_probability=res.raw_probability,
        calibrated_probability=res.calibrated_probability,
        confidence=res.confidence, decision=res.decision,
        calibration_version=res.calibration_version, **base)

    ok, reason, _ = cutoff_mod.validate(snap)
    if not ok:
        # Прогноз, про который нельзя доказать отсутствие будущего,
        # непригоден целиком — он сохраняется как INVALID, а не чинится.
        return PredictionSnapshot(
            features=dict(d.features), state=states.INVALID,
            invalid_reason=reason, raw_probability=res.raw_probability,
            calibrated_probability=res.calibrated_probability,
            confidence=res.confidence, decision=res.decision,
            calibration_version=res.calibration_version, **base)
    return snap


def store_predictions(conn, snapshots: Iterable[PredictionSnapshot]) -> Dict[str, int]:
    stats = {"inserted": 0, "duplicate": 0, "invalid": 0}
    for s in snapshots:
        try:
            repo.insert_snapshot(conn, s)
            stats["inserted"] += 1
            if s.state == states.INVALID:
                stats["invalid"] += 1
        except repo.DuplicatePrediction:
            stats["duplicate"] += 1
    return stats


def resolve(conn, outcomes: Dict[int, Tuple[bool, Optional[datetime]]],
            now: Optional[datetime] = None,
            source: Optional[str] = None) -> Dict[str, int]:
    """Записывает исходы для неразрешённых снимков.

    `outcomes`: match_id -> (radiant_win, actual_start_time). Снимок при
    этом не меняется — пишется отдельная запись разрешения, а состояние
    переводится в RESOLVED.
    """
    now = now or datetime.now(timezone.utc)
    stats = {"resolved": 0, "skipped_no_outcome": 0, "skipped_invalid": 0,
             "duplicate": 0, "mismatched_start": 0}
    for row in repo.list_unresolved(conn, source=source):
        if row.state == states.INVALID:
            stats["skipped_invalid"] += 1
            continue
        if row.match_id is None or row.match_id not in outcomes:
            stats["skipped_no_outcome"] += 1
            continue
        won, actual_start = outcomes[row.match_id]
        # Проверка, что фактическое начало матча позже прогноза. Если матч
        # перенесли НАЗАД и он начался раньше — прогноз к нему неприменим.
        if actual_start is not None and actual_start <= row.prediction_timestamp:
            repo.advance_state(conn, row.prediction_id, states.INVALID)
            stats["mismatched_start"] += 1
            continue
        rec = score_resolution(row, won, now, int(row.match_id), actual_start)
        try:
            repo.insert_resolution(conn, rec)
        except repo.DuplicateResolution:
            stats["duplicate"] += 1
            continue
        repo.advance_state(conn, row.prediction_id, states.RESOLVED)
        stats["resolved"] += 1
    return stats
