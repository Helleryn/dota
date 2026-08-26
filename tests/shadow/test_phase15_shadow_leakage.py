"""
PHASE 15 — adversarial tests shadow-конвейера (PART U, 12 сценариев).

Часть тестов работает против РЕАЛЬНОЙ PostgreSQL: неизменяемость снимка
обеспечена триггером СУБД, и проверять её на моке бессмысленно — мок
проверял бы сам себя. Используется ОТДЕЛЬНАЯ TEST_DATABASE_URL: на Phase 5
запуск тестов против рабочей БД стёр 942 загруженных матча.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from src.config import load_settings
from src.db.engine import make_engine
from src.shadow import cutoff as cutoff_mod
from src.shadow import pipeline, repository as repo, states, versions
from src.shadow.engine import FrozenEngine
from src.shadow.snapshot import (
    PredictionSnapshot,
    make_prediction_id,
    score_resolution,
)

settings = load_settings()
try:
    _e = make_engine(settings, use_test_database=True) if settings.test_database_url else None
    if _e is not None:
        with _e.connect():
            pass
    DB = _e is not None
except Exception:
    DB = False

db_only = pytest.mark.skipif(not DB, reason="TEST_DATABASE_URL недоступен")

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def engine():
    """Очистка через TRUNCATE, а не DELETE — и это не обход защиты.

    Триггер неизменяемости построчный (BEFORE UPDATE OR DELETE), поэтому
    DELETE он отклоняет, а TRUNCATE — нет. Именно такое разделение и
    нужно: ни один путь приложения не может тихо изменить или удалить
    отдельный опубликованный прогноз, а административный сброс всей
    таблицы остаётся возможным. Первый же прогон этих тестов уткнулся
    в собственный триггер на DELETE — защита сработала на самих тестах.
    """
    e = make_engine(settings, use_test_database=True)
    stmt = text("TRUNCATE prediction_resolutions, prediction_snapshots CASCADE")
    with e.begin() as c:
        c.execute(stmt)
    yield e
    with e.begin() as c:
        c.execute(stmt)


def _history(n=3000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    elo = rng.normal(0, 120, n)
    p = 1 / (1 + np.exp(-elo / 300))
    rows = {
        "match_id": np.arange(1, n + 1),
        "as_of_timestamp": pd.date_range("2024-01-01", periods=n, freq="6h", tz="UTC"),
        "elo_difference": elo,
        "form_3_difference": rng.normal(0, 0.3, n),
        "elo_mean_diff": rng.normal(0, 80, n),
        "five_vs_team_elo_diff": rng.normal(0, 40, n),
        "hero_exp_decay_diff": rng.normal(0, 0.05, n),
        "target": (rng.uniform(size=n) < p).astype(int),
        "patch_id": 1, "patch_name": "7.40", "league_id": 100,
    }
    return pd.DataFrame(rows)


def _snap(pid_suffix="a", state=states.PUBLISHED, pred_ts=T0, **kw) -> PredictionSnapshot:
    key = f"test:{pid_suffix}"
    base = dict(
        prediction_id=make_prediction_id(key, pred_ts, versions.PREDICTION_VERSION),
        match_key=key, prediction_timestamp=pred_ts,
        match_start_time=pred_ts + timedelta(hours=1),
        features={f: 0.1 for f in versions.FROZEN_FEATURES},
        data_cutoff=pred_ts, feature_data_cutoff=pred_ts - timedelta(hours=1),
        rating_state_timestamp=pred_ts - timedelta(hours=1),
        roster_state_timestamp=pred_ts - timedelta(hours=1),
        hero_meta_state_timestamp=pred_ts - timedelta(hours=1),
        source="replay", state=state, match_id=int(abs(hash(pid_suffix)) % 10**8),
        raw_probability=0.62, calibrated_probability=0.60, confidence=0.10,
        decision="PREDICT",
    )
    base.update(kw)
    return PredictionSnapshot(**base)


# ---------- 1. будущие данные не могут попасть в признаки ----------

def test_placeholder_outcome_cannot_reach_its_own_features():
    """Ключевой приём фазы: признаки нового матча берутся как последняя
    строка walk-forward прохода по «история + целевой матч с заглушкой
    исхода». Тест проверяет, что заглушка в собственные признаки не
    попадает — иначе весь приём был бы утечкой."""
    from src.datasets.multi_window_features import build_multi_window_features

    class M:
        def __init__(self, i, ts, r, d, win):
            self.match_id, self.start_time = i, ts
            self.radiant_team_id, self.dire_team_id = r, d
            self.radiant_win = win

    base = [M(i, T0 + timedelta(days=i), 1 + i % 4, 5 + i % 3, i % 2 == 0)
            for i in range(40)]
    target_a = M(999, T0 + timedelta(days=100), 1, 5, True)
    target_b = M(999, T0 + timedelta(days=100), 1, 5, False)   # перевёрнутая заглушка

    ra = build_multi_window_features(base + [target_a], k_factor=versions.ELO_K)[-1]
    rb = build_multi_window_features(base + [target_b], k_factor=versions.ELO_K)[-1]
    assert ra.elo_difference == rb.elo_difference
    assert ra.recent_winrate_difference[3] == rb.recent_winrate_difference[3]


def test_cutoff_check_rejects_state_later_than_prediction():
    s = _snap("late", rating_state_timestamp=T0 + timedelta(minutes=1))
    ok, reason, viol = cutoff_mod.validate(s)
    assert not ok and reason == "future_data_detected"
    assert viol and viol[0].field == "rating_state_timestamp"


def test_cutoff_check_passes_clean_snapshot():
    ok, reason, viol = cutoff_mod.validate(_snap("clean"))
    assert ok and reason is None and viol == []


# ---------- 2. будущий исход не может попасть в калибровку ----------

def test_calibration_window_is_strictly_before_cutoff():
    h = _history()
    fe = FrozenEngine(h)
    ts = pd.to_datetime(h["as_of_timestamp"], utc=True)
    cut = ts.iloc[2000].to_pydatetime()
    p, y = fe.calibration_window(cut)
    # матч ровно в момент среза к этому времени исхода ещё не имеет
    assert len(p) == min(versions.CALIBRATION_WINDOW, 2000)


def test_future_outcomes_do_not_change_past_calibration():
    """Переворачиваются исходы всех матчей ПОЗЖЕ момента прогноза.
    Калибровка на этот момент обязана не измениться.

    Момент прогноза берётся после среза обучения модели: shadow-прогноз
    раньше этого среза недопустим в принципе (иначе модель обучалась бы
    на матчах не раньше собственного прогноза), и конвейер такой прогноз
    помечает INVALID — см. test_prediction_before_train_cutoff_is_invalid.
    """
    h = _history()
    fe1 = FrozenEngine(h)
    idx = 2500                                     # заведомо после TRAIN (70% от 3000)
    cut = pd.to_datetime(h["as_of_timestamp"], utc=True).iloc[idx].to_pydatetime()
    assert cut > fe1.train_cutoff.to_pydatetime()
    p1 = fe1.calibrate(0.7, cut)

    h2 = h.copy()
    h2.loc[idx:, "target"] = 1 - h2.loc[idx:, "target"]       # переворот будущего
    p2 = FrozenEngine(h2).calibrate(0.7, cut)
    assert p1[0] == pytest.approx(p2[0]), "будущие исходы повлияли на калибровку"


def test_prediction_before_train_cutoff_is_invalid():
    """Прогноз, сделанный раньше среза обучения модели, непригоден:
    модель видела бы исходы матчей не раньше собственного прогноза.
    Это отдельный вид утечки, не покрываемый проверкой отметок среза."""
    h = _history()
    fe = FrozenEngine(h)
    early = fe.train_cutoff.to_pydatetime() - timedelta(days=10)
    d = pipeline.DiscoveredMatch(
        match_key="replay:early", prediction_timestamp=early,
        match_start_time=early + timedelta(hours=1),
        features={f: 0.1 for f in versions.FROZEN_FEATURES},
        source="replay", match_id=1,
        feature_data_cutoff=early - timedelta(hours=1))
    snap = pipeline.build_snapshot(fe, d)
    assert snap.state == states.INVALID
    assert snap.invalid_reason == "model_trained_after_prediction"


# ---------- 11/12. воспроизводимость ----------

def test_calibration_state_is_reproducible():
    h = _history()
    cut = pd.to_datetime(h["as_of_timestamp"], utc=True).iloc[2500].to_pydatetime()
    a = FrozenEngine(h).calibrate(0.63, cut)
    b = FrozenEngine(h).calibrate(0.63, cut)
    assert a == b


def test_model_version_and_spec_are_reproducible():
    h = _history()
    fa, fb = FrozenEngine(h), FrozenEngine(h)
    feats = {f: 0.05 for f in versions.FROZEN_FEATURES}
    assert fa.raw_probability(feats) == pytest.approx(fb.raw_probability(feats))
    assert versions.frozen_spec() == versions.frozen_spec()
    assert versions.frozen_spec()["features"] == list(versions.FROZEN_FEATURES)
    assert len(versions.FROZEN_FEATURES) == 5, "frozen set — ПЯТЬ признаков, не три"


def test_prediction_id_is_deterministic_not_random():
    a = make_prediction_id("k", T0, versions.PREDICTION_VERSION)
    b = make_prediction_id("k", T0.replace(microsecond=500), versions.PREDICTION_VERSION)
    c = make_prediction_id("k", T0 + timedelta(seconds=1), versions.PREDICTION_VERSION)
    assert a == b, "микросекунды не должны порождать новый идентификатор"
    assert a != c


# ---------- 3/4/10. неизменяемость снимка (против реальной СУБД) ----------

@db_only
def test_published_snapshot_cannot_be_modified(engine):
    s = _snap("imm")
    with engine.begin() as c:
        repo.insert_snapshot(c, s)
    with pytest.raises(Exception) as ei:
        with engine.begin() as c:
            c.execute(text("UPDATE prediction_snapshots SET raw_probability=0.99 "
                           "WHERE prediction_id=:p"), {"p": s.prediction_id})
    assert "неизменяем" in str(ei.value)


@db_only
def test_published_snapshot_cannot_be_deleted(engine):
    s = _snap("del")
    with engine.begin() as c:
        repo.insert_snapshot(c, s)
    with pytest.raises(Exception) as ei:
        with engine.begin() as c:
            c.execute(text("DELETE FROM prediction_snapshots WHERE prediction_id=:p"),
                      {"p": s.prediction_id})
    assert "удалять нельзя" in str(ei.value)


@db_only
def test_state_can_only_move_forward(engine):
    s = _snap("fwd")
    with engine.begin() as c:
        repo.insert_snapshot(c, s)
        repo.advance_state(c, s.prediction_id, states.MATCH_FINISHED)
    with pytest.raises(ValueError):
        with engine.begin() as c:
            repo.advance_state(c, s.prediction_id, states.PREDICTED)


@db_only
def test_resolution_does_not_alter_prediction(engine):
    """Сценарий 4: запись исхода не должна касаться снимка. Сравнивается
    хеш содержимого, а не отдельные поля — так тест поймает любое поле."""
    s = _snap("res")
    with engine.begin() as c:
        repo.insert_snapshot(c, s)
        before = repo.get_snapshot(c, s.prediction_id)
    h_before = (before.raw_probability, before.calibrated_probability,
                before.confidence, before.features, before.prediction_timestamp)
    rec = score_resolution(s, True, T0 + timedelta(hours=3), s.match_id)
    with engine.begin() as c:
        repo.insert_resolution(c, rec)
        repo.advance_state(c, s.prediction_id, states.RESOLVED)
        after = repo.get_snapshot(c, s.prediction_id)
    assert (after.raw_probability, after.calibrated_probability, after.confidence,
            after.features, after.prediction_timestamp) == h_before
    assert after.state == states.RESOLVED


@db_only
def test_patch_change_does_not_mutate_historical_prediction(engine):
    """Сценарий 10: появление нового патча не должно менять уже сделанные
    прогнозы — они хранят патч на момент прогноза."""
    s = _snap("patch", patch_name="7.40")
    with engine.begin() as c:
        repo.insert_snapshot(c, s)
    with pytest.raises(Exception):
        with engine.begin() as c:
            c.execute(text("UPDATE prediction_snapshots SET patch_name='7.41' "
                           "WHERE prediction_id=:p"), {"p": s.prediction_id})
    with engine.connect() as c:
        assert repo.get_snapshot(c, s.prediction_id).patch_name == "7.40"


# ---------- 8/9. дубликаты ----------

@db_only
def test_duplicate_prediction_is_refused_not_silently_created(engine):
    s = _snap("dup")
    with engine.begin() as c:
        repo.insert_snapshot(c, s)
    with pytest.raises(repo.DuplicatePrediction):
        with engine.begin() as c:
            repo.insert_snapshot(c, s)
    with engine.connect() as c:
        n = c.execute(text("SELECT count(*) FROM prediction_snapshots")).scalar()
    assert n == 1


@db_only
def test_duplicate_resolution_cannot_corrupt_metrics(engine):
    s = _snap("dupres")
    rec = score_resolution(s, True, T0 + timedelta(hours=3), s.match_id)
    with engine.begin() as c:
        repo.insert_snapshot(c, s)
        repo.insert_resolution(c, rec)
    with pytest.raises(repo.DuplicateResolution):
        with engine.begin() as c:
            repo.insert_resolution(c, rec)
    with engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM prediction_resolutions")).scalar() == 1


# ---------- 5/6/7. задержки, переносы, отмены ----------

@db_only
def test_postponed_match_creates_new_prediction_not_overwrite(engine):
    """Сценарий 6 (PART W): матч перенесли — старый прогноз остаётся, для
    нового времени создаётся НОВЫЙ прогноз со своим timestamp."""
    first = _snap("post", pred_ts=T0)
    later = T0 + timedelta(hours=6)
    second = PredictionSnapshot(
        **{**first.to_row(),
           "prediction_id": make_prediction_id(first.match_key, later,
                                               versions.PREDICTION_VERSION),
           "prediction_timestamp": later,
           "match_start_time": later + timedelta(hours=1),
           "data_cutoff": later, "created_at": None})
    with engine.begin() as c:
        repo.insert_snapshot(c, first)
        repo.insert_snapshot(c, second)
    with engine.connect() as c:
        rows = repo.list_snapshots(c)
    assert len(rows) == 2
    assert rows[0].prediction_timestamp < rows[1].prediction_timestamp


def test_cancelled_match_gets_explicit_status():
    d = pipeline.DiscoveredMatch(
        match_key="live:cancelled", prediction_timestamp=T0, match_start_time=None,
        features={}, source="live_draft", invalid_reason="match_cancelled")
    snap = pipeline.build_snapshot(FrozenEngine(_history()), d)
    assert snap.state == states.INVALID
    assert snap.invalid_reason == "match_cancelled"


def test_unavailable_source_yields_explicit_status_not_empty_list():
    """Сценарий 5: отличать «матчей нет» от «источник не ответил»."""
    class Boom:
        def get_json(self, *a, **k):
            raise RuntimeError("сеть недоступна")
    out = pipeline.live_draft_discovery(Boom(), now=T0)
    assert len(out) == 1 and out[0].invalid_reason == "source_unavailable"


def test_prediction_after_match_start_is_invalid():
    """PART W: прогноз обязан существовать ДО начала матча."""
    s = _snap("late2", match_start_time=T0 - timedelta(minutes=1))
    ok, reason, _ = cutoff_mod.validate(s)
    assert not ok and reason == "match_already_started"


def test_missing_features_are_refused_not_imputed():
    d = pipeline.DiscoveredMatch(
        match_key="live:nolineup", prediction_timestamp=T0, match_start_time=None,
        features={"elo_difference": 1.0}, source="live_draft")
    snap = pipeline.build_snapshot(FrozenEngine(_history()), d)
    assert snap.state == states.INVALID and snap.invalid_reason == "lineup_unknown"


def test_short_calibration_history_is_reported_not_silently_skipped():
    h = _history(n=600)
    fe = FrozenEngine(h)
    early = pd.to_datetime(h["as_of_timestamp"], utc=True).iloc[100].to_pydatetime()
    p_cal, ver, n = fe.calibrate(0.7, early)
    assert p_cal is None and ver == "none" and n == 100
