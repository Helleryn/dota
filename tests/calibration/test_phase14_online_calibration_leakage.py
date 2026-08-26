"""
PHASE 14 — adversarial leakage tests для онлайн-слоя калибровки.

Слой калибровки — самое удобное место для утечки во всём проекте:
«обучить калибровку на всей выборке» даёт прекрасный ECE и не значит
ничего. Поэтому проверки здесь — условие осмысленности всей фазы, а не
формальность.
"""

import numpy as np
import pytest

from src.evaluation.calibration import expected_calibration_error, calibration_slope_intercept
from src.evaluation.calibrators import (
    BetaCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
)
from src.evaluation.online_calibration import run_walk_forward, rolling_drift_metrics

DAY = 86400.0


def _stream(n=4000, seed=0, distort=2.0):
    """Поток прогнозов, СИСТЕМАТИЧЕСКИ переуверенных: логиты растянуты в
    `distort` раз. Исход генерируется по ИСТИННОЙ вероятности, поэтому
    правильный калибратор обязан этот поток чинить."""
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(0.15, 0.85, size=n)
    y = (rng.uniform(size=n) < p_true).astype(int)
    z = np.log(p_true / (1 - p_true))
    p_raw = 1 / (1 + np.exp(-distort * z))
    ts = np.arange(n, dtype=float) * DAY / 4.0
    patch = np.array([f"p{int(i // 800)}" for i in range(n)], dtype=object)
    return p_raw, y, ts, patch


def _split(p, y, ts, patch, k):
    return (p[:k], y[:k], ts[:k], patch[:k]), (p[k:], y[k:], ts[k:], patch[k:])


# ---------- S1/S9: прогноз не зависит от собственного исхода ----------

def test_future_labels_do_not_change_past_predictions():
    """Перемешиваем метки ВСЕХ матчей после k-го. Калиброванные прогнозы
    первых k обязаны совпасть до последнего знака."""
    p, y, ts, patch = _stream()
    warm, ev = _split(p, y, ts, patch, 1500)
    base = run_walk_forward(*ev, mode="decay", method="platt", half_life=30,
                            warm_p=warm[0], warm_y=warm[1], warm_ts=warm[2], warm_patch=warm[3])
    rng = np.random.default_rng(7)
    y2 = ev[1].copy()
    k = 1000
    y2[k:] = rng.permutation(y2[k:])
    after = run_walk_forward(ev[0], y2, ev[2], ev[3], mode="decay", method="platt",
                             half_life=30, warm_p=warm[0], warm_y=warm[1],
                             warm_ts=warm[2], warm_patch=warm[3])
    assert np.allclose(base[:k], after[:k], atol=0, rtol=0), \
        "УТЕЧКА: будущие метки повлияли на прошлые калиброванные прогнозы"


def test_own_outcome_does_not_affect_own_prediction():
    """Переворачиваем исход ОДНОГО матча — его собственный калиброванный
    прогноз меняться не должен."""
    p, y, ts, patch = _stream()
    warm, ev = _split(p, y, ts, patch, 1500)
    base = run_walk_forward(*ev, mode="rolling", method="platt", window=1000,
                            warm_p=warm[0], warm_y=warm[1], warm_ts=warm[2], warm_patch=warm[3])
    idx = 900
    y2 = ev[1].copy()
    y2[idx] = 1 - y2[idx]
    after = run_walk_forward(ev[0], y2, ev[2], ev[3], mode="rolling", method="platt",
                             window=1000, warm_p=warm[0], warm_y=warm[1],
                             warm_ts=warm[2], warm_patch=warm[3])
    assert base[idx] == after[idx]
    assert np.allclose(base[:idx + 1], after[:idx + 1])


# ---------- S2: prefix-stability ----------

@pytest.mark.parametrize("mode,kw", [
    ("frozen", {}),
    ("rolling", {"window": 1000}),
    ("decay", {"half_life": 30}),
    ("patch_local", {"half_life": 30}),
])
def test_prefix_stability_all_modes(mode, kw):
    """Формальное определение отсутствия утечки: прогнозы на префиксе
    тождественны соответствующей части прогнозов на полном потоке."""
    p, y, ts, patch = _stream()
    warm, ev = _split(p, y, ts, patch, 1500)
    full = run_walk_forward(*ev, mode=mode, method="platt",
                            warm_p=warm[0], warm_y=warm[1], warm_ts=warm[2],
                            warm_patch=warm[3], **kw)
    k = 1200
    pref = run_walk_forward(ev[0][:k], ev[1][:k], ev[2][:k], ev[3][:k], mode=mode,
                            method="platt", warm_p=warm[0], warm_y=warm[1],
                            warm_ts=warm[2], warm_patch=warm[3], **kw)
    assert np.allclose(full[:k], pref), f"режим {mode}: префикс не совпал"


def test_appending_future_matches_changes_nothing():
    p, y, ts, patch = _stream(n=3000)
    warm, ev = _split(p, y, ts, patch, 1200)
    short = run_walk_forward(ev[0][:800], ev[1][:800], ev[2][:800], ev[3][:800],
                             mode="decay", method="platt", half_life=60,
                             warm_p=warm[0], warm_y=warm[1], warm_ts=warm[2], warm_patch=warm[3])
    long = run_walk_forward(*ev, mode="decay", method="platt", half_life=60,
                            warm_p=warm[0], warm_y=warm[1], warm_ts=warm[2], warm_patch=warm[3])
    assert np.allclose(short, long[:800])


# ---------- слой обязан РАБОТАТЬ, иначе тесты на утечку тривиальны ----------

def test_calibration_actually_fixes_a_broken_stream():
    """Обратный контроль. Если бы слой ничего не делал, все проверки на
    утечку проходили бы тождественно."""
    p, y, ts, patch = _stream(distort=2.5)
    warm, ev = _split(p, y, ts, patch, 1500)
    raw_ece = expected_calibration_error(ev[1], ev[0])
    cal = run_walk_forward(*ev, mode="decay", method="platt", half_life=90,
                           warm_p=warm[0], warm_y=warm[1], warm_ts=warm[2], warm_patch=warm[3])
    cal_ece = expected_calibration_error(ev[1], cal)
    assert raw_ece > 0.05, "поток должен быть заметно сломан"
    assert cal_ece < raw_ece / 2, f"слой не починил поток: {raw_ece:.4f} -> {cal_ece:.4f}"


def test_calibration_recovers_slope_towards_one():
    p, y, ts, patch = _stream(distort=2.5)
    warm, ev = _split(p, y, ts, patch, 1500)
    assert calibration_slope_intercept(ev[1], ev[0])["slope"] < 0.6
    cal = run_walk_forward(*ev, mode="decay", method="platt", half_life=90,
                           warm_p=warm[0], warm_y=warm[1], warm_ts=warm[2], warm_patch=warm[3])
    assert calibration_slope_intercept(ev[1], cal)["slope"] == pytest.approx(1.0, abs=0.15)


def test_mode_none_returns_raw_untouched():
    p, y, ts, patch = _stream(n=1000)
    out = run_walk_forward(p, y, ts, patch, mode="none")
    assert np.array_equal(out, p)


# ---------- монотонность: калибровка не должна ломать ранжирование ----------

def _auc(y, s):
    import pandas as pd
    y = np.asarray(y, dtype=int); s = np.asarray(s, dtype=float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@pytest.mark.parametrize("method", ["platt", "temperature", "beta"])
def test_parametric_calibration_preserves_ranking(method):
    """Строго монотонное преобразование не может изменить ROC-AUC.
    Если изменило — в коде ошибка."""
    p, y, ts, patch = _stream()
    warm, ev = _split(p, y, ts, patch, 1500)
    cal = run_walk_forward(*ev, mode="frozen", method=method,
                           warm_p=warm[0], warm_y=warm[1], warm_ts=warm[2], warm_patch=warm[3])
    assert _auc(ev[1], cal) == pytest.approx(_auc(ev[1], ev[0]), abs=1e-9)


def test_isotonic_may_change_ranking_and_that_is_expected():
    """Изотоническая регрессия кусочно-постоянна и склеивает разные
    вероятности. Тест фиксирует это как известное свойство, а не как баг."""
    p, y, ts, patch = _stream()
    warm, ev = _split(p, y, ts, patch, 1500)
    cal = run_walk_forward(*ev, mode="frozen", method="isotonic",
                           warm_p=warm[0], warm_y=warm[1], warm_ts=warm[2], warm_patch=warm[3])
    assert len(np.unique(cal)) < len(np.unique(ev[0])), \
        "изотоническая калибровка обязана склеивать значения"


# ---------- частота переобучения слоя ----------

def test_refit_frequency_barely_matters():
    """Обоснование выбранного по умолчанию refit_every=10. Пороги взяты по
    измерению (см. докстринг модуля), а не назначены на глаз: отдельный
    прогноз отклоняется не более чем на 1.6 пп, ECE — на 0.0021."""
    p, y, ts, patch = _stream(n=2500)
    warm, ev = _split(p, y, ts, patch, 1200)
    kw = dict(mode="decay", method="platt", half_life=60, warm_p=warm[0],
              warm_y=warm[1], warm_ts=warm[2], warm_patch=warm[3])
    a = run_walk_forward(*ev, refit_every=1, **kw)
    b = run_walk_forward(*ev, refit_every=10, **kw)
    assert np.max(np.abs(a - b)) < 0.016
    assert abs(expected_calibration_error(ev[1], a)
               - expected_calibration_error(ev[1], b)) < 0.003


def test_coarser_refit_degrades_monotonically():
    """Контроль к предыдущему тесту: если бы частота пересчёта не влияла
    вовсе, выбор refit_every был бы бессмысленным."""
    p, y, ts, patch = _stream(n=2500)
    warm, ev = _split(p, y, ts, patch, 1200)
    kw = dict(mode="decay", method="platt", half_life=60, warm_p=warm[0],
              warm_y=warm[1], warm_ts=warm[2], warm_patch=warm[3])
    ref = run_walk_forward(*ev, refit_every=1, **kw)
    devs = [np.max(np.abs(run_walk_forward(*ev, refit_every=r, **kw) - ref))
            for r in (5, 10, 25, 50)]
    assert devs == sorted(devs), "отклонение обязано расти с разрежением пересчёта"


# ---------- холодный старт ----------

def test_no_history_means_no_calibration_not_garbage():
    """Без истории слой обязан вернуть сырой прогноз, а не выдумать
    отображение по трём матчам."""
    p, y, ts, patch = _stream(n=300)
    out = run_walk_forward(p, y, ts, patch, mode="decay", method="platt",
                           half_life=30, min_history=500)
    assert np.array_equal(out, p)


# ---------- S6/S7: детектор дрейфа смотрит только назад ----------

def test_drift_metrics_use_only_past_matches():
    p, y, ts, patch = _stream(n=3000)
    base = rolling_drift_metrics(p, y, window=500, step=250)
    y2 = y.copy()
    y2[2000:] = 1 - y2[2000:]
    after = rolling_drift_metrics(p, y2, window=500, step=250)
    for a, b in zip(base, after):
        if a["end_index"] <= 2000:
            assert a == b, "детектор дрейфа увидел будущее"


def test_drift_metrics_detect_a_real_break():
    """Обратный контроль к предыдущему тесту."""
    p, y, ts, patch = _stream(n=4000)
    y2 = y.copy()
    y2[2000:] = 1 - y2[2000:]
    m = rolling_drift_metrics(p, y2, window=500, step=250)
    before = [x["ece"] for x in m if x["end_index"] <= 2000]
    after = [x["ece"] for x in m if x["end_index"] >= 2600]
    assert max(after) > max(before) * 2, "перелом не замечен детектором"


# ---------- поведение отдельных калибраторов ----------

def test_calibrators_refuse_to_fit_on_tiny_samples():
    for cls in (PlattCalibrator, TemperatureCalibrator, BetaCalibrator, IsotonicCalibrator):
        c = cls().fit(np.array([0.4, 0.6]), np.array([0, 1]))
        assert not c.fitted
        assert np.allclose(c.transform(np.array([0.3])), np.array([0.3]))


def test_calibrators_refuse_single_class():
    p = np.linspace(0.2, 0.8, 500)
    for cls in (PlattCalibrator, TemperatureCalibrator, BetaCalibrator, IsotonicCalibrator):
        assert not cls().fit(p, np.ones(500)).fitted


def test_weights_shift_the_fit_towards_weighted_subset():
    """Веса обязаны влиять: без этого экспоненциальное затухание было бы
    декоративным."""
    rng = np.random.default_rng(3)
    p = rng.uniform(0.2, 0.8, 4000)
    y_a = (rng.uniform(size=4000) < p).astype(int)              # калибровано
    z = np.log(p / (1 - p))
    y_b = (rng.uniform(size=4000) < 1 / (1 + np.exp(-2 * z))).astype(int)  # нет
    P = np.concatenate([p, p]); Y = np.concatenate([y_a, y_b])
    w_first = np.concatenate([np.ones(4000), np.full(4000, 0.01)])
    w_second = np.concatenate([np.full(4000, 0.01), np.ones(4000)])
    b1 = PlattCalibrator().fit(P, Y, w_first).b
    b2 = PlattCalibrator().fit(P, Y, w_second).b
    assert abs(b1 - 1.0) < abs(b2 - 1.0), "веса не сместили подгонку"
