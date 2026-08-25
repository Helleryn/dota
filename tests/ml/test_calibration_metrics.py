"""PHASE 13 — тесты метрик калибровки на данных с ИЗВЕСТНЫМ ответом."""

import numpy as np
import pytest

from src.evaluation.calibration import (
    calibration_report,
    calibration_slope_intercept,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_curve,
)


def _perfect(n=40000, seed=0):
    """Идеально откалиброванные прогнозы: исход генерируется САМИМ прогнозом."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, size=n)
    y = (rng.uniform(size=n) < p).astype(int)
    return y, p


def test_perfect_calibration_gives_near_zero_ece():
    y, p = _perfect()
    assert expected_calibration_error(y, p) < 0.01


def test_perfect_calibration_gives_slope_one_intercept_zero():
    y, p = _perfect()
    si = calibration_slope_intercept(y, p)
    assert si["slope"] == pytest.approx(1.0, abs=0.05)
    assert si["intercept"] == pytest.approx(0.0, abs=0.05)


def test_overconfident_model_has_slope_below_one():
    """Растягиваем логиты вдвое — прогнозы становятся слишком крайними.
    Наклон обязан упасть заметно ниже 1, иначе метрика бесполезна."""
    y, p = _perfect()
    z = np.log(p / (1 - p))
    p_over = 1 / (1 + np.exp(-2.0 * z))
    assert calibration_slope_intercept(y, p_over)["slope"] < 0.7


def test_underconfident_model_has_slope_above_one():
    y, p = _perfect()
    z = np.log(p / (1 - p))
    p_under = 1 / (1 + np.exp(-0.5 * z))
    assert calibration_slope_intercept(y, p_under)["slope"] > 1.5


def test_systematic_shift_shows_up_in_intercept():
    y, p = _perfect()
    z = np.log(p / (1 - p)) + 1.0     # сдвиг в пользу класса 1
    p_shift = 1 / (1 + np.exp(-z))
    si = calibration_slope_intercept(y, p_shift)
    assert si["intercept"] < -0.5, "систематический сдвиг не пойман intercept"


def test_mce_catches_one_bad_bin_that_ece_hides():
    """ECE усредняет по размеру бина, поэтому один маленький, но
    катастрофически плохой бин в нём тонет. MCE обязан его показать."""
    rng = np.random.default_rng(1)
    n_good = 20000
    p_good = rng.uniform(0.4, 0.6, size=n_good)
    y_good = (rng.uniform(size=n_good) < p_good).astype(int)
    n_bad = 200
    p_bad = np.full(n_bad, 0.95)
    y_bad = np.zeros(n_bad, dtype=int)       # обещали 95%, случилось 0%
    y = np.concatenate([y_good, y_bad]); p = np.concatenate([p_good, p_bad])
    assert expected_calibration_error(y, p) < 0.02, "ECE не должен был заметить"
    assert maximum_calibration_error(y, p, min_bin_size=30) > 0.9, "MCE обязан заметить"


def test_reliability_bins_partition_all_points():
    y, p = _perfect(n=5000)
    assert sum(b.n for b in reliability_curve(y, p, bins=10)) == len(y)


def test_min_bin_size_filters_noise_bins():
    """Бин из трёх матчей не должен объявлять модель плохо откалиброванной."""
    rng = np.random.default_rng(2)
    p = np.concatenate([rng.uniform(0.4, 0.6, 5000), np.full(3, 0.95)])
    y = np.concatenate([(rng.uniform(size=5000) < p[:5000]).astype(int), np.zeros(3, int)])
    assert maximum_calibration_error(y, p, min_bin_size=30) < 0.2


def test_report_is_consistent_with_parts():
    y, p = _perfect(n=8000)
    r = calibration_report(y, p)
    assert r["n"] == len(y)
    assert r["ece"] == pytest.approx(expected_calibration_error(y, p))
    assert r["base_rate"] == pytest.approx(float(y.mean()))


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        calibration_report(np.zeros(10), np.zeros(11))
