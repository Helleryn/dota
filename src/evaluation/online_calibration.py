"""
PHASE 14 — walk-forward (онлайн) слой калибровки.

## Инвариант, ради которого написан модуль

Слой калибровки — самое удобное место для утечки во всём проекте.
«Обучить калибровку на всей выборке и применить» даёт прекрасный ECE и
не значит ничего: используются исходы тех самых матчей, которые
калибруются.

Здесь порядок операций строго такой:

    для матча t:
        p_cal = calibrate(p_raw, состояние из матчей < t)
        ...наблюдаем исход...
        состояние += (p_raw, y_t, t)          # ПОСЛЕ, не раньше

Формальная проверка — prefix-stability: калиброванные прогнозы на первых
k матчах не зависят от того, существуют ли матчи k+1, k+2, ….

## Режимы (PART C задания)

| Режим | Что берётся в обучение слоя |
|---|---|
| `none` | ничего, сырой прогноз |
| `frozen` | история до начала отрезка, один раз, дальше не меняется |
| `rolling` | последние N наблюдений |
| `decay` | вся история с весом 0.5^(Δдней / half_life) |
| `patch_local` | только текущий патч; при малой выборке — запасной режим |

## О частоте переобучения слоя

Пересчёт слоя после КАЖДОГО матча (`refit_every=1`) — эталон, но на
реальных объёмах он неприменим: при затухании с half-life 30 дней в
обучение попадает порядка 30 тысяч матчей, и 16 тысяч пересчётов дают
часы счёта на одну конфигурацию.

Компромисс измерен, а не выбран на глаз (синтетический поток, 1300
матчей, затухание hl=60):

    refit_every=1    ECE=0.01216   max|Δ|=0.00000   0.74 c
    refit_every=5    ECE=0.01489   max|Δ|=0.01066   0.16 c
    refit_every=10   ECE=0.01418   max|Δ|=0.01537   0.09 c
    refit_every=25   ECE=0.01678   max|Δ|=0.02766   0.04 c
    refit_every=50   ECE=0.01417   max|Δ|=0.03854   0.03 c

По умолчанию берётся **10**: отклонение отдельного прогноза не превышает
1.6 пп, разница в ECE — 0.0021. Финальная выбранная
конфигурация дополнительно проверяется при `refit_every=1`.

Заодно это ближе к реальному продакшену, где слой пересчитывается по
расписанию, а не после каждой игры.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.evaluation.calibrators import CALIBRATORS, BaseCalibrator, IdentityCalibrator

MODES = ("none", "frozen", "rolling", "decay", "patch_local")


@dataclass
class CalibrationState:
    """История, доступная слою. Хранится в порядке появления матчей."""

    p: List[float] = field(default_factory=list)
    y: List[int] = field(default_factory=list)
    ts: List[float] = field(default_factory=list)
    patch: List[object] = field(default_factory=list)

    def add(self, p: float, y: int, ts: float, patch: object) -> None:
        self.p.append(float(p))
        self.y.append(int(y))
        self.ts.append(float(ts))
        self.patch.append(patch)

    def __len__(self) -> int:
        return len(self.p)

    def arrays(self):
        return (np.asarray(self.p), np.asarray(self.y, dtype=float),
                np.asarray(self.ts), np.asarray(self.patch, dtype=object))


def _select(state: CalibrationState, mode: str, now_ts: float, now_patch: object,
            window: Optional[int], half_life: Optional[float],
            min_patch_n: int, max_fit_n: int = 20000):
    """Возвращает (p, y, weights) для обучения слоя, либо None."""
    if len(state) == 0:
        return None
    p, y, ts, patch = state.arrays()
    # Ограничение объёма подгонки. При затухании с большим half-life в выборку
    # попадала бы вся история (до 93 тысяч матчей) на каждом пересчёте, что
    # делает эксперимент неисполнимым. Берутся последние max_fit_n наблюдений:
    # для малых half-life отбрасываемые веса пренебрежимы, для больших это
    # осознанное приближение, а не незамеченное.
    if len(p) > max_fit_n and mode in ("decay", "patch_local", "frozen"):
        p, y, ts, patch = p[-max_fit_n:], y[-max_fit_n:], ts[-max_fit_n:], patch[-max_fit_n:]

    if mode == "frozen":
        return p, y, np.ones_like(p)

    if mode == "rolling":
        n = int(window or len(p))
        return p[-n:], y[-n:], np.ones(min(n, len(p)))

    if mode == "decay":
        hl = float(half_life or 90.0)
        dt = np.maximum((now_ts - ts) / 86400.0, 0.0)
        w = 0.5 ** (dt / hl)
        keep = w > 1e-4          # вклад меньше 0.01% — не считаем
        if keep.sum() < 20:
            keep = np.ones_like(w, dtype=bool)
        return p[keep], y[keep], w[keep]

    if mode == "patch_local":
        sel = patch == now_patch
        if sel.sum() >= min_patch_n:
            return p[sel], y[sel], np.ones(int(sel.sum()))
        # Запасной вариант — затухание. Это сознательное решение: патч-локальная
        # калибровка в первые дни патча физически не имеет данных, а именно там
        # она нужнее всего. Отказ от запасного варианта означал бы «в первые дни
        # калибровки нет», что делает режим бессмысленным.
        dt = np.maximum((now_ts - ts) / 86400.0, 0.0)
        w = 0.5 ** (dt / float(half_life or 90.0))
        keep = w > 1e-4
        if keep.sum() < 20:
            keep = np.ones_like(w, dtype=bool)
        return p[keep], y[keep], w[keep]

    raise ValueError(f"неизвестный режим калибровки: {mode}")


def run_walk_forward(
    p_raw: Sequence[float],
    y: Sequence[int],
    ts: Sequence[float],
    patch: Sequence[object],
    *,
    mode: str = "decay",
    method: str = "platt",
    window: Optional[int] = None,
    half_life: Optional[float] = None,
    warm_p: Optional[Sequence[float]] = None,
    warm_y: Optional[Sequence[int]] = None,
    warm_ts: Optional[Sequence[float]] = None,
    warm_patch: Optional[Sequence[object]] = None,
    refit_every: int = 10,
    min_history: int = 500,
    min_patch_n: int = 300,
    max_fit_n: int = 20000,
) -> np.ndarray:
    """
    Калибрует последовательность прогнозов онлайн.

    `warm_*` — история ДО оцениваемого отрезка (например, TRAIN при оценке
    на VALIDATION). Она доступна слою с самого начала: в проде к моменту
    запуска история уже есть. Матчи самого отрезка добавляются в состояние
    строго ПОСЛЕ того, как их прогноз откалиброван.

    Возвращает массив калиброванных вероятностей той же длины, что p_raw.
    """
    if mode not in MODES:
        raise ValueError(f"неизвестный режим: {mode}")
    p_raw = np.asarray(p_raw, dtype=float)
    y = np.asarray(y, dtype=int)
    ts = np.asarray(ts, dtype=float)
    patch = np.asarray(patch, dtype=object)
    n = len(p_raw)

    if mode == "none":
        return p_raw.copy()

    state = CalibrationState()
    if warm_p is not None:
        for a, b, c, d in zip(warm_p, warm_y, warm_ts, warm_patch):
            state.add(a, b, c, d)

    cls = CALIBRATORS[method]
    cal: BaseCalibrator = IdentityCalibrator().fit(None, None)
    out = np.empty(n)
    fitted_once = False

    for i in range(n):
        need_refit = (i % refit_every == 0) and not (mode == "frozen" and fitted_once)
        if need_refit and len(state) >= min_history:
            sel = _select(state, mode, ts[i], patch[i], window, half_life,
                          min_patch_n, max_fit_n)
            if sel is not None:
                sp, sy, sw = sel
                candidate = cls().fit(sp, sy, sw)
                if candidate.fitted:
                    cal = candidate
                    fitted_once = True
        out[i] = float(np.asarray(cal.transform(np.array([p_raw[i]])))[0])
        # обновление состояния — строго ПОСЛЕ выдачи прогноза
        state.add(p_raw[i], y[i], ts[i], patch[i])

    return out


def rolling_drift_metrics(
    p: Sequence[float],
    y: Sequence[int],
    window: int = 1000,
    step: int = 200,
) -> List[Dict[str, float]]:
    """
    Скользящие метрики калибровки для ДЕТЕКТОРА дрейфа (PART G).

    Каждая точка считается по УЖЕ СЫГРАННЫМ матчам окна — ровно та
    информация, что доступна в проде. Никаких будущих меток.
    """
    from src.evaluation.calibration import calibration_report

    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    out: List[Dict[str, float]] = []
    for end in range(window, len(p) + 1, step):
        sl = slice(end - window, end)
        r = calibration_report(y[sl], p[sl])
        out.append({"end_index": end, "n": r["n"], "ece": r["ece"],
                    "brier": r["brier"], "log_loss": r["log_loss"],
                    "slope": r["slope"], "intercept": r["intercept"],
                    "residual_bias": float(np.mean(y[sl] - p[sl]))})
    return out
