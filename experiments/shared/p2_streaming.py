"""Streaming quantile utilities for sequential inference experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - fallback when numba is unavailable
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def _decorator(func):
            return func

        return _decorator


@dataclass
class P2State:
    """Serializable state for one P2 quantile estimator."""

    quantile: float
    n_obs: int
    markers: np.ndarray
    positions: np.ndarray
    desired: np.ndarray


class P2Estimator:
    """Streaming P2 quantile estimator (Jain and Chlamtac, 1985)."""

    def __init__(self, quantile: float):
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        self.quantile = float(quantile)
        self._buffer: list[float] = []
        self._n_obs = 0
        self._q = np.zeros(5, dtype=np.float64)
        self._n = np.zeros(5, dtype=np.float64)
        self._np = np.zeros(5, dtype=np.float64)
        self._dn = np.array(
            [0.0, self.quantile / 2.0, self.quantile, (1.0 + self.quantile) / 2.0, 1.0],
            dtype=np.float64,
        )

    @property
    def n_obs(self) -> int:
        return self._n_obs

    def update(self, value: float) -> None:
        x = float(value)
        self._n_obs += 1

        if self._n_obs <= 5:
            self._buffer.append(x)
            if self._n_obs == 5:
                self._initialize()
            return

        q = self._q
        n = self._n

        if x < q[0]:
            q[0] = x
            k = 0
        elif x < q[1]:
            k = 0
        elif x < q[2]:
            k = 1
        elif x < q[3]:
            k = 2
        elif x < q[4]:
            k = 3
        else:
            q[4] = x
            k = 3

        for j in range(k + 1, 5):
            n[j] += 1.0

        self._np += self._dn

        for i in range(1, 4):
            d = self._np[i] - n[i]
            if (d >= 1.0 and n[i + 1] - n[i] > 1.0) or (d <= -1.0 and n[i - 1] - n[i] < -1.0):
                step = 1.0 if d > 0 else -1.0
                q_new = self._parabolic(i, step)
                if q[i - 1] < q_new < q[i + 1]:
                    q[i] = q_new
                else:
                    q[i] = self._linear(i, step)
                n[i] += step

    def estimate(self) -> float:
        if self._n_obs == 0:
            return float("nan")
        if self._n_obs < 5:
            buf = np.sort(np.asarray(self._buffer, dtype=np.float64))
            idx = int(np.floor(self.quantile * (len(buf) - 1)))
            return float(buf[idx])
        return float(self._q[2])

    def state(self) -> P2State:
        return P2State(
            quantile=self.quantile,
            n_obs=self._n_obs,
            markers=self._q.copy(),
            positions=self._n.copy(),
            desired=self._np.copy(),
        )

    def _initialize(self) -> None:
        buf = np.sort(np.asarray(self._buffer, dtype=np.float64))
        self._q[:] = buf
        self._n[:] = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        self._np[:] = np.array(
            [
                1.0,
                1.0 + 2.0 * self.quantile,
                1.0 + 4.0 * self.quantile,
                3.0 + 2.0 * self.quantile,
                5.0,
            ],
            dtype=np.float64,
        )

    def _parabolic(self, idx: int, step: float) -> float:
        q = self._q
        n = self._n
        return q[idx] + (step / (n[idx + 1] - n[idx - 1])) * (
            (n[idx] - n[idx - 1] + step) * (q[idx + 1] - q[idx]) / (n[idx + 1] - n[idx])
            + (n[idx + 1] - n[idx] - step) * (q[idx] - q[idx - 1]) / (n[idx] - n[idx - 1])
        )

    def _linear(self, idx: int, step: float) -> float:
        q = self._q
        n = self._n
        j = idx + int(step)
        return q[idx] + step * (q[j] - q[idx]) / (n[j] - n[idx])


def streaming_iqr_scale(values: Iterable[float], warmup: int = 10) -> np.ndarray:
    """Compute streaming IQR-based scale with P2 quantiles."""

    seq = np.asarray(list(values), dtype=np.float64)
    if seq.ndim != 1:
        raise ValueError("values must be 1D")

    q25 = P2Estimator(0.25)
    q75 = P2Estimator(0.75)
    scales = np.full(seq.shape[0], np.nan, dtype=np.float64)

    min_samples = max(5, int(warmup))

    for t, val in enumerate(seq):
        q25.update(float(val))
        q75.update(float(val))
        if t + 1 >= min_samples:
            scales[t] = (q75.estimate() - q25.estimate()) / 1.349

    return scales


def detect_scale_changepoint(
    scales: np.ndarray,
    threshold: float = 0.10,
    ref_window: Tuple[int, int] = (10, 30),
    min_trees: int = 20,
) -> Tuple[int, np.ndarray]:
    """Detect first time relative scale falls below a threshold."""

    if scales.ndim != 1:
        raise ValueError("scales must be a 1D array")

    start, end = ref_window
    start = max(0, int(start))
    end = min(len(scales), int(end))
    if end <= start:
        return len(scales) - 1, np.full_like(scales, np.nan)

    sigma_ref = np.nanmean(scales[start:end])
    if not np.isfinite(sigma_ref) or sigma_ref <= 0:
        return len(scales) - 1, np.full_like(scales, np.nan)

    relative = scales / sigma_ref
    begin = max(int(min_trees), end)
    for idx in range(begin, len(scales)):
        value = relative[idx]
        if np.isfinite(value) and value < threshold:
            return idx, relative

    return len(scales) - 1, relative


def rolling_iqr_scale(
    values: Iterable[float],
    window: int = 20,
    warmup: int = 10,
) -> np.ndarray:
    """Compute IQR-based scale using a rolling window of recent observations.

    Unlike ``streaming_iqr_scale`` (which uses the full prefix), this keeps
    only the last *window* observations, allowing the scale estimate to
    respond quickly to distributional changes.
    """

    seq = np.asarray(list(values), dtype=np.float64)
    if seq.ndim != 1:
        raise ValueError("values must be 1D")

    scales = np.full(seq.shape[0], np.nan, dtype=np.float64)
    min_samples = max(5, int(warmup))

    for t in range(len(seq)):
        # Collect the most recent `window` observations up to index t (inclusive)
        start = max(0, t - window + 1)
        buf = seq[start : t + 1]
        if len(buf) < min_samples:
            continue
        # Use exact percentiles on the window (small window → exact is fine)
        q25 = float(np.percentile(buf, 25))
        q75 = float(np.percentile(buf, 75))
        iqr_scale = (q75 - q25) / 1.349
        scales[t] = iqr_scale

    return scales


def detect_cusum_changepoint(
    scales: np.ndarray,
    k: float = 0.5,
    h: float = 4.0,
    ref_window: Tuple[int, int] = (10, 30),
    min_trees: int = 20,
) -> Tuple[int, np.ndarray]:
    """Detect a downward shift in scale using Page's CUSUM.

    Parameters
    ----------
    scales : 1D array of IQR-based scale estimates.
    k : Allowance parameter (fraction of reference σ).
    h : Decision threshold – CUSUM signals when S_t > h * σ_ref.
    ref_window : (start, end) index range used to estimate reference σ.
    min_trees : Earliest index at which a detection is permitted.

    Returns
    -------
    stop_index : First index where CUSUM triggers (or len-1 if none).
    cusum_stats : Array of CUSUM statistic values.
    """

    if scales.ndim != 1:
        raise ValueError("scales must be a 1D array")

    start, end = ref_window
    start = max(0, int(start))
    end = min(len(scales), int(end))
    if end <= start:
        return len(scales) - 1, np.full_like(scales, np.nan)

    ref_values = scales[start:end]
    ref_values = ref_values[np.isfinite(ref_values)]
    if len(ref_values) < 3:
        return len(scales) - 1, np.full_like(scales, np.nan)

    mu_ref = float(np.mean(ref_values))
    sigma_ref = float(np.std(ref_values))
    if not np.isfinite(mu_ref) or mu_ref <= 0:
        return len(scales) - 1, np.full_like(scales, np.nan)
    if sigma_ref <= 0:
        sigma_ref = mu_ref * 0.1  # fallback

    cusum = np.zeros(len(scales), dtype=np.float64)
    threshold = h * sigma_ref
    allowance = k * sigma_ref
    begin = max(int(min_trees), end)

    for t in range(begin, len(scales)):
        if not np.isfinite(scales[t]):
            cusum[t] = cusum[t - 1] if t > 0 else 0.0
            continue
        # Detect downward shift: accumulate (mu_ref - scale_t - allowance)
        cusum[t] = max(0.0, cusum[t - 1] + (mu_ref - scales[t] - allowance))
        if cusum[t] > threshold:
            return t, cusum

    return len(scales) - 1, cusum


@njit(cache=True, fastmath=True)
def _percentile_linear_sorted(sorted_vals: np.ndarray, n: int, p: float) -> float:
    """Linear percentile interpolation on pre-sorted values."""
    if n <= 0:
        return np.nan
    if n == 1:
        return sorted_vals[0]

    rank = p * (n - 1)
    lo = int(np.floor(rank))
    hi = int(np.ceil(rank))
    if hi == lo:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


@njit(cache=True, fastmath=True)
def update_p2stop_state_numba(
    active: np.ndarray,
    deltas: np.ndarray,
    t: int,
    threshold: float,
    start: int,
    end: int,
    begin: int,
    min_trees: int,
    min_samples: int,
    rolling_window: int,
    ring_values: np.ndarray,
    ring_count: np.ndarray,
    ring_pos: np.ndarray,
    ref_sum: np.ndarray,
    ref_count: np.ndarray,
    counts: np.ndarray,
    stopped: np.ndarray,
    tau: np.ndarray,
) -> None:
    """Numba kernel for online P2-STOP state updates.

    Parameters are pre-allocated state arrays owned by the caller.
    The function mutates state in place.
    """
    if active.size == 0:
        return

    for j in range(active.size):
        idx = int(active[j])

        if stopped[idx]:
            continue

        pos = int(ring_pos[idx])
        ring_values[idx, pos] = float(deltas[j])
        pos += 1
        if pos >= rolling_window:
            pos = 0
        ring_pos[idx] = pos

        cnt = int(ring_count[idx])
        if cnt < rolling_window:
            cnt += 1
            ring_count[idx] = cnt

        if cnt < min_samples:
            continue

        buf = np.empty(cnt, dtype=np.float64)
        for k in range(cnt):
            buf[k] = ring_values[idx, k]
        buf.sort()

        q25 = _percentile_linear_sorted(buf, cnt, 0.25)
        q75 = _percentile_linear_sorted(buf, cnt, 0.75)
        scale = (q75 - q25) / 1.349
        if not np.isfinite(scale):
            continue

        if start <= t < end:
            ref_sum[idx] += scale
            ref_count[idx] += 1

        if t >= begin and counts[idx] >= min_trees and ref_count[idx] > 0:
            sigma_ref = ref_sum[idx] / ref_count[idx]
            if sigma_ref > 0.0:
                rel = scale / sigma_ref
                if np.isfinite(rel) and rel < threshold:
                    stopped[idx] = True
                    tau[idx] = t
