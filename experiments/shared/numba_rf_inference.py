"""Numba-accelerated RF inference kernels for timing proof-of-concept.

This module provides an optimized inference path that compiles both
tree traversal and online stopping logic with Numba. It is intentionally
used as an implementation proof-of-concept for timing studies and does
not replace the primary sklearn-based reference path.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

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
def _sorted_insert_inplace(buf: np.ndarray, count: int, value: float) -> None:
    """Insert `value` into sorted `buf[:count]` (in-place), shifting right."""
    idx = count
    while idx > 0 and buf[idx - 1] > value:
        buf[idx] = buf[idx - 1]
        idx -= 1
    buf[idx] = value


@njit(cache=True, fastmath=True)
def _sorted_remove_inplace(buf: np.ndarray, count: int, value: float) -> None:
    """Remove one occurrence of `value` from sorted `buf[:count]` (in-place)."""
    idx = -1
    for j in range(count):
        if buf[j] == value:
            idx = j
            break

    # In practice value should be present; use nearest-value fallback for safety.
    if idx == -1:
        idx = 0
        best = abs(buf[0] - value)
        for j in range(1, count):
            diff = abs(buf[j] - value)
            if diff < best:
                best = diff
                idx = j

    for j in range(idx, count - 1):
        buf[j] = buf[j + 1]


@njit(cache=True, fastmath=True)
def rolling_iqr_scale_numba(values: np.ndarray, window: int = 20, warmup: int = 10) -> np.ndarray:
    """Exact rolling-window IQR scale computed with an incremental sorted buffer.

    This avoids per-step allocations/sorts by maintaining a sorted window
    under O(window) insert/remove updates. It matches the semantics of
    `experiments.shared.p2_streaming.rolling_iqr_scale` (linear interpolation).
    """
    n = values.shape[0]
    scales = np.full(n, np.nan, dtype=np.float64)
    w = max(1, int(window))
    min_samples = max(5, int(warmup))

    ring = np.zeros(w, dtype=np.float64)
    sorted_buf = np.zeros(w, dtype=np.float64)
    count = 0
    pos = 0

    for t in range(n):
        x = float(values[t])
        if count == w:
            old = ring[pos]
            _sorted_remove_inplace(sorted_buf, count, old)
            insert_count = count - 1
        else:
            insert_count = count

        ring[pos] = x
        pos += 1
        if pos >= w:
            pos = 0
        if count < w:
            count += 1

        _sorted_insert_inplace(sorted_buf, insert_count, x)

        if count >= min_samples:
            q25 = _percentile_linear_sorted(sorted_buf, count, 0.25)
            q75 = _percentile_linear_sorted(sorted_buf, count, 0.75)
            scales[t] = (q75 - q25) / 1.349

    return scales


@njit(cache=True, fastmath=True)
def _rf_full_kernel(
    X: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    node_probs: np.ndarray,
) -> np.ndarray:
    """Full RF inference (sample-major).

    Kept for reference. For fair timing comparisons against the P2-STOP
    kernel (tree-major), prefer `_rf_full_kernel_tree_major`.
    """
    n_samples = X.shape[0]
    n_trees = children_left.shape[0]
    n_classes = node_probs.shape[2]

    cumulative = np.zeros((n_samples, n_classes), dtype=np.float64)
    pred_idx = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        for t in range(n_trees):
            node = 0
            while children_left[t, node] != -1:
                feat = feature[t, node]
                if X[i, feat] <= threshold[t, node]:
                    node = children_left[t, node]
                else:
                    node = children_right[t, node]

            for c in range(n_classes):
                cumulative[i, c] += node_probs[t, node, c]

        best_c = 0
        best_v = cumulative[i, 0]
        for c in range(1, n_classes):
            v = cumulative[i, c]
            if v > best_v:
                best_v = v
                best_c = c
        pred_idx[i] = best_c

    return pred_idx


@njit(cache=True, fastmath=True)
def _rf_full_kernel_tree_major(
    X: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    node_probs: np.ndarray,
) -> np.ndarray:
    """Full RF inference (tree-major) to match the P2-STOP kernel's ordering."""
    n_samples = X.shape[0]
    n_trees = children_left.shape[0]
    n_classes = node_probs.shape[2]

    cumulative = np.zeros((n_samples, n_classes), dtype=np.float64)
    pred_idx = np.zeros(n_samples, dtype=np.int64)

    for t in range(n_trees):
        for i in range(n_samples):
            node = 0
            while children_left[t, node] != -1:
                feat = feature[t, node]
                if X[i, feat] <= threshold[t, node]:
                    node = children_left[t, node]
                else:
                    node = children_right[t, node]

            for c in range(n_classes):
                cumulative[i, c] += node_probs[t, node, c]

    for i in range(n_samples):
        best_c = 0
        best_v = cumulative[i, 0]
        for c in range(1, n_classes):
            v = cumulative[i, c]
            if v > best_v:
                best_v = v
                best_c = c
        pred_idx[i] = best_c

    return pred_idx


@njit(cache=True, fastmath=True)
def _rf_p2stop_kernel(
    X: np.ndarray,
    threshold_stop: float,
    start: int,
    end: int,
    begin: int,
    min_trees: int,
    min_samples: int,
    rolling_window: int,
    children_left: np.ndarray,
    children_right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    node_probs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = X.shape[0]
    n_trees = children_left.shape[0]
    n_classes = node_probs.shape[2]

    cumulative = np.zeros((n_samples, n_classes), dtype=np.float64)
    counts = np.zeros(n_samples, dtype=np.int64)
    prev_scalar = np.zeros(n_samples, dtype=np.float64)
    stopped = np.zeros(n_samples, dtype=np.bool_)
    tau = np.full(n_samples, n_trees - 1, dtype=np.int64)

    ring_values = np.zeros((n_samples, rolling_window), dtype=np.float64)
    ring_sorted = np.zeros((n_samples, rolling_window), dtype=np.float64)
    ring_count = np.zeros(n_samples, dtype=np.int64)
    ring_pos = np.zeros(n_samples, dtype=np.int64)
    ref_sum = np.zeros(n_samples, dtype=np.float64)
    ref_count = np.zeros(n_samples, dtype=np.int64)

    pred_idx = np.zeros(n_samples, dtype=np.int64)

    for t in range(n_trees):
        active_count = 0

        for i in range(n_samples):
            if stopped[i]:
                continue
            active_count += 1

            node = 0
            while children_left[t, node] != -1:
                feat = feature[t, node]
                if X[i, feat] <= threshold[t, node]:
                    node = children_left[t, node]
                else:
                    node = children_right[t, node]

            counts[i] += 1
            inv_count = 1.0 / counts[i]

            scalar_now = 0.0
            for c in range(n_classes):
                cumulative[i, c] += node_probs[t, node, c]
                avg = cumulative[i, c] * inv_count
                if c == 0 or avg > scalar_now:
                    scalar_now = avg

            if t == 0:
                delta = 0.0
            else:
                delta = scalar_now - prev_scalar[i]
            prev_scalar[i] = scalar_now

            pos = ring_pos[i]

            cnt = ring_count[i]
            if cnt == rolling_window:
                old = ring_values[i, pos]
                _sorted_remove_inplace(ring_sorted[i], cnt, old)
                insert_count = cnt - 1
            else:
                insert_count = cnt

            ring_values[i, pos] = delta
            pos += 1
            if pos >= rolling_window:
                pos = 0
            ring_pos[i] = pos

            if cnt < rolling_window:
                cnt += 1
                ring_count[i] = cnt

            if cnt < min_samples:
                _sorted_insert_inplace(ring_sorted[i], insert_count, delta)
                continue

            _sorted_insert_inplace(ring_sorted[i], insert_count, delta)

            q25 = _percentile_linear_sorted(ring_sorted[i], cnt, 0.25)
            q75 = _percentile_linear_sorted(ring_sorted[i], cnt, 0.75)
            scale = (q75 - q25) / 1.349
            if not np.isfinite(scale):
                continue

            if start <= t < end:
                ref_sum[i] += scale
                ref_count[i] += 1

            if t >= begin and counts[i] >= min_trees and ref_count[i] > 0:
                sigma_ref = ref_sum[i] / ref_count[i]
                if sigma_ref > 0.0:
                    rel = scale / sigma_ref
                    if np.isfinite(rel) and rel < threshold_stop:
                        stopped[i] = True
                        tau[i] = t

        if active_count == 0:
            break

    for i in range(n_samples):
        cnt = counts[i] if counts[i] > 0 else 1
        inv_count = 1.0 / cnt
        best_c = 0
        best_v = cumulative[i, 0] * inv_count
        for c in range(1, n_classes):
            v = cumulative[i, c] * inv_count
            if v > best_v:
                best_v = v
                best_c = c
        pred_idx[i] = best_c

    return pred_idx, tau


def compile_rf_forest_for_numba(model: Any) -> Dict[str, np.ndarray]:
    """Extract RF tree arrays into dense tensors for Numba kernels."""
    n_trees = len(model.estimators_)
    n_classes = len(model.classes_)
    classes = np.asarray(model.classes_)
    max_nodes = max(est.tree_.node_count for est in model.estimators_)

    children_left = np.full((n_trees, max_nodes), -1, dtype=np.int32)
    children_right = np.full((n_trees, max_nodes), -1, dtype=np.int32)
    feature = np.full((n_trees, max_nodes), -1, dtype=np.int32)
    threshold = np.zeros((n_trees, max_nodes), dtype=np.float64)
    node_probs = np.zeros((n_trees, max_nodes, n_classes), dtype=np.float64)

    for t, est in enumerate(model.estimators_):
        tree = est.tree_
        n_nodes = tree.node_count

        children_left[t, :n_nodes] = tree.children_left.astype(np.int32, copy=False)
        children_right[t, :n_nodes] = tree.children_right.astype(np.int32, copy=False)
        feature[t, :n_nodes] = tree.feature.astype(np.int32, copy=False)
        threshold[t, :n_nodes] = tree.threshold.astype(np.float64, copy=False)

        values = tree.value[:, 0, :].astype(np.float64, copy=False)
        sums = values.sum(axis=1, keepdims=True)
        probs = np.divide(values, sums, out=np.zeros_like(values), where=sums > 0)
        node_probs[t, :n_nodes, :] = probs

    return {
        "classes": classes,
        "children_left": children_left,
        "children_right": children_right,
        "feature": feature,
        "threshold": threshold,
        "node_probs": node_probs,
        "n_trees": np.asarray([n_trees], dtype=np.int64),
    }


def rf_full_inference_numba(
    forest: Dict[str, np.ndarray],
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run full-ensemble RF inference with Numba tree traversal."""
    X = np.asarray(X_test, dtype=np.float64, order="C")
    # Use the tree-major kernel to match the P2-STOP ordering in the timing PoC.
    pred_idx = _rf_full_kernel_tree_major(
        X,
        forest["children_left"],
        forest["children_right"],
        forest["feature"],
        forest["threshold"],
        forest["node_probs"],
    )
    preds = forest["classes"][pred_idx]
    tau = np.full(X.shape[0], int(forest["n_trees"][0]) - 1, dtype=np.int64)
    return preds, tau


def rf_p2stop_inference_numba(
    forest: Dict[str, np.ndarray],
    X_test: np.ndarray,
    threshold: float,
    ref_window: Tuple[int, int] = (10, 30),
    warmup: int = 10,
    min_trees: int = 10,
    rolling_window: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run P2-STOP RF inference with Numba tree traversal + stopping loop."""
    X = np.asarray(X_test, dtype=np.float64, order="C")
    n_trees = int(forest["n_trees"][0])

    start = max(0, int(ref_window[0]))
    end = min(n_trees, int(ref_window[1]))
    begin = max(int(min_trees), end)
    min_samples = max(5, int(warmup))
    window = max(1, int(rolling_window))

    pred_idx, tau = _rf_p2stop_kernel(
        X=X,
        threshold_stop=float(threshold),
        start=start,
        end=end,
        begin=begin,
        min_trees=int(min_trees),
        min_samples=min_samples,
        rolling_window=window,
        children_left=forest["children_left"],
        children_right=forest["children_right"],
        feature=forest["feature"],
        threshold=forest["threshold"],
        node_probs=forest["node_probs"],
    )
    preds = forest["classes"][pred_idx]
    return preds, tau
