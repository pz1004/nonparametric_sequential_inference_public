#!/usr/bin/env python3
"""
================================================================================
Phase D — Revision Experiments
================================================================================

Single script executing all four Phase D tasks for the MDPI revision:

  II-1  Extract Credit Card threshold-sweep data → add to Table 4.
  II-2  Rerun ablation with W_ref=[10,30) included → expand to 4-column table.
  II-3  Run P² (full-prefix) vs rolling-window comparison table.
  II-4  Implement and run QWYC baseline on all 4 datasets × {RF, XGBoost} × 5 seeds.

Outputs are written to  results/phase_d/  with per-task subdirectories
and a consolidated LaTeX-ready table file for each task.

Usage
-----
    python scripts/run_phase_d.py [--data-dir data] [--quick]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_rel, wilcoxon

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.shared.local_data_loader import load_local_dataset
from experiments.shared.p2_streaming import (
    detect_scale_changepoint,
    rolling_iqr_scale,
    streaming_iqr_scale,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("phase_d")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATE))
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATE))
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Constants (manuscript-consistent)
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 1024]
RF_DATASETS = ["mnist", "covertype", "higgs", "credit"]
GBM_DATASETS = ["mnist", "covertype", "higgs", "credit"]
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

RF_N_TREES = 200
GBM_N_TREES = 500
GBM_LR = 0.05
GBM_MAX_DEPTH = 6

MAX_TRAIN = 20_000
MAX_TEST = 5_000
GBM_MAX_TRAIN = 10_000
GBM_MAX_TEST = 2_000
HIGGS_MAX_ROWS = 500_000

DEFAULT_REF_WINDOW = (10, 30)
ROLLING_WINDOW = 20
WARMUP = 10
MIN_TREES = 10
DIRICHLET_THRESHOLD = 0.95

ABLATION_WINDOWS = {
    "tref10": (5, 10),
    "tref20": (10, 20),
    "tref30": (10, 30),   # NEW — the recommended default
    "tref40": (20, 40),
}

# ---------------------------------------------------------------------------
# Statistics helpers (same as run_all_experiments.py)
# ---------------------------------------------------------------------------
BOOTSTRAP_CI_SEED = 12345
BOOTSTRAP_CI_N = 20000


def _mean_ci95(values) -> Dict[str, float]:
    from scipy.stats import norm
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = float(np.mean(arr))
    if arr.size == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}
    std = float(np.std(arr, ddof=1))
    sem = std / np.sqrt(arr.size)
    tcrit = float(student_t.ppf(0.975, arr.size - 1))
    half = tcrit * sem
    return {"n": int(arr.size), "mean": mean, "std": std,
            "ci95_low": float(mean - half), "ci95_high": float(mean + half)}


def _mean_bca_ci95(values, *, bounds=None) -> Dict[str, float]:
    from scipy.stats import norm
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = float(np.mean(arr))
    if arr.size == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}
    std = float(np.std(arr, ddof=1))
    n = int(arr.size)
    rng = np.random.default_rng(BOOTSTRAP_CI_SEED)
    idx = rng.integers(0, n, size=(BOOTSTRAP_CI_N, n))
    boot_means = np.mean(arr[idx], axis=1)
    prop = float(np.mean(boot_means < mean))
    prop = min(max(prop, 0.5 / BOOTSTRAP_CI_N), 1.0 - 0.5 / BOOTSTRAP_CI_N)
    z0 = float(norm.ppf(prop))
    total = float(np.sum(arr))
    jack = np.array([(total - arr[i]) / (n - 1) for i in range(n)])
    jm = float(np.mean(jack))
    diffs = jm - jack
    denom = float(np.sum(diffs ** 2)) ** 1.5
    a = float(np.sum(diffs ** 3)) / (6.0 * denom) if denom > 0 else 0.0

    def _adj(alpha):
        z = float(norm.ppf(alpha))
        num = z0 + z
        den = 1.0 - a * num
        if den == 0:
            den = 1e-12
        return float(norm.cdf(z0 + num / den))

    a1 = min(max(_adj(0.025), 0.0), 1.0)
    a2 = min(max(_adj(0.975), 0.0), 1.0)
    if a2 < a1:
        a1, a2 = a2, a1
    lo = float(np.quantile(boot_means, a1, method="linear"))
    hi = float(np.quantile(boot_means, a2, method="linear"))
    if bounds is not None:
        lo = float(np.clip(lo, bounds[0], bounds[1]))
        hi = float(np.clip(hi, bounds[0], bounds[1]))
    return {"n": n, "mean": mean, "std": std, "ci95_low": lo, "ci95_high": hi}


def _paired_stats(x, y) -> Dict[str, float]:
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    v = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[v], ya[v]
    if xa.size == 0:
        return {"n": 0, "delta_mean": float("nan"), "delta_std": float("nan"),
                "delta_ci95_low": float("nan"), "delta_ci95_high": float("nan"),
                "p_ttest": float("nan"), "p_wilcoxon": float("nan")}
    diff = xa - ya
    ci = _mean_ci95(diff)
    p_t = p_w = float("nan")
    if xa.size >= 2 and np.std(diff, ddof=1) > 0:
        try:
            p_t = float(ttest_rel(xa, ya).pvalue)
        except Exception:
            pass
    if xa.size >= 2 and not np.allclose(diff, 0):
        try:
            p_w = float(wilcoxon(xa, ya, zero_method="wilcox", alternative="two-sided").pvalue)
        except Exception:
            pass
    return {"n": ci["n"], "delta_mean": ci["mean"], "delta_std": ci["std"],
            "delta_ci95_low": ci["ci95_low"], "delta_ci95_high": ci["ci95_high"],
            "p_ttest": p_t, "p_wilcoxon": p_w}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def stratified_subsample(X, y, max_rows, seed):
    if max_rows <= 0 or len(y) <= max_rows:
        return X, y
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max_rows, replace=False)
        return X[idx], y[idx]
    proportions = counts / counts.sum()
    alloc = np.floor(proportions * max_rows).astype(int)
    alloc = np.maximum(alloc, 1)
    alloc = np.minimum(alloc, counts)
    while alloc.sum() > max_rows:
        cands = np.where(alloc > 1)[0]
        if cands.size == 0:
            break
        alloc[cands[np.argmax(alloc[cands])]] -= 1
    rem = max_rows - int(alloc.sum())
    if rem > 0:
        spare = counts - alloc
        for idx in np.argsort(-spare):
            if rem <= 0:
                break
            take = min(int(spare[idx]), rem)
            alloc[idx] += take
            rem -= take
    rng = np.random.default_rng(seed)
    parts = []
    for cls, take in zip(classes, alloc):
        ci = np.where(y == cls)[0]
        chosen = ci if take >= ci.size else rng.choice(ci, size=int(take), replace=False)
        parts.append(chosen)
    sel = np.concatenate(parts)
    rng.shuffle(sel)
    return X[sel], y[sel]


def prepare_splits(bundle, seed, max_train, max_test):
    """70/10/20 train/val/test split (same as Phase 1)."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_tv, X_test, y_tv, y_test = train_test_split(
        bundle.X, bundle.y, test_size=0.3, random_state=seed, stratify=bundle.y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=1.0 / 7.0, random_state=seed + 100, stratify=y_tv)

    if max_train > 0 and len(X_train) > max_train:
        X_train, y_train = stratified_subsample(X_train, y_train, max_train, seed)
    if max_test > 0 and len(X_test) > max_test:
        X_test, y_test = stratified_subsample(X_test, y_test, max_test, seed + 1)
    if max_test > 0 and len(X_val) > max_test:
        X_val, y_val = stratified_subsample(X_val, y_val, max_test, seed + 2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def dataset_max_rows(ds_name: str) -> int:
    return HIGGS_MAX_ROWS if ds_name == "higgs" else 0


def _ds_max_rows_dict(ds_name: str) -> Optional[Dict[str, int]]:
    """Return dataset_max_rows dict expected by load_local_dataset."""
    if ds_name == "higgs":
        return {"higgs": HIGGS_MAX_ROWS}
    return None


# ---------------------------------------------------------------------------
# Trajectory cache — avoids refitting the same model across tasks
# ---------------------------------------------------------------------------
_TRAJ_CACHE: Dict[str, Dict] = {}


def _cache_key(ds_name: str, model_type: str, seed: int) -> str:
    return f"{ds_name}__{model_type}__{seed}"


def get_cached_trajectory(
    ds_name: str,
    model_type: str,  # "rf" or "gbm"
    seed: int,
    data_dir: Path,
    logger: logging.Logger,
) -> Dict:
    """Return cached trajectory dict; fit & compute if not present.

    Returns dict with keys: traj, X_train, X_val, X_test, y_train, y_val,
    y_test, n_trees, n_classes, model (RF only), tree_preds (RF only)
    """
    key = _cache_key(ds_name, model_type, seed)
    if key in _TRAJ_CACHE:
        return _TRAJ_CACHE[key]

    bundle = load_local_dataset(ds_name, data_dir=data_dir, seed=42,
                                dataset_max_rows=_ds_max_rows_dict(ds_name))

    if model_type == "rf":
        mt, mte = MAX_TRAIN, MAX_TEST
    else:
        mt, mte = GBM_MAX_TRAIN, GBM_MAX_TEST

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_splits(
        bundle, seed, mt, mte)
    n_classes = len(np.unique(y_train))

    if model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier
        n_trees = RF_N_TREES
        rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        traj = _compute_rf_trajectory(rf, X_test)
        traj_val = _compute_rf_trajectory(rf, X_val)
        # Per-tree hard predictions for Dirichlet
        tree_preds = np.array([np.argmax(tree.predict_proba(X_test), axis=1)
                               for tree in rf.estimators_]).T  # (N, T)
        entry = {
            "traj": traj, "traj_val": traj_val,
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "n_trees": n_trees, "n_classes": n_classes,
            "model": rf, "tree_preds": tree_preds,
        }
    else:
        n_trees = GBM_N_TREES
        traj = _compute_gbm_trajectory(X_train, y_train, X_test, n_trees, seed)
        traj_val = _compute_gbm_trajectory(X_train, y_train, X_val, n_trees, seed)
        entry = {
            "traj": traj, "traj_val": traj_val,
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "n_trees": n_trees, "n_classes": n_classes,
        }

    _TRAJ_CACHE[key] = entry
    return entry


def _compute_p2_stops(traj, n_trees, theta, ref_window, scale_mode="rolling"):
    """Compute P2-STOP stop times for (N, T, C) trajectory."""
    straj = np.max(traj, axis=2)
    delts = np.diff(straj, axis=1, prepend=straj[:, :1])
    N = traj.shape[0]
    stops = np.full(N, n_trees - 1, dtype=np.int64)
    for i in range(N):
        if scale_mode == "rolling":
            sc = rolling_iqr_scale(delts[i], window=ROLLING_WINDOW, warmup=WARMUP)
        else:
            sc = streaming_iqr_scale(delts[i], warmup=WARMUP)
        tau, _ = detect_scale_changepoint(
            sc, threshold=theta, ref_window=ref_window, min_trees=MIN_TREES)
        stops[i] = tau
    return stops


# ===================================================================
# II-1  Credit Card threshold-sweep extraction
# ===================================================================

def task_ii1_credit_card_sweep(
    output_dir: Path, data_dir: Path, logger: logging.Logger
) -> pd.DataFrame:
    """Extract or compute Credit Card threshold-sweep results."""

    logger.info("=" * 72)
    logger.info("II-1: Credit Card threshold-sweep")

    # First try reusing existing raw Phase 1 data
    existing = ROOT / "results" / "run_20260228_151229" / "phase1_main" / "run_20260228_151237" / "phase1_summary.csv"
    if existing.exists():
        df = pd.read_csv(existing)
        credit_df = df[df["dataset"] == "credit"]
        if len(credit_df) >= len(SEEDS) * len(THRESHOLDS):
            logger.info("  Reusing existing Credit Card data from %s (%d rows)", existing, len(credit_df))
            credit_df.to_csv(output_dir / "credit_sweep_raw.csv", index=False)
            return credit_df

    logger.info("  Running Credit Card experiment from scratch ...")
    return _run_rf_sweep_cached("credit", output_dir, data_dir, logger)


def _run_rf_sweep_cached(
    ds_name: str, output_dir: Path, data_dir: Path, logger: logging.Logger
) -> pd.DataFrame:
    """Run the standard RF P2-STOP sweep using trajectory cache."""
    from sklearn.metrics import accuracy_score

    rows = []
    for seed in SEEDS:
        logger.info("    Seed %d ...", seed)
        c = get_cached_trajectory(ds_name, "rf", seed, data_dir, logger)
        traj, y_test = c["traj"], c["y_test"]
        traj_val, y_val = c["traj_val"], c["y_val"]
        n_classes, n_trees = c["n_classes"], c["n_trees"]
        tree_preds = c["tree_preds"]

        acc_full = accuracy_score(y_test, np.argmax(traj[:, -1], axis=1))
        dir_stops = _compute_dirichlet_stops(tree_preds, n_classes, DIRICHLET_THRESHOLD, MIN_TREES)
        wr_dir = 1.0 - float(np.mean((dir_stops + 1) / n_trees))

        for theta in THRESHOLDS:
            stops = _compute_p2_stops(traj, n_trees, theta, DEFAULT_REF_WINDOW)
            stopped_probs = traj[np.arange(len(y_test)), stops]
            acc_p2 = accuracy_score(y_test, np.argmax(stopped_probs, axis=1))
            wr_p2 = 1.0 - float(np.mean((stops + 1) / n_trees))
            elbow = float(np.mean(stops < n_trees - 1))

            # Val split
            stops_val = _compute_p2_stops(traj_val, n_trees, theta, DEFAULT_REF_WINDOW)
            acc_full_val = accuracy_score(y_val, np.argmax(traj_val[:, -1], axis=1))
            acc_p2_val = accuracy_score(y_val, np.argmax(traj_val[np.arange(len(y_val)), stops_val], axis=1))
            wr_p2_val = 1.0 - float(np.mean((stops_val + 1) / n_trees))
            elbow_val = float(np.mean(stops_val < n_trees - 1))

            rows.append({
                "dataset": ds_name, "seed": seed, "threshold": theta,
                "n_train": len(c["X_train"]), "n_val": len(c["X_val"]), "n_test": len(c["X_test"]),
                "n_trees": n_trees,
                "accuracy_full": acc_full, "accuracy_p2": acc_p2,
                "accuracy_dirichlet": acc_full,  # placeholder
                "delta_acc_vs_full": acc_p2 - acc_full,
                "mean_work_p2": float(np.mean((stops + 1) / n_trees)),
                "mean_work_dirichlet": float(np.mean((dir_stops + 1) / n_trees)),
                "work_reduction_p2": wr_p2,
                "work_reduction_dirichlet": wr_dir,
                "elbow_fraction": elbow,
                "accuracy_full_val": acc_full_val,
                "accuracy_p2_val": acc_p2_val,
                "delta_acc_vs_full_val": acc_p2_val - acc_full_val,
                "work_reduction_p2_val": wr_p2_val,
                "elbow_fraction_val": elbow_val,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"{ds_name}_sweep_raw.csv", index=False)
    return df


def _compute_rf_trajectory(model, X):
    """(N, T, C) cumulative probability trajectory."""
    per_tree = [tree.predict_proba(X) for tree in model.estimators_]
    stacked = np.stack(per_tree, axis=1)
    cum = np.cumsum(stacked, axis=1)
    denom = np.arange(1, stacked.shape[1] + 1, dtype=np.float64)[None, :, None]
    return cum / denom


def _compute_dirichlet_stops(tree_preds, n_classes, threshold, min_trees):
    """Dirichlet-based stopping times. tree_preds: (N, T) int."""
    from scipy.special import betainc
    N, T = tree_preds.shape
    stops = np.full(N, T - 1, dtype=np.int64)
    for i in range(N):
        counts = np.ones(n_classes, dtype=np.float64)
        for t in range(T):
            counts[int(tree_preds[i, t])] += 1.0
            if t + 1 < min_trees:
                continue
            best_idx = int(np.argmax(counts))
            best = counts[best_idx]
            total = float(np.sum(counts))
            if n_classes == 2:
                second = total - best
                p_stable = 1.0 - betainc(best, second, 0.5)
            else:
                masked = counts.copy()
                masked[best_idx] = -1.0
                second = float(np.max(masked))
                mu_diff = (best - second) / total
                var_b = best * (total - best)
                var_s = second * (total - second)
                cov = -best * second
                sigma = np.sqrt(max(var_b + var_s - 2 * cov, 0) / (total ** 2 * (total + 1)))
                z = mu_diff / (sigma + 1e-12)
                p_stable = 0.5 * (1.0 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
            if p_stable > threshold:
                stops[i] = t
                break
    return stops


# ===================================================================
# II-2  Reference-window ablation with [10,30) included
# ===================================================================

def task_ii2_ablation(
    output_dir: Path, data_dir: Path, logger: logging.Logger
) -> Dict[str, pd.DataFrame]:
    """Run P2-STOP at θ=0.10 with 4 reference windows, all 4 datasets × 5 seeds.

    Uses trajectory cache so RF is fitted only once per (dataset, seed).
    """

    logger.info("=" * 72)
    logger.info("II-2: Reference-window ablation (4 windows)")

    ablation_results: Dict[str, pd.DataFrame] = {}
    theta = 0.10

    # Collect all rows per window by iterating (ds, seed) in outer loop
    window_rows: Dict[str, List[Dict]] = {label: [] for label in ABLATION_WINDOWS}

    for ds_name in RF_DATASETS:
        for seed in SEEDS:
            logger.info("    %s / seed=%d — fitting model and computing 4 windows ...", ds_name, seed)
            c = get_cached_trajectory(ds_name, "rf", seed, data_dir, logger)
            traj, y_test = c["traj"], c["y_test"]
            traj_val, y_val = c["traj_val"], c["y_val"]
            n_trees = c["n_trees"]
            acc_full = float(np.mean(np.argmax(traj[:, -1], axis=1) == y_test))

            for label, (t0, t1) in ABLATION_WINDOWS.items():
                stops = _compute_p2_stops(traj, n_trees, theta, (t0, t1))
                stopped_probs = traj[np.arange(len(y_test)), stops]
                acc_p2 = float(np.mean(np.argmax(stopped_probs, axis=1) == y_test))
                wr_p2 = 1.0 - float(np.mean((stops + 1) / n_trees))

                # Val
                stops_val = _compute_p2_stops(traj_val, n_trees, theta, (t0, t1))
                acc_full_val = float(np.mean(np.argmax(traj_val[:, -1], axis=1) == y_val))
                acc_p2_val = float(np.mean(np.argmax(traj_val[np.arange(len(y_val)), stops_val], axis=1) == y_val))
                wr_p2_val = 1.0 - float(np.mean((stops_val + 1) / n_trees))

                window_rows[label].append({
                    "dataset": ds_name, "seed": seed, "threshold": theta,
                    "ref_window": f"{t0},{t1}",
                    "n_train": len(c["X_train"]), "n_test": len(c["X_test"]),
                    "n_trees": n_trees,
                    "accuracy_full": acc_full, "accuracy_p2": acc_p2,
                    "delta_acc_vs_full": acc_p2 - acc_full,
                    "work_reduction_p2": wr_p2,
                    "elbow_fraction": float(np.mean(stops < n_trees - 1)),
                    "accuracy_full_val": acc_full_val,
                    "accuracy_p2_val": acc_p2_val,
                    "delta_acc_vs_full_val": acc_p2_val - acc_full_val,
                    "work_reduction_p2_val": wr_p2_val,
                })

    for label, rows in window_rows.items():
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / f"ablation_{label}.csv", index=False)
        ablation_results[label] = df

    return ablation_results


# ===================================================================
# II-3  P² full-prefix vs rolling-window comparison
# ===================================================================

def task_ii3_p2_vs_rolling(
    output_dir: Path, data_dir: Path, logger: logging.Logger
) -> pd.DataFrame:
    """Compare full-prefix P² backend vs rolling-window exact backend."""

    logger.info("=" * 72)
    logger.info("II-3: P² full-prefix vs rolling-window comparison")

    from sklearn.metrics import accuracy_score

    configs = [
        ("mnist", "rf"),
        ("covertype", "rf"),
        ("mnist", "gbm"),
    ]

    rows = []
    for ds_name, model_type in configs:
        for seed in SEEDS:
            logger.info("  %s / %s / seed=%d ...", ds_name, model_type, seed)
            c = get_cached_trajectory(ds_name, model_type, seed, data_dir, logger)
            traj, y_test = c["traj"], c["y_test"]
            n_trees = c["n_trees"]
            acc_full = accuracy_score(y_test, np.argmax(traj[:, -1], axis=1))

            for theta in [0.05, 0.10, 0.15]:
                for backend in ["rolling", "prefix"]:
                    stops = _compute_p2_stops(traj, n_trees, theta,
                                              DEFAULT_REF_WINDOW, scale_mode=backend)
                    stopped_probs = traj[np.arange(len(y_test)), stops]
                    acc_p2 = accuracy_score(y_test, np.argmax(stopped_probs, axis=1))
                    wr = 1.0 - float(np.mean((stops + 1) / n_trees))

                    rows.append({
                        "dataset": ds_name, "model": model_type, "seed": seed,
                        "backend": backend, "threshold": theta,
                        "accuracy_full": acc_full, "accuracy": acc_p2,
                        "delta_acc": acc_p2 - acc_full,
                        "work_reduction": wr,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "p2_vs_rolling_raw.csv", index=False)
    return df


def _compute_gbm_trajectory(X_train, y_train, X_test, n_trees, seed):
    """Build XGBoost model and extract cumulative trajectory."""
    import xgboost as xgb

    n_classes = len(np.unique(y_train))
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    params = {
        "n_estimators": n_trees,
        "learning_rate": GBM_LR,
        "max_depth": GBM_MAX_DEPTH,
        "random_state": seed,
        "objective": objective,
        "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
        "use_label_encoder": False,
        "n_jobs": -1,
        "verbosity": 0,
    }
    if n_classes > 2:
        params["num_class"] = n_classes

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)

    N = X_test.shape[0]
    C = n_classes
    traj = np.zeros((N, n_trees, C), dtype=np.float64)
    for t in range(1, n_trees + 1):
        proba = model.predict_proba(X_test, iteration_range=(0, t))
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        traj[:, t - 1] = proba

    return traj


# ===================================================================
# II-4  QWYC baseline implementation and experiments
# ===================================================================

def task_ii4_qwyc(
    output_dir: Path, data_dir: Path, logger: logging.Logger
) -> pd.DataFrame:
    """Implement and run QWYC baseline in two configurations.

    QWYC (Quit When You Can) - Wang, Gupta, You (ACM JETC, 2021)
    ---------------------------------------------------------------
    We evaluate QWYC in two configurations:
      - QWYC-margin: Component (ii) alone — margin-based stopping
        under the natural tree ordering.
      - QWYC-full: Components (i)+(ii) — greedy validation-based
        ordering optimization followed by margin-based stopping.
        Applies to RF only; for XGBoost (sequential correctors),
        QWYC-full ≡ QWYC-margin.

    Core idea: Stop as soon as the leading class has an unbeatable
    margin — even if all remaining trees vote for the runner-up,
    the leader cannot be overtaken.
    """

    logger.info("=" * 72)
    logger.info("II-4: QWYC baseline on all 4 datasets × {RF, XGBoost} × 5 seeds")

    from sklearn.metrics import accuracy_score

    rows = []

    # ---- RF experiments ----
    for ds_name in RF_DATASETS:
        for seed in SEEDS:
            logger.info("  QWYC RF: %s / seed=%d", ds_name, seed)
            c = get_cached_trajectory(ds_name, "rf", seed, data_dir, logger)
            traj, y_test = c["traj"], c["y_test"]
            traj_val, y_val = c["traj_val"], c["y_val"]
            n_trees, n_classes = c["n_trees"], c["n_classes"]
            tree_preds = c["tree_preds"]
            model = c["model"]

            acc_full = accuracy_score(y_test, np.argmax(traj[:, -1], axis=1))

            # --- QWYC margin-only (component ii, fixed order) ---
            qwyc_stops = _qwyc_rf_stops(traj, n_trees)
            acc_qwyc = accuracy_score(y_test, np.argmax(traj[np.arange(len(y_test)), qwyc_stops], axis=1))
            wr_qwyc = 1.0 - float(np.mean((qwyc_stops + 1) / n_trees))

            # --- QWYC full (component i + ii: optimized ordering + margin stop) ---
            # Per-tree soft predictions for ordering optimizer
            per_tree_proba_val = np.stack(
                [tree.predict_proba(c["X_val"]) for tree in model.estimators_], axis=1
            )  # (N_val, T, C)
            per_tree_proba_test = np.stack(
                [tree.predict_proba(c["X_test"]) for tree in model.estimators_], axis=1
            )  # (N_test, T, C)

            logger.info("    Computing QWYC optimized ordering on validation set ...")
            ordering = _qwyc_optimize_ordering(per_tree_proba_val, logger)

            # Reorder test trajectory and apply margin stopping
            traj_reordered = _reorder_rf_trajectory(per_tree_proba_test, ordering)
            qwyc_full_stops = _qwyc_rf_stops(traj_reordered, n_trees)
            acc_qwyc_full = accuracy_score(
                y_test, np.argmax(traj_reordered[np.arange(len(y_test)), qwyc_full_stops], axis=1)
            )
            wr_qwyc_full = 1.0 - float(np.mean((qwyc_full_stops + 1) / n_trees))

            # P2-STOP (θ=0.10) — from cache, just evaluate
            p2_stops = _compute_p2_stops(traj, n_trees, 0.10, DEFAULT_REF_WINDOW)
            acc_p2 = accuracy_score(y_test, np.argmax(traj[np.arange(len(y_test)), p2_stops], axis=1))
            wr_p2 = 1.0 - float(np.mean((p2_stops + 1) / n_trees))

            # Dirichlet baseline
            dir_stops = _compute_dirichlet_stops(tree_preds, n_classes, DIRICHLET_THRESHOLD, MIN_TREES)
            acc_dir = accuracy_score(y_test, np.argmax(traj[np.arange(len(y_test)), dir_stops], axis=1))
            wr_dir = 1.0 - float(np.mean((dir_stops + 1) / n_trees))

            rows.append({
                "dataset": ds_name, "model": "rf", "seed": seed,
                "n_trees": n_trees,
                "accuracy_full": acc_full,
                "accuracy_qwyc": acc_qwyc, "accuracy_qwyc_full": acc_qwyc_full,
                "accuracy_p2stop": acc_p2,
                "accuracy_dirichlet": acc_dir,
                "delta_acc_qwyc": acc_qwyc - acc_full,
                "delta_acc_qwyc_full": acc_qwyc_full - acc_full,
                "delta_acc_p2stop": acc_p2 - acc_full,
                "delta_acc_dirichlet": acc_dir - acc_full,
                "wr_qwyc": wr_qwyc, "wr_qwyc_full": wr_qwyc_full,
                "wr_p2stop": wr_p2, "wr_dirichlet": wr_dir,
                "mean_stop_qwyc": float(np.mean(qwyc_stops + 1)),
                "mean_stop_qwyc_full": float(np.mean(qwyc_full_stops + 1)),
                "mean_stop_p2stop": float(np.mean(p2_stops + 1)),
                "mean_stop_dirichlet": float(np.mean(dir_stops + 1)),
            })

    # ---- GBM (XGBoost) experiments ----
    # NOTE: Boosted trees cannot be reordered (each tree corrects the
    # residual of all preceding trees), so QWYC-full == QWYC margin-only
    # for GBM.  We record qwyc_full columns as identical to qwyc columns
    # to keep the DataFrame schema consistent.
    for ds_name in GBM_DATASETS:
        for seed in SEEDS:
            logger.info("  QWYC XGBoost: %s / seed=%d", ds_name, seed)
            c = get_cached_trajectory(ds_name, "gbm", seed, data_dir, logger)
            traj, y_test = c["traj"], c["y_test"]
            n_trees = c["n_trees"]

            acc_full = accuracy_score(y_test, np.argmax(traj[:, -1], axis=1))

            qwyc_stops = _qwyc_gbm_stops(traj, n_trees)
            acc_qwyc = accuracy_score(y_test, np.argmax(traj[np.arange(len(y_test)), qwyc_stops], axis=1))
            wr_qwyc = 1.0 - float(np.mean((qwyc_stops + 1) / n_trees))

            p2_stops = _compute_p2_stops(traj, n_trees, 0.10, DEFAULT_REF_WINDOW)
            acc_p2 = accuracy_score(y_test, np.argmax(traj[np.arange(len(y_test)), p2_stops], axis=1))
            wr_p2 = 1.0 - float(np.mean((p2_stops + 1) / n_trees))

            rows.append({
                "dataset": ds_name, "model": "xgboost", "seed": seed,
                "n_trees": n_trees,
                "accuracy_full": acc_full,
                "accuracy_qwyc": acc_qwyc, "accuracy_qwyc_full": acc_qwyc,
                "accuracy_p2stop": acc_p2,
                "accuracy_dirichlet": float("nan"),
                "delta_acc_qwyc": acc_qwyc - acc_full,
                "delta_acc_qwyc_full": acc_qwyc - acc_full,
                "delta_acc_p2stop": acc_p2 - acc_full,
                "delta_acc_dirichlet": float("nan"),
                "wr_qwyc": wr_qwyc, "wr_qwyc_full": wr_qwyc,
                "wr_p2stop": wr_p2, "wr_dirichlet": float("nan"),
                "mean_stop_qwyc": float(np.mean(qwyc_stops + 1)),
                "mean_stop_qwyc_full": float(np.mean(qwyc_stops + 1)),
                "mean_stop_p2stop": float(np.mean(p2_stops + 1)),
                "mean_stop_dirichlet": float("nan"),
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "qwyc_raw.csv", index=False)
    return df


def _qwyc_optimize_ordering(
    per_tree_proba_val: np.ndarray,
    logger: logging.Logger,
) -> np.ndarray:
    """Greedy validation-based tree ordering optimizer (QWYC component i).

    For independently trained ensembles (e.g., RF), trees can be freely
    permuted.  This implements the greedy strategy from Wang et al. (2021):
    at each position k, select the remaining tree that maximizes the number
    of validation instances stopped (via the margin criterion) at or before
    position k.

    The objective is prediction-based (maximize count of margin-stopped
    instances on the validation set); true labels are not needed.

    Parameters
    ----------
    per_tree_proba_val : (N_val, T, C) array
        Per-tree probability predictions on the validation set.
    logger : Logger

    Returns
    -------
    ordering : (T,) int array
        Permutation of tree indices [0..T-1] defining the optimized order.
    """
    N, T, C = per_tree_proba_val.shape
    ordering = np.empty(T, dtype=np.int64)
    remaining = list(range(T))
    # cumulative sum of selected trees' probas: (N, C)
    cum_proba = np.zeros((N, C), dtype=np.float64)

    for k in range(T):
        best_idx_in_remaining = 0
        best_stopped = -1

        for ri, tree_idx in enumerate(remaining):
            # Tentative cumulative proba after adding this tree at position k
            trial_cum = cum_proba + per_tree_proba_val[:, tree_idx, :]
            trial_avg = trial_cum / (k + 1)

            if k + 1 >= MIN_TREES:
                # Compute margin for each instance
                # Sort descending along class axis
                sorted_probs = np.sort(trial_avg, axis=1)[:, ::-1]
                if C > 1:
                    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
                else:
                    margins = sorted_probs[:, 0]
                remaining_trees = T - k - 1
                max_shift = remaining_trees / T
                stopped = int(np.sum(margins > max_shift))
            else:
                stopped = 0

            if stopped > best_stopped:
                best_stopped = stopped
                best_idx_in_remaining = ri

        chosen = remaining.pop(best_idx_in_remaining)
        ordering[k] = chosen
        cum_proba += per_tree_proba_val[:, chosen, :]

        if (k + 1) % 50 == 0 or k == T - 1:
            logger.debug("    Ordering opt: placed %d/%d trees (stopped=%d/%d)",
                         k + 1, T, best_stopped, N)

    return ordering


def _reorder_rf_trajectory(
    per_tree_proba: np.ndarray,
    ordering: np.ndarray,
) -> np.ndarray:
    """Reorder per-tree predictions and compute cumulative average trajectory.

    Parameters
    ----------
    per_tree_proba : (N, T, C) array — raw per-tree probability predictions
    ordering : (T,) int array — tree permutation

    Returns
    -------
    traj : (N, T, C) — cumulative average trajectory under the new ordering
    """
    reordered = per_tree_proba[:, ordering, :]  # (N, T, C)
    cum = np.cumsum(reordered, axis=1)
    denom = np.arange(1, reordered.shape[1] + 1, dtype=np.float64)[None, :, None]
    return cum / denom


def _qwyc_rf_stops(traj: np.ndarray, T: int) -> np.ndarray:
    """QWYC stopping criterion for RF (soft-vote averaging).

    Stop at tree t if: margin(t) > remaining_trees / T
    """
    N, T_actual, C = traj.shape
    stops = np.full(N, T_actual - 1, dtype=np.int64)
    for i in range(N):
        for t in range(MIN_TREES - 1, T_actual):
            probs = traj[i, t]
            sorted_probs = np.sort(probs)[::-1]
            margin = sorted_probs[0] - sorted_probs[1] if C > 1 else sorted_probs[0]
            remaining = T_actual - t - 1
            max_shift = remaining / T_actual
            if margin > max_shift:
                stops[i] = t
                break
    return stops


def _qwyc_gbm_stops(traj: np.ndarray, T: int) -> np.ndarray:
    """QWYC stopping criterion for GBM (staged prediction)."""
    N, T_actual, C = traj.shape
    stops = np.full(N, T_actual - 1, dtype=np.int64)
    for i in range(N):
        for t in range(MIN_TREES - 1, T_actual):
            probs = traj[i, t]
            sorted_probs = np.sort(probs)[::-1]
            margin = sorted_probs[0] - sorted_probs[1] if C > 1 else sorted_probs[0]
            remaining = T_actual - t - 1
            max_shift = remaining * GBM_LR
            rf_bound = remaining / T_actual
            max_shift = min(max_shift, rf_bound)
            if margin > max_shift:
                stops[i] = t
                break
    return stops


# ===================================================================
# Aggregation and LaTeX table generation
# ===================================================================

def aggregate_credit_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Credit Card sweep into the same format as tab_p2stop_sweep."""
    rows = []
    for theta, g in df.groupby("threshold"):
        acc_full = _mean_ci95(g["accuracy_full"])
        acc_p2 = _mean_ci95(g["accuracy_p2"])
        wr_p2 = _mean_bca_ci95(g["work_reduction_p2"], bounds=(0.0, 1.0))
        pair_acc = _paired_stats(g["accuracy_p2"], g["accuracy_full"])
        pair_wr = _paired_stats(g["work_reduction_p2"], g["work_reduction_dirichlet"])
        rows.append({
            "dataset": "credit", "threshold": theta,
            "delta_acc_mean": pair_acc["delta_mean"],
            "delta_acc_ci_lo": pair_acc["delta_ci95_low"],
            "delta_acc_ci_hi": pair_acc["delta_ci95_high"],
            "p_ttest": pair_acc["p_ttest"],
            "wr_mean": wr_p2["mean"],
            "wr_ci_lo": wr_p2["ci95_low"],
            "wr_ci_hi": wr_p2["ci95_high"],
            "p_wr": pair_wr["p_ttest"],
        })
    return pd.DataFrame(rows)


def aggregate_ablation(ablation_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build 4-column ablation table."""
    rows = []
    for label in ["tref10", "tref20", "tref30", "tref40"]:
        df = ablation_data.get(label)
        if df is None or df.empty:
            continue
        for ds in RF_DATASETS:
            ds_df = df[df["dataset"] == ds] if "dataset" in df.columns else df
            if ds_df.empty:
                continue
            acc = _mean_ci95(ds_df["accuracy_p2"])
            wr = _mean_bca_ci95(ds_df["work_reduction_p2"], bounds=(0.0, 1.0))
            rows.append({
                "window": label, "dataset": ds,
                "accuracy_mean": acc["mean"] * 100,
                "accuracy_ci_lo": acc["ci95_low"] * 100,
                "accuracy_ci_hi": acc["ci95_high"] * 100,
                "wr_mean": wr["mean"] * 100,
                "wr_ci_lo": wr["ci95_low"] * 100,
                "wr_ci_hi": wr["ci95_high"] * 100,
            })
    return pd.DataFrame(rows)


def aggregate_p2_vs_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate P² vs rolling comparison."""
    rows = []
    for (ds, model, backend, theta), g in df.groupby(
            ["dataset", "model", "backend", "threshold"]):
        acc = _mean_ci95(g["accuracy"])
        wr = _mean_bca_ci95(g["work_reduction"], bounds=(0.0, 1.0))
        dacc = _mean_ci95(g["delta_acc"])
        rows.append({
            "dataset": ds, "model": model, "backend": backend, "threshold": theta,
            "accuracy_mean": acc["mean"],
            "delta_acc_mean": dacc["mean"],
            "delta_acc_ci_lo": dacc["ci95_low"],
            "delta_acc_ci_hi": dacc["ci95_high"],
            "wr_mean": wr["mean"],
            "wr_ci_lo": wr["ci95_low"],
            "wr_ci_hi": wr["ci95_high"],
        })
    return pd.DataFrame(rows)


def aggregate_qwyc(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate QWYC comparison table."""
    rows = []
    for (ds, model), g in df.groupby(["dataset", "model"]):
        acc_full = _mean_ci95(g["accuracy_full"])
        acc_q = _mean_ci95(g["accuracy_qwyc"])
        acc_p = _mean_ci95(g["accuracy_p2stop"])
        wr_q = _mean_bca_ci95(g["wr_qwyc"], bounds=(0.0, 1.0))
        wr_p = _mean_bca_ci95(g["wr_p2stop"], bounds=(0.0, 1.0))

        pair_acc_qp = _paired_stats(g["accuracy_p2stop"], g["accuracy_qwyc"])
        pair_wr_qp = _paired_stats(g["wr_p2stop"], g["wr_qwyc"])

        # QWYC-full stats (ordering + margin)
        acc_qf = _mean_ci95(g["accuracy_qwyc_full"])
        wr_qf = _mean_bca_ci95(g["wr_qwyc_full"], bounds=(0.0, 1.0))
        pair_acc_qf_p = _paired_stats(g["accuracy_p2stop"], g["accuracy_qwyc_full"])
        pair_wr_qf_p = _paired_stats(g["wr_p2stop"], g["wr_qwyc_full"])
        pair_wr_qf_q = _paired_stats(g["wr_qwyc_full"], g["wr_qwyc"])

        row = {
            "dataset": ds, "model": model,
            "acc_full": acc_full["mean"],
            "acc_qwyc": acc_q["mean"],
            "acc_qwyc_ci_lo": acc_q["ci95_low"],
            "acc_qwyc_ci_hi": acc_q["ci95_high"],
            "acc_qwyc_full": acc_qf["mean"],
            "acc_qwyc_full_ci_lo": acc_qf["ci95_low"],
            "acc_qwyc_full_ci_hi": acc_qf["ci95_high"],
            "acc_p2stop": acc_p["mean"],
            "acc_p2stop_ci_lo": acc_p["ci95_low"],
            "acc_p2stop_ci_hi": acc_p["ci95_high"],
            "wr_qwyc": wr_q["mean"],
            "wr_qwyc_ci_lo": wr_q["ci95_low"],
            "wr_qwyc_ci_hi": wr_q["ci95_high"],
            "wr_qwyc_full": wr_qf["mean"],
            "wr_qwyc_full_ci_lo": wr_qf["ci95_low"],
            "wr_qwyc_full_ci_hi": wr_qf["ci95_high"],
            "wr_p2stop": wr_p["mean"],
            "wr_p2stop_ci_lo": wr_p["ci95_low"],
            "wr_p2stop_ci_hi": wr_p["ci95_high"],
            "delta_acc_p2_vs_qwyc": pair_acc_qp["delta_mean"],
            "delta_acc_p2_vs_qwyc_ci_lo": pair_acc_qp["delta_ci95_low"],
            "delta_acc_p2_vs_qwyc_ci_hi": pair_acc_qp["delta_ci95_high"],
            "p_ttest_acc": pair_acc_qp["p_ttest"],
            "delta_acc_p2_vs_qwyc_full": pair_acc_qf_p["delta_mean"],
            "delta_acc_p2_vs_qwyc_full_ci_lo": pair_acc_qf_p["delta_ci95_low"],
            "delta_acc_p2_vs_qwyc_full_ci_hi": pair_acc_qf_p["delta_ci95_high"],
            "p_ttest_acc_vs_qwyc_full": pair_acc_qf_p["p_ttest"],
            "delta_wr_p2_vs_qwyc": pair_wr_qp["delta_mean"],
            "delta_wr_p2_vs_qwyc_ci_lo": pair_wr_qp["delta_ci95_low"],
            "delta_wr_p2_vs_qwyc_ci_hi": pair_wr_qp["delta_ci95_high"],
            "p_ttest_wr": pair_wr_qp["p_ttest"],
            "delta_wr_p2_vs_qwyc_full": pair_wr_qf_p["delta_mean"],
            "delta_wr_p2_vs_qwyc_full_ci_lo": pair_wr_qf_p["delta_ci95_low"],
            "delta_wr_p2_vs_qwyc_full_ci_hi": pair_wr_qf_p["delta_ci95_high"],
            "p_ttest_wr_vs_qwyc_full": pair_wr_qf_p["p_ttest"],
            "delta_wr_qwyc_full_vs_qwyc": pair_wr_qf_q["delta_mean"],
            "p_ttest_wr_full_vs_margin": pair_wr_qf_q["p_ttest"],
        }

        # Dirichlet stats (RF only)
        if "accuracy_dirichlet" in g.columns:
            dir_valid = g["accuracy_dirichlet"].dropna()
            if len(dir_valid) > 0:
                acc_d = _mean_ci95(g["accuracy_dirichlet"])
                wr_d = _mean_bca_ci95(g["wr_dirichlet"].dropna(), bounds=(0.0, 1.0))
                row["acc_dirichlet"] = acc_d["mean"]
                row["wr_dirichlet"] = wr_d["mean"]

        rows.append(row)
    return pd.DataFrame(rows)


# ===================================================================
# LaTeX generation
# ===================================================================

def _fmt_pval(p: float) -> str:
    if not np.isfinite(p):
        return "---"
    if p < 1e-10:
        s = f"{p:.2e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-")
        return "$" + s + "}$"
    if p < 0.001:
        s = f"{p:.2e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-")
        return "$" + s + "}$"
    return f"${p:.3f}$"


def _fmt_num(x: float, fmt: str = ".4f") -> str:
    if not np.isfinite(x):
        return "---"
    return f"${x:{fmt}}$"


def generate_latex_credit_rows(agg: pd.DataFrame) -> str:
    """Generate LaTeX rows for Credit Card to insert into Table 4."""
    lines = [r"\midrule", r"\multirow{6}{*}{Credit Card}"]
    for _, row in agg.iterrows():
        theta = row["threshold"]
        dacc = f"${row['delta_acc_mean']:+.4f}$" if np.isfinite(row['delta_acc_mean']) else "---"
        ci = f"[${row['delta_acc_ci_lo']:+.4f}$, ${row['delta_acc_ci_hi']:+.4f}$]"
        p = _fmt_pval(row["p_ttest"])
        wr = f"{row['wr_mean']:.3f}"
        wr_ci = f"[{row['wr_ci_lo']:.3f}, {row['wr_ci_hi']:.3f}]"
        p_wr = _fmt_pval(row["p_wr"])
        lines.append(f"  & {theta:.2f} & {dacc} {ci} & {p} & {wr} {wr_ci} & {p_wr} \\\\")
    return "\n".join(lines)


def generate_latex_ablation(agg: pd.DataFrame) -> str:
    """Generate full 4-column ablation table LaTeX."""
    header = r"""\begin{table}[H]
\caption{Sensitivity of P\textsuperscript{2}-STOP to the reference
  window $W_{\textup{ref}}$ ($\theta = 0.10$). Window endpoints are
  reported as half-open intervals $[t_0,t_1)$.}\label{tab:tref_ablation}
\centering
\begin{tabular}{lrrrr}
\toprule
\textbf{Dataset}
  & $[5,10)$
  & $[10,20)$
  & $[10,30)$
  & $[20,40)$ \\
\midrule"""

    ds_names = {"mnist": "MNIST", "covertype": "Covertype",
                "higgs": "HIGGS", "credit": "Credit Card"}
    windows = ["tref10", "tref20", "tref30", "tref40"]

    # Accuracy section
    lines = [header, r"\multicolumn{5}{l}{\textit{Accuracy (\%)}} \\"]
    for ds in RF_DATASETS:
        vals = []
        for w in windows:
            match = agg[(agg["window"] == w) & (agg["dataset"] == ds)]
            if len(match) > 0:
                vals.append(f"{match.iloc[0]['accuracy_mean']:.2f}")
            else:
                vals.append("---")
        name = ds_names.get(ds, ds)
        lines.append(f"{name} & {' & '.join(vals)} \\\\")

    # Work reduction section
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{l}{\textit{Work Reduction (\%)}} \\")
    for ds in RF_DATASETS:
        vals = []
        for w in windows:
            match = agg[(agg["window"] == w) & (agg["dataset"] == ds)]
            if len(match) > 0:
                vals.append(f"{match.iloc[0]['wr_mean']:.2f}")
            else:
                vals.append("---")
        name = ds_names.get(ds, ds)
        lines.append(f"{name} & {' & '.join(vals)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_latex_p2_vs_rolling(agg: pd.DataFrame) -> str:
    """Generate P² vs rolling-window comparison table."""
    ds_names = {"mnist": "MNIST", "covertype": "Covertype"}
    model_names = {"rf": "RF", "gbm": "XGBoost"}

    header = r"""\begin{table}[H]
\caption{Comparison of full-prefix P\textsuperscript{2} backend vs
  rolling-window exact backend ($\theta = 0.10$, 5 seeds).
  Both backends use the same stopping rule; only the scale
  estimation differs.}\label{tab:p2_vs_rolling}
\centering
\begin{tabularx}{\textwidth}{llccc}
\toprule
\textbf{Dataset / Model} & \textbf{Backend}
  & \textbf{$\Delta$Acc vs Full}
  & \textbf{Work Reduction}
  & \textbf{WR [95\% CI]} \\
\midrule"""

    lines = [header]
    sub = agg[agg["threshold"] == 0.10].copy()
    for (ds, model), g in sub.groupby(["dataset", "model"]):
        label = f"{ds_names.get(ds, ds)} / {model_names.get(model, model)}"
        for _, row in g.iterrows():
            backend = row["backend"]
            bname = "Rolling-window" if backend == "rolling" else "Full-prefix P$^2$"
            dacc = f"${row['delta_acc_mean']:+.4f}$" if np.isfinite(row['delta_acc_mean']) else "---"
            wr = f"{row['wr_mean']:.3f}"
            wr_ci = f"[{row['wr_ci_lo']:.3f}, {row['wr_ci_hi']:.3f}]"
            lines.append(f"{label} & {bname} & {dacc} & {wr} & {wr_ci} \\\\")
        lines.append(r"\cmidrule{2-5}")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_latex_qwyc(agg: pd.DataFrame) -> str:
    """Generate QWYC comparison table with margin-only and full variants."""
    ds_names = {"mnist": "MNIST", "covertype": "Covertype",
                "higgs": "HIGGS", "credit": "Credit Card"}

    header = r"""\begin{table}[H]
\caption{Comparison of P\textsuperscript{2}-STOP ($\theta = 0.10$) against
  QWYC~\cite{wang2021qwyc} in two configurations: \emph{margin-only}
  (component~(ii), fixed tree order) and \emph{full} (components~(i)+(ii),
  validation-optimized ordering plus margin stop).
  For XGBoost, ordering optimization is not applicable (trees are
  sequential correctors), so QWYC-full $\equiv$ QWYC-margin.
  (5 seeds, seed-level paired tests).}\label{tab:qwyc_comparison}
\begin{adjustwidth}{-\extralength}{0cm}
{\footnotesize
\begin{tabularx}{\fulllength}{
  l l
  >{\centering\arraybackslash}X
  >{\centering\arraybackslash}X
  >{\centering\arraybackslash}X
  >{\centering\arraybackslash}X
  >{\centering\arraybackslash}X
  >{\centering\arraybackslash}X
  >{\centering\arraybackslash}X
  >{\centering\arraybackslash\hsize=1.15\hsize}X
  >{\centering\arraybackslash\hsize=0.85\hsize}X
}
\toprule
\textbf{Dataset} & \textbf{Model}
  & \textbf{Acc (Full)}
  & \textbf{Acc (QWYC-m)}
  & \textbf{Acc (QWYC-f)}
  & \textbf{Acc (P\textsuperscript{2})}
  & \textbf{WR (QWYC-m)}
  & \textbf{WR (QWYC-f)}
  & \textbf{WR (P\textsuperscript{2})}
  & \textbf{$\Delta$WR(P\textsuperscript{2}$-$Qf)}
  & \textbf{$p$ (paired)} \\
\midrule"""

    lines = [header]
    for _, row in agg.iterrows():
        ds = ds_names.get(row["dataset"], row["dataset"])
        model = "RF" if row["model"] == "rf" else "XGBoost"
        af = f"{row['acc_full']:.4f}"
        aq = f"{row['acc_qwyc']:.4f}"
        aqf = f"{row['acc_qwyc_full']:.4f}"
        ap = f"{row['acc_p2stop']:.4f}"
        wrq = f"{row['wr_qwyc']:.3f}"
        wrqf = f"{row['wr_qwyc_full']:.3f}"
        wrp = f"{row['wr_p2stop']:.3f}"
        dwr = f"${row['delta_wr_p2_vs_qwyc_full']:+.3f}$"
        p_val = row.get("p_ttest_wr_vs_qwyc_full", float("nan"))
        p_str = _fmt_pval(p_val)
        lines.append(f"{ds} & {model} & {af} & {aq} & {aqf} & {ap} & {wrq} & {wrqf} & {wrp} & {dwr} & {p_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"}")
    lines.append(r"\end{adjustwidth}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ===================================================================
# Pareto figure update
# ===================================================================

def generate_pareto_figure(qwyc_df: pd.DataFrame, output_dir: Path, logger: logging.Logger):
    """Generate updated 2×2 Pareto figure including QWYC."""
    import matplotlib.pyplot as plt

    logger.info("  Generating updated Pareto figure with QWYC ...")
    ds_names = {"mnist": "MNIST", "covertype": "Covertype",
                "higgs": "HIGGS", "credit": "Credit Card"}

    rf_data = qwyc_df[qwyc_df["model"] == "rf"]
    datasets = [d for d in RF_DATASETS if d in rf_data["dataset"].values]

    n_ds = len(datasets)
    if n_ds == 0:
        logger.warning("  No RF data for Pareto figure")
        return

    ncols = min(n_ds, 2)
    nrows = (n_ds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), dpi=160)
    if n_ds == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        ds_data = rf_data[rf_data["dataset"] == ds]

        # Aggregate per seed
        for method, color, marker, label in [
            ("qwyc", "#E69F00", "D", "QWYC (margin)"),
            ("qwyc_full", "#CC79A7", "s", "QWYC (full)"),
            ("p2stop", "#0072B2", "o", "P²-STOP"),
            ("dirichlet", "#D55E00", "x", "Dirichlet"),
        ]:
            acc_col = f"accuracy_{method}"
            wr_col = f"wr_{method}"
            if acc_col not in ds_data.columns:
                continue
            valid = ds_data[[acc_col, wr_col]].dropna()
            if valid.empty:
                continue
            acc_mean = valid[acc_col].mean()
            wr_mean = valid[wr_col].mean()
            ax.scatter([wr_mean], [acc_mean], s=100, c=color, marker=marker,
                       label=label, zorder=5, edgecolors="black", linewidths=0.5)

        # Full ensemble point
        acc_full = ds_data["accuracy_full"].mean()
        ax.scatter([0.0], [acc_full], s=120, c="black", marker="*",
                   label="Full ensemble", zorder=5)

        ax.set_title(ds_names.get(ds, ds), fontsize=14)
        ax.set_xlabel("Work Reduction", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for idx in range(n_ds, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "pareto_with_qwyc.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "pareto_with_qwyc.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("  Pareto figure saved to %s", output_dir / "pareto_with_qwyc.png")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase D revision experiments")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "phase_d")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: reduce seeds to 2 and datasets to mnist only")
    args = parser.parse_args()

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir / "phase_d.log")
    logger.info("Phase D experiments — %s", timestamp)
    logger.info("Output directory: %s", output_dir)

    global SEEDS, RF_DATASETS, GBM_DATASETS
    if args.quick:
        SEEDS = [42, 123]
        RF_DATASETS = ["mnist"]
        GBM_DATASETS = ["mnist"]
        logger.info("QUICK MODE: 2 seeds, mnist only")

    t0_total = time.perf_counter()

    # ------------------------------------------------------------------
    # II-1: Credit Card sweep
    # ------------------------------------------------------------------
    ii1_dir = output_dir / "ii1_credit_sweep"
    ii1_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    credit_df = task_ii1_credit_card_sweep(ii1_dir, args.data_dir, logger)
    credit_agg = aggregate_credit_sweep(credit_df)
    credit_agg.to_csv(ii1_dir / "credit_sweep_aggregated.csv", index=False)
    credit_latex = generate_latex_credit_rows(credit_agg)
    (ii1_dir / "credit_sweep_latex_rows.tex").write_text(credit_latex, encoding="utf-8")
    logger.info("  II-1 done in %.1fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # II-2: Ablation with [10,30)
    # ------------------------------------------------------------------
    ii2_dir = output_dir / "ii2_ablation"
    ii2_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ablation_data = task_ii2_ablation(ii2_dir, args.data_dir, logger)
    ablation_agg = aggregate_ablation(ablation_data)
    ablation_agg.to_csv(ii2_dir / "ablation_aggregated.csv", index=False)
    ablation_latex = generate_latex_ablation(ablation_agg)
    (ii2_dir / "ablation_table.tex").write_text(ablation_latex, encoding="utf-8")
    logger.info("  II-2 done in %.1fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # II-3: P² vs rolling-window
    # ------------------------------------------------------------------
    ii3_dir = output_dir / "ii3_p2_vs_rolling"
    ii3_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    p2_vs_rolling_df = task_ii3_p2_vs_rolling(ii3_dir, args.data_dir, logger)
    p2_vs_rolling_agg = aggregate_p2_vs_rolling(p2_vs_rolling_df)
    p2_vs_rolling_agg.to_csv(ii3_dir / "p2_vs_rolling_aggregated.csv", index=False)
    p2_vs_rolling_latex = generate_latex_p2_vs_rolling(p2_vs_rolling_agg)
    (ii3_dir / "p2_vs_rolling_table.tex").write_text(p2_vs_rolling_latex, encoding="utf-8")
    logger.info("  II-3 done in %.1fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # II-4: QWYC baseline
    # ------------------------------------------------------------------
    ii4_dir = output_dir / "ii4_qwyc"
    ii4_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    qwyc_df = task_ii4_qwyc(ii4_dir, args.data_dir, logger)
    qwyc_agg = aggregate_qwyc(qwyc_df)
    qwyc_agg.to_csv(ii4_dir / "qwyc_aggregated.csv", index=False)
    qwyc_latex = generate_latex_qwyc(qwyc_agg)
    (ii4_dir / "qwyc_table.tex").write_text(qwyc_latex, encoding="utf-8")
    generate_pareto_figure(qwyc_df, ii4_dir, logger)
    logger.info("  II-4 done in %.1fs", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = time.perf_counter() - t0_total
    logger.info("=" * 72)
    logger.info("Phase D complete in %.1fs (%.1f min)", total_time, total_time / 60)
    logger.info("Output directory: %s", output_dir)

    summary = {
        "timestamp": timestamp,
        "total_time_sec": total_time,
        "seeds": SEEDS,
        "rf_datasets": RF_DATASETS,
        "gbm_datasets": GBM_DATASETS,
        "tasks": {
            "ii1_credit_sweep": {
                "status": "complete",
                "output": str(ii1_dir),
                "n_rows": len(credit_df),
            },
            "ii2_ablation": {
                "status": "complete",
                "output": str(ii2_dir),
                "windows": list(ABLATION_WINDOWS.keys()),
            },
            "ii3_p2_vs_rolling": {
                "status": "complete",
                "output": str(ii3_dir),
                "n_rows": len(p2_vs_rolling_df),
            },
            "ii4_qwyc": {
                "status": "complete",
                "output": str(ii4_dir),
                "n_rows": len(qwyc_df),
            },
        },
    }
    with open(output_dir / "phase_d_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Summary written to %s", output_dir / "phase_d_summary.json")
    print(f"\n{'='*72}")
    print(f"Phase D complete.  Results in: {output_dir}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
