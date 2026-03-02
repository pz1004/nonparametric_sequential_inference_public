#!/usr/bin/env python3
"""
================================================================================
Unified Phase-1 Experiment Runner for MDPI Manuscript
================================================================================

Runs all experiments needed for the Phase-1 manuscript, with:
- multi-seed execution
- structured logging
- CI/paired-test table exports
- true end-to-end timing benchmark (full RF vs early-stop RF)
- optional GBM cross-model validation (XGBoost/LightGBM)
- CUSUM variant evaluation
- contamination robustness study (5%/15%/25% corrupted trees)

Manuscript artefacts covered
-----------------------------
TABLE  tab:p2stop_sweep        Phase 1 threshold sweep
TABLE  tab:p2stop_vs_dirichlet Best per-seed Phase 1 comparison + paired tests
FIGURE fig:scale_trajectories  Phase 1 scale diagnostics
FIGURE fig:pareto              Accuracy-vs-work Pareto
TABLE  tab:best_summary        Phase-1 best configuration summary
TABLE  tab_gbm_p2stop_sweep    Optional GBM threshold sweep
TABLE  tab_gbm_best_summary    Optional GBM best per-seed summary
TABLE  tab_cusum_vs_relative   CUSUM detector versus relative-threshold detector
TABLE  tab_robustness          Contamination robustness summary
TABLE  tab_robustness_iqr_vs_mean Robust versus non-robust baseline paired comparison
TABLE  tab:timing              End-to-end inference timing (full vs early-stop)
TABLE  tab_timing_numba_poc    Optional Numba timing benchmark (compiled path)
TABLE  tab:tref_ablation       t_ref sensitivity ablation
FIGURE fig_pareto_all_datasets 2×2 Pareto frontier across all datasets
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import betainc
from scipy.stats import norm
from scipy.stats import t as student_t
from scipy.stats import ttest_rel, wilcoxon

# Ensure in-process imports (e.g., timing benchmark) can resolve project packages.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.shared.p2_streaming import HAS_NUMBA, update_p2stop_state_numba
from experiments.shared.numba_rf_inference import (
    compile_rf_forest_for_numba,
    rf_full_inference_numba,
    rf_p2stop_inference_numba,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("run_all_experiments")
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
# Subprocess runner
# ---------------------------------------------------------------------------


def run_command(
    cmd: List[str],
    logger: logging.Logger,
    label: str,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run a command, log it, and raise on failure."""
    logger.info("=" * 72)
    logger.info("STEP: %s", label)
    logger.info("CMD : %s", " ".join(cmd))
    t0 = time.perf_counter()

    run_env = os.environ.copy()
    run_env.setdefault("MPLBACKEND", "Agg")
    if env is not None:
        run_env.update(env)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=run_env,
    )
    elapsed = time.perf_counter() - t0

    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            logger.debug("  [stdout] %s", line)
    if result.stderr.strip():
        for line in result.stderr.strip().splitlines():
            logger.debug("  [stderr] %s", line)

    if result.returncode != 0:
        logger.error("FAILED (exit %d) after %.1fs: %s", result.returncode, elapsed, label)
        logger.error("stderr:\n%s", result.stderr[-2000:] if result.stderr else "(empty)")
        raise RuntimeError(f"Step '{label}' failed with exit code {result.returncode}")

    logger.info("OK  : %s  (%.1fs)", label, elapsed)
    return result


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
DATASETS = "mnist,covertype,higgs,credit"
GBM_DATASETS = "mnist,covertype,higgs"
NUMBA_POC_DATASETS = "all"

# Manuscript-consistent defaults
PHASE1_N_TREES = 200
GBM_N_TREES = 500
MAX_TRAIN = 20_000
MAX_TEST = 5_000
HIGGS_MAX_ROWS = 500_000

# Threshold sweep in manuscript
P2STOP_THRESHOLDS = "0.05,0.10,0.15,0.20,0.25,0.30"
ROBUSTNESS_CONTAMINATION_LEVELS = "0.05,0.15,0.25"

# t_ref ablation
TREF_ABLATION_WINDOWS = {
    "tref10": "5,10",
    "tref20": "10,20",
    "tref40": "20,40",
}


# ---------------------------------------------------------------------------
# Experiment steps (Phase 1 only)
# ---------------------------------------------------------------------------


def step_data_integrity(
    args: argparse.Namespace,
    output_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Path]:
    """Run local data integrity checks and store reports in this run directory."""
    report_json = output_dir / "data_integrity_report.json"
    report_md = output_dir / "data_integrity_report.md"
    cmd = [
        sys.executable,
        "scripts/verify_data_integrity.py",
        "--data-dir",
        str(args.data_dir),
        "--higgs-sample-rows",
        str(args.higgs_integrity_sample_rows),
        "--output-json",
        str(report_json),
        "--output-md",
        str(report_md),
    ]
    run_command(cmd, logger, "Data integrity verification")
    return {"json": report_json, "md": report_md}



def step_phase1_main(
    args: argparse.Namespace,
    output_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Run Phase 1 main experiment."""
    seeds_str = ",".join(str(s) for s in args.seeds)
    cmd = [
        sys.executable,
        "-m",
        "experiments.phase1_changepoint.run_phase1_changepoint",
        "--datasets",
        args.datasets,
        "--data-dir",
        str(args.data_dir),
        "--n-trees",
        str(args.phase1_n_trees),
        "--thresholds",
        P2STOP_THRESHOLDS,
        "--ref-window",
        "10,30",
        "--warmup",
        "10",
        "--min-trees",
        "10",
        "--dirichlet-threshold",
        "0.95",
        "--max-train",
        str(args.max_train),
        "--max-test",
        str(args.max_test),
        "--mnist-max-rows",
        str(args.mnist_max_rows),
        "--covertype-max-rows",
        str(args.covertype_max_rows),
        "--credit-max-rows",
        str(args.credit_max_rows),
        "--higgs-max-rows",
        str(args.higgs_max_rows),
        "--seeds",
        seeds_str,
        "--scale-mode",
        "rolling",
        "--rolling-window",
        "20",
        "--detection-method",
        "relative",
        "--output-dir",
        str(output_dir / "phase1_main"),
    ]
    run_command(cmd, logger, "Phase 1 — main threshold sweep (rolling, relative)")

    runs = sorted((output_dir / "phase1_main").glob("run_*"))
    return runs[-1] if runs else output_dir / "phase1_main"


def step_phase1_gbm_main(
    args: argparse.Namespace,
    output_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Run optional GBM P2-STOP sweep to validate model-agnosticity."""
    requested = str(args.gbm_backend).lower().strip()
    resolved = requested
    has_xgb = importlib.util.find_spec("xgboost") is not None
    has_lgb = importlib.util.find_spec("lightgbm") is not None

    if requested == "xgboost" and not has_xgb:
        resolved = "lightgbm" if has_lgb else "sklearn"
        logger.warning(
            "Requested GBM backend 'xgboost' is unavailable in this environment. "
            "Falling back to '%s'.",
            resolved,
        )
    elif requested == "lightgbm" and not has_lgb:
        resolved = "xgboost" if has_xgb else "sklearn"
        logger.warning(
            "Requested GBM backend 'lightgbm' is unavailable in this environment. "
            "Falling back to '%s'.",
            resolved,
        )
    elif requested == "auto":
        if has_xgb:
            resolved = "xgboost"
        elif has_lgb:
            resolved = "lightgbm"
        else:
            resolved = "sklearn"
        logger.info("GBM backend auto-resolution: %s", resolved)
    elif requested == "sklearn":
        resolved = "sklearn"

    seeds_str = ",".join(str(s) for s in args.seeds)
    cmd = [
        sys.executable,
        "-m",
        "experiments.phase1_changepoint.run_phase1_gbm_changepoint",
        "--datasets",
        args.gbm_datasets,
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(output_dir / "phase1_gbm_main"),
        "--backend",
        resolved,
        "--n-trees",
        str(args.gbm_n_trees),
        "--learning-rate",
        str(args.gbm_learning_rate),
        "--max-depth",
        str(args.gbm_max_depth),
        "--thresholds",
        P2STOP_THRESHOLDS,
        "--ref-window",
        "10,30",
        "--warmup",
        "10",
        "--min-trees",
        "10",
        "--rolling-window",
        "20",
        "--max-train",
        str(args.gbm_max_train),
        "--max-test",
        str(args.gbm_max_test),
        "--mnist-max-rows",
        str(args.mnist_max_rows),
        "--covertype-max-rows",
        str(args.covertype_max_rows),
        "--credit-max-rows",
        str(args.credit_max_rows),
        "--higgs-max-rows",
        str(args.higgs_max_rows),
        "--seeds",
        seeds_str,
    ]
    run_command(cmd, logger, "Phase 1 (GBM) — P2-STOP threshold sweep")

    runs = sorted((output_dir / "phase1_gbm_main").glob("run_*"))
    return runs[-1] if runs else output_dir / "phase1_gbm_main"


def step_phase1_cusum_main(
    args: argparse.Namespace,
    output_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Run Phase 1 with CUSUM detection for explicit experimental comparison."""
    seeds_str = ",".join(str(s) for s in args.seeds)
    cmd = [
        sys.executable,
        "-m",
        "experiments.phase1_changepoint.run_phase1_changepoint",
        "--datasets",
        args.datasets,
        "--data-dir",
        str(args.data_dir),
        "--n-trees",
        str(args.phase1_n_trees),
        "--thresholds",
        "0.10",
        "--ref-window",
        "10,30",
        "--warmup",
        "10",
        "--min-trees",
        "10",
        "--dirichlet-threshold",
        "0.95",
        "--max-train",
        str(args.max_train),
        "--max-test",
        str(args.max_test),
        "--mnist-max-rows",
        str(args.mnist_max_rows),
        "--covertype-max-rows",
        str(args.covertype_max_rows),
        "--credit-max-rows",
        str(args.credit_max_rows),
        "--higgs-max-rows",
        str(args.higgs_max_rows),
        "--seeds",
        seeds_str,
        "--scale-mode",
        "rolling",
        "--rolling-window",
        "20",
        "--detection-method",
        "cusum",
        "--cusum-k",
        str(args.cusum_k),
        "--cusum-h",
        str(args.cusum_h),
        "--output-dir",
        str(output_dir / "phase1_cusum_main"),
    ]
    run_command(cmd, logger, "Phase 1 — CUSUM detector evaluation")

    runs = sorted((output_dir / "phase1_cusum_main").glob("run_*"))
    return runs[-1] if runs else output_dir / "phase1_cusum_main"


def step_phase1_robustness(
    args: argparse.Namespace,
    output_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Run contamination robustness experiment."""
    seeds_str = ",".join(str(s) for s in args.seeds)
    cmd = [
        sys.executable,
        "-m",
        "experiments.phase1_changepoint.run_phase1_robustness_contamination",
        "--datasets",
        args.datasets,
        "--data-dir",
        str(args.data_dir),
        "--n-trees",
        str(args.phase1_n_trees),
        "--thresholds",
        P2STOP_THRESHOLDS,
        "--contamination-levels",
        args.robustness_contamination_levels,
        "--ref-window",
        "10,30",
        "--warmup",
        "10",
        "--min-trees",
        "10",
        "--rolling-window",
        "20",
        "--max-train",
        str(args.max_train),
        "--max-test",
        str(args.max_test),
        "--mnist-max-rows",
        str(args.mnist_max_rows),
        "--covertype-max-rows",
        str(args.covertype_max_rows),
        "--credit-max-rows",
        str(args.credit_max_rows),
        "--higgs-max-rows",
        str(args.higgs_max_rows),
        "--seeds",
        seeds_str,
        "--output-dir",
        str(output_dir / "phase1_robustness"),
    ]
    run_command(cmd, logger, "Phase 1 — contamination robustness experiment")

    runs = sorted((output_dir / "phase1_robustness").glob("run_*"))
    return runs[-1] if runs else output_dir / "phase1_robustness"



def step_phase1_tref_ablation(
    args: argparse.Namespace,
    output_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Path]:
    """Run Phase 1 with different reference windows for tab:tref_ablation."""
    seeds_str = ",".join(str(s) for s in args.seeds)
    run_dirs: Dict[str, Path] = {}

    for label, ref_window in TREF_ABLATION_WINDOWS.items():
        ablation_dir = output_dir / f"phase1_ablation_{label}"
        cmd = [
            sys.executable,
            "-m",
            "experiments.phase1_changepoint.run_phase1_changepoint",
            "--datasets",
            args.datasets,
            "--data-dir",
            str(args.data_dir),
            "--n-trees",
            str(args.phase1_n_trees),
            "--thresholds",
            "0.10",
            "--ref-window",
            ref_window,
            "--warmup",
            "10",
            "--min-trees",
            "10",
            "--dirichlet-threshold",
            "0.95",
            "--max-train",
            str(args.max_train),
            "--max-test",
            str(args.max_test),
            "--mnist-max-rows",
            str(args.mnist_max_rows),
            "--covertype-max-rows",
            str(args.covertype_max_rows),
            "--credit-max-rows",
            str(args.credit_max_rows),
            "--higgs-max-rows",
            str(args.higgs_max_rows),
            "--seeds",
            seeds_str,
            "--scale-mode",
            "rolling",
            "--rolling-window",
            "20",
            "--detection-method",
            "relative",
            "--output-dir",
            str(ablation_dir),
        ]
        run_command(cmd, logger, f"Phase 1 — t_ref ablation ({label}, ref-window={ref_window})")
        runs = sorted(ablation_dir.glob("run_*"))
        run_dirs[label] = runs[-1] if runs else ablation_dir

    return run_dirs


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

BOOTSTRAP_CI_SEED = 12345
BOOTSTRAP_CI_N = 20000


def _mean_bca_ci95(
    values: Iterable[float],
    *,
    n_boot: int = BOOTSTRAP_CI_N,
    seed: int = BOOTSTRAP_CI_SEED,
    bounds: Tuple[float, float] | None = None,
) -> Dict[str, float]:
    """BCa bootstrap CI for the mean (with optional bounds).

    This is mainly used for bounded metrics such as work reduction, where
    Gaussian/t intervals can cross logical bounds with small n (e.g., 5 seeds).
    """
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    mean = float(np.mean(arr))
    if arr.size == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
        }

    std = float(np.std(arr, ddof=1))
    n = int(arr.size)

    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(n_boot), n), dtype=np.int64)
    boot_means = np.mean(arr[idx], axis=1)

    theta_hat = mean
    prop = float(np.mean(boot_means < theta_hat))
    prop = min(max(prop, 1.0 / (2.0 * n_boot)), 1.0 - 1.0 / (2.0 * n_boot))
    z0 = float(norm.ppf(prop))

    # Jackknife acceleration for the mean (stable O(n)).
    total = float(np.sum(arr))
    jack = np.empty(n, dtype=np.float64)
    for i in range(n):
        jack[i] = (total - float(arr[i])) / float(n - 1)
    jack_mean = float(np.mean(jack))
    diffs = jack_mean - jack
    denom = float(np.sum(diffs**2)) ** 1.5
    if denom > 0.0:
        a = float(np.sum(diffs**3)) / (6.0 * denom)
    else:
        a = 0.0

    def _adj(alpha: float) -> float:
        z = float(norm.ppf(alpha))
        num = z0 + z
        den = 1.0 - a * num
        if den == 0.0:
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

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": lo,
        "ci95_high": hi,
    }


def _mean_ci95(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    mean = float(np.mean(arr))
    if arr.size == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
        }

    std = float(np.std(arr, ddof=1))
    sem = std / np.sqrt(arr.size)
    tcrit = float(student_t.ppf(0.975, arr.size - 1))
    half = tcrit * sem
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - half),
        "ci95_high": float(mean + half),
    }



def _paired_stats(x: Iterable[float], y: Iterable[float]) -> Dict[str, float]:
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]

    if x_arr.size == 0:
        return {
            "n": 0,
            "delta_mean": float("nan"),
            "delta_std": float("nan"),
            "delta_ci95_low": float("nan"),
            "delta_ci95_high": float("nan"),
            "p_ttest": float("nan"),
            "p_wilcoxon": float("nan"),
        }

    diff = x_arr - y_arr
    ci = _mean_ci95(diff)

    p_ttest = float("nan")
    p_wilcoxon = float("nan")

    if x_arr.size >= 2 and np.nanstd(diff, ddof=1) > 0:
        try:
            p_ttest = float(ttest_rel(x_arr, y_arr).pvalue)
        except Exception:
            p_ttest = float("nan")

    if x_arr.size >= 2 and not np.allclose(diff, 0.0):
        try:
            p_wilcoxon = float(wilcoxon(x_arr, y_arr, zero_method="wilcox", alternative="two-sided").pvalue)
        except Exception:
            p_wilcoxon = float("nan")

    return {
        "n": ci["n"],
        "delta_mean": ci["mean"],
        "delta_std": ci["std"],
        "delta_ci95_low": ci["ci95_low"],
        "delta_ci95_high": ci["ci95_high"],
        "p_ttest": p_ttest,
        "p_wilcoxon": p_wilcoxon,
    }



def _select_best_phase1(df_ds: pd.DataFrame) -> pd.Series:
    if df_ds.empty:
        raise ValueError("Phase 1 subset is empty")

    has_val = {
        "delta_acc_vs_full_val",
        "work_reduction_p2_val",
        "accuracy_p2_val",
    }.issubset(df_ds.columns)

    if has_val:
        # One-sided validation constraint: allow improvements, limit degradation.
        feasible = df_ds[df_ds["delta_acc_vs_full_val"] >= -0.005]
        order_cols = ["work_reduction_p2_val", "accuracy_p2_val", "threshold"]
        ascending = [False, False, True]
    else:
        feasible = df_ds[df_ds["delta_acc_vs_full"] >= -0.005]
        order_cols = ["work_reduction_p2", "accuracy_p2", "threshold"]
        ascending = [False, False, True]

    if feasible.empty:
        # If no threshold meets the accuracy tolerance, fall back to the
        # best validation accuracy (tie-break by larger work reduction).
        if has_val:
            order_cols = ["accuracy_p2_val", "work_reduction_p2_val", "threshold"]
            ascending = [False, False, True]
        else:
            order_cols = ["accuracy_p2", "work_reduction_p2", "threshold"]
            ascending = [False, False, True]
        feasible = df_ds

    return feasible.sort_values(order_cols, ascending=ascending).iloc[0]


def _select_best_phase1_gbm(df_ds: pd.DataFrame) -> pd.Series:
    if df_ds.empty:
        raise ValueError("Phase 1 GBM subset is empty")
    feasible = df_ds[df_ds["delta_acc_vs_full_val"] >= -0.005]
    if feasible.empty:
        # Same fallback: maximize validation accuracy if all candidates violate
        # the accuracy tolerance.
        feasible = df_ds
        return feasible.sort_values(
            ["accuracy_p2_val", "work_reduction_p2_val", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]
    return feasible.sort_values(
        ["work_reduction_p2_val", "accuracy_p2_val", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]


# ---------------------------------------------------------------------------
# Timing benchmark (true end-to-end path)
# ---------------------------------------------------------------------------


def _stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_rows: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_rows <= 0 or len(y) <= max_rows:
        return X, y

    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=max_rows, replace=False)
        return X[idx], y[idx]

    if max_rows < classes.size:
        raise ValueError(f"max_rows={max_rows} smaller than class count={classes.size}; cannot stratify")

    proportions = counts / counts.sum()
    alloc = np.floor(proportions * max_rows).astype(int)
    alloc = np.maximum(alloc, 1)
    alloc = np.minimum(alloc, counts)

    while alloc.sum() > max_rows:
        candidates = np.where(alloc > 1)[0]
        if candidates.size == 0:
            break
        idx = candidates[np.argmax(alloc[candidates])]
        alloc[idx] -= 1

    remaining = max_rows - int(alloc.sum())
    if remaining > 0:
        spare = counts - alloc
        order = np.argsort(-spare)
        for idx in order:
            if remaining <= 0:
                break
            if spare[idx] <= 0:
                continue
            take = min(int(spare[idx]), int(remaining))
            alloc[idx] += take
            remaining -= take

    rng = np.random.default_rng(seed)
    selected_parts = []
    for cls, take in zip(classes, alloc):
        class_idx = np.where(y == cls)[0]
        if take >= class_idx.size:
            chosen = class_idx
        else:
            chosen = rng.choice(class_idx, size=int(take), replace=False)
        selected_parts.append(chosen)

    selected = np.concatenate(selected_parts)
    rng.shuffle(selected)
    return X[selected], y[selected]



def _prepare_rf_data(
    bundle: Any,
    seed: int,
    max_train: int,
    max_test: int,
):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=0.3,
        random_state=seed,
        stratify=bundle.y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=1.0 / 7.0,
        random_state=seed + 100,
        stratify=y_trainval,
    )

    if max_train > 0 and len(X_train) > max_train:
        X_train, y_train = _stratified_subsample(X_train, y_train, max_train, seed)

    if max_test > 0 and len(X_test) > max_test:
        X_test, y_test = _stratified_subsample(X_test, y_test, max_test, seed + 1)

    # Keep val split behavior aligned with Phase 1 even though timing does not use val.
    if max_test > 0 and len(X_val) > max_test:
        X_val, y_val = _stratified_subsample(X_val, y_val, max_test, seed + 2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test



def _rf_full_inference(model: Any, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = X_test.shape[0]
    n_trees = len(model.estimators_)
    n_classes = len(model.classes_)

    cumulative = np.zeros((n_samples, n_classes), dtype=np.float64)
    for tree in model.estimators_:
        cumulative += tree.predict_proba(X_test)

    probs = cumulative / float(n_trees)
    pred_idx = np.argmax(probs, axis=1)
    preds = model.classes_[pred_idx]
    tau = np.full(n_samples, n_trees - 1, dtype=np.int64)
    return preds, tau



def _rf_p2stop_inference(
    model: Any,
    X_test: np.ndarray,
    threshold: float,
    ref_window: Tuple[int, int] = (10, 30),
    warmup: int = 10,
    min_trees: int = 10,
    rolling_window: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = X_test.shape[0]
    n_trees = len(model.estimators_)
    n_classes = len(model.classes_)
    window = max(1, int(rolling_window))

    cumulative = np.zeros((n_samples, n_classes), dtype=np.float64)
    counts = np.zeros(n_samples, dtype=np.int64)
    prev_scalar = np.zeros(n_samples, dtype=np.float64)
    stopped = np.zeros(n_samples, dtype=bool)
    tau = np.full(n_samples, n_trees - 1, dtype=np.int64)

    # Compact fixed-width ring buffers allow low-overhead JIT state updates.
    ring_values = np.zeros((n_samples, window), dtype=np.float64)
    ring_count = np.zeros(n_samples, dtype=np.int64)
    ring_pos = np.zeros(n_samples, dtype=np.int64)
    ref_sum = np.zeros(n_samples, dtype=np.float64)
    ref_count = np.zeros(n_samples, dtype=np.int64)

    start = max(0, int(ref_window[0]))
    end = min(n_trees, int(ref_window[1]))
    begin = max(int(min_trees), end)
    min_samples = max(5, int(warmup))

    for t, tree in enumerate(model.estimators_):
        active = np.flatnonzero(~stopped)
        if active.size == 0:
            break

        probs = tree.predict_proba(X_test[active])
        counts[active] += 1

        updated = cumulative[active] + probs
        scalar_now = np.max(updated / counts[active, None], axis=1)
        if t == 0:
            deltas = np.zeros_like(scalar_now)
        else:
            deltas = scalar_now - prev_scalar[active]

        cumulative[active] = updated
        prev_scalar[active] = scalar_now

        update_p2stop_state_numba(
            active=active,
            deltas=deltas,
            t=t,
            threshold=float(threshold),
            start=start,
            end=end,
            begin=begin,
            min_trees=int(min_trees),
            min_samples=min_samples,
            rolling_window=window,
            ring_values=ring_values,
            ring_count=ring_count,
            ring_pos=ring_pos,
            ref_sum=ref_sum,
            ref_count=ref_count,
            counts=counts,
            stopped=stopped,
            tau=tau,
        )

    denom = np.maximum(counts, 1)[:, None]
    probs = cumulative / denom
    pred_idx = np.argmax(probs, axis=1)
    preds = model.classes_[pred_idx]
    return preds, tau



def _rf_dirichlet_inference(
    model: Any,
    X_test: np.ndarray,
    confidence_threshold: float = 0.95,
    min_trees: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = X_test.shape[0]
    n_trees = len(model.estimators_)
    classes = model.classes_
    n_classes = len(classes)

    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    counts = np.ones((n_samples, n_classes), dtype=np.float64)
    stopped = np.zeros(n_samples, dtype=bool)
    tau = np.full(n_samples, n_trees - 1, dtype=np.int64)

    for t, tree in enumerate(model.estimators_):
        active = np.flatnonzero(~stopped)
        if active.size == 0:
            break

        preds = tree.predict(X_test[active])
        pred_idx = np.fromiter((class_to_idx[p] for p in preds), dtype=np.int64, count=preds.shape[0])
        counts[active, pred_idx] += 1.0

        if t + 1 < min_trees:
            continue

        c = counts[active]
        best_idx = np.argmax(c, axis=1)
        best = c[np.arange(active.size), best_idx]
        total = np.sum(c, axis=1)

        if n_classes == 2:
            second = total - best
            p_stable = 1.0 - betainc(best, second, 0.5)
        else:
            masked = c.copy()
            masked[np.arange(active.size), best_idx] = -np.inf
            second = np.max(masked, axis=1)

            mu_diff = (best - second) / total
            var_best = best * (total - best)
            var_second = second * (total - second)
            cov = -best * second
            numer = np.maximum(var_best + var_second - 2.0 * cov, 0.0)
            sigma = np.sqrt(numer / (total * total * (total + 1.0)))
            z = mu_diff / (sigma + 1e-12)
            p_stable = 0.5 * (1.0 + np.tanh(0.7978845608 * (z + 0.044715 * z**3)))

        done = p_stable > confidence_threshold
        if np.any(done):
            done_idx = active[done]
            stopped[done_idx] = True
            tau[done_idx] = t

    pred_idx = np.argmax(counts, axis=1)
    preds = classes[pred_idx]
    return preds, tau



def _select_thresholds_for_timing(phase1_run: Path) -> Tuple[Dict[Tuple[str, int], float], Dict[str, float]]:
    summary_path = phase1_run / "phase1_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Phase 1 summary not found: {summary_path}")

    df = pd.read_csv(summary_path)
    if df.empty:
        raise RuntimeError("Phase 1 summary is empty; cannot derive timing thresholds")

    by_seed: Dict[Tuple[str, int], float] = {}
    by_dataset: Dict[str, float] = {}

    for ds in sorted(df["dataset"].unique()):
        ds_df = df[df["dataset"] == ds]

        # Dataset fallback threshold: best mean validation work reduction under accuracy tolerance.
        grouped = (
            ds_df.groupby("threshold", as_index=False)
            .agg(
                delta_acc_vs_full_val_mean=("delta_acc_vs_full_val", "mean"),
                work_reduction_p2_val_mean=("work_reduction_p2_val", "mean"),
                accuracy_p2_val_mean=("accuracy_p2_val", "mean"),
            )
        )
        feasible = grouped[grouped["delta_acc_vs_full_val_mean"] >= -0.005]
        if feasible.empty:
            # If the tolerance is unmet on average, pick best validation accuracy.
            best_ds = grouped.sort_values(
                ["accuracy_p2_val_mean", "work_reduction_p2_val_mean", "threshold"],
                ascending=[False, False, True],
            ).iloc[0]
            by_dataset[str(ds)] = float(best_ds["threshold"])
        else:
            best_ds = feasible.sort_values(
                ["work_reduction_p2_val_mean", "accuracy_p2_val_mean", "threshold"],
                ascending=[False, False, True],
            ).iloc[0]
            by_dataset[str(ds)] = float(best_ds["threshold"])

        if "seed" in ds_df.columns:
            for seed in sorted(ds_df["seed"].unique()):
                s_df = ds_df[ds_df["seed"] == seed]
                row = _select_best_phase1(s_df)
                by_seed[(str(ds), int(seed))] = float(row["threshold"])

    return by_seed, by_dataset



def _time_inference_method(
    func,
    repeats: int,
    warmup_runs: int,
) -> Tuple[List[float], Tuple[np.ndarray, np.ndarray]]:
    output: Tuple[np.ndarray, np.ndarray] | None = None

    for _ in range(warmup_runs):
        output = func()

    durations: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        output = func()
        durations.append(time.perf_counter() - t0)

    if output is None:
        raise RuntimeError("Timing function produced no output")
    return durations, output



def step_timing_benchmark(
    args: argparse.Namespace,
    phase1_run: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Run end-to-end wall-clock timing benchmark for RF inference.

    Methodology (aligned with FORCE benchmarking style):
      1) warm-up runs (excluded from measurement)
      2) repeated measured runs
      3) seed-level statistics with confidence intervals

    For each dataset/seed we measure true inference path latency for:
      - Full Ensemble (RF): evaluate all trees
      - P2-STOP: evaluate trees sequentially and stop instances online
      - Dirichlet: evaluate trees sequentially and stop instances online
    """
    timing_dir = output_dir / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("STEP: Wall-clock timing benchmark (end-to-end inference path)")
    logger.info("  P2 stop kernel: %s", "Numba JIT" if HAS_NUMBA else "Python fallback")
    t0 = time.perf_counter()

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        from experiments.shared.local_data_loader import load_local_dataset
    except ImportError as exc:
        logger.error("Timing benchmark import failed: %s", exc)
        raise

    threshold_by_seed, threshold_by_dataset = _select_thresholds_for_timing(phase1_run)

    dataset_max_rows = {
        "mnist": args.mnist_max_rows,
        "covertype": args.covertype_max_rows,
        "credit": args.credit_max_rows,
        "higgs": args.higgs_max_rows,
    }

    timing_max_test = args.timing_max_test if args.timing_max_test > 0 else args.max_test

    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []

    for ds_name in dataset_names:
        logger.info("  Timing dataset: %s", ds_name)
        for seed in args.seeds:
            logger.info("    Seed: %d", seed)

            bundle = load_local_dataset(
                name=ds_name,
                data_dir=args.data_dir,
                seed=seed,
                dataset_max_rows=dataset_max_rows,
            )

            X_train, X_test, y_train, y_test = _prepare_rf_data(
                bundle=bundle,
                seed=seed,
                max_train=args.max_train,
                max_test=timing_max_test,
            )

            model = RandomForestClassifier(
                n_estimators=args.phase1_n_trees,
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

            threshold = threshold_by_seed.get((ds_name, int(seed)), threshold_by_dataset[ds_name])

            full_fn = lambda: _rf_full_inference(model, X_test)
            p2_fn = lambda: _rf_p2stop_inference(
                model,
                X_test,
                threshold=threshold,
                ref_window=(10, 30),
                warmup=10,
                min_trees=10,
                rolling_window=20,
            )
            dir_fn = lambda: _rf_dirichlet_inference(
                model,
                X_test,
                confidence_threshold=0.95,
                min_trees=10,
            )

            method_fns = [
                ("Full Ensemble (RF)", full_fn),
                ("P2-STOP", p2_fn),
                ("Dirichlet", dir_fn),
            ]

            for method_name, fn in method_fns:
                durations, (preds, tau) = _time_inference_method(
                    fn,
                    repeats=args.timing_repeats,
                    warmup_runs=args.timing_warmup_runs,
                )

                for rep, seconds in enumerate(durations, start=1):
                    raw_rows.append(
                        {
                            "dataset": ds_name,
                            "seed": int(seed),
                            "method": method_name,
                            "repeat": rep,
                            "seconds_total": float(seconds),
                            "ms_per_instance": float(seconds * 1000.0 / len(X_test)),
                        }
                    )

                ms = np.asarray(durations, dtype=np.float64) * 1000.0 / len(X_test)
                mean_trees = float(np.mean(tau + 1))
                n_trees = len(model.estimators_)
                all_rows.append(
                    {
                        "dataset": ds_name,
                        "seed": int(seed),
                        "method": method_name,
                        "ms_per_instance": float(np.mean(ms)),
                        "ms_per_instance_std": float(np.std(ms, ddof=1)) if len(ms) > 1 else 0.0,
                        "accuracy": float(accuracy_score(y_test, preds)),
                        "mean_trees_used": mean_trees,
                        "work_reduction": float(1.0 - mean_trees / n_trees),
                        "n_test": int(len(X_test)),
                        "n_trees": int(n_trees),
                        "threshold": float(threshold) if method_name == "P2-STOP" else float("nan"),
                    }
                )

    timing_df = pd.DataFrame(all_rows)
    timing_csv = timing_dir / "timing_results.csv"
    timing_df.to_csv(timing_csv, index=False)

    raw_df = pd.DataFrame(raw_rows)
    raw_csv = timing_dir / "timing_raw.csv"
    raw_df.to_csv(raw_csv, index=False)

    logger.info("  Timing results saved to %s", timing_csv)
    logger.info("  Timing raw repeats saved to %s", raw_csv)

    elapsed = time.perf_counter() - t0
    logger.info("OK  : Wall-clock timing benchmark  (%.1fs)", elapsed)
    return timing_dir


def step_timing_numba_poc(
    args: argparse.Namespace,
    phase1_run: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Run Numba timing proof-of-concept on selected datasets.

    This step measures a compiled inference path that JITs both:
      1) tree traversal and
      2) the online stopping loop

    It is reported separately from the main timing table to keep the
    reference methodology unchanged.
    """
    timing_dir = output_dir / "timing_numba_poc"
    timing_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("STEP: Numba timing proof-of-concept (selected datasets)")
    logger.info("  Datasets       : %s", args.numba_timing_datasets)
    logger.info("  Numba available: %s", HAS_NUMBA)

    if not HAS_NUMBA:
        logger.warning("  Skipping Numba PoC timing because Numba is unavailable")
        return timing_dir

    t0 = time.perf_counter()

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        from experiments.shared.local_data_loader import load_local_dataset
    except ImportError as exc:
        logger.error("Numba PoC timing import failed: %s", exc)
        raise

    threshold_by_seed, threshold_by_dataset = _select_thresholds_for_timing(phase1_run)
    numba_datasets = [d.strip().lower() for d in str(args.numba_timing_datasets).split(",") if d.strip()]
    if not numba_datasets:
        raise RuntimeError("No datasets resolved for Numba timing proof-of-concept.")
    missing_thresholds = [ds for ds in numba_datasets if ds not in threshold_by_dataset]
    if missing_thresholds:
        raise RuntimeError(
            "Numba PoC datasets missing from Phase 1 thresholds: "
            + ",".join(missing_thresholds)
            + ". Ensure they are included in --datasets."
        )

    dataset_max_rows = {
        "mnist": args.mnist_max_rows,
        "covertype": args.covertype_max_rows,
        "credit": args.credit_max_rows,
        "higgs": args.higgs_max_rows,
    }

    poc_max_test = (
        args.numba_timing_max_test
        if args.numba_timing_max_test > 0
        else (args.timing_max_test if args.timing_max_test > 0 else args.max_test)
    )

    all_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    n_estimators_poc = int(args.numba_timing_n_trees) if int(args.numba_timing_n_trees) > 0 else int(args.phase1_n_trees)
    logger.info("  RF trees (PoC) : %d", n_estimators_poc)

    for ds_name in numba_datasets:
        logger.info("  PoC dataset: %s", ds_name)
        for seed in args.seeds:
            logger.info("    Seed: %d", seed)
            bundle = load_local_dataset(
                name=ds_name,
                data_dir=args.data_dir,
                seed=seed,
                dataset_max_rows=dataset_max_rows,
            )

            X_train, X_test, y_train, y_test = _prepare_rf_data(
                bundle=bundle,
                seed=seed,
                max_train=args.max_train,
                max_test=poc_max_test,
            )

            model = RandomForestClassifier(
                n_estimators=n_estimators_poc,
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

            threshold = threshold_by_seed.get((ds_name, int(seed)), threshold_by_dataset[ds_name])
            forest = compile_rf_forest_for_numba(model)

            full_numba_fn = lambda: rf_full_inference_numba(forest, X_test)
            p2_numba_fn = lambda: rf_p2stop_inference_numba(
                forest,
                X_test,
                threshold=threshold,
                ref_window=(10, 30),
                warmup=10,
                min_trees=10,
                rolling_window=20,
            )
            p2_python_fn = lambda: _rf_p2stop_inference(
                model,
                X_test,
                threshold=threshold,
                ref_window=(10, 30),
                warmup=10,
                min_trees=10,
                rolling_window=20,
            )

            method_fns = [
                ("Full Ensemble (Numba engine)", full_numba_fn),
                ("P2-STOP (Numba engine)", p2_numba_fn),
                ("P2-STOP (Python reference)", p2_python_fn),
            ]

            for method_name, fn in method_fns:
                durations, (preds, tau) = _time_inference_method(
                    fn,
                    repeats=args.timing_repeats,
                    warmup_runs=args.timing_warmup_runs,
                )

                for rep, seconds in enumerate(durations, start=1):
                    raw_rows.append(
                        {
                            "dataset": ds_name,
                            "seed": int(seed),
                            "method": method_name,
                            "repeat": rep,
                            "seconds_total": float(seconds),
                            "ms_per_instance": float(seconds * 1000.0 / len(X_test)),
                        }
                    )

                ms = np.asarray(durations, dtype=np.float64) * 1000.0 / len(X_test)
                mean_trees = float(np.mean(tau + 1))
                n_trees = len(model.estimators_)
                all_rows.append(
                    {
                        "dataset": ds_name,
                        "seed": int(seed),
                        "method": method_name,
                        "ms_per_instance": float(np.mean(ms)),
                        "ms_per_instance_std": float(np.std(ms, ddof=1)) if len(ms) > 1 else 0.0,
                        "accuracy": float(accuracy_score(y_test, preds)),
                        "mean_trees_used": mean_trees,
                        "work_reduction": float(1.0 - mean_trees / n_trees),
                        "n_test": int(len(X_test)),
                        "n_trees": int(n_trees),
                        "threshold": float(threshold) if "P2-STOP" in method_name else float("nan"),
                    }
                )

    timing_df = pd.DataFrame(all_rows)
    timing_csv = timing_dir / "timing_numba_poc.csv"
    timing_df.to_csv(timing_csv, index=False)

    raw_df = pd.DataFrame(raw_rows)
    raw_csv = timing_dir / "timing_numba_poc_raw.csv"
    raw_df.to_csv(raw_csv, index=False)

    logger.info("  Numba PoC timing results saved to %s", timing_csv)
    logger.info("  Numba PoC timing raw repeats saved to %s", raw_csv)
    elapsed = time.perf_counter() - t0
    logger.info("OK  : Numba timing proof-of-concept  (%.1fs)", elapsed)
    return timing_dir


# ---------------------------------------------------------------------------
# Post-processing: collect manuscript tables
# ---------------------------------------------------------------------------


def collect_results(
    phase1_run: Path,
    gbm_run: Optional[Path],
    cusum_run: Optional[Path],
    robustness_run: Optional[Path],
    tref_runs: Dict[str, Path],
    timing_dir: Path,
    numba_timing_dir: Optional[Path],
    output_dir: Path,
    logger: logging.Logger,
    integrity_reports: Optional[Dict[str, Path]] = None,
) -> Path:
    """Read outputs and build manuscript_data.json (Phase 1 only)."""

    logger.info("=" * 72)
    logger.info("STEP: Collecting results into manuscript_data.json")

    data: Dict[str, Any] = {
        "meta": {
            "phase1_run": str(phase1_run),
            "gbm_run": str(gbm_run) if gbm_run is not None else None,
            "cusum_run": str(cusum_run) if cusum_run is not None else None,
            "robustness_run": str(robustness_run) if robustness_run is not None else None,
            "timing_dir": str(timing_dir),
            "numba_timing_dir": str(numba_timing_dir) if numba_timing_dir is not None else None,
            "tref_runs": {k: str(v) for k, v in tref_runs.items()},
        }
    }

    p1_summary = phase1_run / "phase1_summary.csv"
    if not p1_summary.exists():
        raise FileNotFoundError(f"Phase 1 summary CSV not found: {p1_summary}")

    df1 = pd.read_csv(p1_summary)
    logger.info("  Phase 1 summary: %d rows", len(df1))

    # ------------------------------------------------------------------
    # Table: threshold sweep with CI and paired tests (seed-level)
    # ------------------------------------------------------------------
    sweep_rows: List[Dict[str, Any]] = []
    for (ds, theta), g in df1.groupby(["dataset", "threshold"], as_index=False):
        acc_full = _mean_ci95(g["accuracy_full"])
        acc_p2 = _mean_ci95(g["accuracy_p2"])
        wr_p2 = _mean_bca_ci95(g["work_reduction_p2"], bounds=(0.0, 1.0))
        elbow = _mean_ci95(g["elbow_fraction"])

        pair_acc_full = _paired_stats(g["accuracy_p2"], g["accuracy_full"])
        pair_wr_dir = _paired_stats(g["work_reduction_p2"], g["work_reduction_dirichlet"])

        sweep_rows.append(
            {
                "dataset": str(ds),
                "threshold": float(theta),
                "n_seeds": int(acc_p2["n"]),
                "accuracy_full_mean": acc_full["mean"],
                "accuracy_full_std": acc_full["std"],
                "accuracy_full_ci95_low": acc_full["ci95_low"],
                "accuracy_full_ci95_high": acc_full["ci95_high"],
                "accuracy_p2_mean": acc_p2["mean"],
                "accuracy_p2_std": acc_p2["std"],
                "accuracy_p2_ci95_low": acc_p2["ci95_low"],
                "accuracy_p2_ci95_high": acc_p2["ci95_high"],
                "delta_acc_vs_full_mean": pair_acc_full["delta_mean"],
                "delta_acc_vs_full_std": pair_acc_full["delta_std"],
                "delta_acc_vs_full_ci95_low": pair_acc_full["delta_ci95_low"],
                "delta_acc_vs_full_ci95_high": pair_acc_full["delta_ci95_high"],
                "p_ttest_acc_p2_vs_full": pair_acc_full["p_ttest"],
                "p_wilcoxon_acc_p2_vs_full": pair_acc_full["p_wilcoxon"],
                "work_reduction_p2_mean": wr_p2["mean"],
                "work_reduction_p2_std": wr_p2["std"],
                "work_reduction_p2_ci95_low": wr_p2["ci95_low"],
                "work_reduction_p2_ci95_high": wr_p2["ci95_high"],
                "delta_wr_p2_minus_dir_mean": pair_wr_dir["delta_mean"],
                "delta_wr_p2_minus_dir_std": pair_wr_dir["delta_std"],
                "delta_wr_p2_minus_dir_ci95_low": pair_wr_dir["delta_ci95_low"],
                "delta_wr_p2_minus_dir_ci95_high": pair_wr_dir["delta_ci95_high"],
                "p_ttest_wr_p2_vs_dir": pair_wr_dir["p_ttest"],
                "p_wilcoxon_wr_p2_vs_dir": pair_wr_dir["p_wilcoxon"],
                "elbow_fraction_mean": elbow["mean"],
                "elbow_fraction_std": elbow["std"],
                "elbow_fraction_ci95_low": elbow["ci95_low"],
                "elbow_fraction_ci95_high": elbow["ci95_high"],
            }
        )

    data["tab_p2stop_sweep"] = sorted(sweep_rows, key=lambda r: (r["dataset"], r["threshold"]))

    # ------------------------------------------------------------------
    # Table: best per-seed θ vs Dirichlet with CI and paired tests
    # ------------------------------------------------------------------
    p2_vs_dir_rows: List[Dict[str, Any]] = []
    best_summary_rows: List[Dict[str, Any]] = []

    for ds in sorted(df1["dataset"].unique()):
        ds_df = df1[df1["dataset"] == ds]
        seeds = sorted(ds_df["seed"].unique()) if "seed" in ds_df.columns else [None]

        per_seed_rows: List[pd.Series] = []
        for seed in seeds:
            seed_df = ds_df if seed is None else ds_df[ds_df["seed"] == seed]
            per_seed_rows.append(_select_best_phase1(seed_df))

        best_df = pd.DataFrame(per_seed_rows)

        acc_p2 = _mean_ci95(best_df["accuracy_p2"])
        acc_full = _mean_ci95(best_df["accuracy_full"])
        acc_dir = _mean_ci95(best_df["accuracy_dirichlet"])
        wr_p2 = _mean_bca_ci95(best_df["work_reduction_p2"], bounds=(0.0, 1.0))
        wr_dir = _mean_bca_ci95(best_df["work_reduction_dirichlet"], bounds=(0.0, 1.0))
        pear = _mean_ci95(best_df["pearson_tau_vs_dirichlet"])

        pair_acc_full = _paired_stats(best_df["accuracy_p2"], best_df["accuracy_full"])
        pair_acc_dir = _paired_stats(best_df["accuracy_p2"], best_df["accuracy_dirichlet"])
        pair_wr_dir = _paired_stats(best_df["work_reduction_p2"], best_df["work_reduction_dirichlet"])

        theta_mode = float(best_df["threshold"].mode().iloc[0])

        row = {
            "dataset": str(ds),
            "n_seeds": int(acc_p2["n"]),
            "theta_mode": theta_mode,
            "theta_values": ",".join(f"{float(v):.2f}" for v in best_df["threshold"].tolist()),
            "accuracy_full_mean": acc_full["mean"],
            "accuracy_full_std": acc_full["std"],
            "accuracy_full_ci95_low": acc_full["ci95_low"],
            "accuracy_full_ci95_high": acc_full["ci95_high"],
            "accuracy_p2_mean": acc_p2["mean"],
            "accuracy_p2_std": acc_p2["std"],
            "accuracy_p2_ci95_low": acc_p2["ci95_low"],
            "accuracy_p2_ci95_high": acc_p2["ci95_high"],
            "accuracy_dirichlet_mean": acc_dir["mean"],
            "accuracy_dirichlet_std": acc_dir["std"],
            "accuracy_dirichlet_ci95_low": acc_dir["ci95_low"],
            "accuracy_dirichlet_ci95_high": acc_dir["ci95_high"],
            "delta_acc_p2_minus_full_mean": pair_acc_full["delta_mean"],
            "delta_acc_p2_minus_full_ci95_low": pair_acc_full["delta_ci95_low"],
            "delta_acc_p2_minus_full_ci95_high": pair_acc_full["delta_ci95_high"],
            "p_ttest_acc_p2_vs_full": pair_acc_full["p_ttest"],
            "p_wilcoxon_acc_p2_vs_full": pair_acc_full["p_wilcoxon"],
            "delta_acc_p2_minus_dir_mean": pair_acc_dir["delta_mean"],
            "delta_acc_p2_minus_dir_ci95_low": pair_acc_dir["delta_ci95_low"],
            "delta_acc_p2_minus_dir_ci95_high": pair_acc_dir["delta_ci95_high"],
            "p_ttest_acc_p2_vs_dir": pair_acc_dir["p_ttest"],
            "p_wilcoxon_acc_p2_vs_dir": pair_acc_dir["p_wilcoxon"],
            "work_reduction_p2_mean": wr_p2["mean"],
            "work_reduction_p2_std": wr_p2["std"],
            "work_reduction_p2_ci95_low": wr_p2["ci95_low"],
            "work_reduction_p2_ci95_high": wr_p2["ci95_high"],
            "work_reduction_dirichlet_mean": wr_dir["mean"],
            "work_reduction_dirichlet_std": wr_dir["std"],
            "work_reduction_dirichlet_ci95_low": wr_dir["ci95_low"],
            "work_reduction_dirichlet_ci95_high": wr_dir["ci95_high"],
            "delta_wr_p2_minus_dir_mean": pair_wr_dir["delta_mean"],
            "delta_wr_p2_minus_dir_ci95_low": pair_wr_dir["delta_ci95_low"],
            "delta_wr_p2_minus_dir_ci95_high": pair_wr_dir["delta_ci95_high"],
            "p_ttest_wr_p2_vs_dir": pair_wr_dir["p_ttest"],
            "p_wilcoxon_wr_p2_vs_dir": pair_wr_dir["p_wilcoxon"],
            "pearson_tau_vs_dirichlet_mean": pear["mean"],
            "pearson_tau_vs_dirichlet_std": pear["std"],
            "pearson_tau_vs_dirichlet_ci95_low": pear["ci95_low"],
            "pearson_tau_vs_dirichlet_ci95_high": pear["ci95_high"],
        }
        p2_vs_dir_rows.append(row)

        best_summary_rows.append(
            {
                "dataset": str(ds),
                "method": "P2-STOP",
                "param_mode": f"theta={theta_mode:.2f}",
                "n_seeds": int(acc_p2["n"]),
                "accuracy_mean": acc_p2["mean"],
                "accuracy_ci95_low": acc_p2["ci95_low"],
                "accuracy_ci95_high": acc_p2["ci95_high"],
                "delta_acc_vs_full_mean": pair_acc_full["delta_mean"],
                "delta_acc_vs_full_ci95_low": pair_acc_full["delta_ci95_low"],
                "delta_acc_vs_full_ci95_high": pair_acc_full["delta_ci95_high"],
                "p_ttest_acc_vs_full": pair_acc_full["p_ttest"],
                "work_reduction_mean": wr_p2["mean"],
                "work_reduction_ci95_low": wr_p2["ci95_low"],
                "work_reduction_ci95_high": wr_p2["ci95_high"],
            }
        )

    data["tab_p2stop_vs_dirichlet"] = p2_vs_dir_rows
    data["tab_best_summary"] = best_summary_rows

    # ------------------------------------------------------------------
    # Table: t_ref ablation
    # ------------------------------------------------------------------
    tref_payload: Dict[str, Any] = {}
    for label, run_dir in tref_runs.items():
        csv = run_dir / "phase1_summary.csv"
        if not csv.exists():
            logger.warning("  t_ref ablation CSV not found: %s", csv)
            continue

        ab_df = pd.read_csv(csv)
        rows = []
        for ds, g in ab_df.groupby("dataset", as_index=False):
            acc = _mean_ci95(g["accuracy_p2"])
            wr = _mean_bca_ci95(g["work_reduction_p2"], bounds=(0.0, 1.0))
            rows.append(
                {
                    "dataset": str(ds),
                    "n_seeds": int(acc["n"]),
                    "accuracy_p2_mean": acc["mean"],
                    "accuracy_p2_ci95_low": acc["ci95_low"],
                    "accuracy_p2_ci95_high": acc["ci95_high"],
                    "work_reduction_p2_mean": wr["mean"],
                    "work_reduction_p2_ci95_low": wr["ci95_low"],
                    "work_reduction_p2_ci95_high": wr["ci95_high"],
                }
            )
        tref_payload[label] = rows

    data["tab_tref_ablation"] = tref_payload

    # ------------------------------------------------------------------
    # Table: GBM threshold sweep + best-per-seed summary (optional)
    # ------------------------------------------------------------------
    if gbm_run is not None:
        gbm_csv = gbm_run / "phase1_gbm_summary.csv"
        if gbm_csv.exists():
            gbm_df = pd.read_csv(gbm_csv)
            logger.info("  GBM Phase 1 summary: %d rows", len(gbm_df))

            gbm_sweep_rows: List[Dict[str, Any]] = []
            for (ds, theta), g in gbm_df.groupby(["dataset", "threshold"], as_index=False):
                acc_full = _mean_ci95(g["accuracy_full"])
                acc_p2 = _mean_ci95(g["accuracy_p2"])
                wr_p2 = _mean_bca_ci95(g["work_reduction_p2"], bounds=(0.0, 1.0))
                elbow = _mean_ci95(g["elbow_fraction"])
                pair_acc_full = _paired_stats(g["accuracy_p2"], g["accuracy_full"])

                gbm_sweep_rows.append(
                    {
                        "dataset": str(ds),
                        "threshold": float(theta),
                        "n_seeds": int(acc_p2["n"]),
                        "backend_mode": str(g["backend"].mode().iloc[0]) if "backend" in g.columns else "",
                        "accuracy_full_mean": acc_full["mean"],
                        "accuracy_full_ci95_low": acc_full["ci95_low"],
                        "accuracy_full_ci95_high": acc_full["ci95_high"],
                        "accuracy_p2_mean": acc_p2["mean"],
                        "accuracy_p2_ci95_low": acc_p2["ci95_low"],
                        "accuracy_p2_ci95_high": acc_p2["ci95_high"],
                        "delta_acc_vs_full_mean": pair_acc_full["delta_mean"],
                        "delta_acc_vs_full_ci95_low": pair_acc_full["delta_ci95_low"],
                        "delta_acc_vs_full_ci95_high": pair_acc_full["delta_ci95_high"],
                        "p_ttest_acc_p2_vs_full": pair_acc_full["p_ttest"],
                        "p_wilcoxon_acc_p2_vs_full": pair_acc_full["p_wilcoxon"],
                        "work_reduction_p2_mean": wr_p2["mean"],
                        "work_reduction_p2_ci95_low": wr_p2["ci95_low"],
                        "work_reduction_p2_ci95_high": wr_p2["ci95_high"],
                        "elbow_fraction_mean": elbow["mean"],
                        "elbow_fraction_ci95_low": elbow["ci95_low"],
                        "elbow_fraction_ci95_high": elbow["ci95_high"],
                    }
                )
            data["tab_gbm_p2stop_sweep"] = sorted(
                gbm_sweep_rows,
                key=lambda r: (r["dataset"], r["threshold"]),
            )

            gbm_best_rows: List[Dict[str, Any]] = []
            for ds in sorted(gbm_df["dataset"].unique()):
                ds_df = gbm_df[gbm_df["dataset"] == ds]
                per_seed_rows: List[pd.Series] = []
                for seed in sorted(ds_df["seed"].unique()):
                    per_seed_rows.append(_select_best_phase1_gbm(ds_df[ds_df["seed"] == seed]))
                best_df = pd.DataFrame(per_seed_rows)

                acc_full = _mean_ci95(best_df["accuracy_full"])
                acc_p2 = _mean_ci95(best_df["accuracy_p2"])
                wr_p2 = _mean_bca_ci95(best_df["work_reduction_p2"], bounds=(0.0, 1.0))
                pair_acc_full = _paired_stats(best_df["accuracy_p2"], best_df["accuracy_full"])

                gbm_best_rows.append(
                    {
                        "dataset": str(ds),
                        "n_seeds": int(acc_p2["n"]),
                        "backend_mode": str(best_df["backend"].mode().iloc[0]) if "backend" in best_df.columns else "",
                        "theta_mode": float(best_df["threshold"].mode().iloc[0]),
                        "theta_values": ",".join(f"{float(v):.2f}" for v in best_df["threshold"].tolist()),
                        "accuracy_full_mean": acc_full["mean"],
                        "accuracy_full_ci95_low": acc_full["ci95_low"],
                        "accuracy_full_ci95_high": acc_full["ci95_high"],
                        "accuracy_p2_mean": acc_p2["mean"],
                        "accuracy_p2_ci95_low": acc_p2["ci95_low"],
                        "accuracy_p2_ci95_high": acc_p2["ci95_high"],
                        "delta_acc_p2_minus_full_mean": pair_acc_full["delta_mean"],
                        "delta_acc_p2_minus_full_ci95_low": pair_acc_full["delta_ci95_low"],
                        "delta_acc_p2_minus_full_ci95_high": pair_acc_full["delta_ci95_high"],
                        "p_ttest_acc_p2_vs_full": pair_acc_full["p_ttest"],
                        "p_wilcoxon_acc_p2_vs_full": pair_acc_full["p_wilcoxon"],
                        "work_reduction_p2_mean": wr_p2["mean"],
                        "work_reduction_p2_ci95_low": wr_p2["ci95_low"],
                        "work_reduction_p2_ci95_high": wr_p2["ci95_high"],
                    }
                )
            data["tab_gbm_best_summary"] = gbm_best_rows
        else:
            logger.warning("  GBM summary CSV not found: %s", gbm_csv)

    # ------------------------------------------------------------------
    # Table: CUSUM variant vs relative-threshold detector (optional)
    # ------------------------------------------------------------------
    if cusum_run is not None:
        cusum_csv = cusum_run / "phase1_summary.csv"
        if cusum_csv.exists():
            cusum_df = pd.read_csv(cusum_csv)
            logger.info("  CUSUM Phase 1 summary: %d rows", len(cusum_df))

            cusum_rows: List[Dict[str, Any]] = []
            for ds in sorted(df1["dataset"].unique()):
                rel_ds = df1[df1["dataset"] == ds]
                cus_ds = cusum_df[cusum_df["dataset"] == ds]
                if rel_ds.empty or cus_ds.empty:
                    continue

                rel_best_seed: List[pd.Series] = []
                cus_best_seed: List[pd.Series] = []
                for seed in sorted(set(rel_ds["seed"].unique()) & set(cus_ds["seed"].unique())):
                    rel_best_seed.append(_select_best_phase1(rel_ds[rel_ds["seed"] == seed]))
                    cus_best_seed.append(_select_best_phase1(cus_ds[cus_ds["seed"] == seed]))

                if not rel_best_seed or not cus_best_seed:
                    continue

                rel_best = pd.DataFrame(rel_best_seed).set_index("seed")
                cus_best = pd.DataFrame(cus_best_seed).set_index("seed")
                merged = rel_best.join(cus_best, how="inner", lsuffix="_rel", rsuffix="_cusum")
                if merged.empty:
                    continue

                rel_acc = _mean_ci95(merged["accuracy_p2_rel"].values)
                cus_acc = _mean_ci95(merged["accuracy_p2_cusum"].values)
                rel_wr = _mean_bca_ci95(merged["work_reduction_p2_rel"].values, bounds=(0.0, 1.0))
                cus_wr = _mean_bca_ci95(merged["work_reduction_p2_cusum"].values, bounds=(0.0, 1.0))

                pair_acc = _paired_stats(merged["accuracy_p2_cusum"].values, merged["accuracy_p2_rel"].values)
                pair_wr = _paired_stats(
                    merged["work_reduction_p2_cusum"].values,
                    merged["work_reduction_p2_rel"].values,
                )

                cusum_rows.append(
                    {
                        "dataset": str(ds),
                        "n_seeds": int(rel_acc["n"]),
                        "accuracy_relative_mean": rel_acc["mean"],
                        "accuracy_relative_ci95_low": rel_acc["ci95_low"],
                        "accuracy_relative_ci95_high": rel_acc["ci95_high"],
                        "accuracy_cusum_mean": cus_acc["mean"],
                        "accuracy_cusum_ci95_low": cus_acc["ci95_low"],
                        "accuracy_cusum_ci95_high": cus_acc["ci95_high"],
                        "delta_acc_cusum_minus_relative_mean": pair_acc["delta_mean"],
                        "delta_acc_cusum_minus_relative_ci95_low": pair_acc["delta_ci95_low"],
                        "delta_acc_cusum_minus_relative_ci95_high": pair_acc["delta_ci95_high"],
                        "p_ttest_acc_cusum_vs_relative": pair_acc["p_ttest"],
                        "p_wilcoxon_acc_cusum_vs_relative": pair_acc["p_wilcoxon"],
                        "work_reduction_relative_mean": rel_wr["mean"],
                        "work_reduction_relative_ci95_low": rel_wr["ci95_low"],
                        "work_reduction_relative_ci95_high": rel_wr["ci95_high"],
                        "work_reduction_cusum_mean": cus_wr["mean"],
                        "work_reduction_cusum_ci95_low": cus_wr["ci95_low"],
                        "work_reduction_cusum_ci95_high": cus_wr["ci95_high"],
                        "delta_wr_cusum_minus_relative_mean": pair_wr["delta_mean"],
                        "delta_wr_cusum_minus_relative_ci95_low": pair_wr["delta_ci95_low"],
                        "delta_wr_cusum_minus_relative_ci95_high": pair_wr["delta_ci95_high"],
                        "p_ttest_wr_cusum_vs_relative": pair_wr["p_ttest"],
                        "p_wilcoxon_wr_cusum_vs_relative": pair_wr["p_wilcoxon"],
                    }
                )
            data["tab_cusum_vs_relative"] = cusum_rows
        else:
            logger.warning("  CUSUM summary CSV not found: %s", cusum_csv)

    # ------------------------------------------------------------------
    # Table: contamination robustness (optional)
    # ------------------------------------------------------------------
    if robustness_run is not None:
        robust_csv = robustness_run / "robustness_summary.csv"
        if robust_csv.exists():
            robust_df = pd.read_csv(robust_csv)
            logger.info("  Robustness summary: %d rows", len(robust_df))

            robust_rows: List[Dict[str, Any]] = []
            for (ds, level, method), g in robust_df.groupby(
                ["dataset", "contamination_rate", "method"], as_index=False
            ):
                acc_full = _mean_ci95(g["accuracy_full"])
                acc_method = _mean_ci95(g["accuracy_method"])
                delta_acc = _mean_ci95(g["delta_acc_vs_full"])
                wr = _mean_bca_ci95(g["work_reduction"], bounds=(0.0, 1.0))
                elbow = _mean_ci95(g["elbow_fraction"])
                robust_rows.append(
                    {
                        "dataset": str(ds),
                        "contamination_rate": float(level),
                        "method": str(method),
                        "n_seeds": int(acc_method["n"]),
                        "accuracy_full_mean": acc_full["mean"],
                        "accuracy_method_mean": acc_method["mean"],
                        "accuracy_method_ci95_low": acc_method["ci95_low"],
                        "accuracy_method_ci95_high": acc_method["ci95_high"],
                        "delta_acc_vs_full_mean": delta_acc["mean"],
                        "delta_acc_vs_full_ci95_low": delta_acc["ci95_low"],
                        "delta_acc_vs_full_ci95_high": delta_acc["ci95_high"],
                        "work_reduction_mean": wr["mean"],
                        "work_reduction_ci95_low": wr["ci95_low"],
                        "work_reduction_ci95_high": wr["ci95_high"],
                        "elbow_fraction_mean": elbow["mean"],
                        "elbow_fraction_ci95_low": elbow["ci95_low"],
                        "elbow_fraction_ci95_high": elbow["ci95_high"],
                    }
                )
            data["tab_robustness"] = sorted(
                robust_rows,
                key=lambda r: (r["dataset"], r["contamination_rate"], r["method"]),
            )

            pair_rows: List[Dict[str, Any]] = []
            for (ds, level), g in robust_df.groupby(["dataset", "contamination_rate"], as_index=False):
                pivot = g.pivot_table(index="seed", columns="method", values=["delta_acc_vs_full", "work_reduction"])
                if (
                    ("delta_acc_vs_full", "p2_iqr") not in pivot.columns
                    or ("delta_acc_vs_full", "mean_scale") not in pivot.columns
                    or ("work_reduction", "p2_iqr") not in pivot.columns
                    or ("work_reduction", "mean_scale") not in pivot.columns
                ):
                    continue

                acc_pair = _paired_stats(
                    pivot[("delta_acc_vs_full", "p2_iqr")].values,
                    pivot[("delta_acc_vs_full", "mean_scale")].values,
                )
                wr_pair = _paired_stats(
                    pivot[("work_reduction", "p2_iqr")].values,
                    pivot[("work_reduction", "mean_scale")].values,
                )

                pair_rows.append(
                    {
                        "dataset": str(ds),
                        "contamination_rate": float(level),
                        "n_seeds": int(acc_pair["n"]),
                        "delta_acc_iqr_minus_mean_mean": acc_pair["delta_mean"],
                        "delta_acc_iqr_minus_mean_ci95_low": acc_pair["delta_ci95_low"],
                        "delta_acc_iqr_minus_mean_ci95_high": acc_pair["delta_ci95_high"],
                        "p_ttest_acc_iqr_vs_mean": acc_pair["p_ttest"],
                        "p_wilcoxon_acc_iqr_vs_mean": acc_pair["p_wilcoxon"],
                        "delta_wr_iqr_minus_mean_mean": wr_pair["delta_mean"],
                        "delta_wr_iqr_minus_mean_ci95_low": wr_pair["delta_ci95_low"],
                        "delta_wr_iqr_minus_mean_ci95_high": wr_pair["delta_ci95_high"],
                        "p_ttest_wr_iqr_vs_mean": wr_pair["p_ttest"],
                        "p_wilcoxon_wr_iqr_vs_mean": wr_pair["p_wilcoxon"],
                    }
                )
            data["tab_robustness_iqr_vs_mean"] = sorted(
                pair_rows,
                key=lambda r: (r["dataset"], r["contamination_rate"]),
            )
            data["tab_robustness_seed_raw"] = robust_df.to_dict(orient="records")
        else:
            logger.warning("  Robustness summary CSV not found: %s", robust_csv)

    # ------------------------------------------------------------------
    # Table: timing (aggregated across seeds with paired tests)
    # ------------------------------------------------------------------
    timing_csv = timing_dir / "timing_results.csv"
    if timing_csv.exists():
        timing_df = pd.read_csv(timing_csv)
        timing_rows: List[Dict[str, Any]] = []

        for ds in sorted(timing_df["dataset"].unique()):
            ds_df = timing_df[timing_df["dataset"] == ds]

            full_seed = (
                ds_df[ds_df["method"] == "Full Ensemble (RF)"][
                    ["seed", "ms_per_instance", "mean_trees_used", "work_reduction", "accuracy"]
                ]
                .drop_duplicates(subset=["seed"]) 
                .set_index("seed")
            )
            if full_seed.empty:
                continue

            for method in ["Full Ensemble (RF)", "P2-STOP", "Dirichlet"]:
                m_df = ds_df[ds_df["method"] == method][
                    ["seed", "ms_per_instance", "mean_trees_used", "work_reduction", "accuracy"]
                ].drop_duplicates(subset=["seed"])
                if m_df.empty:
                    continue

                m_seed = m_df.set_index("seed")
                ms_stats = _mean_ci95(m_seed["ms_per_instance"].values)
                tree_stats = _mean_ci95(m_seed["mean_trees_used"].values)
                wr_stats = _mean_bca_ci95(m_seed["work_reduction"].values, bounds=(0.0, 1.0))
                acc_stats = _mean_ci95(m_seed["accuracy"].values)

                speed_stats = {
                    "mean": 1.0,
                    "ci95_low": 1.0,
                    "ci95_high": 1.0,
                }
                p_ms = float("nan")
                p_ms_w = float("nan")

                if method != "Full Ensemble (RF)":
                    merged = full_seed.join(m_seed, how="inner", lsuffix="_full", rsuffix="_method")
                    if not merged.empty:
                        speed = merged["ms_per_instance_full"] / merged["ms_per_instance_method"]
                        speed_ci = _mean_ci95(speed.values)
                        speed_stats = {
                            "mean": speed_ci["mean"],
                            "ci95_low": speed_ci["ci95_low"],
                            "ci95_high": speed_ci["ci95_high"],
                        }

                        ms_pair = _paired_stats(
                            merged["ms_per_instance_full"].values,
                            merged["ms_per_instance_method"].values,
                        )
                        p_ms = ms_pair["p_ttest"]
                        p_ms_w = ms_pair["p_wilcoxon"]

                timing_rows.append(
                    {
                        "dataset": str(ds),
                        "method": method,
                        "n_seeds": int(ms_stats["n"]),
                        "ms_per_instance_mean": ms_stats["mean"],
                        "ms_per_instance_ci95_low": ms_stats["ci95_low"],
                        "ms_per_instance_ci95_high": ms_stats["ci95_high"],
                        "speedup_vs_full_mean": speed_stats["mean"],
                        "speedup_vs_full_ci95_low": speed_stats["ci95_low"],
                        "speedup_vs_full_ci95_high": speed_stats["ci95_high"],
                        "p_ttest_ms_vs_full": p_ms,
                        "p_wilcoxon_ms_vs_full": p_ms_w,
                        "mean_trees_used_mean": tree_stats["mean"],
                        "mean_trees_used_ci95_low": tree_stats["ci95_low"],
                        "mean_trees_used_ci95_high": tree_stats["ci95_high"],
                        "work_reduction_mean": wr_stats["mean"],
                        "work_reduction_ci95_low": wr_stats["ci95_low"],
                        "work_reduction_ci95_high": wr_stats["ci95_high"],
                        "accuracy_mean": acc_stats["mean"],
                        "accuracy_ci95_low": acc_stats["ci95_low"],
                        "accuracy_ci95_high": acc_stats["ci95_high"],
                    }
                )

        data["tab_timing"] = timing_rows
        data["tab_timing_seed_raw"] = timing_df.to_dict(orient="records")
    else:
        logger.info("  Timing CSV not found (timing step may be skipped): %s", timing_csv)

    # ------------------------------------------------------------------
    # Table: Numba timing proof-of-concept (optional, one or more datasets)
    # ------------------------------------------------------------------
    if numba_timing_dir is not None:
        numba_timing_csv = numba_timing_dir / "timing_numba_poc.csv"
        if numba_timing_csv.exists():
            nt_df = pd.read_csv(numba_timing_csv)
            nt_rows: List[Dict[str, Any]] = []
            baseline_method = "Full Ensemble (Numba engine)"

            for ds in sorted(nt_df["dataset"].unique()):
                ds_df = nt_df[nt_df["dataset"] == ds]
                full_seed = (
                    ds_df[ds_df["method"] == baseline_method][
                        ["seed", "ms_per_instance", "mean_trees_used", "work_reduction", "accuracy"]
                    ]
                    .drop_duplicates(subset=["seed"])
                    .set_index("seed")
                )
                if full_seed.empty:
                    continue

                for method in sorted(ds_df["method"].unique()):
                    m_df = ds_df[ds_df["method"] == method][
                        ["seed", "ms_per_instance", "mean_trees_used", "work_reduction", "accuracy"]
                    ].drop_duplicates(subset=["seed"])
                    if m_df.empty:
                        continue

                    m_seed = m_df.set_index("seed")
                    ms_stats = _mean_ci95(m_seed["ms_per_instance"].values)
                    tree_stats = _mean_ci95(m_seed["mean_trees_used"].values)
                    wr_stats = _mean_bca_ci95(m_seed["work_reduction"].values, bounds=(0.0, 1.0))
                    acc_stats = _mean_ci95(m_seed["accuracy"].values)

                    speed_stats = {"mean": 1.0, "ci95_low": 1.0, "ci95_high": 1.0}
                    p_ms = float("nan")
                    p_ms_w = float("nan")

                    if method != baseline_method:
                        merged = full_seed.join(m_seed, how="inner", lsuffix="_full", rsuffix="_method")
                        if not merged.empty:
                            speed = merged["ms_per_instance_full"] / merged["ms_per_instance_method"]
                            speed_ci = _mean_ci95(speed.values)
                            speed_stats = {
                                "mean": speed_ci["mean"],
                                "ci95_low": speed_ci["ci95_low"],
                                "ci95_high": speed_ci["ci95_high"],
                            }
                            ms_pair = _paired_stats(
                                merged["ms_per_instance_full"].values,
                                merged["ms_per_instance_method"].values,
                            )
                            p_ms = ms_pair["p_ttest"]
                            p_ms_w = ms_pair["p_wilcoxon"]

                    nt_rows.append(
                        {
                            "dataset": str(ds),
                            "method": method,
                            "n_seeds": int(ms_stats["n"]),
                            "ms_per_instance_mean": ms_stats["mean"],
                            "ms_per_instance_ci95_low": ms_stats["ci95_low"],
                            "ms_per_instance_ci95_high": ms_stats["ci95_high"],
                            "speedup_vs_full_numba_mean": speed_stats["mean"],
                            "speedup_vs_full_numba_ci95_low": speed_stats["ci95_low"],
                            "speedup_vs_full_numba_ci95_high": speed_stats["ci95_high"],
                            "p_ttest_ms_vs_full_numba": p_ms,
                            "p_wilcoxon_ms_vs_full_numba": p_ms_w,
                            "mean_trees_used_mean": tree_stats["mean"],
                            "mean_trees_used_ci95_low": tree_stats["ci95_low"],
                            "mean_trees_used_ci95_high": tree_stats["ci95_high"],
                            "work_reduction_mean": wr_stats["mean"],
                            "work_reduction_ci95_low": wr_stats["ci95_low"],
                            "work_reduction_ci95_high": wr_stats["ci95_high"],
                            "accuracy_mean": acc_stats["mean"],
                            "accuracy_ci95_low": acc_stats["ci95_low"],
                            "accuracy_ci95_high": acc_stats["ci95_high"],
                        }
                    )

            data["tab_timing_numba_poc"] = nt_rows
            data["tab_timing_numba_poc_seed_raw"] = nt_df.to_dict(orient="records")
        else:
            logger.info("  Numba PoC timing CSV not found: %s", numba_timing_csv)

    # ------------------------------------------------------------------
    # Data integrity
    # ------------------------------------------------------------------
    if integrity_reports is not None:
        integrity_json = integrity_reports.get("json")
        if integrity_json is not None and integrity_json.exists():
            data["data_integrity"] = json.loads(integrity_json.read_text())

    out_path = output_dir / "manuscript_data.json"
    out_path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("  Collected results written to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Export tables
# ---------------------------------------------------------------------------


def export_table_artifacts(
    manuscript_data_path: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Export manuscript table payloads into CSV files for direct use."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(manuscript_data_path.read_text())
    manifest: Dict[str, str] = {}

    table_keys = [
        "tab_p2stop_sweep",
        "tab_p2stop_vs_dirichlet",
        "tab_best_summary",
        "tab_gbm_p2stop_sweep",
        "tab_gbm_best_summary",
        "tab_cusum_vs_relative",
        "tab_robustness",
        "tab_robustness_iqr_vs_mean",
        "tab_timing",
        "tab_timing_numba_poc",
    ]
    for key in table_keys:
        records = data.get(key)
        if not isinstance(records, list) or not records:
            continue
        path = tables_dir / f"{key}.csv"
        pd.DataFrame(records).to_csv(path, index=False)
        manifest[key] = str(path)

    tref_payload = data.get("tab_tref_ablation", {})
    if isinstance(tref_payload, dict):
        for label, records in tref_payload.items():
            if not isinstance(records, list) or not records:
                continue
            path = tables_dir / f"tab_tref_ablation_{label}.csv"
            pd.DataFrame(records).to_csv(path, index=False)
            manifest[f"tab_tref_ablation_{label}"] = str(path)

    manifest_path = tables_dir / "table_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("  Table exports manifest: %s", manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _resolve_seed_dir(run_dir: Path, seed: int) -> Optional[Path]:
    direct = run_dir / f"seed_{seed}"
    if direct.exists():
        return direct
    seed_dirs = sorted([p for p in run_dir.glob("seed_*") if p.is_dir()])
    return seed_dirs[0] if seed_dirs else None


def _build_pareto_all_datasets_figure(phase1_run: Path, dst: Path) -> None:
    """Create a 2x2 Pareto figure using seed-averaged Phase-1 metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv = phase1_run / "phase1_summary.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Phase 1 summary missing: {csv}")

    df = pd.read_csv(csv)
    datasets = sorted(df["dataset"].unique())
    if not datasets:
        raise RuntimeError("No datasets found in phase1_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=180)
    flat = axes.flatten()

    for idx, ax in enumerate(flat):
        if idx >= len(datasets):
            ax.axis("off")
            continue

        ds = datasets[idx]
        g = df[df["dataset"] == ds]
        if g.empty:
            ax.axis("off")
            continue

        curve = (
            g.groupby("threshold", as_index=False)
            .agg(
                accuracy_p2_mean=("accuracy_p2", "mean"),
                work_reduction_p2_mean=("work_reduction_p2", "mean"),
            )
            .sort_values("work_reduction_p2_mean")
        )

        ax.plot(
            curve["work_reduction_p2_mean"],
            curve["accuracy_p2_mean"],
            marker="o",
            linestyle="-",
            color="tab:blue",
            label="P2-STOP",
        )
        for _, row in curve.iterrows():
            ax.text(
                float(row["work_reduction_p2_mean"]),
                float(row["accuracy_p2_mean"]),
                f"{float(row['threshold']):.2f}",
                fontsize=7,
            )

        ax.scatter(
            [float(g["work_reduction_dirichlet"].mean())],
            [float(g["accuracy_dirichlet"].mean())],
            marker="x",
            s=55,
            color="tab:orange",
            label="Dirichlet",
        )

        ax.set_title(ds.upper())
        ax.set_xlabel("Work reduction")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.tight_layout()
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst)
    plt.close(fig)



def collect_figures(
    phase1_run: Path,
    output_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Collect all generated Phase-1 figures and write manifest."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    p1_figs = list(phase1_run.rglob("*.png"))

    logger.info("=" * 72)
    logger.info("STEP: Collecting figures")
    logger.info("  Phase 1 figures: %d PNGs", len(p1_figs))
    for f in sorted(p1_figs):
        logger.debug("    %s", f)

    manifest = {
        "phase1": [str(f) for f in sorted(p1_figs)],
    }
    manifest_path = figures_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("  Figure manifest: %s", manifest_path)
    return manifest_path



def export_manuscript_figure_targets(
    phase1_run: Path,
    output_dir: Path,
    logger: logging.Logger,
    dataset: str,
    seed: int,
) -> Path:
    """Copy figure files mapped to manuscript labels (Phase 1 only)."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    p1_seed_dir = _resolve_seed_dir(phase1_run, seed)
    if p1_seed_dir is None:
        raise FileNotFoundError("Could not resolve seed-specific figure directory for Phase 1")

    figure_sources = {
        "fig_scale_trajectories": p1_seed_dir / f"phase1_scale_diagnostics_{dataset}.png",
        "fig_pareto": p1_seed_dir / f"phase1_pareto_{dataset}.png",
    }

    copied: Dict[str, str] = {}
    missing: Dict[str, str] = {}
    for label, src in figure_sources.items():
        if src.exists():
            dst = figures_dir / f"{label}.png"
            shutil.copy2(src, dst)
            copied[label] = str(dst)
        else:
            missing[label] = str(src)

    # Build multi-dataset Pareto frontier panel (2x2) from aggregated results.
    pareto_grid_dst = figures_dir / "fig_pareto_all_datasets.png"
    try:
        _build_pareto_all_datasets_figure(phase1_run=phase1_run, dst=pareto_grid_dst)
        copied["fig_pareto_all_datasets"] = str(pareto_grid_dst)
    except Exception as exc:
        missing["fig_pareto_all_datasets"] = f"{pareto_grid_dst} ({exc})"

    payload = {
        "dataset": dataset,
        "seed": seed,
        "copied": copied,
        "missing": missing,
    }
    out_path = figures_dir / "manuscript_figure_targets.json"
    out_path.write_text(json.dumps(payload, indent=2))

    if missing:
        logger.warning("  Missing manuscript figure sources: %s", missing)
    logger.info("  Manuscript figure targets: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_outputs(
    phase1_run: Path,
    gbm_run: Optional[Path],
    cusum_run: Optional[Path],
    robustness_run: Optional[Path],
    tref_runs: Dict[str, Path],
    timing_dir: Path,
    numba_timing_dir: Optional[Path],
    output_dir: Path,
    logger: logging.Logger,
    expect_timing: bool,
    expect_numba_timing_poc: bool,
    expect_ablation: bool,
    expect_gbm: bool,
    expect_cusum: bool,
    expect_robustness: bool,
    integrity_reports: Optional[Dict[str, Path]] = None,
) -> bool:
    """Verify expected outputs exist and have content."""
    logger.info("=" * 72)
    logger.info("STEP: Verifying outputs")
    ok = True

    checks = [
        ("Phase 1 summary CSV", phase1_run / "phase1_summary.csv"),
        ("Phase 1 aggregated CSV", phase1_run / "phase1_aggregated.csv"),
        ("Phase 1 best thresholds CSV", phase1_run / "phase1_best_thresholds.csv"),
        ("Phase 1 config JSON", phase1_run / "config.json"),
        ("Manuscript data JSON", output_dir / "manuscript_data.json"),
        ("Figure manifest", output_dir / "figures/figure_manifest.json"),
        ("Manuscript figure targets", output_dir / "figures/manuscript_figure_targets.json"),
        ("Pareto 2x2 figure", output_dir / "figures/fig_pareto_all_datasets.png"),
        ("Table manifest", output_dir / "tables/table_manifest.json"),
    ]

    if expect_gbm and gbm_run is not None:
        checks.extend(
            [
                ("GBM summary CSV", gbm_run / "phase1_gbm_summary.csv"),
                ("GBM aggregated CSV", gbm_run / "phase1_gbm_aggregated.csv"),
                ("GBM best thresholds CSV", gbm_run / "phase1_gbm_best_thresholds.csv"),
                ("GBM config JSON", gbm_run / "config.json"),
            ]
        )

    if expect_cusum and cusum_run is not None:
        checks.extend(
            [
                ("CUSUM summary CSV", cusum_run / "phase1_summary.csv"),
                ("CUSUM aggregated CSV", cusum_run / "phase1_aggregated.csv"),
                ("CUSUM config JSON", cusum_run / "config.json"),
            ]
        )

    if expect_robustness and robustness_run is not None:
        checks.extend(
            [
                ("Robustness summary CSV", robustness_run / "robustness_summary.csv"),
                ("Robustness aggregated CSV", robustness_run / "robustness_aggregated.csv"),
                ("Robustness config JSON", robustness_run / "config.json"),
            ]
        )

    if expect_timing:
        checks.append(("Timing CSV", timing_dir / "timing_results.csv"))
        checks.append(("Timing raw CSV", timing_dir / "timing_raw.csv"))
    if expect_numba_timing_poc and numba_timing_dir is not None:
        checks.append(("Numba PoC timing CSV", numba_timing_dir / "timing_numba_poc.csv"))
        checks.append(("Numba PoC timing raw CSV", numba_timing_dir / "timing_numba_poc_raw.csv"))

    if expect_ablation:
        for label, ablation_run in tref_runs.items():
            checks.append((f"t_ref ablation ({label}) CSV", ablation_run / "phase1_summary.csv"))

    if integrity_reports is not None:
        integrity_json = integrity_reports.get("json")
        integrity_md = integrity_reports.get("md")
        if integrity_json is not None:
            checks.append(("Data integrity JSON", integrity_json))
        if integrity_md is not None:
            checks.append(("Data integrity MD", integrity_md))

    for label, path in checks:
        if path.exists():
            size = path.stat().st_size
            if size == 0:
                logger.warning("  EMPTY: %s (%s)", label, path)
                ok = False
            else:
                logger.info("  OK   : %s (%s, %d bytes)", label, path.name, size)
        else:
            logger.error("  MISSING: %s (%s)", label, path)
            ok = False

    p1_csv = phase1_run / "phase1_summary.csv"
    if p1_csv.exists():
        df = pd.read_csv(p1_csv)
        n_datasets = df["dataset"].nunique()
        n_thresholds = df["threshold"].nunique()
        n_seeds = df["seed"].nunique() if "seed" in df.columns else 1
        expected = n_datasets * n_thresholds * n_seeds

        if len(df) != expected:
            logger.warning(
                "  Phase 1 rows: %d (expected %d = %d datasets × %d thresholds × %d seeds)",
                len(df), expected, n_datasets, n_thresholds, n_seeds,
            )
        else:
            logger.info(
                "  Phase 1 rows: %d (%d datasets × %d thresholds × %d seeds) ✓",
                len(df), n_datasets, n_thresholds, n_seeds,
            )

        required_p1_cols = {
            "dataset",
            "threshold",
            "accuracy_full",
            "accuracy_p2",
            "accuracy_dirichlet",
            "delta_acc_vs_full",
            "mean_work_p2",
            "work_reduction_p2",
            "work_reduction_dirichlet",
            "elbow_fraction",
            "pearson_tau_vs_dirichlet",
            "spearman_tau_vs_dirichlet",
            "accuracy_p2_val",
            "work_reduction_p2_val",
            "seed",
        }
        missing = required_p1_cols - set(df.columns)
        if missing:
            logger.warning("  Phase 1 missing columns: %s", missing)
            ok = False
        else:
            logger.info("  Phase 1 columns: all %d required present ✓", len(required_p1_cols))

    if expect_gbm and gbm_run is not None:
        gbm_csv = gbm_run / "phase1_gbm_summary.csv"
        if gbm_csv.exists():
            df = pd.read_csv(gbm_csv)
            n_datasets = df["dataset"].nunique()
            n_thresholds = df["threshold"].nunique()
            n_seeds = df["seed"].nunique() if "seed" in df.columns else 1
            expected = n_datasets * n_thresholds * n_seeds
            if len(df) != expected:
                logger.warning(
                    "  GBM rows: %d (expected %d = %d datasets × %d thresholds × %d seeds)",
                    len(df), expected, n_datasets, n_thresholds, n_seeds,
                )
            else:
                logger.info(
                    "  GBM rows: %d (%d datasets × %d thresholds × %d seeds) ✓",
                    len(df), n_datasets, n_thresholds, n_seeds,
                )

            required_cols = {
                "dataset",
                "backend",
                "threshold",
                "accuracy_full",
                "accuracy_p2",
                "delta_acc_vs_full",
                "mean_work_p2",
                "work_reduction_p2",
                "elbow_fraction",
                "accuracy_p2_val",
                "work_reduction_p2_val",
                "delta_acc_vs_full_val",
                "seed",
            }
            missing = required_cols - set(df.columns)
            if missing:
                logger.warning("  GBM missing columns: %s", missing)
                ok = False
            else:
                logger.info("  GBM columns: all %d required present ✓", len(required_cols))

    if expect_cusum and cusum_run is not None:
        cusum_csv = cusum_run / "phase1_summary.csv"
        if cusum_csv.exists():
            df = pd.read_csv(cusum_csv)
            n_datasets = df["dataset"].nunique()
            n_thresholds = df["threshold"].nunique()
            n_seeds = df["seed"].nunique() if "seed" in df.columns else 1
            expected = n_datasets * n_thresholds * n_seeds
            if len(df) != expected:
                logger.warning(
                    "  CUSUM rows: %d (expected %d = %d datasets × %d thresholds × %d seeds)",
                    len(df), expected, n_datasets, n_thresholds, n_seeds,
                )
            else:
                logger.info(
                    "  CUSUM rows: %d (%d datasets × %d thresholds × %d seeds) ✓",
                    len(df), n_datasets, n_thresholds, n_seeds,
                )

    if expect_robustness and robustness_run is not None:
        robust_csv = robustness_run / "robustness_summary.csv"
        if robust_csv.exists():
            df = pd.read_csv(robust_csv)
            required = {
                "dataset",
                "seed",
                "method",
                "contamination_rate",
                "n_corrupt_trees",
                "threshold",
                "accuracy_full",
                "accuracy_method",
                "delta_acc_vs_full",
                "work_reduction",
            }
            missing = required - set(df.columns)
            if missing:
                logger.warning("  Robustness missing columns: %s", missing)
                ok = False
            else:
                logger.info("  Robustness columns: all %d required present ✓", len(required))

    if ok:
        logger.info("  All verifications PASSED ✓")
    else:
        logger.warning("  Some verifications FAILED — check warnings above")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all Phase-1 experiments for the MDPI manuscript.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated random seeds (default: %(default)s)",
    )
    parser.add_argument("--datasets", type=str, default=DATASETS)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Root output directory for this run.",
    )
    parser.add_argument(
        "--reuse-phase1-run",
        type=Path,
        default=None,
        help=(
            "Reuse an existing Phase-1 run directory (e.g., "
            "results/run_YYYYMMDD_HHMMSS/phase1_main/run_YYYYMMDD_HHMMSS) "
            "instead of rerunning Phase 1."
        ),
    )

    parser.add_argument("--phase1-n-trees", type=int, default=PHASE1_N_TREES)
    parser.add_argument(
        "--enable-gbm",
        action="store_true",
        help="Run optional GBM P2-STOP sweep (XGBoost/LightGBM, 500 rounds by default).",
    )
    parser.add_argument(
        "--gbm-backend",
        type=lambda s: str(s).strip().lower(),
        choices=["auto", "xgboost", "lightgbm", "sklearn"],
        default="auto",
        help="auto|xgboost|lightgbm|sklearn",
    )
    parser.add_argument("--gbm-datasets", type=str, default=GBM_DATASETS)
    parser.add_argument("--gbm-n-trees", type=int, default=GBM_N_TREES)
    parser.add_argument("--gbm-learning-rate", type=float, default=0.05)
    parser.add_argument("--gbm-max-depth", type=int, default=6)
    parser.add_argument("--gbm-max-train", type=int, default=10_000)
    parser.add_argument("--gbm-max-test", type=int, default=2_000)
    parser.add_argument("--cusum-k", type=float, default=0.5)
    parser.add_argument("--cusum-h", type=float, default=4.0)
    parser.add_argument(
        "--robustness-contamination-levels",
        type=str,
        default=ROBUSTNESS_CONTAMINATION_LEVELS,
        help="Comma-separated contamination rates for robustness experiment.",
    )
    parser.add_argument("--max-train", type=int, default=MAX_TRAIN)
    parser.add_argument("--max-test", type=int, default=MAX_TEST)

    parser.add_argument("--mnist-max-rows", type=int, default=0)
    parser.add_argument("--covertype-max-rows", type=int, default=0)
    parser.add_argument("--credit-max-rows", type=int, default=0)
    parser.add_argument("--higgs-max-rows", type=int, default=HIGGS_MAX_ROWS)
    parser.add_argument("--higgs-integrity-sample-rows", type=int, default=100000)

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke-test mode: fewer trees, smaller datasets, 2 seeds.",
    )
    parser.add_argument(
        "--skip-integrity-check",
        action="store_true",
        help="Skip local dataset integrity verification.",
    )
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Skip wall-clock timing benchmark.",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip t_ref ablation experiments.",
    )
    parser.add_argument(
        "--skip-cusum",
        action="store_true",
        help="Skip CUSUM detector experiment.",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip contamination robustness experiment.",
    )

    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=3,
        help="Measured timing repeats per method (default: %(default)s)",
    )
    parser.add_argument(
        "--timing-warmup-runs",
        type=int,
        default=1,
        help="Warmup timing runs per method (default: %(default)s)",
    )
    parser.add_argument(
        "--timing-max-test",
        type=int,
        default=1000,
        help="Max test rows for timing benchmark (0 means use --max-test)",
    )
    parser.add_argument(
        "--enable-numba-timing-poc",
        action="store_true",
        help="Run Numba timing proof-of-concept on selected datasets.",
    )
    parser.add_argument(
        "--numba-timing-datasets",
        type=str,
        default=NUMBA_POC_DATASETS,
        help=(
            "Datasets for Numba timing proof-of-concept as comma-separated list "
            "or 'all' (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--numba-timing-dataset",
        type=str,
        default="",
        help=(
            "Deprecated alias for a single Numba timing dataset. "
            "Use --numba-timing-datasets."
        ),
    )
    parser.add_argument(
        "--numba-timing-max-test",
        type=int,
        default=0,
        help="Max test rows for Numba timing PoC (0 means use timing max-test).",
    )
    parser.add_argument(
        "--numba-timing-n-trees",
        type=int,
        default=0,
        help="RF trees for Numba timing PoC (0 means use --phase1-n-trees).",
    )

    parser.add_argument(
        "--primary-figure-dataset",
        type=str,
        default="mnist",
        help="Dataset used for manuscript figure targets (default: mnist)",
    )
    parser.add_argument(
        "--primary-figure-seed",
        type=int,
        default=None,
        help="Seed used for manuscript figure targets (default: first seed)",
    )

    args = parser.parse_args()

    args.seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not args.seeds:
        parser.error("At least one seed is required")

    if args.quick:
        args.phase1_n_trees = min(args.phase1_n_trees, 50)
        args.gbm_n_trees = min(args.gbm_n_trees, 100)
        args.max_train = min(args.max_train, 5000)
        args.max_test = min(args.max_test, 1000)
        args.gbm_max_train = min(args.gbm_max_train, 4000)
        args.gbm_max_test = min(args.gbm_max_test, 400)
        args.timing_max_test = min(args.timing_max_test if args.timing_max_test > 0 else 1000, 300)
        if args.numba_timing_max_test > 0:
            args.numba_timing_max_test = min(args.numba_timing_max_test, 300)
        if args.mnist_max_rows <= 0:
            args.mnist_max_rows = 20000
        if args.covertype_max_rows <= 0:
            args.covertype_max_rows = 200000
        if args.credit_max_rows <= 0:
            args.credit_max_rows = 200000
        args.higgs_max_rows = min(args.higgs_max_rows, 50000)
        if len(args.seeds) > 2:
            args.seeds = args.seeds[:2]
        args.robustness_contamination_levels = "0.05,0.25"

    args.primary_figure_dataset = str(args.primary_figure_dataset).strip().lower()
    args.numba_timing_datasets = str(args.numba_timing_datasets).strip().lower()
    args.numba_timing_dataset = str(args.numba_timing_dataset).strip().lower()

    # Backward compatibility: allow legacy singular flag.
    if args.numba_timing_dataset:
        if args.numba_timing_datasets not in {"", "all"}:
            parser.error("Use either --numba-timing-datasets or --numba-timing-dataset, not both.")
        args.numba_timing_datasets = args.numba_timing_dataset

    if args.primary_figure_seed is None:
        args.primary_figure_seed = args.seeds[0]

    dataset_list = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    if not dataset_list:
        parser.error("At least one dataset is required")
    supported_datasets = {"mnist", "covertype", "higgs", "credit"}
    invalid_datasets = [d for d in dataset_list if d not in supported_datasets]
    if invalid_datasets:
        parser.error(
            "Unsupported datasets: "
            + ",".join(invalid_datasets)
            + ". Supported: "
            + ",".join(sorted(supported_datasets))
        )
    args.datasets = ",".join(dataset_list)
    if args.primary_figure_dataset not in dataset_list:
        args.primary_figure_dataset = dataset_list[0]
    if args.numba_timing_datasets in {"", "all"}:
        numba_dataset_list = list(dataset_list)
    else:
        numba_dataset_list = [d.strip().lower() for d in args.numba_timing_datasets.split(",") if d.strip()]
    if not numba_dataset_list:
        parser.error("At least one Numba timing dataset is required")
    invalid_numba_datasets = [d for d in numba_dataset_list if d not in supported_datasets]
    if invalid_numba_datasets:
        parser.error(
            "Unsupported Numba timing datasets: "
            + ",".join(invalid_numba_datasets)
            + ". Supported: "
            + ",".join(sorted(supported_datasets))
        )
    if args.enable_numba_timing_poc and args.reuse_phase1_run is None:
        missing_numba = [d for d in numba_dataset_list if d not in dataset_list]
        if missing_numba:
            parser.error(
                "--numba-timing-datasets must be included in --datasets for threshold selection: "
                f"{missing_numba} not in {dataset_list}"
            )
    args.numba_timing_datasets = ",".join(numba_dataset_list)

    gbm_dataset_list = [d.strip().lower() for d in args.gbm_datasets.split(",") if d.strip()]
    if args.enable_gbm and not gbm_dataset_list:
        parser.error("At least one GBM dataset is required when --enable-gbm is set")
    invalid_gbm_datasets = [d for d in gbm_dataset_list if d not in supported_datasets]
    if invalid_gbm_datasets:
        parser.error(
            "Unsupported GBM datasets: "
            + ",".join(invalid_gbm_datasets)
            + ". Supported: "
            + ",".join(sorted(supported_datasets))
        )
    args.gbm_datasets = ",".join(gbm_dataset_list)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(run_dir / "experiment.log")
    logger.info("=" * 72)
    logger.info("MDPI Manuscript Experiment Runner")
    logger.info("=" * 72)
    logger.info("Timestamp       : %s", timestamp)
    logger.info("Output directory: %s", run_dir)
    logger.info("Seeds           : %s", args.seeds)
    logger.info("Datasets        : %s", args.datasets)
    logger.info("Phase 1 trees   : %d", args.phase1_n_trees)
    if args.reuse_phase1_run is not None:
        logger.info("Reuse Phase 1   : %s", args.reuse_phase1_run)
    logger.info("Enable GBM      : %s", args.enable_gbm)
    if args.enable_gbm:
        logger.info("GBM backend     : %s", args.gbm_backend)
        logger.info("GBM datasets    : %s", args.gbm_datasets)
        logger.info("GBM trees       : %d", args.gbm_n_trees)
        logger.info("GBM lr          : %.4f", args.gbm_learning_rate)
        logger.info("GBM max depth   : %d", args.gbm_max_depth)
        logger.info("GBM max train   : %d", args.gbm_max_train)
        logger.info("GBM max test    : %d", args.gbm_max_test)
    logger.info("CUSUM k/h       : %.3f / %.3f", args.cusum_k, args.cusum_h)
    logger.info("Robustness lvls : %s", args.robustness_contamination_levels)
    logger.info("Max train       : %d", args.max_train)
    logger.info("Max test        : %d", args.max_test)
    logger.info("Timing max test : %d", args.timing_max_test)
    logger.info("Timing repeats  : %d", args.timing_repeats)
    logger.info("Timing warmup   : %d", args.timing_warmup_runs)
    logger.info("Numba PoC timing: %s", args.enable_numba_timing_poc)
    if args.enable_numba_timing_poc:
        logger.info("Numba PoC ds    : %s", args.numba_timing_datasets)
        logger.info("Numba PoC max t : %d", args.numba_timing_max_test)
        logger.info("Numba PoC trees : %d", args.numba_timing_n_trees if args.numba_timing_n_trees > 0 else args.phase1_n_trees)
        logger.info("Numba available : %s", HAS_NUMBA)
    logger.info("MNIST max rows  : %d", args.mnist_max_rows)
    logger.info("Cov. max rows   : %d", args.covertype_max_rows)
    logger.info("Credit max rows : %d", args.credit_max_rows)
    logger.info("HIGGS max rows  : %d", args.higgs_max_rows)
    logger.info("Integrity check : %s", not args.skip_integrity_check)
    logger.info("Skip timing     : %s", args.skip_timing)
    logger.info("Skip ablation   : %s", args.skip_ablation)
    logger.info("Skip CUSUM      : %s", args.skip_cusum)
    logger.info("Skip robustness : %s", args.skip_robustness)
    logger.info("Primary figure  : dataset=%s, seed=%s", args.primary_figure_dataset, args.primary_figure_seed)
    logger.info("Quick mode      : %s", args.quick)

    config = {
        "timestamp": timestamp,
        "seeds": args.seeds,
        "datasets": args.datasets,
        "data_dir": str(args.data_dir),
        "phase1_n_trees": args.phase1_n_trees,
        "reuse_phase1_run": str(args.reuse_phase1_run) if args.reuse_phase1_run is not None else None,
        "enable_gbm": args.enable_gbm,
        "gbm_backend": args.gbm_backend,
        "gbm_datasets": args.gbm_datasets,
        "gbm_n_trees": args.gbm_n_trees,
        "gbm_learning_rate": args.gbm_learning_rate,
        "gbm_max_depth": args.gbm_max_depth,
        "gbm_max_train": args.gbm_max_train,
        "gbm_max_test": args.gbm_max_test,
        "cusum_k": args.cusum_k,
        "cusum_h": args.cusum_h,
        "robustness_contamination_levels": args.robustness_contamination_levels,
        "max_train": args.max_train,
        "max_test": args.max_test,
        "timing_max_test": args.timing_max_test,
        "timing_repeats": args.timing_repeats,
        "timing_warmup_runs": args.timing_warmup_runs,
        "enable_numba_timing_poc": args.enable_numba_timing_poc,
        "numba_timing_datasets": args.numba_timing_datasets,
        "numba_timing_dataset": args.numba_timing_datasets.split(",")[0],
        "numba_timing_max_test": args.numba_timing_max_test,
        "numba_timing_n_trees": args.numba_timing_n_trees,
        "numba_available": HAS_NUMBA,
        "mnist_max_rows": args.mnist_max_rows,
        "covertype_max_rows": args.covertype_max_rows,
        "credit_max_rows": args.credit_max_rows,
        "higgs_max_rows": args.higgs_max_rows,
        "higgs_integrity_sample_rows": args.higgs_integrity_sample_rows,
        "quick": args.quick,
        "skip_integrity_check": args.skip_integrity_check,
        "skip_timing": args.skip_timing,
        "skip_ablation": args.skip_ablation,
        "skip_cusum": args.skip_cusum,
        "skip_robustness": args.skip_robustness,
        "primary_figure_dataset": args.primary_figure_dataset,
        "primary_figure_seed": args.primary_figure_seed,
        "thresholds_p2stop": P2STOP_THRESHOLDS,
        "tref_ablation_windows": TREF_ABLATION_WINDOWS,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    total_t0 = time.perf_counter()

    # Step 0: data integrity
    integrity_reports: Optional[Dict[str, Path]] = None
    if not args.skip_integrity_check:
        integrity_reports = step_data_integrity(args, run_dir, logger)
    else:
        logger.info("Skipping data integrity check (--skip-integrity-check)")

    # Step 1: Phase 1 main
    if args.reuse_phase1_run is not None:
        phase1_run = Path(args.reuse_phase1_run)
        if not phase1_run.exists():
            raise FileNotFoundError(f"--reuse-phase1-run not found: {phase1_run}")
        logger.info("Reusing Phase 1 run directory: %s", phase1_run)
    else:
        phase1_run = step_phase1_main(args, run_dir, logger)

    # Step 1b: optional GBM main
    gbm_run: Optional[Path] = None
    if args.enable_gbm:
        gbm_run = step_phase1_gbm_main(args, run_dir, logger)
    else:
        logger.info("Skipping GBM sweep (enable with --enable-gbm)")

    # Step 1c: CUSUM variant
    cusum_run: Optional[Path] = None
    if not args.skip_cusum:
        cusum_run = step_phase1_cusum_main(args, run_dir, logger)
    else:
        logger.info("Skipping CUSUM experiment (--skip-cusum)")

    # Step 1d: contamination robustness
    robustness_run: Optional[Path] = None
    if not args.skip_robustness:
        robustness_run = step_phase1_robustness(args, run_dir, logger)
    else:
        logger.info("Skipping robustness experiment (--skip-robustness)")

    # Step 2: t_ref ablation
    tref_runs: Dict[str, Path] = {}
    if not args.skip_ablation:
        tref_runs = step_phase1_tref_ablation(args, run_dir, logger)
    else:
        logger.info("Skipping t_ref ablation (--skip-ablation)")

    # Step 3: timing
    timing_dir = run_dir / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_timing:
        timing_dir = step_timing_benchmark(args, phase1_run, run_dir, logger)
    else:
        logger.info("Skipping timing benchmark (--skip-timing)")

    # Step 3b: optional Numba timing proof-of-concept
    numba_timing_dir: Optional[Path] = run_dir / "timing_numba_poc"
    numba_timing_dir.mkdir(parents=True, exist_ok=True)
    if args.enable_numba_timing_poc:
        numba_timing_dir = step_timing_numba_poc(args, phase1_run, run_dir, logger)
    else:
        logger.info("Skipping Numba timing proof-of-concept (--enable-numba-timing-poc to run)")

    # Step 4: collect results
    manuscript_data_path = collect_results(
        phase1_run=phase1_run,
        gbm_run=gbm_run,
        cusum_run=cusum_run,
        robustness_run=robustness_run,
        tref_runs=tref_runs,
        timing_dir=timing_dir,
        numba_timing_dir=numba_timing_dir,
        output_dir=run_dir,
        logger=logger,
        integrity_reports=integrity_reports,
    )

    # Step 5: export tables
    export_table_artifacts(manuscript_data_path, run_dir, logger)

    # Step 6: figures
    collect_figures(phase1_run, run_dir, logger)
    export_manuscript_figure_targets(
        phase1_run=phase1_run,
        output_dir=run_dir,
        logger=logger,
        dataset=args.primary_figure_dataset,
        seed=args.primary_figure_seed,
    )

    # Step 7: verify outputs
    all_ok = verify_outputs(
        phase1_run=phase1_run,
        gbm_run=gbm_run,
        cusum_run=cusum_run,
        robustness_run=robustness_run,
        tref_runs=tref_runs,
        timing_dir=timing_dir,
        numba_timing_dir=numba_timing_dir,
        output_dir=run_dir,
        logger=logger,
        expect_timing=not args.skip_timing,
        expect_numba_timing_poc=(args.enable_numba_timing_poc and HAS_NUMBA),
        expect_ablation=not args.skip_ablation,
        expect_gbm=args.enable_gbm,
        expect_cusum=not args.skip_cusum,
        expect_robustness=not args.skip_robustness,
        integrity_reports=integrity_reports,
    )

    total_elapsed = time.perf_counter() - total_t0
    logger.info("=" * 72)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("  Total time  : %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
    logger.info("  Output dir  : %s", run_dir)
    logger.info("  Status      : %s", "ALL PASS ✓" if all_ok else "SOME FAILURES — see log")
    logger.info("=" * 72)

    print(f"\n{'=' * 60}")
    print(f"Experiment run complete: {run_dir}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"Status: {'ALL PASS' if all_ok else 'SOME FAILURES'}")
    print(f"Log: {run_dir / 'experiment.log'}")
    print(f"Data: {run_dir / 'manuscript_data.json'}")
    print(f"Tables: {run_dir / 'tables' / 'table_manifest.json'}")
    print(f"Figures: {run_dir / 'figures' / 'manuscript_figure_targets.json'}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
