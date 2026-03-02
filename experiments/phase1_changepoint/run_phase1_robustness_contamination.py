"""Phase 1 robustness study: contamination stress-test for stopping rules.

Injects corrupted trees (random-noise probability outputs) into RF ensembles
and compares:
  1) P2-STOP style IQR scale detector (robust)
  2) Mean/STD-based scale detector (non-robust baseline)
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from experiments.shared.local_data_loader import (
    SUPPORTED_DATASETS,
    DatasetBundle,
    load_local_dataset,
)
from experiments.shared.p2_streaming import detect_scale_changepoint, rolling_iqr_scale


def parse_float_list(raw: str) -> List[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one value")
    return values


def parse_int_list(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def parse_ref_window(raw: str) -> Tuple[int, int]:
    parts = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError("ref window must have two integers, e.g. 10,30")
    return parts[0], parts[1]


def validate_dataset_bundle(bundle: DatasetBundle) -> None:
    if bundle.X.ndim != 2:
        raise ValueError(f"{bundle.name}: X must be 2D, got shape {bundle.X.shape}")
    if bundle.y.ndim != 1:
        raise ValueError(f"{bundle.name}: y must be 1D, got shape {bundle.y.shape}")
    if bundle.X.shape[0] != bundle.y.shape[0]:
        raise ValueError(
            f"{bundle.name}: row mismatch between X and y ({bundle.X.shape[0]} vs {bundle.y.shape[0]})"
        )
    if bundle.X.shape[0] < 20:
        raise ValueError(f"{bundle.name}: too few rows ({bundle.X.shape[0]}), need at least 20")
    if bundle.X.shape[1] < 2:
        raise ValueError(f"{bundle.name}: too few features ({bundle.X.shape[1]}), need at least 2")
    if not np.isfinite(bundle.X).all():
        raise ValueError(f"{bundle.name}: non-finite values detected in X")
    if not np.isfinite(bundle.y).all():
        raise ValueError(f"{bundle.name}: non-finite values detected in y")
    n_classes = np.unique(bundle.y).size
    if n_classes < 2:
        raise ValueError(f"{bundle.name}: need at least 2 classes, got {n_classes}")


def stratified_subsample(
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
        raise ValueError(f"max_rows={max_rows} smaller than class count={classes.size}; cannot stratify.")

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
            take = min(spare[idx], remaining)
            alloc[idx] += take
            remaining -= int(take)

    rng = np.random.default_rng(seed)
    selected_parts = []
    for cls, take in zip(classes, alloc):
        cls_idx = np.where(y == cls)[0]
        chosen = cls_idx if take >= cls_idx.size else rng.choice(cls_idx, size=int(take), replace=False)
        selected_parts.append(chosen)

    selected = np.concatenate(selected_parts)
    rng.shuffle(selected)
    return X[selected], y[selected]


def rolling_std_scale(values: Iterable[float], window: int = 20, warmup: int = 10) -> np.ndarray:
    seq = np.asarray(list(values), dtype=np.float64)
    scales = np.full(seq.shape[0], np.nan, dtype=np.float64)
    min_samples = max(5, int(warmup))
    for t in range(seq.shape[0]):
        start = max(0, t - window + 1)
        buf = seq[start : t + 1]
        if buf.size < min_samples:
            continue
        if buf.size < 2:
            scales[t] = 0.0
        else:
            scales[t] = float(np.std(buf, ddof=1))
    return scales


def compute_per_tree_probabilities(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    per_tree = [tree.predict_proba(X) for tree in model.estimators_]
    return np.stack(per_tree, axis=1)  # (N, T, C)


def build_trajectory_from_tree_probs(tree_probs: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(tree_probs, axis=1)
    denom = np.arange(1, tree_probs.shape[1] + 1, dtype=np.float64)[None, :, None]
    return cumulative / denom


def contaminate_tree_probs(
    clean_probs: np.ndarray,
    contamination_rate: float,
    tree_indices: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    if contamination_rate <= 0:
        return clean_probs.copy(), 0

    n_samples, n_trees, n_classes = clean_probs.shape
    n_corrupt = int(round(contamination_rate * n_trees))
    n_corrupt = max(0, min(n_corrupt, n_trees))
    if n_corrupt == 0:
        return clean_probs.copy(), 0

    chosen = tree_indices[:n_corrupt]
    corrupted = clean_probs.copy()
    alpha = np.ones(n_classes, dtype=np.float64)
    for t_idx in chosen:
        corrupted[:, t_idx, :] = rng.dirichlet(alpha, size=n_samples)
    return corrupted, int(n_corrupt)


def evaluate_threshold(
    trajectories: np.ndarray,
    y_true: np.ndarray,
    stop_times: np.ndarray,
) -> Tuple[float, float]:
    n_samples, n_trees, _ = trajectories.shape
    rows = np.arange(n_samples)
    probs = trajectories[rows, stop_times]
    preds = np.argmax(probs, axis=1)
    acc = float(accuracy_score(y_true, preds))
    mean_work = float(np.mean((stop_times + 1) / n_trees))
    return acc, mean_work


def compute_stop_times(
    trajectories: np.ndarray,
    thresholds: List[float],
    scale_method: str,
    ref_window: Tuple[int, int],
    warmup: int,
    min_trees: int,
    rolling_window: int,
) -> np.ndarray:
    n_thresholds = len(thresholds)
    n_samples, n_trees, _ = trajectories.shape
    scalar = np.max(trajectories, axis=2)
    deltas = np.diff(scalar, axis=1, prepend=scalar[:, :1])

    out = np.full((n_thresholds, n_samples), n_trees - 1, dtype=np.int64)
    for i in range(n_samples):
        if scale_method == "iqr":
            scales = rolling_iqr_scale(deltas[i], window=rolling_window, warmup=warmup)
        elif scale_method == "mean":
            scales = rolling_std_scale(deltas[i], window=rolling_window, warmup=warmup)
        else:
            raise ValueError(f"Unknown scale_method: {scale_method}")

        for t_idx, theta in enumerate(thresholds):
            tau_i, _ = detect_scale_changepoint(
                scales,
                threshold=theta,
                ref_window=ref_window,
                min_trees=min_trees,
            )
            out[t_idx, i] = tau_i
    return out


def select_best_idx(
    thresholds: List[float],
    acc_full_val: float,
    acc_vals: List[float],
    wr_vals: List[float],
) -> int:
    eps = 0.005  # 0.5 percentage points on accuracy (proportion scale)

    # One-sided constraint: allow improvements, limit degradation.
    idxs = [i for i in range(len(thresholds)) if acc_vals[i] >= acc_full_val - eps]
    if idxs:
        best = max(idxs, key=lambda i: (wr_vals[i], acc_vals[i], -thresholds[i]))
        return int(best)

    # If nothing meets the tolerance, pick best validation accuracy (tie-break by WR).
    best = max(range(len(thresholds)), key=lambda i: (acc_vals[i], wr_vals[i], -thresholds[i]))
    return int(best)


def run_dataset_seed(
    bundle: DatasetBundle,
    thresholds: List[float],
    contamination_levels: List[float],
    ref_window: Tuple[int, int],
    n_trees: int,
    warmup: int,
    min_trees: int,
    rolling_window: int,
    max_train: int,
    max_test: int,
    seed: int,
) -> pd.DataFrame:
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
        X_train, y_train = stratified_subsample(X_train, y_train, max_rows=max_train, seed=seed)
    if max_test > 0 and len(X_test) > max_test:
        X_test, y_test = stratified_subsample(X_test, y_test, max_rows=max_test, seed=seed + 1)
    if max_test > 0 and len(X_val) > max_test:
        X_val, y_val = stratified_subsample(X_val, y_val, max_rows=max_test, seed=seed + 2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)

    clean_val = compute_per_tree_probabilities(rf, X_val)
    clean_test = compute_per_tree_probabilities(rf, X_test)

    n_tree_total = clean_test.shape[1]
    tree_perm = np.random.default_rng(seed + 1337).permutation(n_tree_total)

    rows: List[Dict[str, float | int | str]] = []
    for level in contamination_levels:
        rng_val = np.random.default_rng(seed + int(level * 10_000) + 17)
        rng_test = np.random.default_rng(seed + int(level * 10_000) + 23)

        cont_val, n_corrupt = contaminate_tree_probs(
            clean_probs=clean_val,
            contamination_rate=level,
            tree_indices=tree_perm,
            rng=rng_val,
        )
        cont_test, _ = contaminate_tree_probs(
            clean_probs=clean_test,
            contamination_rate=level,
            tree_indices=tree_perm,
            rng=rng_test,
        )

        traj_val = build_trajectory_from_tree_probs(cont_val)
        traj_test = build_trajectory_from_tree_probs(cont_test)

        full_pred_val = np.argmax(traj_val[:, -1, :], axis=1)
        full_pred_test = np.argmax(traj_test[:, -1, :], axis=1)
        acc_full_val = float(accuracy_score(y_val, full_pred_val))
        acc_full_test = float(accuracy_score(y_test, full_pred_test))

        method_map = {
            "p2_iqr": "iqr",
            "mean_scale": "mean",
        }
        for method_name, scale_method in method_map.items():
            stop_val = compute_stop_times(
                trajectories=traj_val,
                thresholds=thresholds,
                scale_method=scale_method,
                ref_window=ref_window,
                warmup=warmup,
                min_trees=min_trees,
                rolling_window=rolling_window,
            )
            stop_test = compute_stop_times(
                trajectories=traj_test,
                thresholds=thresholds,
                scale_method=scale_method,
                ref_window=ref_window,
                warmup=warmup,
                min_trees=min_trees,
                rolling_window=rolling_window,
            )

            acc_vals: List[float] = []
            wr_vals: List[float] = []
            for idx in range(len(thresholds)):
                acc_v, work_v = evaluate_threshold(traj_val, y_val, stop_val[idx])
                acc_vals.append(float(acc_v))
                wr_vals.append(float(1.0 - work_v))
            best_idx = select_best_idx(thresholds, acc_full_val, acc_vals, wr_vals)

            tau_test = stop_test[best_idx]
            acc_test, mean_work_test = evaluate_threshold(traj_test, y_test, tau_test)
            work_reduction = float(1.0 - mean_work_test)
            elbow_fraction = float(np.mean(tau_test < (n_tree_total - 1)))

            rows.append(
                {
                    "dataset": bundle.name,
                    "seed": int(seed),
                    "method": method_name,
                    "scale_method": scale_method,
                    "contamination_rate": float(level),
                    "n_corrupt_trees": int(n_corrupt),
                    "n_trees": int(n_tree_total),
                    "threshold": float(thresholds[best_idx]),
                    "accuracy_full": float(acc_full_test),
                    "accuracy_method": float(acc_test),
                    "delta_acc_vs_full": float(acc_test - acc_full_test),
                    "mean_work": float(mean_work_test),
                    "work_reduction": float(work_reduction),
                    "elbow_fraction": float(elbow_fraction),
                    "n_test": int(len(y_test)),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness contamination study for P2-STOP")
    parser.add_argument("--datasets", type=str, default="mnist,covertype,higgs,credit")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--n-trees", type=int, default=200)
    parser.add_argument("--thresholds", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--contamination-levels", type=str, default="0.05,0.15,0.25")
    parser.add_argument("--ref-window", type=str, default="10,30")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--min-trees", type=int, default=10)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--max-train", type=int, default=20000)
    parser.add_argument("--max-test", type=int, default=5000)
    parser.add_argument("--mnist-max-rows", type=int, default=0)
    parser.add_argument("--covertype-max-rows", type=int, default=0)
    parser.add_argument("--credit-max-rows", type=int, default=0)
    parser.add_argument("--higgs-max-rows", type=int, default=500000)
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/phase1_changepoint/results_robustness"),
    )
    args = parser.parse_args()

    thresholds = parse_float_list(args.thresholds)
    levels = parse_float_list(args.contamination_levels)
    levels = [0.0] + levels
    ref_window = parse_ref_window(args.ref_window)
    seeds = parse_int_list(args.seeds)
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]

    invalid = [d for d in datasets if d not in SUPPORTED_DATASETS]
    if invalid:
        raise ValueError(f"Unsupported datasets: {invalid}. Available: {SUPPORTED_DATASETS}")

    run_dir = args.output_dir / f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_max_rows = {
        "mnist": args.mnist_max_rows,
        "covertype": args.covertype_max_rows,
        "credit": args.credit_max_rows,
        "higgs": args.higgs_max_rows,
    }

    all_rows: List[pd.DataFrame] = []
    for seed in seeds:
        for name in datasets:
            bundle = load_local_dataset(
                name=name,
                data_dir=args.data_dir,
                seed=seed,
                dataset_max_rows=dataset_max_rows,
            )
            validate_dataset_bundle(bundle)
            print(f"[robustness] dataset={name} seed={seed}")
            df = run_dataset_seed(
                bundle=bundle,
                thresholds=thresholds,
                contamination_levels=levels,
                ref_window=ref_window,
                n_trees=args.n_trees,
                warmup=args.warmup,
                min_trees=args.min_trees,
                rolling_window=args.rolling_window,
                max_train=args.max_train,
                max_test=args.max_test,
                seed=seed,
            )
            all_rows.append(df)

    if not all_rows:
        raise RuntimeError("No robustness results produced")

    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_csv(run_dir / "robustness_summary.csv", index=False)

    grouped = (
        summary.groupby(["dataset", "contamination_rate", "method"], as_index=False)
        .agg(
            n=("seed", "count"),
            accuracy_full_mean=("accuracy_full", "mean"),
            accuracy_method_mean=("accuracy_method", "mean"),
            delta_acc_vs_full_mean=("delta_acc_vs_full", "mean"),
            work_reduction_mean=("work_reduction", "mean"),
            mean_work_mean=("mean_work", "mean"),
            elbow_fraction_mean=("elbow_fraction", "mean"),
        )
        .sort_values(["dataset", "contamination_rate", "method"])
    )
    grouped.to_csv(run_dir / "robustness_aggregated.csv", index=False)

    config = {
        "datasets": datasets,
        "seeds": seeds,
        "data_dir": str(args.data_dir),
        "n_trees": args.n_trees,
        "thresholds": thresholds,
        "contamination_levels": levels,
        "ref_window": list(ref_window),
        "warmup": args.warmup,
        "min_trees": args.min_trees,
        "rolling_window": args.rolling_window,
        "max_train": args.max_train,
        "max_test": args.max_test,
        "dataset_max_rows": dataset_max_rows,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Robustness experiment complete: {run_dir}")


if __name__ == "__main__":
    main()
