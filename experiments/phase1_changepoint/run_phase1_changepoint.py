"""Phase 1: Nonparametric change-point experiment for ensemble trajectories."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import betainc
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from experiments.shared.local_data_loader import (
    SUPPORTED_DATASETS,
    DatasetBundle,
    load_local_dataset,
)
from experiments.shared.p2_streaming import (
    detect_cusum_changepoint,
    detect_scale_changepoint,
    rolling_iqr_scale,
    streaming_iqr_scale,
)


def parse_thresholds(raw: str) -> List[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one threshold is required")
    return values


def parse_ref_window(raw: str) -> Tuple[int, int]:
    parts = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError("ref window must have two integers, e.g. 10,30")
    return parts[0], parts[1]


def safe_correlations(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan"), float("nan")
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)


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
        raise ValueError(
            f"max_rows={max_rows} smaller than class count={classes.size}; cannot stratify."
        )

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
        class_idx = np.where(y == cls)[0]
        chosen = class_idx if take >= class_idx.size else rng.choice(class_idx, size=int(take), replace=False)
        selected_parts.append(chosen)
    selected = np.concatenate(selected_parts)
    rng.shuffle(selected)
    return X[selected], y[selected]


def compute_probability_trajectory(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    per_tree_probs = [tree.predict_proba(X) for tree in model.estimators_]
    stacked = np.stack(per_tree_probs, axis=1)  # (N, T, C)
    cumulative = np.cumsum(stacked, axis=1)
    denominators = np.arange(1, stacked.shape[1] + 1, dtype=np.float64)[None, :, None]
    return cumulative / denominators


def compute_dirichlet_stop_times(
    tree_predictions: np.ndarray,
    n_classes: int,
    confidence_threshold: float,
    min_trees: int,
) -> np.ndarray:
    """Approximate LazyRF stopping with per-instance stopping times."""

    n_samples, n_trees = tree_predictions.shape
    stop_times = np.full(n_samples, n_trees - 1, dtype=np.int64)

    for i in range(n_samples):
        counts = np.ones(n_classes, dtype=np.float64)
        for t in range(n_trees):
            counts[int(tree_predictions[i, t])] += 1.0
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
                var_best = best * (total - best)
                var_second = second * (total - second)
                cov = -best * second
                numer = max(var_best + var_second - 2.0 * cov, 0.0)
                sigma = np.sqrt(numer / (total * total * (total + 1.0)))
                z = mu_diff / (sigma + 1e-12)
                p_stable = 0.5 * (1.0 + np.tanh(0.7978845608 * (z + 0.044715 * z**3)))

            if p_stable > confidence_threshold:
                stop_times[i] = t
                break

    return stop_times


def evaluate_threshold(
    trajectories: np.ndarray,
    y_true: np.ndarray,
    stop_times: np.ndarray,
) -> Tuple[float, float]:
    n_samples, n_trees, _ = trajectories.shape
    rows = np.arange(n_samples)
    stage_probs = trajectories[rows, stop_times]
    predictions = np.argmax(stage_probs, axis=1)
    accuracy = accuracy_score(y_true, predictions)
    mean_work = float(np.mean((stop_times + 1) / n_trees))
    return accuracy, mean_work


def save_dataset_plots(
    dataset: str,
    output_dir: Path,
    scales: np.ndarray,
    relative_scales: np.ndarray,
    p2_stop_times: np.ndarray,
    dirichlet_stop_times: np.ndarray,
    thresholds: Iterable[float],
    work_reductions: Iterable[float],
    accuracies: Iterable[float],
    baseline_work: float,
    baseline_acc: float,
) -> None:
    rng = np.random.default_rng(42)
    n_samples, n_trees = scales.shape

    sample_count = min(20, n_samples)
    sample_indices = rng.choice(n_samples, size=sample_count, replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=160)

    ax = axes[0]
    for idx in sample_indices:
        ax.plot(np.arange(1, n_trees + 1), scales[idx], color="gray", alpha=0.35, linewidth=1.0)
    ax.set_title("Streaming IQR scale per instance")
    ax.set_xlabel("Tree index")
    ax.set_ylabel("Scale")

    ax = axes[1]
    finite = np.isfinite(relative_scales)
    counts = finite.sum(axis=0)
    sum_rel = np.nansum(relative_scales, axis=0)
    mean_rel = np.divide(sum_rel, counts, out=np.full(n_trees, np.nan), where=counts > 0)
    centered = np.where(finite, relative_scales - mean_rel, 0.0)
    var_rel = np.divide(
        np.sum(centered**2, axis=0),
        counts,
        out=np.full(n_trees, np.nan),
        where=counts > 0,
    )
    std_rel = np.sqrt(var_rel)
    xs = np.arange(1, n_trees + 1)
    ax.plot(xs, mean_rel, color="tab:blue", linewidth=2.0)
    ax.fill_between(xs, mean_rel - std_rel, mean_rel + std_rel, color="tab:blue", alpha=0.2)
    ax.axhline(0.10, color="tab:red", linestyle="--", linewidth=1.0, label="theta=0.10")
    ax.set_title("Relative scale mean +/- std")
    ax.set_xlabel("Tree index")
    ax.set_ylabel("Relative scale")
    ax.legend()

    ax = axes[2]
    ax.hist(p2_stop_times + 1, bins=20, alpha=0.6, label="P2-STOP", color="tab:green")
    ax.hist(dirichlet_stop_times + 1, bins=20, alpha=0.6, label="Dirichlet", color="tab:orange")
    ax.set_title("Stopping-time histogram")
    ax.set_xlabel("Trees used")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"phase1_scale_diagnostics_{dataset}.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.scatter(work_reductions, accuracies, s=45, color="tab:blue", label="P2 thresholds")
    for theta, x_value, y_value in zip(thresholds, work_reductions, accuracies):
        ax.text(x_value, y_value, f"{theta:.2f}", fontsize=8)

    ax.scatter([1.0 - baseline_work], [baseline_acc], marker="x", s=120, color="tab:orange", label="Dirichlet")
    ax.set_xlabel("Work reduction")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs work reduction")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"phase1_pareto_{dataset}.png")
    plt.close(fig)


def run_dataset(
    bundle: DatasetBundle,
    thresholds: List[float],
    ref_window: Tuple[int, int],
    n_trees: int,
    warmup: int,
    min_trees: int,
    dirichlet_threshold: float,
    max_train: int,
    max_test: int,
    seed: int,
    output_dir: Path,
    scale_mode: str = "rolling",
    rolling_window: int = 20,
    detection_method: str = "relative",
    cusum_k: float = 0.5,
    cusum_h: float = 4.0,
) -> pd.DataFrame:
    # Three-way split: 60% train, 10% val, 30% test
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
        test_size=1.0 / 7.0,  # 10% of total
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

    rf = RandomForestClassifier(
        n_estimators=n_trees,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Helper: compute stop-times for a given evaluation set
    def _compute_stops(X_eval):
        traj = compute_probability_trajectory(rf, X_eval)
        straj = np.max(traj, axis=2)
        delt = np.diff(straj, axis=1, prepend=straj[:, :1])
        n_eval, total = delt.shape
        sc = np.full((n_eval, total), np.nan, dtype=np.float64)
        rel = np.full((n_thresholds, n_eval, total), np.nan, dtype=np.float64)
        st = np.full((n_thresholds, n_eval), total - 1, dtype=np.int64)
        for i in range(n_eval):
            if scale_mode == "rolling":
                si = rolling_iqr_scale(delt[i], window=rolling_window, warmup=warmup)
            else:
                si = streaming_iqr_scale(delt[i], warmup=warmup)
            sc[i] = si
            if detection_method == "cusum":
                tau_i, cs = detect_cusum_changepoint(
                    si, k=cusum_k, h=cusum_h,
                    ref_window=ref_window, min_trees=min_trees,
                )
                for t_idx in range(n_thresholds):
                    st[t_idx, i] = tau_i
                    rel[t_idx, i] = cs
            else:
                for t_idx, theta in enumerate(thresholds):
                    tau_i, r = detect_scale_changepoint(
                        si, threshold=theta,
                        ref_window=ref_window, min_trees=min_trees,
                    )
                    st[t_idx, i] = tau_i
                    rel[t_idx, i] = r
        return traj, delt, sc, rel, st

    n_thresholds = len(thresholds)

    # --- Validation set ---
    traj_val, deltas_val, scales_val, relative_val, stop_times_val = _compute_stops(X_val)
    n_val = len(X_val)
    total_trees = traj_val.shape[1]

    # --- Test set ---
    trajectories, deltas, scales, relative, stop_times = _compute_stops(X_test)
    n_samples = len(X_test)

    tree_predictions = np.stack([tree.predict(X_test) for tree in rf.estimators_], axis=1)
    dirichlet_stop_times = compute_dirichlet_stop_times(
        tree_predictions=tree_predictions,
        n_classes=len(np.unique(y_train)),
        confidence_threshold=dirichlet_threshold,
        min_trees=min_trees,
    )

    full_predictions = np.argmax(trajectories[:, -1, :], axis=1)
    acc_full = accuracy_score(y_test, full_predictions)
    dirichlet_acc, dirichlet_work = evaluate_threshold(trajectories, y_test, dirichlet_stop_times)

    full_pred_val = np.argmax(traj_val[:, -1, :], axis=1)
    acc_full_val = accuracy_score(y_val, full_pred_val)

    rows: List[Dict[str, float]] = []
    work_reductions = []
    accuracies = []

    for t_idx, theta in enumerate(thresholds):
        # Val metrics
        tau_v = stop_times_val[t_idx]
        acc_val, work_val = evaluate_threshold(traj_val, y_val, tau_v)
        wr_val = 1.0 - work_val
        elbow_val = float(np.mean(tau_v < (total_trees - 1)))

        # Test metrics
        tau = stop_times[t_idx]
        acc_p2, mean_work = evaluate_threshold(trajectories, y_test, tau)
        work_reduction = 1.0 - mean_work
        elbow_fraction = float(np.mean(tau < (total_trees - 1)))

        r_pearson, r_spearman = safe_correlations(tau, dirichlet_stop_times)

        rows.append(
            {
                "dataset": bundle.name,
                "n_train": float(len(X_train)),
                "n_val": float(n_val),
                "n_test": float(n_samples),
                "n_trees": float(total_trees),
                "threshold": float(theta),
                "accuracy_full": float(acc_full),
                "accuracy_p2": float(acc_p2),
                "accuracy_dirichlet": float(dirichlet_acc),
                "delta_acc_vs_full": float(acc_p2 - acc_full),
                "delta_acc_vs_dirichlet": float(acc_p2 - dirichlet_acc),
                "mean_work_p2": float(mean_work),
                "mean_work_dirichlet": float(dirichlet_work),
                "work_reduction_p2": float(work_reduction),
                "work_reduction_dirichlet": float(1.0 - dirichlet_work),
                "elbow_fraction": float(elbow_fraction),
                "pearson_tau_vs_dirichlet": float(r_pearson),
                "spearman_tau_vs_dirichlet": float(r_spearman),
                # Validation set metrics (for threshold selection)
                "accuracy_full_val": float(acc_full_val),
                "accuracy_p2_val": float(acc_val),
                "delta_acc_vs_full_val": float(acc_val - acc_full_val),
                "work_reduction_p2_val": float(wr_val),
                "elbow_fraction_val": float(elbow_val),
            }
        )

        work_reductions.append(work_reduction)
        accuracies.append(acc_p2)

    # Compute scalar trajectory from full probability trajectories
    scalar_traj = np.max(trajectories, axis=2)

    np.savez_compressed(
        output_dir / f"trajectories_{bundle.name}.npz",
        trajectories=trajectories,
        scalar_trajectory=scalar_traj,
        y_true=y_test,
        y_pred_full=full_predictions,
        dirichlet_stop_times=dirichlet_stop_times,
        dataset_name=bundle.name,
        n_trees=total_trees,
    )

    np.savez_compressed(
        output_dir / f"scales_{bundle.name}.npz",
        deltas=deltas,
        scales=scales,
        relative_scales=relative,
        stop_times=stop_times,
        thresholds=np.asarray(thresholds, dtype=np.float64),
    )

    default_idx = int(np.argmin(np.abs(np.asarray(thresholds) - 0.10)))
    save_dataset_plots(
        dataset=bundle.name,
        output_dir=output_dir,
        scales=scales,
        relative_scales=relative[default_idx],
        p2_stop_times=stop_times[default_idx],
        dirichlet_stop_times=dirichlet_stop_times,
        thresholds=thresholds,
        work_reductions=work_reductions,
        accuracies=accuracies,
        baseline_work=dirichlet_work,
        baseline_acc=dirichlet_acc,
    )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 changepoint experiment")
    parser.add_argument("--datasets", type=str, default="mnist,covertype,higgs,credit")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--n-trees", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--min-trees", type=int, default=10)
    parser.add_argument("--thresholds", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--ref-window", type=str, default="10,30")
    parser.add_argument("--dirichlet-threshold", type=float, default=0.95)
    parser.add_argument("--max-train", type=int, default=20000)
    parser.add_argument("--max-test", type=int, default=5000)
    parser.add_argument("--mnist-max-rows", type=int, default=0)
    parser.add_argument("--covertype-max-rows", type=int, default=0)
    parser.add_argument("--credit-max-rows", type=int, default=0)
    parser.add_argument("--higgs-max-rows", type=int, default=500000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/phase1_changepoint/results"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds", type=str, default="",
        help="Comma-separated seeds for multi-seed evaluation. Overrides --seed if non-empty.",
    )
    parser.add_argument(
        "--scale-mode", type=str, default="rolling",
        choices=["rolling", "prefix"],
        help="Scale estimation: 'rolling' (windowed) or 'prefix' (full-prefix P2).",
    )
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument(
        "--detection-method", type=str, default="relative",
        choices=["relative", "cusum"],
        help="Change-point detection: 'relative' (ratio threshold) or 'cusum' (Page's CUSUM).",
    )
    parser.add_argument("--cusum-k", type=float, default=0.5, help="CUSUM allowance parameter")
    parser.add_argument("--cusum-h", type=float, default=4.0, help="CUSUM decision threshold")
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    ref_window = parse_ref_window(args.ref_window)

    # Determine seed list
    if args.seeds.strip():
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed]

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "datasets": [item.strip() for item in args.datasets.split(",") if item.strip()],
        "data_dir": str(args.data_dir),
        "n_trees": args.n_trees,
        "warmup": args.warmup,
        "min_trees": args.min_trees,
        "thresholds": thresholds,
        "ref_window": ref_window,
        "dirichlet_threshold": args.dirichlet_threshold,
        "max_train": args.max_train,
        "max_test": args.max_test,
        "dataset_max_rows": {
            "mnist": args.mnist_max_rows,
            "covertype": args.covertype_max_rows,
            "credit": args.credit_max_rows,
            "higgs": args.higgs_max_rows,
        },
        "seeds": seeds,
        "scale_mode": args.scale_mode,
        "rolling_window": args.rolling_window,
        "detection_method": args.detection_method,
        "cusum_k": args.cusum_k,
        "cusum_h": args.cusum_h,
    }

    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    all_rows: List[pd.DataFrame] = []
    for seed in seeds:
        for dataset_name in config["datasets"]:
            key = dataset_name.strip().lower()
            if key not in SUPPORTED_DATASETS:
                available = ", ".join(SUPPORTED_DATASETS)
                raise ValueError(f"Unsupported dataset '{dataset_name}'. Available: {available}")
            try:
                bundle = load_local_dataset(
                    name=key,
                    data_dir=args.data_dir,
                    seed=seed,
                    dataset_max_rows=config["dataset_max_rows"],
                )
                validate_dataset_bundle(bundle)
            except Exception as exc:
                raise RuntimeError(f"Phase 1 failed while preparing dataset '{key}': {exc}") from exc

            seed_dir = run_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            frame = run_dataset(
                bundle=bundle,
                thresholds=thresholds,
                ref_window=ref_window,
                n_trees=args.n_trees,
                warmup=args.warmup,
                min_trees=args.min_trees,
                dirichlet_threshold=args.dirichlet_threshold,
                max_train=args.max_train,
                max_test=args.max_test,
                seed=seed,
                output_dir=seed_dir,
                scale_mode=args.scale_mode,
                rolling_window=args.rolling_window,
                detection_method=args.detection_method,
                cusum_k=args.cusum_k,
                cusum_h=args.cusum_h,
            )
            frame["seed"] = seed
            all_rows.append(frame)

    if not all_rows:
        raise RuntimeError("Phase 1 produced no dataset results.")

    summary = pd.concat(all_rows, ignore_index=True)
    if summary.empty:
        raise RuntimeError("Phase 1 summary is empty.")
    summary.to_csv(run_dir / "phase1_summary.csv", index=False)

    # Aggregate across seeds: mean ± std for numeric columns
    numeric_cols = [
        "accuracy_full", "accuracy_p2", "accuracy_dirichlet",
        "delta_acc_vs_full", "delta_acc_vs_dirichlet",
        "mean_work_p2", "work_reduction_p2", "elbow_fraction",
        "accuracy_p2_val", "delta_acc_vs_full_val",
        "work_reduction_p2_val", "elbow_fraction_val",
    ]
    agg_funcs = {col: ["mean", "std"] for col in numeric_cols if col in summary.columns}
    grouped = summary.groupby(["dataset", "threshold"], as_index=False).agg(agg_funcs)
    grouped.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in grouped.columns
    ]
    grouped.to_csv(run_dir / "phase1_aggregated.csv", index=False)

    best_rows = (
        summary.sort_values(["dataset", "accuracy_p2"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .head(1)
    )
    best_rows.to_csv(run_dir / "phase1_best_thresholds.csv", index=False)

    print(f"Phase 1 complete. Results: {run_dir}")


if __name__ == "__main__":
    main()
