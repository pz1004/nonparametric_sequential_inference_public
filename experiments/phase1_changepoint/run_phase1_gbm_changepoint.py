"""Phase 1 (GBM): P2-STOP threshold sweep on boosting trajectories."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from experiments.shared.local_data_loader import (
    SUPPORTED_DATASETS,
    DatasetBundle,
    load_local_dataset,
)
from experiments.shared.p2_streaming import detect_scale_changepoint, rolling_iqr_scale


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


def _resolve_backend(name: str) -> str:
    key = name.lower().strip()
    if key not in {"auto", "xgboost", "lightgbm", "sklearn"}:
        raise ValueError(f"Unsupported backend '{name}'. Use auto|xgboost|lightgbm|sklearn")

    def _has_module(mod: str) -> bool:
        try:
            __import__(mod)
            return True
        except Exception:
            return False

    def _fallback(preferred: str) -> str:
        order = [preferred, "xgboost", "lightgbm", "sklearn"]
        seen: set[str] = set()
        for cand in order:
            if cand in seen:
                continue
            seen.add(cand)
            if cand == "sklearn":
                return "sklearn"
            if _has_module(cand):
                return cand
        return "sklearn"

    if key == "xgboost":
        if _has_module("xgboost"):
            return "xgboost"
        chosen = _fallback("lightgbm")
        print(
            "[GBM] Requested backend 'xgboost' is unavailable. "
            f"Falling back to '{chosen}'.",
            flush=True,
        )
        return chosen

    if key == "lightgbm":
        if _has_module("lightgbm"):
            return "lightgbm"
        chosen = _fallback("xgboost")
        print(
            "[GBM] Requested backend 'lightgbm' is unavailable. "
            f"Falling back to '{chosen}'.",
            flush=True,
        )
        return chosen

    if key == "sklearn":
        return "sklearn"

    if _has_module("xgboost"):
        return "xgboost"

    if _has_module("lightgbm"):
        return "lightgbm"

    print("[GBM] No xgboost/lightgbm found. Falling back to 'sklearn'.", flush=True)
    return "sklearn"


def _build_gbm(
    backend: str,
    n_trees: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_classes: int,
):
    if backend == "xgboost":
        from xgboost import XGBClassifier

        objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
        eval_metric = "mlogloss" if n_classes > 2 else "logloss"
        return XGBClassifier(
            n_estimators=int(n_trees),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            subsample=1.0,
            colsample_bytree=1.0,
            objective=objective,
            eval_metric=eval_metric,
            random_state=int(seed),
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )

    if backend == "lightgbm":
        from lightgbm import LGBMClassifier

        kwargs = {
            "n_estimators": int(n_trees),
            "learning_rate": float(learning_rate),
            "max_depth": int(max_depth),
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "random_state": int(seed),
            "n_jobs": -1,
            "verbosity": -1,
        }
        if n_classes > 2:
            kwargs["objective"] = "multiclass"
            kwargs["num_class"] = int(n_classes)
        else:
            kwargs["objective"] = "binary"
        return LGBMClassifier(**kwargs)

    if backend == "sklearn":
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            n_estimators=int(n_trees),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            random_state=int(seed),
        )

    raise ValueError(f"Unhandled backend: {backend}")


def _ensure_proba_2d(probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim == 1:
        p1 = arr.reshape(-1, 1)
        arr = np.hstack([1.0 - p1, p1])
    return arr


def _predict_proba_prefix(model, backend: str, X: np.ndarray, n_rounds: int) -> np.ndarray:
    if backend == "xgboost":
        probs = model.predict_proba(X, iteration_range=(0, int(n_rounds)))
        return _ensure_proba_2d(probs)

    if backend == "lightgbm":
        probs = model.predict_proba(X, num_iteration=int(n_rounds))
        return _ensure_proba_2d(probs)

    if backend == "sklearn":
        stage_iter = model.staged_predict_proba(X)
        out = None
        for idx, probs in enumerate(stage_iter, start=1):
            if idx == int(n_rounds):
                out = probs
                break
        if out is None:
            out = model.predict_proba(X)
        return _ensure_proba_2d(out)

    raise ValueError(f"Unhandled backend: {backend}")


def compute_probability_trajectory(model, backend: str, X: np.ndarray, n_trees: int) -> np.ndarray:
    first = _predict_proba_prefix(model, backend=backend, X=X[:1], n_rounds=1)
    n_classes = first.shape[1]
    traj = np.zeros((X.shape[0], int(n_trees), n_classes), dtype=np.float64)
    for t in range(1, int(n_trees) + 1):
        traj[:, t - 1, :] = _predict_proba_prefix(model, backend=backend, X=X, n_rounds=t)
    return traj


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


def _compute_stop_times(
    trajectories: np.ndarray,
    thresholds: List[float],
    ref_window: Tuple[int, int],
    warmup: int,
    min_trees: int,
    rolling_window: int,
) -> np.ndarray:
    n_thresholds = len(thresholds)
    n_samples, total_trees, _ = trajectories.shape

    scalar_traj = np.max(trajectories, axis=2)
    deltas = np.diff(scalar_traj, axis=1, prepend=scalar_traj[:, :1])
    stop_times = np.full((n_thresholds, n_samples), total_trees - 1, dtype=np.int64)

    for i in range(n_samples):
        scales = rolling_iqr_scale(deltas[i], window=rolling_window, warmup=warmup)
        for t_idx, theta in enumerate(thresholds):
            tau_i, _ = detect_scale_changepoint(
                scales,
                threshold=theta,
                ref_window=ref_window,
                min_trees=min_trees,
            )
            stop_times[t_idx, i] = tau_i

    return stop_times


def _select_best(df_ds: pd.DataFrame) -> pd.Series:
    eps = 0.005  # 0.5 percentage points on accuracy (proportion scale)
    feasible = df_ds[df_ds["delta_acc_vs_full_val"] >= -eps]
    if feasible.empty:
        # If nothing meets the tolerance, pick best validation accuracy
        # (tie-break by larger work reduction, then smaller theta).
        return df_ds.sort_values(
            ["accuracy_p2_val", "work_reduction_p2_val", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]
    return feasible.sort_values(
        ["work_reduction_p2_val", "accuracy_p2_val", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]


def run_dataset(
    bundle: DatasetBundle,
    backend: str,
    n_trees: int,
    learning_rate: float,
    max_depth: int,
    thresholds: List[float],
    ref_window: Tuple[int, int],
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

    n_classes = np.unique(y_train).size
    model = _build_gbm(
        backend=backend,
        n_trees=n_trees,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed,
        n_classes=n_classes,
    )
    model.fit(X_train, y_train)

    traj_val = compute_probability_trajectory(model, backend=backend, X=X_val, n_trees=n_trees)
    traj_test = compute_probability_trajectory(model, backend=backend, X=X_test, n_trees=n_trees)

    stop_val = _compute_stop_times(
        trajectories=traj_val,
        thresholds=thresholds,
        ref_window=ref_window,
        warmup=warmup,
        min_trees=min_trees,
        rolling_window=rolling_window,
    )
    stop_test = _compute_stop_times(
        trajectories=traj_test,
        thresholds=thresholds,
        ref_window=ref_window,
        warmup=warmup,
        min_trees=min_trees,
        rolling_window=rolling_window,
    )

    full_pred_val = np.argmax(traj_val[:, -1, :], axis=1)
    acc_full_val = accuracy_score(y_val, full_pred_val)
    full_pred_test = np.argmax(traj_test[:, -1, :], axis=1)
    acc_full_test = accuracy_score(y_test, full_pred_test)

    rows: List[Dict[str, float]] = []
    for t_idx, theta in enumerate(thresholds):
        tau_val = stop_val[t_idx]
        acc_val, mean_work_val = evaluate_threshold(traj_val, y_val, tau_val)
        wr_val = 1.0 - mean_work_val
        elbow_val = float(np.mean(tau_val < (n_trees - 1)))

        tau = stop_test[t_idx]
        acc_test, mean_work_test = evaluate_threshold(traj_test, y_test, tau)
        wr_test = 1.0 - mean_work_test
        elbow_test = float(np.mean(tau < (n_trees - 1)))

        rows.append(
            {
                "dataset": bundle.name,
                "backend": backend,
                "seed": int(seed),
                "n_train": float(len(X_train)),
                "n_val": float(len(X_val)),
                "n_test": float(len(X_test)),
                "n_trees": float(n_trees),
                "threshold": float(theta),
                "accuracy_full": float(acc_full_test),
                "accuracy_p2": float(acc_test),
                "delta_acc_vs_full": float(acc_test - acc_full_test),
                "mean_work_p2": float(mean_work_test),
                "work_reduction_p2": float(wr_test),
                "elbow_fraction": float(elbow_test),
                "accuracy_p2_val": float(acc_val),
                "delta_acc_vs_full_val": float(acc_val - acc_full_val),
                "work_reduction_p2_val": float(wr_val),
                "elbow_fraction_val": float(elbow_val),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 GBM P2-STOP threshold sweep")
    parser.add_argument("--datasets", type=str, default="mnist,covertype,higgs")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/phase1_changepoint/results_gbm"),
    )
    parser.add_argument(
        "--backend",
        type=lambda s: str(s).strip().lower(),
        choices=["auto", "xgboost", "lightgbm", "sklearn"],
        default="auto",
        help="auto|xgboost|lightgbm|sklearn",
    )
    parser.add_argument("--n-trees", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--thresholds", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--ref-window", type=str, default="10,30")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--min-trees", type=int, default=10)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--max-train", type=int, default=10000)
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    parser.add_argument("--mnist-max-rows", type=int, default=0)
    parser.add_argument("--covertype-max-rows", type=int, default=0)
    parser.add_argument("--credit-max-rows", type=int, default=0)
    parser.add_argument("--higgs-max-rows", type=int, default=200000)
    args = parser.parse_args()

    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    invalid = [d for d in datasets if d not in SUPPORTED_DATASETS]
    if invalid:
        raise ValueError(f"Unsupported datasets: {invalid}. Supported: {SUPPORTED_DATASETS}")

    thresholds = parse_thresholds(args.thresholds)
    ref_window = parse_ref_window(args.ref_window)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")

    backend = _resolve_backend(args.backend)

    run_dir = args.output_dir / f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_max_rows = {
        "mnist": args.mnist_max_rows,
        "covertype": args.covertype_max_rows,
        "credit": args.credit_max_rows,
        "higgs": args.higgs_max_rows,
    }

    all_rows = []
    for ds_name in datasets:
        for seed in seeds:
            bundle = load_local_dataset(
                name=ds_name,
                data_dir=args.data_dir,
                seed=seed,
                dataset_max_rows=dataset_max_rows,
            )
            validate_dataset_bundle(bundle)
            print(f"[GBM] dataset={ds_name} seed={seed} backend={backend}")

            df = run_dataset(
                bundle=bundle,
                backend=backend,
                n_trees=args.n_trees,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                thresholds=thresholds,
                ref_window=ref_window,
                warmup=args.warmup,
                min_trees=args.min_trees,
                rolling_window=args.rolling_window,
                max_train=args.max_train,
                max_test=args.max_test,
                seed=seed,
            )
            seed_dir = run_dir / f"seed_{seed}"
            seed_dir.mkdir(exist_ok=True)
            df.to_csv(seed_dir / f"phase1_gbm_summary_{ds_name}.csv", index=False)
            all_rows.append(df)

    if not all_rows:
        raise RuntimeError("No GBM runs completed.")

    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_csv(run_dir / "phase1_gbm_summary.csv", index=False)

    grouped = (
        summary.groupby(["dataset", "threshold"], as_index=False)
        .agg(
            accuracy_full_mean=("accuracy_full", "mean"),
            accuracy_p2_mean=("accuracy_p2", "mean"),
            delta_acc_vs_full_mean=("delta_acc_vs_full", "mean"),
            mean_work_p2_mean=("mean_work_p2", "mean"),
            work_reduction_p2_mean=("work_reduction_p2", "mean"),
            elbow_fraction_mean=("elbow_fraction", "mean"),
            accuracy_p2_val_mean=("accuracy_p2_val", "mean"),
            delta_acc_vs_full_val_mean=("delta_acc_vs_full_val", "mean"),
            work_reduction_p2_val_mean=("work_reduction_p2_val", "mean"),
            n=("seed", "count"),
        )
        .sort_values(["dataset", "threshold"])
    )
    grouped.to_csv(run_dir / "phase1_gbm_aggregated.csv", index=False)

    best_rows = []
    for ds in sorted(summary["dataset"].unique()):
        ds_df = summary[summary["dataset"] == ds]
        for seed in sorted(ds_df["seed"].unique()):
            best_rows.append(_select_best(ds_df[ds_df["seed"] == seed]))
    best_df = pd.DataFrame(best_rows).reset_index(drop=True)
    best_df.to_csv(run_dir / "phase1_gbm_best_thresholds.csv", index=False)

    config = {
        "datasets": datasets,
        "seeds": seeds,
        "data_dir": str(args.data_dir),
        "backend_requested": args.backend,
        "backend_resolved": backend,
        "n_trees": args.n_trees,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "thresholds": thresholds,
        "ref_window": list(ref_window),
        "warmup": args.warmup,
        "min_trees": args.min_trees,
        "rolling_window": args.rolling_window,
        "max_train": args.max_train,
        "max_test": args.max_test,
        "dataset_max_rows": dataset_max_rows,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"[GBM] Run complete: {run_dir}")
    print(f"[GBM] Summary: {run_dir / 'phase1_gbm_summary.csv'}")


if __name__ == "__main__":
    main()
