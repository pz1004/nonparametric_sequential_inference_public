"""Verify integrity of local datasets in data/ folder."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shared.local_data_loader import (
    _read_idx_images,
    _read_idx_labels,
    load_local_dataset,
)


@dataclass
class DatasetIntegrity:
    dataset: str
    status: str
    rows: int | None
    columns: int | None
    classes: int | None
    missing_values: int | None
    notes: Dict[str, Any]


def verify_mnist(data_dir: Path) -> DatasetIntegrity:
    raw_dir = data_dir / "mnist" / "MNIST" / "raw"

    train_images = _read_idx_images(raw_dir / "train-images-idx3-ubyte")
    train_labels = _read_idx_labels(raw_dir / "train-labels-idx1-ubyte")
    test_images = _read_idx_images(raw_dir / "t10k-images-idx3-ubyte")
    test_labels = _read_idx_labels(raw_dir / "t10k-labels-idx1-ubyte")

    if train_images.shape[0] != train_labels.shape[0]:
        raise ValueError("MNIST train image/label count mismatch")
    if test_images.shape[0] != test_labels.shape[0]:
        raise ValueError("MNIST test image/label count mismatch")

    X = np.vstack([train_images, test_images])
    y = np.concatenate([train_labels, test_labels])

    label_counts = np.bincount(y, minlength=10).tolist()

    return DatasetIntegrity(
        dataset="mnist",
        status="PASS",
        rows=int(X.shape[0]),
        columns=int(X.shape[1]),
        classes=int(len(np.unique(y))),
        missing_values=int(np.isnan(X).sum()),
        notes={
            "train_rows": int(train_images.shape[0]),
            "test_rows": int(test_images.shape[0]),
            "image_dim": [28, 28],
            "label_counts": label_counts,
        },
    )


def verify_covertype(data_dir: Path) -> DatasetIntegrity:
    cov_dir = data_dir / "covertype"
    X = np.asarray(joblib.load(cov_dir / "samples_py3"))
    y = np.asarray(joblib.load(cov_dir / "targets_py3"))

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Covertype row mismatch X={X.shape[0]} y={y.shape[0]}")

    missing = int(np.isnan(X).sum()) + int(np.isnan(y.astype(np.float64)).sum())
    class_values, class_counts = np.unique(y, return_counts=True)

    return DatasetIntegrity(
        dataset="covertype",
        status="PASS",
        rows=int(X.shape[0]),
        columns=int(X.shape[1]),
        classes=int(class_values.size),
        missing_values=missing,
        notes={
            "dtype_X": str(X.dtype),
            "dtype_y": str(y.dtype),
            "class_values": class_values.tolist(),
            "class_counts": class_counts.tolist(),
        },
    )


def verify_credit(data_dir: Path) -> DatasetIntegrity:
    path = data_dir / "creditcard" / "creditcard.csv"
    df = pd.read_csv(path)

    target_col = "Class" if "Class" in df.columns else df.columns[-1]
    y = df[target_col]
    X = df.drop(columns=[target_col])

    class_values, class_counts = np.unique(y.to_numpy(), return_counts=True)

    return DatasetIntegrity(
        dataset="credit",
        status="PASS",
        rows=int(df.shape[0]),
        columns=int(X.shape[1]),
        classes=int(class_values.size),
        missing_values=int(df.isna().sum().sum()),
        notes={
            "target_col": target_col,
            "dtype_summary": {col: str(dtype) for col, dtype in df.dtypes.items() if col in [target_col, "Time", "Amount"]},
            "class_values": [int(v) if float(v).is_integer() else float(v) for v in class_values],
            "class_counts": class_counts.tolist(),
        },
    )


def verify_higgs(data_dir: Path, sample_rows: int) -> DatasetIntegrity:
    path = data_dir / "higgs" / "HIGGS.csv"
    df_sample = pd.read_csv(path, header=None, nrows=sample_rows)

    if df_sample.shape[1] != 29:
        raise ValueError(f"HIGGS expected 29 columns, got {df_sample.shape[1]}")

    y = df_sample.iloc[:, 0].to_numpy()
    label_values, label_counts = np.unique(y, return_counts=True)

    # Fast approximate row count from full pass; no parsing performed.
    with path.open("rb") as f:
        approx_rows = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""))

    return DatasetIntegrity(
        dataset="higgs",
        status="PASS",
        rows=int(approx_rows),
        columns=28,
        classes=int(label_values.size),
        missing_values=int(df_sample.isna().sum().sum()),
        notes={
            "sample_rows_checked": int(df_sample.shape[0]),
            "sample_label_values": [float(v) for v in label_values],
            "sample_label_counts": label_counts.tolist(),
            "sample_dtypes": [str(dtype) for dtype in df_sample.dtypes[:5]],
        },
    )


def run_loader_smoke(data_dir: Path) -> Dict[str, str]:
    limits = {"mnist": 2000, "covertype": 5000, "credit": 5000, "higgs": 5000}
    out: Dict[str, str] = {}
    for name in ["mnist", "covertype", "credit", "higgs"]:
        bundle = load_local_dataset(name=name, data_dir=data_dir, seed=42, dataset_max_rows=limits)
        out[name] = f"X={bundle.X.shape}, y={bundle.y.shape}, classes={len(np.unique(bundle.y))}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify local dataset integrity")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--higgs-sample-rows", type=int, default=200000)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/data_integrity_report.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("experiments/data_integrity_report.md"),
    )
    args = parser.parse_args()

    checks = [
        verify_mnist(args.data_dir),
        verify_covertype(args.data_dir),
        verify_credit(args.data_dir),
        verify_higgs(args.data_dir, sample_rows=args.higgs_sample_rows),
    ]

    smoke = run_loader_smoke(args.data_dir)

    report = {
        "data_dir": str(args.data_dir),
        "checks": [asdict(item) for item in checks],
        "loader_smoke": smoke,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    args.output_json.write_text(json.dumps(report, indent=2))

    lines = ["# Data Integrity Report", "", f"Data dir: `{args.data_dir}`", ""]
    for item in checks:
        lines.extend(
            [
                f"## {item.dataset}",
                f"- Status: **{item.status}**",
                f"- Rows: `{item.rows}`",
                f"- Columns: `{item.columns}`",
                f"- Classes: `{item.classes}`",
                f"- Missing values: `{item.missing_values}`",
                f"- Notes: `{json.dumps(item.notes)}`",
                "",
            ]
        )

    lines.append("## Loader Smoke Test")
    for name, message in smoke.items():
        lines.append(f"- {name}: `{message}`")

    args.output_md.write_text("\n".join(lines))

    print(f"Integrity report written:\n- {args.output_json}\n- {args.output_md}")


if __name__ == "__main__":
    main()
