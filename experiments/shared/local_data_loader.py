"""Local dataset loaders for MNIST/Covertype/HIGGS/CreditCard under data/ directory."""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@dataclass
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray


SUPPORTED_DATASETS = ("mnist", "covertype", "higgs", "credit")


def _subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_rows: Optional[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_rows is None or max_rows <= 0 or len(X) <= max_rows:
        return X, y
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        return X[idx], y[idx]
    if max_rows < classes.size:
        raise ValueError(
            f"max_rows={max_rows} is smaller than number of classes={classes.size}; "
            "cannot keep at least one sample per class."
        )

    proportions = counts / counts.sum()
    alloc = np.floor(proportions * max_rows).astype(int)
    alloc = np.maximum(alloc, 1)
    alloc = np.minimum(alloc, counts)

    # Adjust allocation to match max_rows exactly.
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
        if take >= cls_idx.size:
            chosen = cls_idx
        else:
            chosen = rng.choice(cls_idx, size=int(take), replace=False)
        selected_parts.append(chosen)

    selected = np.concatenate(selected_parts)
    rng.shuffle(selected)
    return X[selected], y[selected]


def _encode_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.ravel()
    encoder = LabelEncoder()
    return encoder.fit_transform(y)


def _open_idx_file(path_plain: Path):
    if path_plain.exists():
        return open(path_plain, "rb")
    path_gz = Path(str(path_plain) + ".gz")
    if path_gz.exists():
        return gzip.open(path_gz, "rb")
    raise FileNotFoundError(f"Missing IDX file: {path_plain} (or .gz)")


def _read_idx_images(path: Path) -> np.ndarray:
    with _open_idx_file(path) as f:
        magic, n_images, n_rows, n_cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image magic number in {path}: {magic}")
        raw = f.read()
    images = np.frombuffer(raw, dtype=np.uint8)
    expected = n_images * n_rows * n_cols
    if images.size != expected:
        raise ValueError(
            f"MNIST image byte mismatch in {path}: got {images.size}, expected {expected}"
        )
    return images.reshape(n_images, n_rows * n_cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with _open_idx_file(path) as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label magic number in {path}: {magic}")
        raw = f.read()
    labels = np.frombuffer(raw, dtype=np.uint8)
    if labels.size != n_labels:
        raise ValueError(
            f"MNIST label byte mismatch in {path}: got {labels.size}, expected {n_labels}"
        )
    return labels


def load_mnist(data_dir: Path, max_rows: Optional[int], seed: int) -> DatasetBundle:
    raw_dir = data_dir / "mnist" / "MNIST" / "raw"
    X_train = _read_idx_images(raw_dir / "train-images-idx3-ubyte")
    y_train = _read_idx_labels(raw_dir / "train-labels-idx1-ubyte")
    X_test = _read_idx_images(raw_dir / "t10k-images-idx3-ubyte")
    y_test = _read_idx_labels(raw_dir / "t10k-labels-idx1-ubyte")

    X = np.vstack([X_train, X_test]).astype(np.float32) / 255.0
    y = np.concatenate([y_train, y_test]).astype(np.int64)

    X, y = _subsample(X, y, max_rows=max_rows, seed=seed)
    y = _encode_labels(y)
    return DatasetBundle(name="mnist", X=X, y=y)


def load_covertype(data_dir: Path, max_rows: Optional[int], seed: int) -> DatasetBundle:
    cov_dir = data_dir / "covertype"
    samples_path = cov_dir / "samples_py3"
    targets_path = cov_dir / "targets_py3"

    X = joblib.load(samples_path)
    y = joblib.load(targets_path)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Covertype row mismatch: X={X.shape[0]} y={y.shape[0]}")

    X, y = _subsample(X, y, max_rows=max_rows, seed=seed)
    y = _encode_labels(y)
    return DatasetBundle(name="covertype", X=X, y=y)


def load_higgs(data_dir: Path, max_rows: Optional[int]) -> DatasetBundle:
    higgs_path = data_dir / "higgs" / "HIGGS.csv"
    nrows = None if max_rows is None or max_rows <= 0 else int(max_rows)

    df = pd.read_csv(higgs_path, header=None, nrows=nrows)
    if df.shape[1] != 29:
        raise ValueError(f"HIGGS expected 29 columns (label + 28 features), got {df.shape[1]}")

    y_raw = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32)

    y = np.asarray(y_raw)
    if np.issubdtype(y.dtype, np.floating):
        y = (y > 0.5).astype(np.int64)
    y = _encode_labels(y)

    return DatasetBundle(name="higgs", X=X, y=y)


def load_credit(data_dir: Path, max_rows: Optional[int], seed: int) -> DatasetBundle:
    credit_path = data_dir / "creditcard" / "creditcard.csv"
    nrows = None if max_rows is None or max_rows <= 0 else int(max_rows)
    df = pd.read_csv(credit_path, nrows=nrows)

    target_col = "Class" if "Class" in df.columns else df.columns[-1]
    y = df[target_col].to_numpy()
    X = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)

    X, y = _subsample(X, y, max_rows=max_rows, seed=seed)
    y = _encode_labels(y)
    return DatasetBundle(name="credit", X=X, y=y)


def load_local_dataset(
    name: str,
    data_dir: Path,
    seed: int,
    dataset_max_rows: Optional[Dict[str, Optional[int]]] = None,
) -> DatasetBundle:
    key = name.lower().strip()
    if key not in SUPPORTED_DATASETS:
        available = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    limits = dataset_max_rows or {}

    if key == "mnist":
        return load_mnist(data_dir=data_dir, max_rows=limits.get("mnist"), seed=seed)
    if key == "covertype":
        return load_covertype(data_dir=data_dir, max_rows=limits.get("covertype"), seed=seed)
    if key == "higgs":
        return load_higgs(data_dir=data_dir, max_rows=limits.get("higgs"))
    if key == "credit":
        return load_credit(data_dir=data_dir, max_rows=limits.get("credit"), seed=seed)

    raise RuntimeError(f"Unhandled dataset key: {key}")
