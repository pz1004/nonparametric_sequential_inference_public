"""Compatibility wrapper for the Phase-1-only MDPI pipeline.

This script now delegates to `scripts/run_all_experiments.py` so there is
one canonical experiment runner for manuscript artifacts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the canonical Phase-1 MDPI experiment pipeline",
        allow_abbrev=False,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))

    # Parse known args and forward all other flags transparently.
    args, remaining = parser.parse_known_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")

    cmd = [
        sys.executable,
        "scripts/run_all_experiments.py",
        "--data-dir",
        str(args.data_dir),
        *remaining,
    ]

    print(f"[INFO] Delegating to canonical runner: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
