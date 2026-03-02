#!/usr/bin/env python3
"""Re-export manuscript tables from an existing results run directory.

This is a lightweight post-processing helper: it recomputes aggregated
statistics (including confidence intervals) from the per-seed CSV artefacts
already present in a run directory, without rerunning any experiments.

Example:
  python scripts/reexport_tables_from_run.py results/run_20260228_151229
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root is on sys.path when this file is executed as a script.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_all_experiments import collect_results, export_table_artifacts, setup_logging


def _latest_run_dir(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    runs = [p for p in parent.glob("run_*") if p.is_dir()]
    return sorted(runs)[-1] if runs else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Existing results run directory (results/run_*)")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    logger = setup_logging(run_dir / "reexport_tables.log")
    logger.info("Re-exporting tables for existing run: %s", run_dir)

    phase1_run = _latest_run_dir(run_dir / "phase1_main")
    if phase1_run is None:
        raise FileNotFoundError(f"Missing phase1_main/run_* under {run_dir}")

    gbm_run = _latest_run_dir(run_dir / "phase1_gbm_main")
    cusum_run = _latest_run_dir(run_dir / "phase1_cusum_main")
    robustness_run = _latest_run_dir(run_dir / "phase1_robustness")

    tref_runs = {}
    for label in ["tref10", "tref20", "tref40"]:
        p = _latest_run_dir(run_dir / f"phase1_ablation_{label}")
        if p is not None:
            tref_runs[label] = p

    timing_dir = run_dir / "timing"
    numba_timing_dir = run_dir / "timing_numba_poc"

    integrity_reports = None
    integrity_json = run_dir / "data_integrity_report.json"
    if integrity_json.exists():
        integrity_reports = {"json": integrity_json}

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
    export_table_artifacts(manuscript_data_path, run_dir, logger)


if __name__ == "__main__":
    main()
