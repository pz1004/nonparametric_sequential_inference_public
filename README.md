# Nonparametric Sequential Inference (Public Release Bundle)

## Status
This repository is prepared as the **official code and artifact bundle** for the manuscript.

To avoid premature disclosure, please treat this as a **pre-release package**. Public archival metadata (citation/announcement) will be finalized after acceptance.

## Scope
This bundle targets reproducibility of the current manuscript, including:
- RF Phase-1 threshold sweep and Dirichlet comparison,
- adaptive threshold selection rule,
- CUSUM detector evaluation,
- contamination robustness experiment (5%/15%/25%),
- GBM (XGBoost) cross-model validation,
- Numba compiled timing proof-of-concept,
- manuscript table/figure artifacts and source files.

## Included Code
- Canonical runner:
  - `scripts/run_all_experiments.py`
  - `scripts/run_mdpi_pipeline.py`
  - `scripts/reexport_tables_from_run.py`
  - `scripts/verify_data_integrity.py`
- Experiment modules:
  - `experiments/phase1_changepoint/run_phase1_changepoint.py`
  - `experiments/phase1_changepoint/run_phase1_gbm_changepoint.py`
  - `experiments/phase1_changepoint/run_phase1_robustness_contamination.py`
  - `experiments/shared/local_data_loader.py`
  - `experiments/shared/p2_streaming.py`
  - `experiments/shared/numba_rf_inference.py`


## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Requirement (User-Provided)
Place datasets under local `data/` with loaders compatible with:
- `mnist`
- `covertype`
- `higgs`
- `credit`

Dataset loading behavior is defined in:
- `experiments/shared/local_data_loader.py`

## Reproducing Full Paper Pipeline (From Scratch)
```bash
python scripts/run_all_experiments.py \
  --data-dir data \
  --seeds 42,123,456,789,1024 \
  --max-train 20000 \
  --max-test 5000 \
  --enable-gbm \
  --enable-numba-timing-poc \
  --numba-timing-datasets mnist,covertype,higgs,credit \
  --numba-timing-n-trees 500
```

## Reproducing the Numba Timing Refresh Only
```bash
python scripts/run_all_experiments.py \
  --reuse-phase1-run artifacts/run_20260228_151229/phase1_main/run_20260228_151237 \
  --skip-integrity-check --skip-timing --skip-ablation --skip-cusum --skip-robustness \
  --enable-numba-timing-poc \
  --numba-timing-datasets mnist,covertype,higgs,credit \
  --numba-timing-n-trees 500 \
  --seeds 42,123,456,789,1024 \
  --max-train 20000 \
  --numba-timing-max-test 1000
```

## Re-Exporting Tables from an Existing Run
```bash
python scripts/reexport_tables_from_run.py artifacts/run_20260228_151229
```

## Notes
- This bundle is aligned to the current manuscript draft which will be disclosed after the acceptance.
- Timing interpretation is implementation-dependent: Python reference timing and Numba compiled timing are both included and reported separately.
