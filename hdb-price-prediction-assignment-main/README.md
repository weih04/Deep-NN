# HDB resale price regression (SC4001 Part A)

Predicts HDB resale prices from `hdb_price_prediction.csv`: train on 2017–2020,
validate and select on 2021, test on 2022.

## Environment

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/); exact versions are pinned in `uv.lock`
- Dependencies (versions used): `torch` 2.13.0, `numpy` 2.5.2, `pandas` 3.0.5,
  `matplotlib` 3.11.1, `ray[tune]` 2.58.0, `captum` 0.9.0; `pytest` 9.1.1 for tests
- Runs on CPU. `hdb_price_prediction.csv` must sit in the repository root.

```bash
uv sync
```

## Seeds

- Default seed `42` (`src/config.py`), used by the baseline, every grid-search trial,
  the final retrain, the λ search and the A4 attribution models.
- A2 ablations and A3 wide-and-deep evaluation: seeds `42, 43, 44, 45, 46`
  (`SEEDS` in `src/ablation.py` and `src/wide_deep.py`).

## Run

Run every command from the repository root, **in this order**. Later steps read
the outputs of earlier ones (`outputs/tune/best_config.json`, `outputs/ablation/runs.json`).

| question | command | writes |
|---|---|---|
| baseline | `uv run python src/baseline.py` | `outputs/baseline/` |
| A1 grid search (25 trials, Ray Tune) | `uv run python src/tune_search.py` | `outputs/tune/` |
| A1 final retrain + test | `uv run python src/final.py` | `outputs/final/` |
| A1 report numbers | `uv run python src/report.py` | `outputs/report_evidence.md` |
| A2 ablations (3 variants × 5 seeds) | `uv run python src/ablation.py` | `outputs/ablation/` |
| A3 wide-and-deep (λ search + 5 seeds) | `uv run python src/wide_deep.py` | `outputs/wide_deep/` |
| A4 FeatureAblation | `uv run python src/attribution.py` | `outputs/attribution/` |


## Method

- **Split and preprocessing** (`src/data.py`): category IDs, continuous-feature
  z-scores and target statistics are fitted on the training rows only. ID 0 is
  reserved for unseen categories. The final retrain refits them on 2017–2021.
- **Training** (`src/baseline.py`, `src/final.py`): Adam, MSE on the standardised
  price, at most 20 epochs, early stopping with patience 4 and minimum improvement
  1e-4. The final retrain, ablations and wide-and-deep runs train for the best
  epoch selected in A1.
- **Selection:** A1 picks the lowest best validation RMSE; A3 picks λ by
  validation RMSE. Because λ = 0 makes the wide branch inert, the best non-zero λ
  (0.5) is also evaluated as a supplementary diagnostic and used in A4.

## Code map

```text
src/config.py       run configuration (split years, features, model size, training settings)
src/data.py         split, fit preprocessing, tensor conversion
src/model.py        categorical embeddings + MLP, optional wide branch (A3)
src/baseline.py     training loop with early stopping, RMSE / R² evaluation
src/plots.py        shared plotting
src/tune_search.py  A1 Ray Tune grid search over hidden width and embedding dim
src/final.py        A1 final retrain on train+validation, test evaluation
src/report.py       A1 parameter counts and report evidence
src/ablation.py     A2 no-embeddings and Sigmoid ablations
src/wide_deep.py    A3 λ search and five-seed wide-and-deep evaluation
src/attribution.py  A4 Captum FeatureAblation on the test set
```
