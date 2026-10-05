# HDB price baseline

## Run it

Install the exact environment recorded in `uv.lock`:

```bash
uv sync
```

Run the baseline with:

```bash
uv run python src/baseline.py
```

Run the hyperparameter grid search (25 trials, Ray Tune):

```bash
uv run python src/tune_search.py
```

Retrain the selected configuration on train + validation and evaluate on the test year:

```bash
uv run python src/final.py
```

Collect the numbers cited in the report:

```bash
uv run python src/report.py
```

Run the tests:

```bash
uv run pytest tests/ -v
```

Run the command from the repository root. The input CSV and output directory are constants at the top of `src/baseline.py`, while the data split, features, model size, and training hyperparameters are defined in `src/config.py`.

## Code map

```text
src/config.py       run configuration
src/data.py         split, fit preprocessing, tensor conversion
src/model.py        categorical embeddings and MLP
src/baseline.py     single-run training, RMSE evaluation, and model export
src/plots.py        shared RMSE curve plotting
src/tune_search.py  Ray Tune grid search over hidden width and embedding dim
src/final.py        final retrain on train+validation, test evaluation
src/report.py       parameter counts and report evidence
```
