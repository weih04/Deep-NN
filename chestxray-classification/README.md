# Chest X-ray classification (SC4001 Part B)

Binary classification of 64×64 grey-level chest X-rays: label 1 = pneumonia
(positive class), label 0 = normal.

## Environment

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/); exact versions in `uv.lock`
- Dependencies: `torch`, `numpy`, `matplotlib` (`pytest` for tests)
- Runs on CPU. Seed: `42` (`src/config.py`); deterministic algorithms are enabled.

```bash
uv sync
```

`chestxray_binary_64.npz` must sit in the repository root.

## Run

Run from the repository root.

B1 — AlexNet-like CNN, learning-rate sweep {1e-5, 1e-4, 1e-3, 1e-2, 1e-1} at batch size 128
(about 15 minutes on an Apple M1 CPU):

```bash
uv run python src/b1.py
```

Tests:

```bash
uv run pytest tests/ -v
```

## Method

- **Preprocessing:** pixels scaled to [0, 1], then normalised with the single mean and std
  of the training set (`src/data.py`).
- **Model (Figure B1):** every conv layer is conv (no bias, SAME padding) → BatchNorm → ReLU;
  the fc layers keep their bias; dropout 0.5 on the two hidden fc layers only (`src/models.py`).
- **Training:** Adam, `BCEWithLogitsLoss`, at most 20 epochs, early stopping on validation loss
  with patience 4 and minimum improvement 1e-4; the best-epoch weights are restored
  (`src/train.py`). Every learning rate starts from the same seed and initialisation.
- **Metrics:** the prediction is pneumonia when sigmoid(logit) ≥ 0.5. Sensitivity = pneumonia recall,
  specificity = normal recall. Table values are taken at the best epoch.
- **Selection:** lowest validation classification error (ties broken by lower validation loss).
  Only the selected model is evaluated on the test set, once.

## Outputs (`outputs/b1/`)

```text
table.md          lr, best epoch, val loss, val error, sensitivity, specificity
results.json      the table, selected lr, parameter count, normalisation stats, test metrics
histories.json    per-epoch train loss and validation metrics for every lr
val_loss.png      validation loss vs epoch, five learning rates
val_accuracy.png  validation accuracy vs epoch, five learning rates
alexnet_best.pt   selected model weights (reused for B3/B4)
```

## Code map

```text
src/config.py   seed, batch size, epochs, early stopping, threshold
src/data.py     load the pre-split npz, fit normalisation on train, tensor datasets
src/models.py   conv block and the AlexNet-like network
src/train.py    Adam training loop, early stopping, accuracy/sensitivity/specificity
src/plots.py    validation curves, one line per swept value
src/b1.py       B1 learning-rate sweep, selection, and the single test evaluation
```
