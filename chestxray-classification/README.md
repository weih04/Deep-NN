# Chest X-ray classification (SC4001 Part B)

Binary classification of 64×64 grey-level chest X-rays: label 1 = pneumonia
(positive class), label 0 = normal.

## Environment

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/); exact versions are pinned in `uv.lock`
- Dependencies (versions used): `torch` 2.14.1, `numpy` 2.5.3, `matplotlib` 3.11.2,
  `captum` 0.9.0; `pytest` 9.1.1 for tests
- Runs on CPU. `chestxray_binary_64.npz` must sit in the repository root.

```bash
uv sync
```

## Seeds

Seed `42` (`src/config.py`) for model initialisation, batch shuffling and the B4
image selection. Deterministic algorithms are enabled (`src/main.py`).

## Run

Run every command from the repository root, **in this order**. Each question
reads the selections and model weights written by the earlier ones.

| question | command | writes |
|---|---|---|
| B1 AlexNet-like, learning rates {1e-5 … 1e-1} at batch 128 (about 15 min on an Apple M1 CPU) | `uv run python src/main.py b1` | `outputs/b1/` |
| B2 VGG-like, batch sizes {16 … 256} at B1's learning rate | `uv run python src/main.py b2` | `outputs/b2/` |
| B3 ResNet-like at B1's learning rate and B2's batch size, plus the B1–B3 comparison | `uv run python src/main.py b3` | `outputs/b3/` |
| B4 Grad-CAM on five random pneumonia test X-rays | `uv run python src/main.py b4` | `outputs/b4/` |

## Method

- **Preprocessing:** pixels scaled to [0, 1], then normalised with the single mean and std
  of the training set (`src/data.py`).
- **Models (Figure B):** every conv layer is conv (no bias, SAME padding) → BatchNorm → ReLU;
  the fc layers keep their bias; dropout 0.5 on the two hidden fc layers of B1 only. In B3, ReLU is
  applied after each residual addition, and 1×1 stride-2 projections match shapes (`src/models.py`).
- **Training:** Adam, `BCEWithLogitsLoss`, at most 20 epochs, early stopping on validation loss
  with patience 4 and minimum improvement 1e-4; the best-epoch weights are restored
  (`src/train.py`). Every swept value starts from the same seed and initialisation.
- **Metrics:** the prediction is pneumonia when sigmoid(logit) ≥ 0.5. Sensitivity = pneumonia recall,
  specificity = normal recall. Table values are taken at the best epoch. FLOPs count conv and
  linear layers for one 64×64 image.
- **Selection:** lowest validation classification error (ties broken by lower validation loss).
  Only the selected model is evaluated on the test set, once.
- **Grad-CAM (B4):** Captum `LayerGradCam` on each model's last convolutional stage, upsampled to
  64×64 and scaled to [0, 1] per image (`src/gradcam.py`).

## Outputs

```text
outputs/b1/  table.md, results.json, histories.json, val_loss.png, val_accuracy.png, alexnet_best.pt
outputs/b2/  table.md, results.json, histories.json, val_loss.png, val_accuracy.png, vgg_best.pt
outputs/b3/  results.json, history.json, val_loss.png, val_accuracy.png, resnet_best.pt,
             comparison.md / comparison.json (B1–B3 accuracy, sensitivity, specificity, error,
             parameters, FLOPs)
outputs/b4/  gradcam.png (overlays), gradcam_stats.md / .json, samples.json (chosen test indices)
```

## Code map

```text
src/config.py   seed, batch size, epochs, early stopping, threshold
src/data.py     load the pre-split npz, fit normalisation on train, tensor datasets
src/models.py   conv block, AlexNet-like, VGG-like and ResNet-like networks, parameter and FLOP counts
src/train.py    Adam training loop, early stopping, accuracy/sensitivity/specificity
src/gradcam.py  Grad-CAM heatmaps and region statistics
src/plots.py    validation curves and Grad-CAM grid
src/main.py     question switch (b1-b4); sweeps, selection, and the single test evaluation
```
