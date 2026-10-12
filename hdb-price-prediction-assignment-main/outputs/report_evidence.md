# Report evidence

## Part 1 - baseline parameter count

Input width: 4*8 + 6 = 38

| block | parameters |
|---|---|
| embeddings | 816 |
| linear_1 | 2,496 |
| layer_norm | 128 |
| linear_2 | 65 |
| **total** | **3,505** |

General form: `102*d + (4d+6)*h + 4h + 1`

## Parts 2-3 - grid search (best validation RMSE, SGD)

| hidden_width \ embedding_dim | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| 16 | 66,426 | 68,880 | 67,520 | 62,655 | 67,165 |
| 32 | 69,204 | 62,886 | 66,200 | 62,480 | 64,593 |
| 64 | 63,492 | 64,634 | 63,422 | 64,105 | 63,033 |
| 128 | 66,329 | 61,741 | 66,832 | 60,332 | 65,600 |
| 256 | 60,306 | 60,578 | 63,324 | 65,746 | 62,915 |

Selected: hidden_width=256, embedding_dim=2, best epoch=10, validation RMSE=60,306, parameters=4,813

## Part 5 - final model on the 2022 test year

- train RMSE (2017-2021): 40,128
- test RMSE (2022): 107,819
- test R2: 0.5990

### Error decomposition on the test year

- systematic bias (predicted - actual): -98,076
- spread around that bias: 44,789
- these combine to the reported RMSE: sqrt(bias^2 + spread^2) = 107,819

## Part 6 - mean resale price by year

| year | mean price | rows |
|---|---|---|
| 2017 | 443,743 | 20,337 |
| 2018 | 441,310 | 21,552 |
| 2019 | 432,180 | 22,168 |
| 2020 | 452,313 | 23,313 |
| 2021 | 511,417 | 29,057 |
| 2022 | 549,701 | 26,702 |
| 2023 | 568,215 | 16,424 |

Fitting set (2017-2021) target mean: 459,697

## Part 6 - diagnostics to cite

- The 25 trials span 60,306 to 69,204 validation RMSE (a 14.8% range, sd 2,516) with no monotonic trend in either axis, so the search surface is close to flat relative to run-to-run noise. Treat the winner as one of several statistically indistinguishable configurations, not a decisive optimum.
- The selected hidden_width (256) sits on the edge of the searched range, so the optimum may lie outside the grid.
- Selected epoch 10 against max_epochs 20.
- Capacity: the baseline has 3,505 parameters for 87,370 training rows (24 rows per parameter), so memorisation is an unlikely explanation for the train/test gap.
- The test error is dominated by bias, not spread: the spread is close to the training RMSE, meaning the model ranks flats well but prices them at the wrong level. That is temporal distribution shift, not overfitting.
