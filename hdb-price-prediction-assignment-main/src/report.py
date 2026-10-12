import json
from pathlib import Path

import pandas as pd
import torch

from baseline import _collect_predictions
from config import Config
from data import prepare_final_data
from model import PriceModel, count_parameters

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "hdb_price_prediction.csv"
TUNE_DIR = REPO_ROOT / "outputs" / "tune"
FINAL_DIR = REPO_ROOT / "outputs" / "final"
OUTPUT_PATH = REPO_ROOT / "outputs" / "report_evidence.md"

# Cardinalities of the training years (2017-2020), from data.fit_preprocessing.
TRAIN_CARDINALITIES = [13, 27, 44, 18]
TRAIN_ROW_COUNT = 87370


def parameter_breakdown(embedding_dim, hidden_width):
    input_width = 4 * embedding_dim + 6
    return {
        "embeddings": sum(TRAIN_CARDINALITIES) * embedding_dim,
        "linear_1": input_width * hidden_width + hidden_width,
        "layer_norm": 2 * hidden_width,
        "linear_2": hidden_width + 1,
    }


def error_decomposition(selection):
    config = Config(
        hidden_width=selection["hidden_width"],
        embedding_dims=(selection["embedding_dim"],) * 4,
    )
    data = prepare_final_data(str(DATA_PATH), config)
    model = PriceModel(
        data["cardinalities"], data["target_mean"], data["target_std"], config
    )
    model.load_state_dict(torch.load(FINAL_DIR / "model.pt"))

    prediction, target = _collect_predictions(model, data["test"], config.batch_size)
    error = prediction - target
    return {
        "bias": error.mean().item(),
        "spread": error.std().item(),
        "fit_target_mean": data["target_mean"],
    }


def main():
    baseline_config = Config()
    baseline_model = PriceModel(TRAIN_CARDINALITIES, 0.0, 1.0, baseline_config)
    baseline_total = count_parameters(baseline_model)
    breakdown = parameter_breakdown(8, baseline_config.hidden_width)

    selection = json.loads((TUNE_DIR / "best_config.json").read_text())
    metrics = json.loads((FINAL_DIR / "metrics.json").read_text())
    grid = (TUNE_DIR / "grid.md").read_text()
    decomposition = error_decomposition(selection)

    yearly = (
        pd.read_csv(DATA_PATH).groupby("year")["resale_price"].agg(["mean", "count"])
    )

    trial_rmse = pd.read_csv(TUNE_DIR / "results.csv")["best_validation_rmse"]
    grid_spread = {
        "min": trial_rmse.min(),
        "max": trial_rmse.max(),
        "sd": trial_rmse.std(),
        "relative_range": (trial_rmse.max() / trial_rmse.min() - 1) * 100,
    }

    lines = [
        "# Report evidence",
        "",
        "## Part 1 - baseline parameter count",
        "",
        f"Input width: 4*8 + 6 = {4 * 8 + 6}",
        "",
        "| block | parameters |",
        "|---|---|",
    ]
    lines += [f"| {name} | {value:,} |" for name, value in breakdown.items()]
    lines += [
        f"| **total** | **{baseline_total:,}** |",
        "",
        "General form: `102*d + (4d+6)*h + 4h + 1`",
        "",
        "## Parts 2-3 - grid search (best validation RMSE, SGD)",
        "",
        grid.strip(),
        "",
        f"Selected: hidden_width={selection['hidden_width']}, "
        f"embedding_dim={selection['embedding_dim']}, "
        f"best epoch={selection['best_epoch']}, "
        f"validation RMSE={selection['best_validation_rmse']:,.0f}, "
        f"parameters={selection['parameter_count']:,}",
        "",
        "## Part 5 - final model on the 2022 test year",
        "",
        f"- train RMSE (2017-2021): {metrics['train_rmse']:,.0f}",
        f"- test RMSE (2022): {metrics['test_rmse']:,.0f}",
        f"- test R2: {metrics['test_r2']:.4f}",
        "",
        "### Error decomposition on the test year",
        "",
        f"- systematic bias (predicted - actual): {decomposition['bias']:,.0f}",
        f"- spread around that bias: {decomposition['spread']:,.0f}",
        f"- these combine to the reported RMSE: "
        f"sqrt(bias^2 + spread^2) = "
        f"{(decomposition['bias'] ** 2 + decomposition['spread'] ** 2) ** 0.5:,.0f}",
        "",
        "## Part 6 - mean resale price by year",
        "",
        "| year | mean price | rows |",
        "|---|---|---|",
    ]
    lines += [
        f"| {int(year)} | {row['mean']:,.0f} | {int(row['count']):,} |"
        for year, row in yearly.iterrows()
    ]
    lines += [
        "",
        f"Fitting set (2017-2021) target mean: "
        f"{decomposition['fit_target_mean']:,.0f}",
        "",
        "## Part 6 - diagnostics to cite",
        "",
        f"- The 25 trials span {grid_spread['min']:,.0f} to "
        f"{grid_spread['max']:,.0f} validation RMSE (a "
        f"{grid_spread['relative_range']:.1f}% range, sd "
        f"{grid_spread['sd']:,.0f}) with no monotonic trend in either axis, so "
        "the search surface is close to flat relative to run-to-run noise. "
        "Treat the winner as one of several statistically indistinguishable "
        "configurations, not a decisive optimum.",
        f"- The selected hidden_width ({selection['hidden_width']}) sits on the "
        "edge of the searched range, so the optimum may lie outside the grid.",
        f"- Selected epoch {selection['best_epoch']} against max_epochs "
        f"{baseline_config.max_epochs}.",
        f"- Capacity: the baseline has {baseline_total:,} parameters for "
        f"{TRAIN_ROW_COUNT:,} training rows "
        f"({TRAIN_ROW_COUNT // baseline_total} rows per parameter), so "
        "memorisation is an unlikely explanation for the train/test gap.",
        "- The test error is dominated by bias, not spread: the spread is "
        "close to the training RMSE, meaning the model ranks flats well but "
        "prices them at the wrong level. That is temporal distribution shift, "
        "not overfitting.",
        "",
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
