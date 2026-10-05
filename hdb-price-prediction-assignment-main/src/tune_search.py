import json
from pathlib import Path

import ray
import torch
from ray import tune
from ray.tune import TuneConfig, Tuner

from baseline import train_model
from config import Config
from data import prepare_data
from model import PriceModel, count_parameters
from plots import plot_rmse_curves

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
# Ray gives each trial its own working directory, so relative paths break.
DATA_PATH = REPO_ROOT / "hdb_price_prediction.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "tune"

HIDDEN_WIDTHS = [16, 32, 64, 128, 256]
EMBEDDING_DIMS = [2, 4, 8, 12, 16]


def build_config(search_config):
    return Config(
        hidden_width=search_config["hidden_width"],
        embedding_dims=(search_config["embedding_dim"],) * 4,
    )


def trainable(search_config, data):
    config = build_config(search_config)
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)

    model = PriceModel(
        data["cardinalities"], data["target_mean"], data["target_std"], config
    )
    best_epoch, history = train_model(model, data["train"], data["validation"], config)
    best_record = history[best_epoch - 1]

    tune.report(
        {
            "hidden_width": search_config["hidden_width"],
            "embedding_dim": search_config["embedding_dim"],
            "best_epoch": best_epoch,
            "best_validation_rmse": best_record["validation_rmse"],
            "final_train_rmse": history[-1]["train_rmse"],
            "parameter_count": count_parameters(model),
            "history": history,
        }
    )


def format_grid_table(records):
    """5x5 markdown table of best validation RMSE."""
    by_key = {
        (record["hidden_width"], record["embedding_dim"]): record[
            "best_validation_rmse"
        ]
        for record in records
    }
    widths = sorted({record["hidden_width"] for record in records})
    dims = sorted({record["embedding_dim"] for record in records})

    lines = [
        "| hidden_width \\ embedding_dim | " + " | ".join(str(d) for d in dims) + " |"
    ]
    lines.append("|---" * (len(dims) + 1) + "|")
    for width in widths:
        cells = [
            f"{by_key[(width, dim)]:,.0f}" if (width, dim) in by_key else "-"
            for dim in dims
        ]
        lines.append(f"| {width} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    config = Config()

    data = prepare_data(str(DATA_PATH), config)

    ray.init(runtime_env={"env_vars": {"PYTHONPATH": str(SRC_DIR)}})

    tuner = Tuner(
        tune.with_resources(tune.with_parameters(trainable, data=data), {"cpu": 1}),
        param_space={
            "hidden_width": tune.grid_search(HIDDEN_WIDTHS),
            "embedding_dim": tune.grid_search(EMBEDDING_DIMS),
        },
        tune_config=TuneConfig(metric="best_validation_rmse", mode="min"),
    )
    results = tuner.fit()

    records = [result.metrics for result in results if result.metrics]
    records.sort(key=lambda record: record["best_validation_rmse"])
    best = results.get_best_result()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selection = {
        "hidden_width": best.metrics["hidden_width"],
        "embedding_dim": best.metrics["embedding_dim"],
        "best_epoch": best.metrics["best_epoch"],
        "best_validation_rmse": best.metrics["best_validation_rmse"],
        "parameter_count": best.metrics["parameter_count"],
    }
    (OUTPUT_DIR / "best_config.json").write_text(json.dumps(selection, indent=2))

    results.get_dataframe().to_csv(OUTPUT_DIR / "results.csv", index=False)
    (OUTPUT_DIR / "grid.md").write_text(format_grid_table(records) + "\n")

    plot_rmse_curves(
        best.metrics["history"],
        {"train_rmse": "train", "validation_rmse": "validation"},
        OUTPUT_DIR / "best_curves.png",
        f"hidden_width={selection['hidden_width']}, "
        f"embedding_dim={selection['embedding_dim']}",
        marker_epoch=selection["best_epoch"],
    )

    print("\n" + format_grid_table(records))
    print(f"\nselected: {selection}")
    print(f"wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
