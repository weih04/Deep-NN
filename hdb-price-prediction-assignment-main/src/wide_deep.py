import json
from pathlib import Path

import torch

from ablation import summarise
from baseline import calculate_r2, calculate_rmse
from config import Config
from data import prepare_data, prepare_final_data
from final import train_fixed_epochs
from model import PriceModel, count_parameters
from plots import plot_ablation_summary, plot_lambda_search

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "hdb_price_prediction.csv"
SELECTION_PATH = REPO_ROOT / "outputs" / "tune" / "best_config.json"
ABLATION_RUNS = REPO_ROOT / "outputs" / "ablation" / "runs.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "wide_deep"

LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0]
SEEDS = [42, 43, 44, 45, 46]


def build_config(selection, wide_lambda, seed=42):
    return Config(
        hidden_width=selection["hidden_width"],
        embedding_dims=(selection["embedding_dim"],) * 4,
        wide_lambda=wide_lambda,
        seed=seed,
    )


def search_lambda(selection):
    data = prepare_data(str(DATA_PATH), Config())
    epochs = selection["best_epoch"]
    records = []
    for wide_lambda in LAMBDAS:
        config = build_config(selection, wide_lambda)
        torch.manual_seed(config.seed)
        model = PriceModel(
            data["cardinalities"], data["target_mean"], data["target_std"], config
        )
        history = train_fixed_epochs(
            model,
            data["train"],
            data["validation"],
            config,
            epochs,
            eval_key="validation_rmse",
        )
        records.append(
            {
                "lambda": wide_lambda,
                "epochs": epochs,
                "validation_rmse": history[-1]["validation_rmse"],
                "parameter_count": count_parameters(model),
            }
        )
        print(
            f"lambda {wide_lambda:<5} validation RMSE "
            f"{records[-1]['validation_rmse']:,.0f} after {epochs} epochs"
        )
    return records


def evaluate_seeds(selection, wide_lambda, label, data=None):
    if data is None:
        data = prepare_final_data(str(DATA_PATH), Config())
    records = []
    for seed in SEEDS:
        config = build_config(selection, wide_lambda, seed)
        torch.manual_seed(seed)
        model = PriceModel(
            data["cardinalities"], data["target_mean"], data["target_std"], config
        )
        train_fixed_epochs(
            model, data["train"], data["test"], config, epochs=selection["best_epoch"]
        )
        records.append(
            {
                "variant": label,
                "lambda": wide_lambda,
                "seed": seed,
                "parameter_count": count_parameters(model),
                "test_rmse": calculate_rmse(model, data["test"], config.batch_size),
                "test_r2": calculate_r2(model, data["test"], config.batch_size),
            }
        )
        print(
            f"{label} seed {seed}: test RMSE {records[-1]['test_rmse']:,.0f} "
            f"| R2 {records[-1]['test_r2']:.4f}"
        )
    return records


def main():
    selection = json.loads(SELECTION_PATH.read_text())

    print("--- lambda search (train 2017-2020, select on 2021) ---")
    search = search_lambda(selection)
    best = min(search, key=lambda record: record["validation_rmse"])
    print(f"\nselected lambda: {best['lambda']}")

    data = prepare_final_data(str(DATA_PATH), Config())

    print(f"\n--- five seeds at the selected lambda={best['lambda']} ---")
    wide_records = evaluate_seeds(
        selection, best["lambda"], f"wide_deep (lambda={best['lambda']})", data
    )

    # If the search selects 0 the wide branch is inert, so the comparison would
    # be baseline against baseline. Also evaluate the best non-zero lambda so a
    # genuine wide-and-deep model is measured. Supplementary, not a reselection.
    supplementary = []
    if best["lambda"] == 0.0:
        non_zero = min(
            (record for record in search if record["lambda"] > 0),
            key=lambda record: record["validation_rmse"],
        )
        print(f"\n--- five seeds at the best non-zero lambda={non_zero['lambda']} ---")
        supplementary = evaluate_seeds(
            selection, non_zero["lambda"], f"wide_deep (lambda={non_zero['lambda']})",
            data,
        )

    # A1 baseline reference: the five-seed baseline runs already produced in A2.
    baseline_records = [
        record
        for record in json.loads(ABLATION_RUNS.read_text())
        if record["variant"] == "baseline"
    ]
    combined = baseline_records + wide_records + supplementary
    summary = summarise(combined)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "lambda_search.json").write_text(json.dumps(search, indent=2))
    (OUTPUT_DIR / "runs.json").write_text(
        json.dumps(wide_records + supplementary, indent=2)
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps({"selected_lambda": best["lambda"], **summary}, indent=2)
    )
    plot_lambda_search(search, OUTPUT_DIR / "lambda_search.png")
    plot_ablation_summary(
        combined,
        OUTPUT_DIR / "comparison.png",
        title="A3 wide-and-deep vs A1 baseline: five seeds (seeds paired)",
    )

    print(f"\n{'variant':<28}{'params':>8}{'RMSE mean':>12}{'RMSE sd':>10}"
          f"{'R2 mean':>10}{'R2 sd':>9}")
    for variant, entry in summary.items():
        params = next(
            r["parameter_count"] for r in combined if r["variant"] == variant
        )
        print(
            f"{variant:<28}{params:>8,}{entry['test_rmse_mean']:>12,.0f}"
            f"{entry['test_rmse_sd']:>10,.0f}{entry['test_r2_mean']:>10.4f}"
            f"{entry['test_r2_sd']:>9.4f}"
        )
    print(f"\nwrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
