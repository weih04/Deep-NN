import json
import statistics
from pathlib import Path

import torch

from baseline import calculate_r2, calculate_rmse
from config import Config
from data import prepare_final_data
from final import train_fixed_epochs
from model import PriceModel, count_parameters
from plots import plot_ablation_summary

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "hdb_price_prediction.csv"
SELECTION_PATH = REPO_ROOT / "outputs" / "tune" / "best_config.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "ablation"

SEEDS = [42, 43, 44, 45, 46]

VARIANTS = {
    "baseline": {},
    "no_embeddings": {"use_embeddings": False},
    "sigmoid": {"activation": "sigmoid"},
}


def build_variant_config(selection, variant, seed=42):
    return Config(
        hidden_width=selection["hidden_width"],
        embedding_dims=(selection["embedding_dim"],) * 4,
        seed=seed,
        **VARIANTS[variant],
    )


def summarise(records):
    """Mean and sample standard deviation per variant."""
    summary = {}
    for variant in dict.fromkeys(record["variant"] for record in records):
        rows = [record for record in records if record["variant"] == variant]
        entry = {"seeds": len(rows)}
        for metric in ("test_rmse", "test_r2"):
            values = [row[metric] for row in rows]
            entry[f"{metric}_mean"] = statistics.fmean(values)
            entry[f"{metric}_sd"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary[variant] = entry
    return summary


def run_one(data, selection, variant, seed):
    config = build_variant_config(selection, variant, seed)
    torch.manual_seed(seed)
    model = PriceModel(
        data["cardinalities"], data["target_mean"], data["target_std"], config
    )
    train_fixed_epochs(
        model, data["train"], data["test"], config, epochs=selection["best_epoch"]
    )
    return {
        "variant": variant,
        "seed": seed,
        "parameter_count": count_parameters(model),
        "test_rmse": calculate_rmse(model, data["test"], config.batch_size),
        "test_r2": calculate_r2(model, data["test"], config.batch_size),
    }


def main():
    selection = json.loads(SELECTION_PATH.read_text())
    data = prepare_final_data(str(DATA_PATH), Config())

    records = []
    for variant in VARIANTS:
        for seed in SEEDS:
            print(f"\n--- {variant} | seed {seed} ---")
            record = run_one(data, selection, variant, seed)
            print(
                f"{variant} seed {seed}: "
                f"test RMSE {record['test_rmse']:,.0f} | R2 {record['test_r2']:.4f}"
            )
            records.append(record)

    summary = summarise(records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "runs.json").write_text(json.dumps(records, indent=2))
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_ablation_summary(records, OUTPUT_DIR / "ablation.png")

    print(f"\n{'variant':<15}{'params':>8}{'RMSE mean':>12}{'RMSE sd':>10}"
          f"{'R2 mean':>10}{'R2 sd':>9}")
    for variant, entry in summary.items():
        params = next(r["parameter_count"] for r in records if r["variant"] == variant)
        print(
            f"{variant:<15}{params:>8,}{entry['test_rmse_mean']:>12,.0f}"
            f"{entry['test_rmse_sd']:>10,.0f}{entry['test_r2_mean']:>10.4f}"
            f"{entry['test_r2_sd']:>9.4f}"
        )
    print(f"\nwrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
