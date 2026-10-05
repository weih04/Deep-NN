import pandas as pd
import torch
from torch.utils.data import TensorDataset

from config import Config


def make_dataset(rows, category_maps, means, stds, config):
    categorical = torch.tensor(
        [
            [
                category_maps[name].get(row[name], 0)
                for name in config.categorical_features
            ]
            for row in rows
        ],
        dtype=torch.long,
    )
    continuous = torch.tensor(
        [[float(row[name]) for name in config.continuous_features] for row in rows],
        dtype=torch.float32,
    )
    continuous = (continuous - means) / stds

    target = torch.tensor([float(row["resale_price"]) for row in rows])
    return TensorDataset(categorical, continuous, target)


def load_rows(csv_path):
    return pd.read_csv(csv_path).to_dict(orient="records")


def fit_preprocessing(fit_rows, config: Config):
    """Fit every preprocessing statistic on fit_rows alone.

    The sweep fits on the training years; the final run fits on training +
    validation. Keeping one implementation is what stops the two paths from
    silently drifting apart.
    """
    # ID 0 means "unseen category", so real IDs start at 1.
    category_maps = {}
    for name in config.categorical_features:
        values = sorted({row[name] for row in fit_rows})
        category_maps[name] = {value: index + 1 for index, value in enumerate(values)}

    fit_continuous = torch.tensor(
        [[float(row[name]) for name in config.continuous_features] for row in fit_rows]
    )
    means = fit_continuous.mean(dim=0)
    stds = fit_continuous.std(dim=0).clamp_min(1e-8)

    fit_target = torch.tensor([float(row["resale_price"]) for row in fit_rows])
    return {
        "category_maps": category_maps,
        "means": means,
        "stds": stds,
        "target_mean": fit_target.mean().item(),
        "target_std": fit_target.std(correction=0).clamp_min(1e-8).item(),
        "cardinalities": [
            len(category_maps[name]) + 1 for name in config.categorical_features
        ],
    }


def _build(rows, stats, config):
    return make_dataset(
        rows, stats["category_maps"], stats["means"], stats["stds"], config
    )


def prepare_data(csv_path, config: Config):
    rows = load_rows(csv_path)

    train_rows = [row for row in rows if int(row["year"]) in config.train_years]
    validation_rows = [
        row for row in rows if int(row["year"]) == config.validation_year
    ]
    test_rows = [row for row in rows if int(row["year"]) == config.test_year]

    stats = fit_preprocessing(train_rows, config)
    return {
        "train": _build(train_rows, stats, config),
        "validation": _build(validation_rows, stats, config),
        "test": _build(test_rows, stats, config),
        "cardinalities": stats["cardinalities"],
        "continuous_mean": stats["means"],
        "continuous_std": stats["stds"],
        "target_mean": stats["target_mean"],
        "target_std": stats["target_std"],
    }


def prepare_final_data(csv_path, config: Config):
    """Train on train + validation years, hold out the test year.

    Statistics are refitted on the combined rows, so this deliberately does
    not reuse the scaling produced by prepare_data.
    """
    rows = load_rows(csv_path)

    fit_years = set(config.train_years) | {config.validation_year}
    train_rows = [row for row in rows if int(row["year"]) in fit_years]
    test_rows = [row for row in rows if int(row["year"]) == config.test_year]

    stats = fit_preprocessing(train_rows, config)
    return {
        "train": _build(train_rows, stats, config),
        "test": _build(test_rows, stats, config),
        "cardinalities": stats["cardinalities"],
        "target_mean": stats["target_mean"],
        "target_std": stats["target_std"],
    }
