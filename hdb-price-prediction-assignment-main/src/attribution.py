import json
from pathlib import Path

import torch
from captum.attr import FeatureAblation

from config import Config
from data import prepare_final_data
from final import train_fixed_epochs
from model import PriceModel
from plots import plot_feature_importance

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "hdb_price_prediction.csv"
SELECTION_PATH = REPO_ROOT / "outputs" / "tune" / "best_config.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "attribution"

SEED = 42
# A3 selected lambda=0, which leaves the wide branch inert and identical to the
# baseline. The best non-zero lambda is used so a genuine wide-and-deep model is
# explained; this is stated in the report rather than treated as a reselection.
WIDE_LAMBDA = 0.5


def feature_names(config):
    """The ten original input features, categorical first to match the tensors."""
    return list(config.categorical_features) + list(config.continuous_features)


def build_masks(config):
    """Group each original feature into one ablation unit.

    An embedding occupies several input dimensions, so ablating one column of
    the categorical tensor ablates that whole embedding. Mask ids are unique
    across both tensors, as Captum requires.
    """
    categorical_count = len(config.categorical_features)
    continuous_count = len(config.continuous_features)
    categorical_mask = torch.arange(categorical_count).unsqueeze(0)
    continuous_mask = (
        torch.arange(continuous_count) + categorical_count
    ).unsqueeze(0)
    return categorical_mask, continuous_mask


def train(config, data, seed=SEED, epochs=10):
    torch.manual_seed(seed)
    model = PriceModel(
        data["cardinalities"], data["target_mean"], data["target_std"], config
    )
    train_fixed_epochs(model, data["train"], data["test"], config, epochs)
    model.eval()
    return model


def modal_categories(train_dataset):
    """Most frequent training id per categorical column.

    Id 0 is reserved for unseen categories and therefore never occurs in
    training, so its embedding row stays at initialisation. Ablating to it would
    substitute an untrained random vector and measure distance to a random point
    rather than feature importance. The modal category is a trained, typical
    value, so the attribution reads as "if this flat had the most common value".
    """
    categorical = train_dataset.tensors[0]
    modes = [
        torch.bincount(categorical[:, index]).argmax().item()
        for index in range(categorical.shape[1])
    ]
    return torch.tensor(modes).unsqueeze(0)


def attribute(model, dataset, config, categorical_baseline):
    """Mean absolute FeatureAblation attribution per original feature, in SGD."""
    categorical, continuous, _ = dataset.tensors

    def forward(categorical_input, continuous_input):
        return model.predict_price(categorical_input, continuous_input)

    ablator = FeatureAblation(forward)
    attributions = ablator.attribute(
        inputs=(categorical, continuous),
        baselines=(categorical_baseline, 0.0),
        feature_mask=build_masks(config),
    )

    scores = []
    for tensor in attributions:
        scores.extend(tensor.abs().mean(dim=0).tolist())
    return dict(zip(feature_names(config), scores))


def main():
    selection = json.loads(SELECTION_PATH.read_text())
    base_kwargs = {
        "hidden_width": selection["hidden_width"],
        "embedding_dims": (selection["embedding_dim"],) * 4,
        "seed": SEED,
    }
    data = prepare_final_data(str(DATA_PATH), Config())
    categorical_baseline = modal_categories(data["train"])

    variants = {
        "A1 baseline": Config(**base_kwargs),
        f"A3 wide-and-deep (lambda={WIDE_LAMBDA})": Config(
            **base_kwargs, wide_lambda=WIDE_LAMBDA
        ),
    }

    results = {}
    for label, config in variants.items():
        print(f"\n--- training {label} (seed {SEED}) ---")
        model = train(config, data, epochs=selection["best_epoch"])
        scores = attribute(model, data["test"], config, categorical_baseline)
        results[label] = scores

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        print(f"\n{label}: mean |attribution| (SGD)")
        for name, value in ranked:
            print(f"  {name:<24}{value:>12,.0f}")
        print(f"  top three: {', '.join(name for name, _ in ranked[:3])}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "importance.json").write_text(json.dumps(results, indent=2))
    plot_feature_importance(results, OUTPUT_DIR / "feature_importance.png")
    print(f"\nwrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
