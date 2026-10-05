import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from baseline import calculate_r2, calculate_rmse
from config import Config
from data import prepare_final_data
from model import PriceModel, count_parameters
from plots import plot_rmse_curves

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "hdb_price_prediction.csv"
SELECTION_PATH = REPO_ROOT / "outputs" / "tune" / "best_config.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "final"


def train_fixed_epochs(model, train_data, eval_data, config, epochs,
                       eval_key="test_rmse"):
    """Train for a fixed number of epochs."""
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True)
    loss_function = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for categorical, continuous, target in train_loader:
            optimiser.zero_grad()
            standard_prediction = model(categorical, continuous)
            standard_target = (target - model.target_mean) / model.target_std
            loss = loss_function(standard_prediction, standard_target)
            loss.backward()
            optimiser.step()

        train_rmse = calculate_rmse(model, train_data, config.batch_size)
        eval_rmse = calculate_rmse(model, eval_data, config.batch_size)
        label = eval_key.replace("_rmse", "")
        print(
            f"epoch {epoch:02d} | train RMSE {train_rmse:,.0f} "
            f"| {label} RMSE {eval_rmse:,.0f}"
        )
        history.append(
            {"epoch": epoch, "train_rmse": train_rmse, eval_key: eval_rmse}
        )

    return history


def main():
    selection = json.loads(SELECTION_PATH.read_text())
    config = Config(
        hidden_width=selection["hidden_width"],
        embedding_dims=(selection["embedding_dim"],) * 4,
    )
    torch.manual_seed(config.seed)

    data = prepare_final_data(str(DATA_PATH), config)
    model = PriceModel(
        data["cardinalities"], data["target_mean"], data["target_std"], config
    )

    history = train_fixed_epochs(
        model, data["train"], data["test"], config, epochs=selection["best_epoch"]
    )

    metrics = {
        "hidden_width": selection["hidden_width"],
        "embedding_dim": selection["embedding_dim"],
        "epochs": selection["best_epoch"],
        "parameter_count": count_parameters(model),
        "train_rmse": calculate_rmse(model, data["train"], config.batch_size),
        "test_rmse": calculate_rmse(model, data["test"], config.batch_size),
        "test_r2": calculate_r2(model, data["test"], config.batch_size),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save(model.state_dict(), OUTPUT_DIR / "model.pt")
    plot_rmse_curves(
        history,
        {"train_rmse": "train", "test_rmse": "test"},
        OUTPUT_DIR / "curves.png",
        f"final model: hidden_width={selection['hidden_width']}, "
        f"embedding_dim={selection['embedding_dim']}",
    )

    print(f"\ntest RMSE: {metrics['test_rmse']:,.0f}")
    print(f"test R2:   {metrics['test_r2']:.4f}")
    print(f"wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
