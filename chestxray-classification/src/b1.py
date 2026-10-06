"""B1: AlexNet-like CNN, learning-rate sweep at batch size 128.

Each learning rate trains from the same seed and initialisation. The rate with
the lowest validation classification error (ties: lower validation loss) is
selected, and only that model is evaluated on the test set, once.
"""
import json
from dataclasses import replace
from pathlib import Path

import torch

from config import Config
from data import prepare_data
from models import AlexNetLike, count_parameters
from plots import plot_validation_curves
from train import evaluate, train_model

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "b1"

LEARNING_RATES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def summarise_run(learning_rate, best_epoch, history):
    """One table row: validation metrics at the restored (best) epoch."""
    best = history[best_epoch - 1]
    return {
        "learning_rate": learning_rate,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "val_loss": best["val_loss"],
        "val_error": 1.0 - best["val_accuracy"],
        "val_sensitivity": best["val_sensitivity"],
        "val_specificity": best["val_specificity"],
    }


def select_best(rows):
    return min(rows, key=lambda row: (row["val_error"], row["val_loss"]))


def format_table(rows):
    lines = [
        "| learning rate | best epoch | val loss | val error | sensitivity | specificity |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['learning_rate']:g} | {row['best_epoch']} "
            f"| {row['val_loss']:.4f} | {row['val_error']:.4f} "
            f"| {row['val_sensitivity']:.4f} | {row['val_specificity']:.4f} |"
        )
    return "\n".join(lines)


def main():
    base_config = Config()
    torch.use_deterministic_algorithms(True)
    data = prepare_data(str(REPO_ROOT / base_config.data_path))

    rows = []
    histories = {}
    models = {}
    for learning_rate in LEARNING_RATES:
        print(f"\n=== learning rate {learning_rate:g}")
        config = replace(base_config, learning_rate=learning_rate)
        torch.manual_seed(config.seed)
        model = AlexNetLike(config.dropout)
        best_epoch, history = train_model(model, data["train"], data["val"], config)

        rows.append(summarise_run(learning_rate, best_epoch, history))
        histories[learning_rate] = history
        models[learning_rate] = model

    selected = select_best(rows)
    final_model = models[selected["learning_rate"]]
    test = evaluate(
        final_model, data["test"], base_config.batch_size, base_config.decision_threshold
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "table.md").write_text(format_table(rows) + "\n")
    (OUTPUT_DIR / "histories.json").write_text(
        json.dumps({f"{lr:g}": h for lr, h in histories.items()}, indent=2)
    )
    results = {
        "selected_learning_rate": selected["learning_rate"],
        "selected_best_epoch": selected["best_epoch"],
        "parameter_count": count_parameters(final_model),
        "normalisation": {"mean": data["mean"], "std": data["std"]},
        "validation_table": rows,
        "test": {
            "loss": test["loss"],
            "accuracy": test["accuracy"],
            "error": 1.0 - test["accuracy"],
            "sensitivity": test["sensitivity"],
            "specificity": test["specificity"],
        },
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    # Kept for B3's comparison table and B4's Grad-CAM.
    torch.save(final_model.state_dict(), OUTPUT_DIR / "alexnet_best.pt")

    plot_validation_curves(
        histories, "val_loss", "validation loss (BCE)",
        OUTPUT_DIR / "val_loss.png", "B1: validation loss per learning rate",
        label_format="lr={:g}",
    )
    plot_validation_curves(
        histories, "val_accuracy", "validation accuracy",
        OUTPUT_DIR / "val_accuracy.png", "B1: validation accuracy per learning rate",
        label_format="lr={:g}",
    )

    print("\n" + format_table(rows))
    print(f"\nselected learning rate: {selected['learning_rate']:g}")
    print(
        f"test accuracy {test['accuracy']:.4f} | sensitivity {test['sensitivity']:.4f} "
        f"| specificity {test['specificity']:.4f}"
    )
    print(f"wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
