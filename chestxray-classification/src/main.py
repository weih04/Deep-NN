"""Part B entry point: run one question with `python src/main.py {b1,b2,b3,b4}`.

B1: AlexNet-like CNN, learning-rate sweep at batch size 128.
B2: VGG-like CNN, batch-size sweep at the learning rate selected in B1.
B3: ResNet-like CNN at B1's learning rate and B2's batch size, plus a
    comparison of the three architectures.
B4: Grad-CAM on five random pneumonia test X-rays for the B1-B3 models.

Each swept value trains from the same seed and initialisation. The value with
the lowest validation classification error (ties: lower validation loss) is
selected, and only that model is evaluated on the test set, once.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from config import Config
from data import prepare_data
from models import AlexNetLike, ResNetLike, VGGLike, count_flops, count_parameters
from gradcam import (
    chest_regions,
    grad_cam,
    hot_region_brightness,
    mean_correlation,
    pneumonia_probabilities,
    region_shares,
)
from plots import plot_gradcam_grid, plot_validation_curves
from train import evaluate, train_model

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"

LEARNING_RATES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
BATCH_SIZES = [16, 32, 64, 128, 256]
GRADCAM_SAMPLES = 5

GRADCAM_MODELS = [
    ("AlexNet-like", "b1", AlexNetLike, "alexnet_best.pt", "features.4"),
    ("VGG-like", "b2", VGGLike, "vgg_best.pt", "features.7"),
    ("ResNet-like", "b3", ResNetLike, "resnet_best.pt", "features.4"),
]

SWEEP_LABELS = {
    "learning_rate": ("learning rate", "lr={:g}", "{:g}"),
    "batch_size": ("batch size", "bs={}", "{}"),
}


def summarise_run(sweep_key, value, best_epoch, history):
    best = history[best_epoch - 1]
    return {
        sweep_key: value,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "val_loss": best["val_loss"],
        "val_error": 1.0 - best["val_accuracy"],
        "val_sensitivity": best["val_sensitivity"],
        "val_specificity": best["val_specificity"],
    }


def summarise_test(test):
    return {
        "loss": test["loss"],
        "accuracy": test["accuracy"],
        "error": 1.0 - test["accuracy"],
        "sensitivity": test["sensitivity"],
        "specificity": test["specificity"],
    }


def select_best(rows):
    return min(rows, key=lambda row: (row["val_error"], row["val_loss"]))


def format_table(rows, sweep_key):
    header, _, value_format = SWEEP_LABELS[sweep_key]
    lines = [
        f"| {header} | best epoch | val loss | val error | sensitivity | specificity |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {value_format.format(row[sweep_key])} | {row['best_epoch']} "
            f"| {row['val_loss']:.4f} | {row['val_error']:.4f} "
            f"| {row['val_sensitivity']:.4f} | {row['val_specificity']:.4f} |"
        )
    return "\n".join(lines)


def run_sweep(question, build_model, sweep_key, values, base_config, checkpoint_name):
    """Train one model per swept value, select on validation, test the winner once."""
    output_dir = OUTPUTS_DIR / question
    header, label_format, value_format = SWEEP_LABELS[sweep_key]
    data = prepare_data(str(REPO_ROOT / base_config.data_path))

    rows = []
    histories = {}
    models = {}
    configs = {}
    for value in values:
        print(f"\n=== {header} {value_format.format(value)}")
        config = replace(base_config, **{sweep_key: value})
        torch.manual_seed(config.seed)
        model = build_model(config)
        best_epoch, history = train_model(model, data["train"], data["val"], config)

        rows.append(summarise_run(sweep_key, value, best_epoch, history))
        histories[value] = history
        models[value] = model
        configs[value] = config

    selected = select_best(rows)
    final_model = models[selected[sweep_key]]
    final_config = configs[selected[sweep_key]]
    test = evaluate(
        final_model, data["test"], final_config.batch_size, final_config.decision_threshold
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "table.md").write_text(format_table(rows, sweep_key) + "\n")
    (output_dir / "histories.json").write_text(
        json.dumps({value_format.format(v): h for v, h in histories.items()}, indent=2)
    )
    results = {
        f"selected_{sweep_key}": selected[sweep_key],
        "selected_best_epoch": selected["best_epoch"],
        "learning_rate": final_config.learning_rate,
        "batch_size": final_config.batch_size,
        "parameter_count": count_parameters(final_model),
        "normalisation": {"mean": data["mean"], "std": data["std"]},
        "validation_table": rows,
        "test": summarise_test(test),
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    # Kept for B3's comparison table and B4's Grad-CAM.
    torch.save(final_model.state_dict(), output_dir / checkpoint_name)

    title = question.upper()
    plot_validation_curves(
        histories, "val_loss", "validation loss (BCE)",
        output_dir / "val_loss.png", f"{title}: validation loss per {header}",
        label_format=label_format,
    )
    plot_validation_curves(
        histories, "val_accuracy", "validation accuracy",
        output_dir / "val_accuracy.png", f"{title}: validation accuracy per {header}",
        label_format=label_format,
    )

    print("\n" + format_table(rows, sweep_key))
    print(f"\nselected {header}: {value_format.format(selected[sweep_key])}")
    print(
        f"test accuracy {test['accuracy']:.4f} | sensitivity {test['sensitivity']:.4f} "
        f"| specificity {test['specificity']:.4f}"
    )
    print(f"wrote {output_dir}")


def load_selected(question, key):
    """Read a hyperparameter selected by an earlier question; that question must run first."""
    path = OUTPUTS_DIR / question / "results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python src/main.py {question}` first"
        )
    return json.loads(path.read_text())[key]

def load_trained(question, model_class, checkpoint):
    path = OUTPUTS_DIR / question / checkpoint
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python src/main.py {question}` first"
        )
    model = model_class()
    model.load_state_dict(torch.load(path, weights_only=True))
    return model.eval()


def run_b1():
    run_sweep(
        "b1", lambda config: AlexNetLike(config.dropout), "learning_rate",
        LEARNING_RATES, Config(), "alexnet_best.pt",
    )


def run_b2():
    learning_rate = load_selected("b1", "selected_learning_rate")
    print(f"using B1's selected learning rate {learning_rate:g}")
    run_sweep(
        "b2", lambda config: VGGLike(), "batch_size",
        BATCH_SIZES, Config(learning_rate=learning_rate), "vgg_best.pt",
    )


def format_comparison(rows):
    lines = [
        "| architecture | accuracy | sensitivity | specificity | error "
        "| parameters | FLOPs |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['architecture']} | {row['accuracy']:.4f} "
            f"| {row['sensitivity']:.4f} | {row['specificity']:.4f} "
            f"| {row['error']:.4f} | {row['parameters']:,} | {row['flops']:,} |"
        )
    return "\n".join(lines)


def run_b3():
    learning_rate = load_selected("b1", "selected_learning_rate")
    batch_size = load_selected("b2", "selected_batch_size")
    print(f"using B1's learning rate {learning_rate:g} and B2's batch size {batch_size}")
    config = Config(learning_rate=learning_rate, batch_size=batch_size)
    output_dir = OUTPUTS_DIR / "b3"
    data = prepare_data(str(REPO_ROOT / config.data_path))

    torch.manual_seed(config.seed)
    model = ResNetLike()
    best_epoch, history = train_model(model, data["train"], data["val"], config)
    test = evaluate(model, data["test"], config.batch_size, config.decision_threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    results = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "parameter_count": count_parameters(model),
        "flops": count_flops(model),
        "normalisation": {"mean": data["mean"], "std": data["std"]},
        "validation": summarise_run("batch_size", batch_size, best_epoch, history),
        "test": summarise_test(test),
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
  
    torch.save(model.state_dict(), output_dir / "resnet_best.pt")

    histories = {"ResNet-like": history}
    plot_validation_curves(
        histories, "val_loss", "validation loss (BCE)",
        output_dir / "val_loss.png", "B3: validation loss",
    )
    plot_validation_curves(
        histories, "val_accuracy", "validation accuracy",
        output_dir / "val_accuracy.png", "B3: validation accuracy",
    )

    architectures = [
        ("AlexNet-like (B1)", "b1", AlexNetLike()),
        ("VGG-like (B2)", "b2", VGGLike()),
        ("ResNet-like (B3)", "b3", ResNetLike()),
    ]
    comparison = []
    for name, question, architecture in architectures:
        test_metrics = load_selected(question, "test")
        comparison.append({
            "architecture": name,
            "accuracy": test_metrics["accuracy"],
            "sensitivity": test_metrics["sensitivity"],
            "specificity": test_metrics["specificity"],
            "error": test_metrics["error"],
            "parameters": count_parameters(architecture),
            "flops": count_flops(architecture),
        })
    (output_dir / "comparison.md").write_text(format_comparison(comparison) + "\n")
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))

    print(f"\nbest epoch {best_epoch} of {len(history)}")
    print(
        f"test accuracy {test['accuracy']:.4f} | sensitivity {test['sensitivity']:.4f} "
        f"| specificity {test['specificity']:.4f}"
    )
    print("\n" + format_comparison(comparison))
    print(f"wrote {output_dir}")


def gradcam_statistics(heatmaps, images):
    """Where each model's heatmap weight falls, and how similar the models' maps are."""
    regions = chest_regions(images.shape[-1])
    names = list(heatmaps)
    return {
        "region_area": {name: mask.float().mean().item() for name, mask in regions.items()},
        "region_share": {name: region_shares(h, regions) for name, h in heatmaps.items()},
        "hot_region_brightness": {
            name: hot_region_brightness(h, images) for name, h in heatmaps.items()
        },
        "correlation": {
            f"{first} vs {second}": mean_correlation(heatmaps[first], heatmaps[second])
            for i, first in enumerate(names)
            for second in names[i + 1:]
        },
    }


def format_gradcam_statistics(stats):
    area = stats["region_area"]
    lines = [
        "| | central strip | lung fields |",
        "|---|---|---|",
        f"| share of image area | {area['central_strip']:.2f} | {area['lung_fields']:.2f} |",
    ]
    for name, share in stats["region_share"].items():
        lines.append(
            f"| {name} | {share['central_strip']:.2f} | {share['lung_fields']:.2f} |"
        )
    lines.append("")
    lines += [
        f"correlation {pair}: {value:+.2f}" for pair, value in stats["correlation"].items()
    ]
    lines += [
        f"hot-region brightness {name}: {value:+.3f}"
        for name, value in stats["hot_region_brightness"].items()
    ]
    return "\n".join(lines)


def run_b4():
    config = Config()
    output_dir = OUTPUTS_DIR / "b4"
    data = prepare_data(str(REPO_ROOT / config.data_path))
    images, labels = data["test"].tensors

    # Seeded so the same five X-rays are drawn on every run.
    pneumonia_ids = torch.nonzero(labels == 1).squeeze(1)
    generator = torch.Generator().manual_seed(config.seed)
    picks = torch.randperm(len(pneumonia_ids), generator=generator)[:GRADCAM_SAMPLES]
    sample_ids = pneumonia_ids[picks].sort().values
    samples = images[sample_ids]

    heatmaps = {}
    probabilities = {}
    for name, question, model_class, checkpoint, layer_name in GRADCAM_MODELS:
        model = load_trained(question, model_class, checkpoint)
        layer = dict(model.named_modules())[layer_name]
        heatmaps[name] = grad_cam(model, layer, samples)
        probabilities[name] = pneumonia_probabilities(model, samples).numpy()

    # Undo the normalisation for display: back to [0, 1] grey levels.
    display = (samples.squeeze(1) * data["std"] + data["mean"]).clamp(0, 1)
    ids = sample_ids.tolist()
    stats = gradcam_statistics(heatmaps, display)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_gradcam_grid(
        display.numpy(), {name: h.numpy() for name, h in heatmaps.items()},
        probabilities, ids, output_dir / "gradcam.png",
    )
    (output_dir / "gradcam_stats.json").write_text(json.dumps(stats, indent=2))
    (output_dir / "gradcam_stats.md").write_text(format_gradcam_statistics(stats) + "\n")
    summary = {
        "seed": config.seed,
        "test_indices": ids,
        "layers": {name: layer for name, _, _, _, layer in GRADCAM_MODELS},
        "pneumonia_probability": {
            name: [round(float(p), 4) for p in probs] for name, probs in probabilities.items()
        },
    }
    (output_dir / "samples.json").write_text(json.dumps(summary, indent=2))

    print(f"test indices: {ids}")
    for name, probs in probabilities.items():
        print(f"{name:13s} p(pneumonia): " + ", ".join(f"{p:.2f}" for p in probs))
    print("\n" + format_gradcam_statistics(stats))
    print(f"wrote {output_dir}")


QUESTIONS = {"b1": run_b1, "b2": run_b2, "b3": run_b3, "b4": run_b4}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", choices=QUESTIONS)
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True)
    QUESTIONS[args.question]()


if __name__ == "__main__":
    main()
