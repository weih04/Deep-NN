from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: these scripts only ever write files
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def plot_lambda_search(records, output_path):
    """Validation RMSE against lambda, with the selected value marked."""
    output_path = Path(output_path)
    lambdas = [record["lambda"] for record in records]
    values = [record["validation_rmse"] for record in records]
    best = min(range(len(values)), key=lambda index: values[index])

    figure, axis = plt.subplots(figsize=(6.5, 4.0))
    axis.plot(range(len(lambdas)), values, marker="o", color="tab:blue")
    axis.scatter([best], [values[best]], s=140, facecolors="none",
                 edgecolors="tab:red", linewidths=2, zorder=4,
                 label=f"selected $\\lambda$ = {lambdas[best]}")
    axis.set_xticks(range(len(lambdas)))
    axis.set_xticklabels([str(value) for value in lambdas])
    axis.set_xlabel("$\\lambda$")
    axis.set_ylabel("best validation RMSE (SGD)")
    axis.set_title("A3 wide-and-deep: $\\lambda$ search on 2021 validation")
    axis.grid(alpha=0.3)
    axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_rmse_curves(history, series, output_path, title, marker_epoch=None):
    output_path = Path(output_path)
    epochs = [record["epoch"] for record in history]

    figure, axes = plt.subplots(figsize=(7.0, 4.5))
    for key, label in series.items():
        axes.plot(epochs, [record[key] for record in history], marker="o", label=label)

    if marker_epoch is not None:
        axes.axvline(marker_epoch, color="grey", linestyle="--", linewidth=1)
        axes.annotate(
            f"selected epoch {marker_epoch}",
            xy=(marker_epoch, axes.get_ylim()[1]),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=8,
            color="grey",
        )

    axes.set_xlabel("epoch")
    axes.set_ylabel("RMSE (SGD)")
    axes.set_title(title)
    axes.legend()
    axes.grid(alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_ablation_summary(records, output_path,
                          title="A2 ablations: five seeds per variant (seeds paired)"):
    output_path = Path(output_path)
    variants = list(dict.fromkeys(record["variant"] for record in records))
    seeds = sorted({record["seed"] for record in records})
    positions = list(range(len(variants)))
    colours = plt.get_cmap("tab10")

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    panels = [("test_rmse", "test RMSE (SGD)"), ("test_r2", "test $R^2$")]

    for axis, (metric, label) in zip(axes, panels):
        lookup = {(r["variant"], r["seed"]): r[metric] for r in records}
        means = [
            sum(lookup[(v, s)] for s in seeds) / len(seeds) for v in variants
        ]
        # A horizontal marker, not a bar: the axis is zoomed to the data, and
        # truncated bars misrepresent magnitude.
        axis.scatter(positions, means, marker="_", s=2600, linewidths=2.5,
                     color="dimgrey", zorder=2)

        for index, seed in enumerate(seeds):
            values = [lookup[(variant, seed)] for variant in variants]
            axis.plot(positions, values, marker="o", markersize=6, linewidth=1.2,
                      color=colours(index), alpha=0.85, zorder=3,
                      label=f"seed {seed}")

        axis.set_xticks(positions)
        axis.set_xticklabels(variants, rotation=15)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.3)
        # Zoom to the data: bars from zero would flatten the differences.
        flat = [lookup[key] for key in lookup]
        span = max(flat) - min(flat)
        axis.set_ylim(min(flat) - 0.25 * span, max(flat) + 0.15 * span)

    handles, labels = axes[0].get_legend_handles_labels()
    # Proxy handle: the real mean marker is too wide to render in a legend.
    handles.append(Line2D([], [], color="dimgrey", linewidth=2.5))
    labels.append("mean")
    figure.legend(handles, labels, loc="center right", frameon=False, fontsize=8)
    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=[0, 0, 0.88, 1])
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_feature_importance(results, output_path):
    """One horizontal bar panel per model, each ordered highest to lowest."""
    output_path = Path(output_path)
    labels = list(results)
    figure, axes = plt.subplots(
        1, len(labels), figsize=(6.0 * len(labels), 4.8), sharex=True
    )
    if len(labels) == 1:
        axes = [axes]

    for axis, label in zip(axes, labels):
        ranked = sorted(results[label].items(), key=lambda item: item[1])
        names = [name for name, _ in ranked]
        values = [value for _, value in ranked]
        axis.barh(range(len(names)), values, color="tab:blue")
        axis.set_yticks(range(len(names)))
        axis.set_yticklabels(names)
        axis.set_xlabel("mean |attribution| (SGD)")
        axis.set_title(label, fontsize=10)
        axis.grid(axis="x", alpha=0.3)
        for index, value in enumerate(values):
            axis.text(value, index, f" {value:,.0f}", va="center", fontsize=8)

    figure.suptitle("A4 FeatureAblation: global feature importance on the test set")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
