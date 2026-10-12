import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_validation_curves(histories, metric, ylabel, path, title, label_format="{}"):
    figure, axes = plt.subplots(figsize=(7, 4.5))
    for key, history in histories.items():
        epochs = [record["epoch"] for record in history]
        values = [record[metric] for record in history]
        axes.plot(epochs, values, marker="o", markersize=3, label=label_format.format(key))

    axes.set_xlabel("epoch")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.set_xticks(range(1, max(len(h) for h in histories.values()) + 1))
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_gradcam_grid(images, heatmaps, probabilities, sample_ids, path, alpha=0.45):
    """Rows: X-rays. Columns: the original, then one Grad-CAM overlay per model.

    images: (N, H, W) in [0, 1]; heatmaps / probabilities: {model name: (N, H, W) / (N,)}.
    """
    names = list(heatmaps)
    rows, cols = len(images), len(names) + 1
    figure, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
    for row in range(rows):
        axes[row, 0].imshow(images[row], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"test #{sample_ids[row]}", fontsize=9)
        for col, name in enumerate(names, start=1):
            axes[row, col].imshow(images[row], cmap="gray", vmin=0, vmax=1)
            axes[row, col].imshow(heatmaps[name][row], cmap="jet", alpha=alpha, vmin=0, vmax=1)
            axes[row, col].set_title(
                f"{name}\np(pneumonia)={probabilities[name][row]:.2f}", fontsize=9
            )
    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
