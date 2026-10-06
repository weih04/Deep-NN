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
