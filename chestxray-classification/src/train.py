import copy
import math

import torch
from torch import nn
from torch.utils.data import DataLoader


def classification_metrics(logits, targets, threshold=0.5):
    """Accuracy, sensitivity (pneumonia recall) and specificity (normal recall)."""
    predictions = (torch.sigmoid(logits) >= threshold).float()
    positives = targets == 1
    negatives = targets == 0

    true_positives = (predictions[positives] == 1).sum().item()
    true_negatives = (predictions[negatives] == 0).sum().item()
    return {
        "accuracy": (predictions == targets).float().mean().item(),
        "sensitivity": true_positives / positives.sum().item(),
        "specificity": true_negatives / negatives.sum().item(),
    }


@torch.no_grad()
def evaluate(model, dataset, batch_size, threshold=0.5):
    model.eval()
    logits = []
    targets = []
    for images, labels in DataLoader(dataset, batch_size=batch_size):
        logits.append(model(images))
        targets.append(labels)
    logits = torch.cat(logits)
    targets = torch.cat(targets)

    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets).item()
    return {"loss": loss} | classification_metrics(logits, targets, threshold)


def train_model(model, train_data, val_data, config):
    """Adam + early stopping on validation loss, restores the best-epoch weights."""
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_data, config.batch_size, shuffle=True, generator=generator
    )
    loss_function = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = float("inf")
    best_weights = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        row_count = 0
        for images, labels in train_loader:
            optimiser.zero_grad()
            loss = loss_function(model(images), labels)
            loss.backward()
            optimiser.step()
            loss_sum += loss.item() * len(labels)
            row_count += len(labels)

        val = evaluate(model, val_data, config.batch_size, config.decision_threshold)
        record = {"epoch": epoch, "train_loss": loss_sum / row_count} | {
            f"val_{name}": value for name, value in val.items()
        }
        history.append(record)
        print(
            f"epoch {epoch:02d} | train loss {record['train_loss']:.4f} "
            f"| val loss {val['loss']:.4f} | val acc {val['accuracy']:.4f}"
        )

        # A diverged (NaN/inf) loss must never count as the best so far.
        monitored = val["loss"] if math.isfinite(val["loss"]) else float("inf")
        if (
            best_weights is None
            or monitored < best_loss - config.early_stopping_threshold
        ):
            best_loss = monitored
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == config.early_stopping_patience:
                print(f"early stopping after epoch {epoch}")
                break

    model.load_state_dict(best_weights)
    return best_epoch, history
