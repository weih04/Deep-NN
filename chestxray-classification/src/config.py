from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    # Label 1 = pneumonia (the positive class for sensitivity), 0 = normal.
    data_path: str = "chestxray_binary_64.npz"

    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 20
    # Early stopping monitors validation BCE loss.
    early_stopping_patience: int = 4
    early_stopping_threshold: float = 1e-4

    dropout: float = 0.5
    # A 0.5 cut on the sigmoid output turns logits into class predictions.
    decision_threshold: float = 0.5
