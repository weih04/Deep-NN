import torch
from torch import nn

from config import Config

class PriceModel(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        target_mean: float,
        target_std: float,
        config: Config,
    ) -> None:
        super().__init__()
        self.use_embeddings = config.use_embeddings
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, width)
            for cardinality, width in zip(cardinalities, config.embedding_dims)
        ) if config.use_embeddings else nn.ModuleList()

        input_width = len(config.continuous_features)
        if config.use_embeddings:
            input_width += sum(config.embedding_dims)

        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid}
        if config.activation not in activations:
            raise ValueError(f"unknown activation: {config.activation}")

        self.mlp = nn.Sequential(
            nn.Linear(input_width, config.hidden_width),
            nn.LayerNorm(config.hidden_width),
            activations[config.activation](),
            nn.Linear(config.hidden_width, 1),
        )

        # A3: the wide branch sees the same concatenated input as the MLP.
        self.wide_lambda = config.wide_lambda
        if config.wide_lambda is not None:
            self.wide = nn.Linear(input_width, 1)

        self.target_mean = target_mean
        self.target_std = target_std

    def _features(
        self, categorical: torch.Tensor, continuous: torch.Tensor
    ) -> torch.Tensor:
        if not self.use_embeddings:
            return continuous

        embedded = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        return torch.cat(embedded + [continuous], dim=1)

    def forward(
        self, categorical: torch.Tensor, continuous: torch.Tensor
    ) -> torch.Tensor:
        features = self._features(categorical, continuous)
        output = self.mlp(features)
        if self.wide_lambda is not None:
            output = output + self.wide_lambda * self.wide(features)
        return output.squeeze(1)

    def predict_price(
        self, categorical: torch.Tensor, continuous: torch.Tensor
    ) -> torch.Tensor:
        return self(categorical, continuous) * self.target_std + self.target_mean


def count_parameters(model: nn.Module) -> int:

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
