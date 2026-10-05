import pytest
import torch
from torch.utils.data import TensorDataset

from baseline import calculate_r2, train_model
from config import Config
from model import PriceModel

CARDINALITIES = [3, 3, 3, 3]


def make_dataset(row_count=64):
    generator = torch.Generator().manual_seed(0)
    categorical = torch.randint(0, 3, (row_count, 4), generator=generator)
    continuous = torch.randn(row_count, 6, generator=generator)
    target = 400000.0 + 50000.0 * continuous[:, 0]
    return TensorDataset(categorical, continuous, target)


class ConstantModel(PriceModel):
    """Predicts the training mean for every row, so R2 must be 0."""

    def forward(self, categorical, continuous):
        return torch.zeros(len(categorical))


def test_should_return_one_history_record_per_epoch_run():
    config = Config(max_epochs=3, early_stopping_patience=99)
    model = PriceModel(CARDINALITIES, 400000.0, 50000.0, config)

    returned_epoch, returned_history = train_model(
        model, make_dataset(), make_dataset(), config
    )

    assert len(returned_history) == 3
    assert [record["epoch"] for record in returned_history] == [1, 2, 3]
    assert set(returned_history[0]) == {"epoch", "train_rmse", "validation_rmse"}
    assert 1 <= returned_epoch <= 3


def test_should_score_zero_when_predicting_the_mean():
    config = Config()
    dataset = make_dataset()
    target_mean = dataset.tensors[2].mean().item()
    model = ConstantModel(CARDINALITIES, target_mean, 1.0, config)

    returned_r2 = calculate_r2(model, dataset, config.batch_size)

    assert returned_r2 == pytest.approx(0.0, abs=1e-6)


def test_should_score_one_for_perfect_predictions():
    config = Config()
    dataset = make_dataset()
    target = dataset.tensors[2]
    model = ConstantModel(CARDINALITIES, 0.0, 1.0, config)
    model.forward = lambda categorical, continuous: target[: len(categorical)]

    returned_r2 = calculate_r2(model, dataset, config.batch_size)

    assert returned_r2 == pytest.approx(1.0, abs=1e-6)
