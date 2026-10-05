import torch
from torch.utils.data import TensorDataset

from config import Config
from final import train_fixed_epochs
from model import PriceModel

CARDINALITIES = [3, 3, 3, 3]


def make_dataset(row_count=64):
    generator = torch.Generator().manual_seed(0)
    categorical = torch.randint(0, 3, (row_count, 4), generator=generator)
    continuous = torch.randn(row_count, 6, generator=generator)
    target = 400000.0 + 50000.0 * continuous[:, 0]
    return TensorDataset(categorical, continuous, target)


def test_should_run_exactly_the_requested_epochs_without_early_stopping():
    config = Config(max_epochs=99, early_stopping_patience=1)
    model = PriceModel(CARDINALITIES, 400000.0, 50000.0, config)

    returned_history = train_fixed_epochs(
        model, make_dataset(), make_dataset(), config, epochs=4
    )

    assert [record["epoch"] for record in returned_history] == [1, 2, 3, 4]
    assert set(returned_history[0]) == {"epoch", "train_rmse", "test_rmse"}


def test_should_name_the_held_out_series_when_asked():
    config = Config()
    model = PriceModel(CARDINALITIES, 400000.0, 50000.0, config)

    returned_history = train_fixed_epochs(
        model, make_dataset(), make_dataset(), config, epochs=2,
        eval_key="validation_rmse",
    )

    assert set(returned_history[0]) == {"epoch", "train_rmse", "validation_rmse"}
