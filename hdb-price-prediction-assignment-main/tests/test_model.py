from config import Config
from model import PriceModel, count_parameters

TRAIN_CARDINALITIES = [13, 27, 44, 18]


def build_model(embedding_dim, hidden_width):
    config = Config(
        hidden_width=hidden_width,
        embedding_dims=(embedding_dim,) * 4,
    )
    return PriceModel(TRAIN_CARDINALITIES, 450000.0, 150000.0, config)


def test_should_match_hand_computed_count_for_baseline_size():
    # 102*8 embeddings + (38*64 + 64) + 2*64 layer norm + (64 + 1)
    expected_count = 3505

    returned_count = count_parameters(build_model(8, 64))

    assert returned_count == expected_count


def test_should_match_closed_form_across_grid_corners():
    for embedding_dim, hidden_width in [(2, 16), (16, 256), (4, 128)]:
        input_width = 4 * embedding_dim + 6
        expected_count = (
            sum(TRAIN_CARDINALITIES) * embedding_dim
            + input_width * hidden_width
            + hidden_width
            + 2 * hidden_width
            + hidden_width
            + 1
        )

        returned_count = count_parameters(build_model(embedding_dim, hidden_width))

        assert returned_count == expected_count


def test_should_drop_embeddings_and_use_only_continuous_features():
    config = Config(hidden_width=256, embedding_dims=(2,) * 4, use_embeddings=False)

    returned_model = PriceModel(TRAIN_CARDINALITIES, 450000.0, 150000.0, config)

    assert len(returned_model.embeddings) == 0
    # 6 continuous inputs only, hidden width unchanged.
    assert returned_model.mlp[0].in_features == 6
    assert returned_model.mlp[0].out_features == 256
    assert count_parameters(returned_model) == 6 * 256 + 256 + 2 * 256 + 256 + 1


def test_should_forward_without_embeddings():
    import torch

    config = Config(hidden_width=32, embedding_dims=(2,) * 4, use_embeddings=False)
    model = PriceModel(TRAIN_CARDINALITIES, 450000.0, 150000.0, config)

    returned = model(torch.zeros(5, 4, dtype=torch.long), torch.randn(5, 6))

    assert returned.shape == (5,)


def test_should_replace_relu_with_sigmoid_keeping_parameter_count():
    from torch import nn

    relu_config = Config(hidden_width=256, embedding_dims=(2,) * 4)
    sigmoid_config = Config(
        hidden_width=256, embedding_dims=(2,) * 4, activation="sigmoid"
    )

    returned_model = PriceModel(TRAIN_CARDINALITIES, 450000.0, 150000.0, sigmoid_config)

    assert any(isinstance(layer, nn.Sigmoid) for layer in returned_model.mlp)
    assert not any(isinstance(layer, nn.ReLU) for layer in returned_model.mlp)
    assert count_parameters(returned_model) == count_parameters(
        PriceModel(TRAIN_CARDINALITIES, 450000.0, 150000.0, relu_config)
    )


def test_should_add_a_wide_branch_over_the_same_concatenated_input():
    import torch

    config = Config(hidden_width=256, embedding_dims=(2,) * 4, wide_lambda=0.5)

    returned_model = PriceModel(TRAIN_CARDINALITIES, 450000.0, 150000.0, config)

    # Same input as the MLP branch: 4*2 embeddings + 6 continuous.
    assert returned_model.wide.in_features == 14
    assert returned_model.wide.out_features == 1
    returned = returned_model(torch.zeros(5, 4, dtype=torch.long), torch.randn(5, 6))
    assert returned.shape == (5,)


def test_should_equal_mlp_output_when_lambda_is_zero():
    import torch

    config = Config(hidden_width=32, embedding_dims=(2,) * 4, wide_lambda=0.0)
    model = PriceModel(TRAIN_CARDINALITIES, 450000.0, 150000.0, config)
    categorical = torch.zeros(4, 4, dtype=torch.long)
    continuous = torch.randn(4, 6)

    embedded = [e(categorical[:, i]) for i, e in enumerate(model.embeddings)]
    features = torch.cat(embedded + [continuous], dim=1)
    expected = model.mlp(features).squeeze(1)

    returned = model(categorical, continuous)

    assert torch.allclose(returned, expected)


def test_should_scale_the_wide_branch_by_lambda():
    import torch

    config = Config(hidden_width=32, embedding_dims=(2,) * 4, wide_lambda=2.0)
    model = PriceModel(TRAIN_CARDINALITIES, 450000.0, 150000.0, config)
    categorical = torch.zeros(4, 4, dtype=torch.long)
    continuous = torch.randn(4, 6)

    embedded = [e(categorical[:, i]) for i, e in enumerate(model.embeddings)]
    features = torch.cat(embedded + [continuous], dim=1)
    expected = (model.mlp(features) + 2.0 * model.wide(features)).squeeze(1)

    returned = model(categorical, continuous)

    assert torch.allclose(returned, expected)
