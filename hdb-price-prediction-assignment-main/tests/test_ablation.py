import pytest

from ablation import VARIANTS, build_variant_config, summarise

SELECTION = {"hidden_width": 256, "embedding_dim": 2, "best_epoch": 10}


def test_should_define_baseline_and_two_ablations():
    assert list(VARIANTS) == ["baseline", "no_embeddings", "sigmoid"]


def test_should_keep_a1_architecture_for_the_baseline_variant():
    returned = build_variant_config(SELECTION, "baseline")

    assert returned.hidden_width == 256
    assert returned.embedding_dims == (2, 2, 2, 2)
    assert returned.use_embeddings is True
    assert returned.activation == "relu"


def test_should_drop_embeddings_but_keep_hidden_width_for_ablation_a():
    returned = build_variant_config(SELECTION, "no_embeddings")

    assert returned.use_embeddings is False
    assert returned.hidden_width == 256


def test_should_swap_activation_but_keep_inputs_for_ablation_b():
    returned = build_variant_config(SELECTION, "sigmoid")

    assert returned.activation == "sigmoid"
    assert returned.use_embeddings is True
    assert returned.hidden_width == 256


def test_should_report_mean_and_sample_standard_deviation_per_variant():
    records = [
        {"variant": "baseline", "seed": 1, "test_rmse": 100.0, "test_r2": 0.5},
        {"variant": "baseline", "seed": 2, "test_rmse": 110.0, "test_r2": 0.7},
        {"variant": "sigmoid", "seed": 1, "test_rmse": 200.0, "test_r2": 0.1},
    ]

    returned = summarise(records)

    assert returned["baseline"]["test_rmse_mean"] == pytest.approx(105.0)
    # sample sd (n-1), not population sd
    assert returned["baseline"]["test_rmse_sd"] == pytest.approx(7.0710678)
    assert returned["baseline"]["test_r2_mean"] == pytest.approx(0.6)
    assert returned["sigmoid"]["test_rmse_mean"] == pytest.approx(200.0)
