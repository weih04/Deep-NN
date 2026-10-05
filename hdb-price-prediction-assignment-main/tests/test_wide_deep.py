from wide_deep import LAMBDAS, SEEDS, build_config

SELECTION = {"hidden_width": 256, "embedding_dim": 2, "best_epoch": 10}


def test_should_search_the_lambda_grid_given_in_the_handout():
    assert LAMBDAS == [0.0, 0.25, 0.5, 1.0, 2.0]


def test_should_use_five_distinct_seeds():
    assert len(SEEDS) == 5
    assert len(set(SEEDS)) == 5


def test_should_keep_a1_hyperparameters_while_setting_lambda():
    returned = build_config(SELECTION, 0.5, seed=43)

    assert returned.hidden_width == 256
    assert returned.embedding_dims == (2, 2, 2, 2)
    assert returned.wide_lambda == 0.5
    assert returned.seed == 43
    # Untouched A1 settings.
    assert returned.learning_rate == 0.01
    assert returned.batch_size == 512
    assert returned.use_embeddings is True
    assert returned.activation == "relu"
