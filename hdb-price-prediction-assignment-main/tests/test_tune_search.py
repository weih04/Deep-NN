from tune_search import build_config, format_grid_table


def test_should_apply_one_embedding_dim_to_all_four_features():
    returned_config = build_config({"hidden_width": 128, "embedding_dim": 12})

    assert returned_config.hidden_width == 128
    assert returned_config.embedding_dims == (12, 12, 12, 12)
    # Untouched fields keep their defaults.
    assert returned_config.seed == 42


def test_should_lay_out_validation_rmse_as_a_width_by_dim_grid():
    records = [
        {"hidden_width": 16, "embedding_dim": 2, "best_validation_rmse": 90000.0},
        {"hidden_width": 16, "embedding_dim": 4, "best_validation_rmse": 88000.0},
        {"hidden_width": 32, "embedding_dim": 2, "best_validation_rmse": 85000.0},
        {"hidden_width": 32, "embedding_dim": 4, "best_validation_rmse": 84000.0},
    ]

    returned_table = format_grid_table(records)

    lines = returned_table.strip().splitlines()
    assert lines[0].startswith("| hidden_width \\ embedding_dim |")
    assert "| 16 |" in returned_table
    assert "90,000" in returned_table
