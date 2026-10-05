from plots import plot_rmse_curves

HISTORY = [
    {"epoch": 1, "train_rmse": 90000.0, "validation_rmse": 95000.0},
    {"epoch": 2, "train_rmse": 70000.0, "validation_rmse": 78000.0},
    {"epoch": 3, "train_rmse": 60000.0, "validation_rmse": 77000.0},
]


def test_should_write_a_figure_to_the_requested_path(tmp_path):
    output_path = tmp_path / "nested" / "curves.png"

    returned_path = plot_rmse_curves(
        HISTORY,
        {"train_rmse": "train", "validation_rmse": "validation"},
        output_path,
        "curves",
        marker_epoch=3,
    )

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
