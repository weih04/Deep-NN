import csv

import pytest

from config import Config
from data import fit_preprocessing, prepare_final_data

FIELDNAMES = [
    "month", "year", "town", "dist_to_nearest_stn", "dist_to_dhoby",
    "degree_centrality", "eigenvector_centrality", "flat_model_type",
    "remaining_lease_years", "floor_area_sqm", "storey_range", "resale_price",
]


def make_row(year, town, price):
    return {
        "month": "01", "year": year, "town": town,
        "dist_to_nearest_stn": 1.0, "dist_to_dhoby": 5.0,
        "degree_centrality": 0.1, "eigenvector_centrality": 0.2,
        "flat_model_type": "4 ROOM Improved", "remaining_lease_years": 70.0,
        "floor_area_sqm": 90.0, "storey_range": "04 TO 06",
        "resale_price": price,
    }


@pytest.fixture
def config():
    return Config()


def test_should_reserve_id_zero_so_cardinality_exceeds_distinct_count(config):
    rows = [make_row(2017, "BEDOK", 400000.0), make_row(2017, "YISHUN", 300000.0)]

    returned_stats = fit_preprocessing(rows, config)

    town_index = config.categorical_features.index("town")
    assert returned_stats["cardinalities"][town_index] == 3
    assert 0 not in returned_stats["category_maps"]["town"].values()


def test_should_fit_target_statistics_only_on_supplied_rows(config):
    rows = [make_row(2017, "BEDOK", 400000.0), make_row(2017, "BEDOK", 600000.0)]

    returned_stats = fit_preprocessing(rows, config)

    assert returned_stats["target_mean"] == pytest.approx(500000.0)


def test_should_include_validation_year_in_final_fitting_set(tmp_path, config):
    csv_path = tmp_path / "rows.csv"
    rows = (
        [make_row(2017, "BEDOK", 400000.0)] * 2
        + [make_row(2021, "YISHUN", 600000.0)] * 2
        + [make_row(2022, "TAMPINES", 700000.0)] * 2
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    returned_data = prepare_final_data(str(csv_path), config)

    # 2017 + 2021 rows are fitted on; 2022 is held out entirely.
    assert returned_data["target_mean"] == pytest.approx(500000.0)
    assert len(returned_data["train"]) == 4
    assert len(returned_data["test"]) == 2
    town_index = config.categorical_features.index("town")
    assert returned_data["cardinalities"][town_index] == 3
