import torch

from attribution import build_masks, feature_names
from config import Config


def test_should_list_the_ten_original_features_categorical_first():
    config = Config()

    returned = feature_names(config)

    assert len(returned) == 10
    assert returned[:4] == list(config.categorical_features)
    assert returned[4:] == list(config.continuous_features)


def test_should_give_every_original_feature_its_own_ablation_group():
    config = Config()

    categorical_mask, continuous_mask = build_masks(config)

    # One group per original feature, ids unique across both tensors.
    assert categorical_mask.tolist() == [[0, 1, 2, 3]]
    assert continuous_mask.tolist() == [[4, 5, 6, 7, 8, 9]]
    ids = torch.cat([categorical_mask.flatten(), continuous_mask.flatten()])
    assert len(set(ids.tolist())) == 10
