import pytest
import torch
from dead_leaves import DeadLeavesImage, DeadLeavesModel


test_shape_params = [
    (
        "circular",
        {"area": {"powerlaw": {"min": 100.0, "max": 10000.0, "k": 1.5}}},
    )
]
test_size_params = [(100, 100), (10, 20)]
test_color_params = [
    {
        "H": {"normal": {"mean": 0.5, "std": 0.2}},
        "S": {"normal": {"mean": 0.5, "std": 0.2}},
        "V": {"normal": {"mean": 0.5, "std": 0.2}},
    },
    {
        "R": {"normal": {"mean": 0.5, "std": 0.7}},
        "G": {"normal": {"mean": 0.5, "std": 0.7}},
        "B": {"normal": {"mean": 0.5, "std": 0.7}},
    },
]
test_texture_params = [
    {"gray": {"normal": {"mean": 0, "std": 0.1}}},
    {
        "H": {"normal": {"mean": 0, "std": 0.1}},
        "S": {"normal": {"mean": 0, "std": 0.1}},
        "V": {"normal": {"mean": 0, "std": 0.1}},
    },
]
test_seeds = list(range(20))


@pytest.mark.parametrize("shape, shape_params", test_shape_params)
@pytest.mark.parametrize("size", test_size_params)
@pytest.mark.parametrize("color_params", test_color_params)
@pytest.mark.parametrize("texture_params", test_texture_params)
@pytest.mark.parametrize("seed", test_seeds)
def test_DeadLeavesModel(shape, shape_params, size, color_params, texture_params, seed):
    torch.manual_seed(seed)
    model = DeadLeavesModel(shape, shape_params, size)
    leaves, partition = model.sample_partition()
    # all pixel assigned to a leaf
    assert (partition > 0).all()
    # leaf indices as integers
    assert not torch.is_floating_point(partition)
    # leaf indices in df and partition match
    assert leaves.leaf_idx.max() == partition.max()
    assert leaves.leaf_idx.min() == partition.min()

    color_model = DeadLeavesImage(leaves, partition, color_params, texture_params)
    image = color_model.sample_image()
    # no empty pixel
    assert not torch.isnan(image).any()
    assert torch.all((0 <= image) & (image <= 1))
