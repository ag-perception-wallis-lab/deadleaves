import pytest
import torch
from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer


test_shape_params = [
    (
        "circular",
        {"area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}},
    )
]
test_size_params = [(100, 100), (10, 20)]
test_color_params = [
    {
        "H": {"normal": {"loc": 0.5, "scale": 0.2}},
        "S": {"normal": {"loc": 0.5, "scale": 0.2}},
        "V": {"normal": {"loc": 0.5, "scale": 0.2}},
    },
    {
        "R": {"normal": {"loc": 0.5, "scale": 0.7}},
        "G": {"normal": {"loc": 0.5, "scale": 0.7}},
        "B": {"normal": {"loc": 0.5, "scale": 0.7}},
    },
]
test_texture_params = [
    {"gray": {"normal": {"loc": 0, "scale": 0.1}}},
    {
        "H": {"normal": {"loc": 0, "scale": 0.1}},
        "S": {"normal": {"loc": 0, "scale": 0.1}},
        "V": {"normal": {"loc": 0, "scale": 0.1}},
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
    geometry_model = LeafGeometryGenerator(shape, shape_params, size)
    leaf_table, segmentation_map = geometry_model.generate_segmentation()
    # all pixel assigned to a leaf
    assert (segmentation_map > 0).all()
    # leaf indices as integers
    assert not torch.is_floating_point(segmentation_map)
    # leaf indices in df and partition match
    assert leaf_table.leaf_idx.max() == segmentation_map.max()
    assert leaf_table.leaf_idx.min() == segmentation_map.min()

    color_model = LeafAppearanceSampler(leaf_table)
    color_model.sample_color(color_params)

    image = ImageRenderer(color_model.leaf_table, segmentation_map).render_image()
    # no empty pixel
    assert not torch.isnan(image).any()
    assert torch.all((0 <= image) & (image <= 1))
