import pytest
from deadleaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

color_spaces = [("H", "S", "V"), ("R", "G", "B"), ("gray",)]
texture_spaces = [("H", "S", "V"), ("R", "G", "B"), ("gray",)]


@pytest.mark.parametrize("texture_space", texture_spaces)
@pytest.mark.parametrize("color_space", color_spaces)
def test_color_space_recovery(texture_space, color_space):
    print(color_space)
    geometry_model = LeafGeometryGenerator(
        "circular",
        {"radius": {"powerlaw": {"low": 4.0, "high": 16.0, "k": 3}}},
        (10, 10),
    )
    leaf_table, segmentation_map = geometry_model.generate_segmentation()
    color_model = LeafAppearanceSampler(leaf_table)
    color_model.sample_color({k: {"constant": {"value": 0.5}} for k in color_space})
    color_model.sample_texture(
        {k: {"normal": {"loc": 0, "scale": 0.05}} for k in texture_space}
    )

    renderer = ImageRenderer(color_model.leaf_table, segmentation_map)
    assert renderer.color_space == color_space or (renderer.color_space,) == color_space
    assert (
        renderer.texture_space == texture_space
        or (renderer.texture_space,) == texture_space
    )
