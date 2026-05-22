import pytest
from deadleaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

color_values = [-0.1, 0.1, 0.3]
texture_values = [-0.2, 0.2]
color_spaces = [("H", "S", "V"), ("R", "G", "B")]


@pytest.mark.parametrize("color_value", color_values)
@pytest.mark.parametrize("texture_value", texture_values)
@pytest.mark.parametrize("color_space", color_spaces)
def test_HSVtransformation(color_value, texture_value, color_space):
    geometry_model = LeafGeometryGenerator(
        "circular",
        {"radius": {"powerlaw": {"low": 4.0, "high": 16.0, "k": 3}}},
        (10, 10),
    )
    leaf_table, segmentation_map = geometry_model.generate_segmentation()
    color_model = LeafAppearanceSampler(leaf_table)
    color_model.sample_color(
        {
            color_space[0]: {"constant": {"value": color_value}},
            color_space[1]: {"constant": {"value": color_value}},
            color_space[2]: {"constant": {"value": color_value}},
        }
    )
    color_model.sample_texture(
        {
            "H": {"constant": {"value": texture_value}},
            "S": {"constant": {"value": texture_value}},
            "V": {"constant": {"value": texture_value}},
        }
    )

    image = ImageRenderer(color_model.leaf_table, segmentation_map).render_image()
    assert (image >= 0.0).all()
