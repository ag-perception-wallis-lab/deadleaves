import pytest
from deadleaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer
import numpy as np

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
    assert (image <= 1.0).all()


grayscale_mean_values = [0.4, 0.5, 0.6]
grayscale_variance_values = [0.1, 0.2]
noise_ranges = [(0.001, 0.01), (0.01, 0.05)]


@pytest.mark.parametrize("grayscale_mean_value", grayscale_mean_values)
@pytest.mark.parametrize("grayscale_variance_value", grayscale_variance_values)
@pytest.mark.parametrize("noise_range", noise_ranges)
def test_luminance_recovery(
    grayscale_mean_value, grayscale_variance_value, noise_range
):
    geometry_model = LeafGeometryGenerator(
        "circular",
        {"radius": {"powerlaw": {"low": 4.0, "high": 16.0, "k": 3}}},
        (50, 50),
    )
    mean_luminances = []
    std_luminances = []
    for _ in range(50):
        leaf_table, segmentation_map = geometry_model.generate_segmentation()
        color_model = LeafAppearanceSampler(leaf_table)
        color_model.sample_color(
            {
                "gray": {
                    "normal": {
                        "loc": grayscale_mean_value,
                        "scale": grayscale_variance_value,
                    }
                }
            }
        )
        color_model.sample_texture(
            {
                "gray": {
                    "normal": {
                        "loc": 0.0,
                        "scale": {
                            "uniform": {"low": noise_range[0], "high": noise_range[1]}
                        },
                    }
                }
            }
        )
        image = ImageRenderer(color_model.leaf_table, segmentation_map).render_image()
        mean_luminances.append(image.mean().item())
        std_luminances.append(image.std().item())
    assert np.mean(mean_luminances) == pytest.approx(grayscale_mean_value, abs=0.05)
    assert np.mean(std_luminances) == pytest.approx(grayscale_variance_value, abs=0.05)
