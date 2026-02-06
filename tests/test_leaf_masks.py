import pytest
import torch
import pandas as pd
from deadleaves.leaf_masks import circular, rectangular, ellipsoid, polygon

X, Y = torch.meshgrid(torch.arange(10), torch.arange(10), indexing="ij")

# --- Test: Circular leaf mask --------------------------------------------------------


def test_circles():
    params = {
        "x_pos": torch.tensor(5.0),
        "y_pos": torch.tensor(5.0),
        "area": torch.tensor(torch.pi * 4),
    }
    mask = circular((X, Y), params)

    # output shape and type
    assert mask.shape == X.shape
    assert mask.dtype == torch.bool
    # check individual values
    # center point
    assert mask[5, 5]
    # points on boundary
    assert mask[5, 7]
    assert mask[7, 5]
    # points outside of leaf
    assert not mask[7, 7]
    # symmetry
    assert mask[5 - 1, 5] == mask[5 + 1, 5]
    assert mask[5, 5 - 2] == mask[5, 5 + 2]


def test_circles_zero_area():
    params = {
        "x_pos": torch.tensor(4.0),
        "y_pos": torch.tensor(6.0),
        "area": torch.tensor(0.0),
    }
    mask = circular((X, Y), params)
    assert mask.sum() == 1
    assert mask[4, 6]


def test_circles_small_area():
    params = {
        "x_pos": torch.tensor(4.0),
        "y_pos": torch.tensor(4.0),
        "area": torch.tensor(1e-6),
    }
    mask = circular((X, Y), params)

    assert mask.sum() == 1
    assert mask[4, 4]


def test_circles_large_area():
    params = {
        "x_pos": torch.tensor(5.0),
        "y_pos": torch.tensor(5.0),
        "area": torch.tensor(torch.pi * 100),
    }
    mask = circular((X, Y), params)
    assert mask.all()


def test_circles_non_integer_center():
    params = {
        "x_pos": torch.tensor(4.5),
        "y_pos": torch.tensor(4.5),
        "area": torch.tensor(torch.pi * 4),
    }
    mask = circular((X, Y), params)
    assert mask[4, 4]
    assert mask[5, 5]
    assert not mask[7, 7]


def test_circles_empty():
    params = {
        "x_pos": torch.tensor(4.5),
        "y_pos": torch.tensor(4.5),
        "area": torch.tensor(0.0),
    }
    mask = circular((X, Y), params)
    assert not mask.any()


def test_circles_pandas():
    params = pd.Series(
        {
            "x_pos": torch.tensor(5.0),
            "y_pos": torch.tensor(5.0),
            "area": torch.tensor(torch.pi * 4),
        }
    )
    mask = circular((X, Y), params)
    assert mask.shape == X.shape
    assert mask.dtype == torch.bool
    assert mask[5, 5]


def test_circles_invalid_args():
    params = {
        "x_pos": torch.tensor(5.0),
        "y_pos": torch.tensor(5.0),
        "area": torch.tensor(-1.0),
    }
    with pytest.raises(ValueError):
        circular((X, Y), params)


# --- Test: Rectangular leaf mask ------------------------------------------------------


def test_squares():
    params = {
        "x_pos": torch.tensor(5.0),
        "y_pos": torch.tensor(5.0),
        "area": torch.tensor(4),
        "aspect_ratio": torch.tensor(1.0),
        "orientation": torch.tensor(0.0),
    }
    mask = rectangular((X, Y), params)

    # output shape and type
    assert mask.shape == X.shape
    assert mask.dtype == torch.bool
    # check individual values
    # center point
    assert mask[5, 5]
    # points on boundary
    assert mask[4, 6]
    assert mask[6, 4]
    # points outside of leaf
    assert not mask[7, 7]


def test_rectangles():
    params = {
        "x_pos": torch.tensor(5.0),
        "y_pos": torch.tensor(5.0),
        "area": torch.tensor(4),
        "aspect_ratio": torch.tensor(4.0),
        "orientation": torch.tensor(0.0),
    }
    mask = rectangular((X, Y), params)

    # center point
    assert mask[5, 5]
    # points on boundary
    assert mask[6, 5]
    assert mask[4, 5]
    # points outside of leaf
    assert not mask[5, 6]
    assert not mask[5, 4]

    params = {
        "x_pos": torch.tensor(5.0),
        "y_pos": torch.tensor(5.0),
        "area": torch.tensor(4),
        "aspect_ratio": torch.tensor(0.25),
        "orientation": torch.tensor(0.0),
    }
    mask = rectangular((X, Y), params)

    # center point
    assert mask[5, 5]
    # points on boundary
    assert mask[5, 6]
    assert mask[5, 4]
    # points outside of leaf
    assert not mask[6, 5]
    assert not mask[4, 5]

    params = {
        "x_pos": torch.tensor(5.0),
        "y_pos": torch.tensor(5.0),
        "area": torch.tensor(4),
        "aspect_ratio": torch.tensor(0.25),
        "orientation": torch.tensor(0.5 * torch.pi),
    }
    mask = rectangular((X, Y), params)

    # center point
    assert mask[5, 5]
    # points on boundary
    assert mask[6, 5]
    assert mask[4, 5]
    # points outside of leaf
    assert not mask[5, 6]
    assert not mask[5, 4]


def test_rectangles_zero_area():
    params = {
        "x_pos": torch.tensor(4.0),
        "y_pos": torch.tensor(6.0),
        "area": torch.tensor(0.0),
        "aspect_ratio": torch.tensor(1.0),
        "orientation": torch.tensor(0.0),
    }
    mask = rectangular((X, Y), params)
    assert mask.sum() == 1
    assert mask[4, 6]


def test_rectangles_small_area():
    params = {
        "x_pos": torch.tensor(4.0),
        "y_pos": torch.tensor(4.0),
        "area": torch.tensor(1e-6),
        "aspect_ratio": torch.tensor(1.0),
        "orientation": torch.tensor(0.0),
    }
    mask = rectangular((X, Y), params)

    assert mask.sum() == 1
    assert mask[4, 4]


def test_rectangles_large_area():
    params = {
        "x_pos": torch.tensor(5.0),
        "y_pos": torch.tensor(5.0),
        "area": torch.tensor(torch.pi * 100),
        "aspect_ratio": torch.tensor(1.0),
        "orientation": torch.tensor(0.0),
    }
    mask = rectangular((X, Y), params)
    assert mask.all()


def test_rectangles_non_integer_center():
    params = {
        "x_pos": torch.tensor(4.5),
        "y_pos": torch.tensor(4.5),
        "area": torch.tensor(torch.pi * 4),
        "aspect_ratio": torch.tensor(1.0),
        "orientation": torch.tensor(0.0),
    }
    mask = rectangular((X, Y), params)
    assert mask[4, 4]
    assert mask[5, 5]
    assert not mask[7, 7]


def test_rectangles_empty():
    params = {
        "x_pos": torch.tensor(4.5),
        "y_pos": torch.tensor(4.5),
        "area": torch.tensor(0.0),
        "aspect_ratio": torch.tensor(1.0),
        "orientation": torch.tensor(0.0),
    }
    mask = rectangular((X, Y), params)
    assert not mask.any()


def test_rectangles_pandas():
    params = pd.Series(
        {
            "x_pos": torch.tensor(5.0),
            "y_pos": torch.tensor(5.0),
            "area": torch.tensor(torch.pi * 4),
            "aspect_ratio": torch.tensor(1.0),
            "orientation": torch.tensor(0.0),
        }
    )
    mask = rectangular((X, Y), params)
    assert mask.shape == X.shape
    assert mask.dtype == torch.bool
    assert mask[5, 5]


test_invalid_args_rectangles = [(-1.0, 1.0), (1.0, 0.0)]


@pytest.mark.parametrize("area, aspect_ratio", test_invalid_args_rectangles)
def test_rectangles_invalid_args(area, aspect_ratio):
    params = {
        "x_pos": torch.tensor(5.0),
        "y_pos": torch.tensor(5.0),
        "area": torch.tensor(area),
        "aspect_ratio": torch.tensor(aspect_ratio),
        "orientation": torch.tensor(0.0),
    }
    with pytest.raises(ValueError):
        rectangular((X, Y), params)


# --- Test: Ellipsoidal leaf mask ------------------------------------------------------


def test_ellipsoids():
    params = {
        "x_pos": torch.tensor(5.0),
        "y_pos": torch.tensor(5.0),
        "area": torch.tensor(4),
        "aspect_ratio": torch.tensor(4.0),
        "orientation": torch.tensor(0.0),
    }
    mask = ellipsoid((X, Y), params)

    # output shape and type
    assert mask.shape == X.shape
    assert mask.dtype == torch.bool
    # check individual values
    # center point
    assert mask[5, 5]
    # points on boundary
    assert mask[6, 5]
    assert mask[4, 5]
    # points outside of leaf
    assert not mask[5, 6]
    assert not mask[5, 4]

    params = {
        "x_pos": torch.tensor(5.0),
        "y_pos": torch.tensor(5.0),
        "area": torch.tensor(4),
        "aspect_ratio": torch.tensor(0.25),
        "orientation": torch.tensor(0.0),
    }
    mask = rectangular((X, Y), params)

    # center point
    assert mask[5, 5]
    # points on boundary
    assert mask[5, 6]
    assert mask[5, 4]
    # points outside of leaf
    assert not mask[6, 5]
    assert not mask[4, 5]

    params = {
        "x_pos": torch.tensor(5.0),
        "y_pos": torch.tensor(5.0),
        "area": torch.tensor(4),
        "aspect_ratio": torch.tensor(0.25),
        "orientation": torch.tensor(0.5 * torch.pi),
    }
    mask = rectangular((X, Y), params)

    # center point
    assert mask[5, 5]
    # points on boundary
    assert mask[6, 5]
    assert mask[4, 5]
    # points outside of leaf
    assert not mask[5, 6]
    assert not mask[5, 4]


def test_ellipsoids_small_area():
    params = {
        "x_pos": torch.tensor(4.0),
        "y_pos": torch.tensor(4.0),
        "area": torch.tensor(1e-6),
        "aspect_ratio": torch.tensor(1.0),
        "orientation": torch.tensor(0.0),
    }
    mask = ellipsoid((X, Y), params)

    assert mask.sum() == 1
    assert mask[4, 4]


def test_ellipsoids_large_area():
    params = {
        "x_pos": torch.tensor(5.0),
        "y_pos": torch.tensor(5.0),
        "area": torch.tensor(torch.pi * 100),
        "aspect_ratio": torch.tensor(1.0),
        "orientation": torch.tensor(0.0),
    }
    mask = ellipsoid((X, Y), params)
    assert mask.all()


def test_ellipsoids_non_integer_center():
    params = {
        "x_pos": torch.tensor(4.5),
        "y_pos": torch.tensor(4.5),
        "area": torch.tensor(torch.pi * 4),
        "aspect_ratio": torch.tensor(1.0),
        "orientation": torch.tensor(0.0),
    }
    mask = ellipsoid((X, Y), params)
    assert mask[4, 4]
    assert mask[5, 5]
    assert not mask[7, 7]


def test_ellipsoids_empty():
    params = {
        "x_pos": torch.tensor(4.5),
        "y_pos": torch.tensor(4.5),
        "area": torch.tensor(1e-6),
        "aspect_ratio": torch.tensor(1.0),
        "orientation": torch.tensor(0.0),
    }
    mask = ellipsoid((X, Y), params)
    assert not mask.any()


def test_ellipsoids_pandas():
    params = pd.Series(
        {
            "x_pos": torch.tensor(5.0),
            "y_pos": torch.tensor(5.0),
            "area": torch.tensor(torch.pi * 4),
            "aspect_ratio": torch.tensor(1.0),
            "orientation": torch.tensor(0.0),
        }
    )
    mask = ellipsoid((X, Y), params)
    assert mask.shape == X.shape
    assert mask.dtype == torch.bool
    assert mask[5, 5]


test_invalid_args_ellipsoids = [(0.0, 1.0), (1.0, 0.0)]


@pytest.mark.parametrize("area, aspect_ratio", test_invalid_args_ellipsoids)
def test_ellipsoids_invalid_args(area, aspect_ratio):
    params = {
        "x_pos": torch.tensor(5.0),
        "y_pos": torch.tensor(5.0),
        "area": torch.tensor(area),
        "aspect_ratio": torch.tensor(aspect_ratio),
        "orientation": torch.tensor(0.0),
    }
    with pytest.raises(ValueError):
        ellipsoid((X, Y), params)


# --- Test: Polygon leaf mask ----------------------------------------------------------


def test_polygon():
    params = {
        "x_pos": torch.tensor(5.0),
        "y_pos": torch.tensor(5.0),
        "area": torch.tensor(9),
        "n_vertices": torch.tensor(4),
    }
    mask = polygon((X, Y), params)

    # output shape and type
    assert mask.shape == X.shape
    assert mask.dtype == torch.bool
    # check individual values
    # center point
    assert mask[5, 5]
    # points on boundary
    assert mask[6, 5]
    assert mask[5, 6]
    # points outside of leaf
    assert not mask[7, 7]
    # symmetry
    assert mask[5 - 1, 5] == mask[5 + 1, 5]
    assert mask[5, 5 - 2] == mask[5, 5 + 2]


def test_polygon_small_area():
    params = {
        "x_pos": torch.tensor(4.0),
        "y_pos": torch.tensor(4.0),
        "area": torch.tensor(1e-6),
        "n_vertices": torch.tensor(4),
    }
    mask = polygon((X, Y), params)

    assert mask.sum() == 1
    assert mask[4, 4]


def test_polygon_large_area():
    params = {
        "x_pos": torch.tensor(5.0),
        "y_pos": torch.tensor(5.0),
        "area": torch.tensor(1000),
        "n_vertices": torch.tensor(4),
    }
    mask = polygon((X, Y), params)
    assert mask.all()


def test_polygon_non_integer_center():
    params = {
        "x_pos": torch.tensor(4.5),
        "y_pos": torch.tensor(4.5),
        "area": torch.tensor(9),
        "n_vertices": torch.tensor(4),
    }
    mask = polygon((X, Y), params)
    assert mask[4, 4]
    assert mask[5, 5]
    assert not mask[7, 7]


def test_polygon_empty():
    params = {
        "x_pos": torch.tensor(4.5),
        "y_pos": torch.tensor(4.5),
        "area": torch.tensor(1e-6),
        "n_vertices": torch.tensor(4),
    }
    mask = polygon((X, Y), params)
    assert not mask.any()


def test_polygon_pandas():
    params = pd.Series(
        {
            "x_pos": torch.tensor(5.0),
            "y_pos": torch.tensor(5.0),
            "area": torch.tensor(4),
            "n_vertices": torch.tensor(4),
        }
    )
    mask = polygon((X, Y), params)
    assert mask.shape == X.shape
    assert mask.dtype == torch.bool
    assert mask[5, 5]


test_invalid_args_polygon = [(0.0, 1.0), (1.0, 0.0), (1.0, 1.5)]


@pytest.mark.parametrize("area, n_vertices", test_invalid_args_polygon)
def test_polygon_invalid_args(area, n_vertices):
    params = {
        "x_pos": torch.tensor(5.0),
        "y_pos": torch.tensor(5.0),
        "area": torch.tensor(area),
        "n_vertices": torch.tensor(n_vertices),
    }
    with pytest.raises(ValueError):
        polygon((X, Y), params)
