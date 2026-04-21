"""
Quadtree equivalence tests for dead leaves sampling.

Verifies two properties:

1. **Deterministic equivalence** — given the same RNG seed, the quadtree
   implementation produces bit-identical segmentation maps and leaf tables
   to the original brute-force loop, across all leaf shapes.

2. **Distributional equivalence** — when sampling with independent seeds,
   the marginal distributions of leaf parameters (area, position, count)
   are statistically indistinguishable between the two implementations.
   A control comparison (original vs. original) is included to calibrate
   the sensitivity of the test.
"""

import math

import pytest
import torch
import pandas as pd
import numpy as np
from scipy import stats

from deadleaves import LeafGeometryGenerator


# Reference implementation
def _generate_segmentation_original(model):
    """Verbatim copy of the pre-quadtree generate_segmentation."""
    leaves_params = []
    segmentation_map = torch.zeros(
        *model.image_shape, device=model.device, dtype=torch.int
    )
    leaf_idx = 1

    while torch.any((segmentation_map == 0) & (model.position_mask == 1)):
        params = model._sample_parameters()
        try:
            leaf_mask = model.generate_leaf_mask((model.X, model.Y), params)
        except ValueError:
            continue
        mask = leaf_mask & (segmentation_map == 0)
        if (mask.sum() > 0) & model.position_mask[
            params["y_pos"].to(torch.int), params["x_pos"].to(torch.int)
        ]:
            segmentation_map[mask] = leaf_idx
            leaves_params.append(params)
            leaf_idx += 1
        if (model.n_sample is not None) and leaf_idx >= model.n_sample:
            break

    leaf_table = pd.DataFrame(leaves_params, columns=model.params)
    leaf_table["leaf_idx"] = torch.tensor(range(leaf_idx - 1)) + 1
    leaf_table["leaf_shape"] = model.leaf_shape
    return leaf_table, segmentation_map


CANVAS_SIZE = (64, 64)
SHAPE_CONFIGS = {
    "circular": {
        "area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}},
    },
    "ellipsoid": {
        "area": {"powerlaw": {"low": 100.0, "high": 8000.0, "k": 1.5}},
        "aspect_ratio": {"uniform": {"low": 0.3, "high": 3.0}},
        "orientation": {"uniform": {"low": 0, "high": math.pi}},
    },
    "rectangular": {
        "area": {"powerlaw": {"low": 100.0, "high": 8000.0, "k": 1.5}},
        "aspect_ratio": {"uniform": {"low": 0.5, "high": 2.0}},
        "orientation": {"uniform": {"low": 0, "high": math.pi}},
    },
    "polygon": {
        "area": {"powerlaw": {"low": 200.0, "high": 5000.0, "k": 1.5}},
        "n_vertices": {"constant": {"value": 5}},
    },
}


@pytest.mark.parametrize("shape", SHAPE_CONFIGS.keys())
@pytest.mark.parametrize("seed", range(10))
def test_deterministic_equivalence(shape, seed):
    """Same seed → bit-identical segmentation map and leaf table."""
    params = SHAPE_CONFIGS[shape]

    torch.manual_seed(seed)
    model = LeafGeometryGenerator(shape, params, CANVAS_SIZE, device="cpu")
    lt_orig, seg_orig = _generate_segmentation_original(model)

    torch.manual_seed(seed)
    model = LeafGeometryGenerator(shape, params, CANVAS_SIZE, device="cpu")
    lt_qt, seg_qt = model.generate_segmentation()

    # Segmentation maps must be identical
    assert torch.equal(seg_orig, seg_qt), (
        f"Segmentation maps differ for {shape} seed={seed}"
    )

    # Leaf tables must have the same number of rows
    assert len(lt_orig) == len(lt_qt), (
        f"Leaf count differs: {len(lt_orig)} vs {len(lt_qt)}"
    )

    # Every numeric column must match exactly
    for col in lt_orig.columns:
        if col == "leaf_shape":
            assert (lt_orig[col] == lt_qt[col]).all()
        else:
            v1 = lt_orig[col].apply(float).values
            v2 = lt_qt[col].apply(float).values
            np.testing.assert_array_equal(
                v1, v2, err_msg=f"Column {col} differs for {shape} seed={seed}"
            )


@pytest.mark.parametrize("shape", SHAPE_CONFIGS.keys())
@pytest.mark.parametrize("seed", range(5))
def test_aabb_contains_all_mask_pixels(shape, seed):
    """The AABB must contain every pixel where the mask is True."""
    from deadleaves.acceleration.quadtree import leaf_aabb

    params = SHAPE_CONFIGS[shape]

    torch.manual_seed(seed)
    model = LeafGeometryGenerator(shape, params, CANVAS_SIZE, device="cpu")
    H, W = CANVAS_SIZE

    for _ in range(50):
        leaf_params = model._sample_parameters()
        try:
            mask = model.generate_leaf_mask((model.X, model.Y), leaf_params)
        except ValueError:
            continue

        if not mask.any():
            continue

        y_min, x_min, y_max, x_max = leaf_aabb(leaf_params, shape)
        y_min_c = max(y_min, 0)
        x_min_c = max(x_min, 0)
        y_max_c = min(y_max, H)
        x_max_c = min(x_max, W)

        # Zero out pixels inside the AABB — anything left is a leak
        outside = mask.clone()
        if y_min_c < y_max_c and x_min_c < x_max_c:
            outside[y_min_c:y_max_c, x_min_c:x_max_c] = False

        assert not outside.any(), (
            f"AABB leak for {shape} seed={seed}: "
            f"AABB=[{y_min},{x_min},{y_max},{x_max}], "
            f"leaked pixels at {torch.stack(torch.where(outside), dim=1).tolist()}"
        )


def _collect_leaf_stats(gen_fn, shape, params, n_realizations, seed_offset):
    """Run n_realizations and collect pooled leaf statistics."""
    areas, x_positions, y_positions, counts = [], [], [], []
    for i in range(n_realizations):
        torch.manual_seed(seed_offset + i)
        model = LeafGeometryGenerator(shape, params, CANVAS_SIZE, device="cpu")
        lt, _ = gen_fn(model)
        areas.extend(lt["area"].apply(float).tolist())
        x_positions.extend(lt["x_pos"].apply(float).tolist())
        y_positions.extend(lt["y_pos"].apply(float).tolist())
        counts.append(len(lt))
    return {
        "area": np.array(areas),
        "x_pos": np.array(x_positions),
        "y_pos": np.array(y_positions),
        "n_leaves": np.array(counts),
    }


N_REALIZATIONS = 200
KS_ALPHA = 0.01  # conservative threshold — see control test


@pytest.mark.parametrize("shape", ["circular"])
def test_distributional_equivalence(shape):
    """Independent-seed KS test: original vs quadtree distributions match."""
    params = SHAPE_CONFIGS[shape]

    orig_stats = _collect_leaf_stats(
        _generate_segmentation_original, shape, params, N_REALIZATIONS, 30000
    )
    qt_stats = _collect_leaf_stats(
        lambda m: m.generate_segmentation(), shape, params, N_REALIZATIONS, 70000
    )

    for key in ("area", "x_pos", "y_pos", "n_leaves"):
        ks_stat, p_value = stats.ks_2samp(orig_stats[key], qt_stats[key])
        assert p_value > KS_ALPHA, (  # pyright: ignore[reportOperatorIssue]
            f"KS test failed for {key}: stat={ks_stat:.4f}, p={p_value:.4f}"
        )


@pytest.mark.parametrize("shape", ["circular"])
def test_control_original_vs_original(shape):
    """Sanity check: two independent batches of the original should also pass."""
    params = SHAPE_CONFIGS[shape]

    stats_a = _collect_leaf_stats(
        _generate_segmentation_original, shape, params, N_REALIZATIONS, 10000
    )
    stats_b = _collect_leaf_stats(
        _generate_segmentation_original, shape, params, N_REALIZATIONS, 50000
    )

    for key in ("area", "x_pos", "y_pos", "n_leaves"):
        ks_stat, p_value = stats.ks_2samp(stats_a[key], stats_b[key])
        # This should pass — if it doesn't, the test setup is miscalibrated
        assert p_value > KS_ALPHA, (  # pyright: ignore[reportOperatorIssue]
            f"Control KS test failed for {key}: stat={ks_stat:.4f}, p={p_value:.4f}. "
            f"This suggests the test is miscalibrated, not a real difference."
        )
