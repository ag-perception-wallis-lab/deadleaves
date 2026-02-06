---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Leaf table manipulations

This section focusses on *working with existing scenes* rather than generating new ones.

In `dead_leaves`, each scene is represented by a *leaf table*: a pandas DataFrame where each row contains a single leaf's geometry parameters (and optionally appearance).
The `LeafTopology` class treats this table as a mutable scene description, allowing you ti rebuild or modify segmentation maps directly from the stored parameters - without resampling leaves.

`LeafTopology` is designed for advanced workflows where scenes need to be edited, combined, or reinterpreted after generation,
Typical use cases include merging independently generated scenes, changing occlusion order by reassigning `leaf_idx`, applying transformations or motion to existing leaves, and editing or relabeling individual leaves.

## (Re)Construct a segmentation map

A segmentation map can always be reconstructed from a leaf table. 

Rebuilding the segmentation is useful whenever the table changes or is reused in a new context. 
For example, you might load a previously saved table from disk, edit leaf parameters such as position or size, merge multiple scenes into a composite structure, or generate a sequence of frames where leaves move over time. In all of these cases, the geometry does not need to be resampled — the segmentation is simply recomputed from the updated table.

This separation lets you treat the leaf table as an editable scene graph and the segmentation as an implementation of that graph.

```python
topology = LeafTopology(image_shape=(512,512))
segmentation_map = topology.segmentation_map_from_table(leaf_table)
```

This feature is in particular used if a leaf table is passed to the `ImageRenderer` without a segmentation map.

**Example: Circular motion**

```{code-cell}
:tags: [hide-input]
import pandas as pd
import torch

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

from dead_leaves import (
    LeafGeometryGenerator,
    LeafAppearanceSampler,
    ImageRenderer,
    LeafTopology,
)

image_shape = (512, 512)
area = image_shape[0] * image_shape[1]

geometry = LeafGeometryGenerator(
    leaf_shape="circular",
    shape_param_distributions={
        "area": {"uniform": {"low": area * 0.005, "high": area * 0.01}},
    },
    image_shape=image_shape,
    n_sample=250,
    position_mask={
        "shape": "circular",
        "params": {
            "area": area * 0.35,
        },
    },
)

leaf_table, segmentation_map = geometry.generate_segmentation()

color_params = {
    "H": {"uniform": {"low": 0.0, "high": 0.2}},
    "S": {"uniform": {"low": 0.0, "high": 1.0}},
    "V": {"uniform": {"low": 0.0, "high": 1.0}},
}

appearance = LeafAppearanceSampler(
    leaf_table=leaf_table,
)

table = appearance.sample_color(color_param_distributions=color_params)

def apply_circular_motion(
    leaf_table: pd.DataFrame,
    image_size: tuple[int, int] | torch.Size,
    angle_step: float,
    angular_jitter: float = 0.0,
    radial_jitter: float = 0.0,
) -> pd.DataFrame:

    table = leaf_table.copy()

    cy = image_size[0] / 2
    cx = image_size[1] / 2

    x = torch.tensor(table["x_pos"]) - cx
    y = torch.tensor(table["y_pos"]) - cy

    # convert to Polar coordinates
    r = torch.sqrt(x**2 + y**2)
    theta = torch.arctan2(y, x)

    # update angle
    theta += angle_step
    if angular_jitter > 0:
        theta += torch.distributions.normal.Normal(0.0, angular_jitter).sample((len(theta),))

    # update radius
    if radial_jitter > 0:
        r += torch.distributions.normal.Normal(0.0, radial_jitter).sample((len(r),))
        r = torch.clip(r, 0.0, None)

    # back to Cartesian coordinates
    table["x_pos"] = cx + r * torch.cos(theta)
    table["y_pos"] = cy + r * torch.sin(theta)

    return table


frames = []
for t in range(20):
    table = apply_circular_motion(
        leaf_table=table,
        image_size=segmentation_map.shape,
        angle_step=torch.pi / 10,
        angular_jitter=0.1,
    )

    renderer = ImageRenderer(
        leaf_table=table,
        image_shape=image_shape,
        background_color=torch.tensor(1.0),
    )
    image = renderer.render_image()
    frames.append(image.cpu())


fig, ax = plt.subplots(figsize=(3, 3))
im = ax.imshow(frames[0])
ax.axis("off")
ax.set_position((0.0, 0.0, 1.0, 1.0))


def update(i):
    im.set_data(frames[i])
    return [im]


ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=100,
    blit=True,
)

plt.close(fig)
HTML(ani.to_jshtml())
```

## Merge leaf tables

Merging leaf tables allows you to combine multiple independently generated dead leaves scenes into a single composite scene. 
The operation concatenates all rows, preserves every geometric and appearance attribute, and assigns a consistent `leaf_idx` so that the combined scene has a valid global occlusion order.

This is useful when different parts of a scene should follow different statistics or roles. 
For example, you might generate one table describing a structured proto-object and another describing background clutter, then merge them to create a scene where both regimes coexist. 
Because all attributes are preserved, the merged table still contains the full parametric description of every leaf and can immediately be turned back into a segmentation map.

**Example: Merging images with different appearances**

```{code-cell}
:tags: [hide-input]
import pandas as pd
import matplotlib.pyplot as plt

from dead_leaves import (
    LeafGeometryGenerator,
    LeafAppearanceSampler,
    ImageRenderer,
    LeafTopology,
)

image_shape = (512, 512)
area = image_shape[0] * image_shape[1]

geometry = LeafGeometryGenerator(
    leaf_shape="circular",
    shape_param_distributions={
        "area": {"uniform": {"low": area * 0.005, "high": area * 0.01}},
    },
    image_shape=image_shape,
)

leaf_table, segmentation_map = geometry.generate_segmentation()

color_params_green = {
    "H": {"normal": {"loc": 0.3, "scale": 0.01}},
    "S": {"normal": {"loc": 0.6, "scale": 0.1}},
    "V": {"normal": {"loc": 0.6, "scale": 0.1}},
}

color_params_blue = {
    "H": {"normal": {"loc": 0.6, "scale": 0.01}},
    "S": {"normal": {"loc": 0.6, "scale": 0.1}},
    "V": {"normal": {"loc": 0.6, "scale": 0.1}},
}

colormodel_green = LeafAppearanceSampler(leaf_table)
colormodel_green.sample_color(color_params_green)

colormodel_blue = LeafAppearanceSampler(leaf_table)
colormodel_blue.sample_color(color_params_blue)

merged_leaf_table = colormodel_blue.leaf_table.copy()
merged_leaf_table.iloc[1::2] = colormodel_green.leaf_table.iloc[1::2]

renderer_blue = ImageRenderer(colormodel_blue.leaf_table, segmentation_map)
renderer_green = ImageRenderer(colormodel_green.leaf_table, segmentation_map)
renderer_merged = ImageRenderer(merged_leaf_table, segmentation_map)
renderer_blue.render_image()
renderer_green.render_image()
renderer_merged.render_image()

fig,ax = plt.subplots(1,3, facecolor='none', frameon=False)
ax[0].imshow(renderer_blue.image.cpu(), vmax=1, vmin=0)
ax[1].imshow(renderer_green.image.cpu(), vmax=1, vmin=0)
ax[2].imshow(renderer_merged.image.cpu(), vmax=1, vmin=0)

for a in ax:
    a.axis('off')
    
fig.patch.set_facecolor(None)
fig.tight_layout()
plt.show()
```

## Reassign leaf indices

Leaf indices (`leaf_idx`) determine occlusion order and therefore control which leaves appear in front of others. 
Reassigning indices lets you modify depth structure without changing any geometric or appearance parameters.
This is useful for generating control conditions, testing robustness to layering, or reorganizing scenes while preserving semantic grouping.

Several index operations are available depending on how much structure you want to keep.

### Global randomization
The method ignores all prior structure and shuffles depth across the entire scene.
All attributes are preserved, but layering becomes fully random.

```python
randomized_table = LeafTopology.randomize_index(leaf_table, seed=42)
```

**Example: Shuffle leaf depth**

```{code-cell}
:tags: [hide-input]
image_shape = (512, 512)
area = image_shape[0] * image_shape[1]

geometry = LeafGeometryGenerator(
    leaf_shape="circular",
    shape_param_distributions={
        "area": {"uniform": {"low": area * 0.005, "high": area * 0.01}},
    },
    image_shape=image_shape,
)

leaf_table, segmentation_map = geometry.generate_segmentation()

n_leaves = len(leaf_table)

color_params = {
    "gray": {"constant": {"value": {"from": "leaf_idx", "fn": lambda x: x/n_leaves}}},
}


colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color(color_params)

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()

randomized_table = LeafTopology.randomize_index(colormodel.leaf_table)
renderer_randomized = ImageRenderer(randomized_table, image_shape=image_shape)
renderer_randomized.render_image()

fig,ax = plt.subplots(1,3, facecolor='none', frameon=False, gridspec_kw={'width_ratios': [1, 7, 7]})
tmp = ax[1].imshow(renderer.image.cpu(), cmap='binary_r', vmax=1, vmin=0)
ax[2].imshow(renderer_randomized.image.cpu(), vmax=1, vmin=0)

for a in ax[1:3]:
     a.axis('off')
    
pos_img = ax[1].get_position()
new_pos = [ax[0].get_position().x0, pos_img.y0, ax[0].get_position().width, pos_img.height]
ax[0].set_position(new_pos)
plt.colorbar(tmp, cax=ax[0])
ax[0].set_ylabel('Original depth')
ax[0].yaxis.set_ticks_position('left')
ax[0].yaxis.set_label_position('left')
fig.patch.set_facecolor(None)
plt.show()
```

### Reindex by group
This method reindexes or shuffles depth locally within semantic groups, defined in one of the table columns. 
This preserves object-level structure but varies internal layering. 
Useful if you want a certain set of leaves in the foreground.


```python
shuffled = LeafTopology.reindex_by_group(
    leaf_table, 
    groupby="type", 
    shuffle=True, 
    group_order="ascending", 
    seed=42
    )
```

**Example: Order leaves by shape**

```{code-cell}
:tags: [hide-input]
image_shape = (512, 512)
area = image_shape[0] * image_shape[1]

geometry = LeafGeometryGenerator(
    leaf_shape="polygon",
    shape_param_distributions={
        "area": {"uniform": {"low": area * 0.005, "high": area * 0.01}},
        "n_vertices": {"poisson": {"rate": 5}}
    },
    image_shape=image_shape,
)

leaf_table, segmentation_map = geometry.generate_segmentation()

color_params = {
    "gray": {"uniform": {"low": 0.0, "high": 1.0}},
}


colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color(color_params)

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()

leaf_table_by_shape = LeafTopology.reindex_by_group(colormodel.leaf_table, groupby="n_vertices", shuffle=True, group_order="descending")
renderer_sorted = ImageRenderer(leaf_table_by_shape, image_shape=image_shape)
renderer_sorted.render_image()

fig,ax = plt.subplots(1,2, facecolor='none', frameon=False)
ax[0].imshow(renderer.image.cpu(), vmax=1, vmin=0)
ax[1].imshow(renderer_sorted.image.cpu(), vmax=1, vmin=0)

for a in ax:
      a.axis('off')
    
fig.patch.set_facecolor(None)
fig.tight_layout()
plt.show()
```