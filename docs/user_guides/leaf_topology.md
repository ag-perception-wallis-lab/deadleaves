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

# Leaf topology and leaf table manipulations

As previously outlined, the deadleaves package separates  
(1) leaf geometry, (2) appearance, and (3) rendering into independent stages.

All generated scene information is stored in a pandas DataFrame called a
`leaf_table`. Each row represents one leaf and fully describes its geometry
(and optionally appearance).

The `LeafGeometryGenerator` normally produces both a table *and* a
segmentation map in one step.  
The `LeafTopology` class sits one level above this process: it treats the
leaf table as a manipulable scene description and lets you rebuild or modify
the segmentation without resampling the leaves.

This is useful whenever you want to **edit**, **combine**, or **reinterpret**
an existing scene instead of generating a new one from scratch.

With `LeafTopology` you can:

- regenerate segmentation maps from an edited table  
- build composite scenes by merging multiple independently generated leaf instances  
- change or randomize depth layering by reassigning leaf identities (`leaf_idx`)    
- apply motion or parametric transformations to existing leaves
- edit individual leaves


---

## The leaf table

A *leaf table* is a complete, structured description of a dead leaves scene.
Each row corresponds to a single leaf and contains all information needed to
reconstruct its geometry and appearance.

A typical row includes:

- a unique index (`leaf_idx`) that determines occlusion order / layering  
- the leaf shape (`leaf_shape`)  
- its position (`x_pos`, `y_pos`)  
- shape-specific geometric parameters  
  (e.g. area, orientation, aspect ratio, number of vertices)  
- optional color and texture attributes  

Because the table contains a full parametric description of every leaf, it is possible to fully *derive* the two-dimensional segmentation map from the leaf table by re-evaluating the geometry encoded in each row.

This makes the leaf table the canonical scene representation: editing the table and regenerating the segmentation produces a new, consistent dead leaves image without resampling the underlying leaves.

```{code-cell}
from dead_leaves import LeafGeometryGenerator

leaf_table, _ = LeafGeometryGenerator(
    leaf_shape = "circular", 
    shape_param_distributions = {
        "area": {"uniform": {"low": 1000, "high": 2500}},
        },
    image_shape = (512,512),
).generate_segmentation()


leaf_table.head()
```
---


## Import

```{code-cell}
from dead_leaves import LeafTopology
```

---

## (Re)Construct a segmentation map

A segmentation map can always be reconstructed from a leaf table. 

Rebuilding the segmentation is useful whenever the table changes or is reused
in a new context. For example, you might load a previously saved table from
disk, edit leaf parameters such as position or size, merge multiple scenes
into a composite structure, or generate a sequence of frames where leaves
move over time. In all of these cases, the geometry does not need to be
resampled — the segmentation is simply recomputed from the updated table.

This separation lets you treat the leaf table as an editable scene graph and
the segmentation as an implementation of that graph.


```{code-cell}
topology = LeafTopology(image_shape=(512,512))
segmentation_map = topology.segmentation_map_from_table(leaf_table)
```

---

## Merge leaf tables

Merging leaf tables allows you to combine multiple independently generated
dead leaves scenes into a single composite scene. The operation concatenates
all rows, preserves every geometric and appearance attribute, and assigns a
consistent `leaf_idx` so that the combined scene has a valid global
occlusion order.

This is useful when different parts of a scene should follow different
statistics or roles. For example, you might generate one table describing a
structured proto-object and another describing background clutter, then merge
them to create a scene where both regimes coexist. Because all attributes are
preserved, the merged table still contains the full parametric description of
every leaf and can immediately be turned back into a segmentation map.

See the proto-object example (link) for a complete workflow.


---

## Reassign leaf indices

Leaf indices (`leaf_idx`) determine occlusion order and therefore control
which leaves appear in front of others. Reassigning indices lets you modify
depth structure without changing any geometric or appearance parameters.
This is useful for generating control conditions, testing robustness to
layering, or reorganizing scenes while preserving semantic grouping.

Several index operations are available depending on how much structure you
want to keep.

**Global randomization** ignores all prior structure and shuffles depth across
the entire scene. All attributes are preserved, but layering becomes fully
random.

```{code-cell}
randomized = LeafTopology.randomize_index(leaf_table, seed=42)
randomized.head(10)
```

**Reindex by group** reindexes or shuffles depth locally within semantic groups, defined in one of the table columns. This preserves object-level structure but varies internal layering. Useful if you want a certain set of leaves in the foreground.


```{code-cell}
import numpy as np 

# add column for grouping
leaf_table["type"] = np.random.choice([1, 2], size=len(leaf_table), p=[0.7, 0.3])

shuffled = LeafTopology.reindex_by_group(leaf_table, groupby="type", shuffle=True, group_order="ascending", seed=42)
shuffled.head(10)
```

---

## Examples 

Look at these examples in the [Gallery](../gallery/index.rst) to browse the source code. 


**Example 1: A proto-object moving through clutter**

```{code-cell}
:tags: [remove-input]
import numpy as np
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


image_shape = (512, 731)
area = image_shape[0] * image_shape[1]

shared_params = {
    "leaf_shape": "circular",
    "shape_param_distributions": {
        "area": {"uniform": {"low": area * 0.005, "high": area * 0.01}}
    },
    "image_shape": image_shape,
}

table_background, _ = LeafGeometryGenerator(
    **shared_params,
    n_sample=200,
).generate_segmentation()

table_target, _ = LeafGeometryGenerator(
    **shared_params,
    n_sample=10,
    position_mask={"shape": "circular", "params": {"area": area * 0.05}},
).generate_segmentation()

table_background = LeafAppearanceSampler(leaf_table=table_background).sample_color(
    color_param_distributions={"gray": {"normal": {"loc": 0.5, "scale": 0.2}}},
)

table_target = LeafAppearanceSampler(leaf_table=table_target).sample_color(
    color_param_distributions={
        "R": {"uniform": {"low": 0.2, "high": 1.0}},
        "G": {"constant": {"value": 0.0}},
        "B": {"constant": {"value": 0.0}},
    },
)

table_background["type"] = "background"
table_target["type"] = "target"

topology = LeafTopology(image_shape=image_shape)

table_merged = topology.merge_leaf_tables(table_background, table_target)
table_merged = LeafTopology.randomize_index(table_merged, seed=42)


def move_leaves_one_step(
    leaf_table: pd.DataFrame,
    frame_idx: int,
    image_shape: tuple[int, int],
    radius: float = 5.0,
    target_velocity: tuple[float, float] = (1.0, 0.0),
    bg_angles: np.ndarray | None = None,
    bg_angular_velocities: np.ndarray | None = None,
) -> pd.DataFrame:
    table = leaf_table.copy()

    # Identify background and target leaves
    bg_mask = table["type"] != "target"
    target_mask = table["type"] == "target"

    n_bg = bg_mask.sum()
    bg_indices = table.index[bg_mask]

    # Initialize angles and angular velocities if not provided
    if bg_angles is None:
        bg_angles = np.random.uniform(0, 2 * np.pi, size=(n_bg,))
    if bg_angular_velocities is None:
        bg_angular_velocities = np.random.uniform(0.02, 0.05, size=(n_bg,))

    # Move background leaves
    for i, idx in enumerate(bg_indices):
        angle = bg_angles[i] + frame_idx * bg_angular_velocities[i]
        dx = radius * np.cos(angle)
        dy = radius * np.sin(angle)
        table.loc[idx, "x_pos"] = table.loc[idx, "x_pos"] + dx
        table.loc[idx, "y_pos"] = table.loc[idx, "y_pos"] + dy

    # Move target leaves linearly
    table.loc[target_mask, "x_pos"] = (
        leaf_table.loc[target_mask, "x_pos"] + frame_idx * target_velocity[0]
    ) % image_shape[1]
    table.loc[target_mask, "y_pos"] = (
        leaf_table.loc[target_mask, "y_pos"] + frame_idx * target_velocity[1]
    ) % image_shape[0]
    return table


frames = []
n_frames = 80
for t in range(n_frames):
    table = move_leaves_one_step(
        leaf_table=table_merged,
        image_shape=image_shape,
        frame_idx=t,
        radius=5.0,
        target_velocity=(image_shape[1] / n_frames, 0.0),
    )

    renderer = ImageRenderer(
        leaf_table=table,
        image_shape=image_shape,
        background_color=torch.tensor(1),
    )
    image = renderer.render_image()
    frames.append(image)

fig, ax = plt.subplots(figsize=(7.31, 5.12))
im = ax.imshow(frames[0])
ax.axis("off")
ax.set_position((0.0, 0.0, 1.0, 1.0))


def update(i):
    im.set_data(frames[i])
    return [im]


ani = animation.FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=100,
    blit=True,
)

plt.close(fig)
HTML(ani.to_jshtml())
```


**Example 2: Circular moving leaves**

```{code-cell}
:tags: [remove-input]
import pandas as pd
import numpy as np
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

image_shape = (512, 731)
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

topology = LeafTopology(image_shape=image_shape)


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

    # convert to numpy for vectorized math
    x = table["x_pos"].to_numpy(dtype=float) - cx
    y = table["y_pos"].to_numpy(dtype=float) - cy

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # update angle
    theta += angle_step
    if angular_jitter > 0:
        theta += np.random.normal(0.0, angular_jitter, size=len(theta))

    # update radius
    if radial_jitter > 0:
        r += np.random.normal(0.0, radial_jitter, size=len(r))
        r = np.clip(r, 0.0, None)

    # back to Cartesian
    table["x_pos"] = cx + r * np.cos(theta)
    table["y_pos"] = cy + r * np.sin(theta)

    return table


frames = []
for t in range(20):
    table = apply_circular_motion(
        leaf_table=table,
        image_size=segmentation_map.shape,
        angle_step=np.pi / 10,
        angular_jitter=0.1,
    )

    renderer = ImageRenderer(
        leaf_table=table,
        image_shape=image_shape,
        background_color=torch.tensor(1),
    )
    image = renderer.render_image()
    frames.append(image)


# Generate gif
fig, ax = plt.subplots(figsize=(7.31, 5.12))
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

---

## When to use LeafTopology

Use this class when you want to:

- combine multiple dead leaves models
- build hierarchical scenes
- create motion or animation by editing the leaf positions in the table
- changing or randomizing the layering occlusion structure
- generate certain experimental control conditions
- edit or relabel existing leaf tables

It turns the leaf table into a flexible scene graph rather than a fixed output.
