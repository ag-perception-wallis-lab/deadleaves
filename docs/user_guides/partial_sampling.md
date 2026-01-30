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

# Partial sampling

The Dead Leaves Model allows controlling how many leaves are sampled and where they appear.

## Sparse sampling

Specifying the argument `n_sample` in the `LeafGeometryGenerator` limits the number of leaves to sample.
If the sampling stops before the entire image is filled, the resulting segmentation will contain empty pixels.

**Example**

```{code-cell}
:tags: [hide-input]
from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer
import torch

model = LeafGeometryGenerator(
    "circular", 
    {"area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}},
    (512,512),
    n_sample = 150
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color({"gray": {"uniform": {"low": 0.0, "high": 1.0}}})

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
image = renderer.render_image()
renderer.show(image, figsize = (3,3))
```

## Masking

Passing a `position_mask` to the `LeafGeometryGenerator` will exclude masked positions from the sampling process.
Any pixel where sampling is prohibited may remain empty in the segmentation.

The position mask may be passed as a boolean numpy array or torch tensor.
Alternatively, a dictionary can be passed which contains a `shape` key to select a leaf mask shape and a `params` key specifying the parameters for the respective shape of leaf mask. 

**Example**

```{code-cell}
:tags: [hide-input]
from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer

model = LeafGeometryGenerator(
    "circular", 
    {"area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}},
    (512,512),
    position_mask = {
        "shape": "circular",
        "params": {"area": 512*512*0.4}
    }
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color({"gray": {"uniform": {"low": 0.0, "high": 1.0}}})

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
image = renderer.render_image()
renderer.show(image, figsize = (3,3))
```

## Background color

Empty pixels resulting from sparse sampling or masked positions are filled with a RGB background color.
The background color can be specified explicitly via the `ImageRenderer` argument `background_color`, if not specified the background will be black.

**Example**

```{code-cell}
:tags: [hide-input]
from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer
import torch

model = LeafGeometryGenerator(
    "circular", 
    {"area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}},
    (512,512),
    n_sample = 150
)
leaf_table, segmentation_map = model.generate_segmentation()

colormodel = LeafAppearanceSampler(leaf_table)
colormodel.sample_color({"gray": {"uniform": {"low": 0.0, "high": 1.0}}})

renderer = ImageRenderer(colormodel.leaf_table, segmentation_map, background_color=torch.tensor([0.1,0.5,0.1]))
image = renderer.render_image()
renderer.show(image, figsize = (3,3))
```