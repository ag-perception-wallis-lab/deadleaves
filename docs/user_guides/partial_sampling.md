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

Specifying the argument `n_sample` in the `DeadLeavesModel` limits the number of leaves to sample.
If the sampling stops before the entire image is filled, the resulting partition will contain empty pixels.

**Example**

```{code-cell}
:tags: [hide-input]
from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    shape = "circular", 
    param_distributions = {"area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}},
    size = (512,512),
    n_sample = 150
)
leaves, partition = model.sample_partition()

colormodel = DeadLeavesImage(
    leaves = leaves, 
    partition = partition, 
    color_param_distributions = {"gray": {"uniform": {"low": 0.0, "high": 1.0}}}
    )
image = colormodel.sample_image()

colormodel.show(image, figsize = (3,3))
```

## Masking

Passing a `position_mask` to the `DeadLeavesModel` will exclude masked positions from the sampling process.
Any pixel where sampling is prohibited may remain empty in the partition.

**Example**

```{code-cell}
:tags: [hide-input]
from dead_leaves import DeadLeavesModel, DeadLeavesImage
from dead_leaves.leaf_masks import circular
import torch

X, Y = torch.meshgrid(
    torch.arange(512),
    torch.arange(512),
    indexing="xy",
)
position_mask = circular((X,Y), {"area": torch.tensor(80000), "x_pos": 256, "y_pos": 256})

model = DeadLeavesModel(
    shape = "circular", 
    param_distributions = {"area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}},
    size = (512,512),
    position_mask = position_mask
)
leaves, partition = model.sample_partition()

colormodel = DeadLeavesImage(
    leaves = leaves, 
    partition = partition, 
    color_param_distributions = {"gray": {"uniform": {"low": 0.0, "high": 1.0}}}
    )
image = colormodel.sample_image()

colormodel.show(image, figsize = (3,3))
```

## Background color

Empty pixels resulting from sparse sampling or masked positions are filled with a background color (RGB).
The background color can be specified explicitly via the `DeadLeavesImage` argument `background_color`, if not specified the background will be black.

**Example**

```{code-cell}
:tags: [hide-input]
from dead_leaves import DeadLeavesModel, DeadLeavesImage

model = DeadLeavesModel(
    shape = "circular", 
    param_distributions = {"area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}},
    size = (512,512),
    n_sample = 150
)
leaves, partition = model.sample_partition()

colormodel = DeadLeavesImage(
    leaves = leaves, 
    partition = partition, 
    color_param_distributions = {"gray": {"uniform": {"low": 0.0, "high": 1.0}}},
    background_color = torch.tensor([0.5,0.5,0.5])
    )
image = colormodel.sample_image()

colormodel.show(image, figsize = (3,3))
```