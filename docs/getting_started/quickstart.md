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

# Quickstart

This package generates synthetic images using the Dead Leaves Model, a process in which random shapes (*leaves*) are repeatedly layered until the canvas is fully covered.

The workflow consists of three steps:

1. **Geometry**: A `LeafGeometryGenerator` defines the geometry of the leaves, e.g. shape, size, orientation.
2. **Appearance**: The `LeafAppearanceSampler` defines the surfaces of the leaves, i.e. color and texture specifications.
3. **Rendering**: The `ImageRenderer` is used to generate the final image.

This Quickstart walks you through the basic usage.

## Import

Begin by importing the two core classes:

```{code-cell}
from dead_leaves import LeafGeometryGenerator, LeafAppearanceSampler, ImageRenderer
```

## Setting up a Model

The `LeafGeometryGenerator` controls only the **geometry** of the leaves and the **canvas size**.
You choose a leaf shape (circles, ellipses, rectangles, or regular polygons) and assign sampling distributions to that shape’s parameters.
Leaf positions are sampled uniformly on the canvas.

For example, circles require only one parameter: `area`.
Here we use a power-law distribution for the area:

```{code-cell}
model = LeafGeometryGenerator(
    leaf_shape = "circular", 
    shape_param_distributions = {"area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}},
    image_shape = (512,512)
)
```

To generate a sample, use `.generate_segmentation()`.
This will trigger a sampling process where in each iteration a sample for each leaf parameter is drawn (in our example the parameters are `x_pos`, `y_pos`, and `area`).
The parameters samples are then used to generate a leaf mask which is placed on top of the canvas and each pixel which does not jet belong to a leaf will be assigned to the current leaf through a `leaf_idx`, i.e. we add leaves to the canvas from front to back with occlusion.
This process is repeated until all pixels belong to an object.
After the sampling process we receive two outputs:
- a **pandas DataFrame** describing all generated leaves (location + shape parameters), and
- a **segmentation map** assigning each pixel to the leaf on top at that location.

```{code-cell}
leaf_table, segmentation_map = model.generate_segmentation()
leaf_table.head()
```

```{code-cell}
:tags: [remove-input]
import matplotlib.pyplot as plt

plt.imshow(segmentation_map.cpu(), cmap='terrain')
plt.axis('off')
plt.colorbar(label = "Leaf index")
plt.show()
```

Each shape type has required parameters, and each of those parameters must be assigned a sampling distribution.

```{tip}
Since the sampling process is random you will generate a new segmentation map each time the command `.generate_segmentation()` is called.
For reproducibility you may set a seed with `torch.manual_seed()`.
```

## Using distributions

You can specify parameter distributions using dictionaries.
The package includes common PyTorch distributions and several custom ones:

| **Name** | **Required keys** | **Notes** |
|--|--|--|
| `"uniform"` | `low`, `high` | |
| `"normal"` | `loc`, `scale` | |
| `"beta"` | `concentration0`, `concentration1` | |
| `"poisson"` | `rate` | |
| `"powerlaw"` | `low`, `high`, `k` | |
| `"cosine"` | `amplitude`, `frequency` | |
| `"expcosine"` | `frequency`, `exponential_constant` | |
| `"image"` | `dir` | |
| `"constant"` | `value` | deterministic |

Each parameter is defined by a dictionary of the form:

```
{"distribution_name": { ... parameters ...}}
```

## Define leaf appearance

The `LeafAppearanceSampler` is used to sample a colors and optionally textures for each leaf in the leaf_table.
To trigger sampling you pass the color mode and the color distributions to the `.sample_color()` method.
This will draw a sample from your color distribution for each leaf index and save the value to the `leaf_table`.

```{code-cell}
colormodel = LeafAppearanceSampler(leaf_table)
leaf_table_w_o_texture = colormodel.sample_color(color_param_distributions = {"gray": {"normal": {"loc": 0.5, "scale": 0.2}}})
leaf_table_w_o_texture.head()
```

### Adding texture (optional)

You can optionally add texture on top of the leaf colors.
The simplest example uses pixel-wise Gaussian noise:

```{code-cell}
leaf_table_w_texture = colormodel.sample_texture(texture_param_distributions = {"gray": {"normal": {"loc": 0, "scale": {"uniform": {"low": 0.01, "high": 0.05}}}}})
leaf_table_w_texture.head()
```

## Rendering the image

To render our sampled image we pass the `leaf_table` and `segmentation_map` to the `ImageRenderer`.
If no `segmentation_map` is provided the segmentation will be regenerated based on the leaf table.
In this case an `image_shape` has to be given.

The rendering is performed by assigning each pixel in the canvas the color value of the leaf it belongs to.
We can render the dead leaves image without texture by using our leaf table without texture.

```{code-cell}
renderer = ImageRenderer(leaf_table_w_o_texture, segmentation_map)
renderer.render_image()
renderer.show()
```

To render the image with texture we use the leaf table which also contains texture information.
Since the texture adds pixelwise information we only resolve the texture at rendering time, such that the specific noise pattern will be resampled each time we render.

```{code-cell}
renderer = ImageRenderer(leaf_table_w_texture, segmentation_map)
renderer.render_image()
renderer.show()
```

```{note}
Since we added the texture to our existing model the two above images only differ in the leaf texture.
```

## Full example

Putting everything together, a complete workflow looks like this:

```{code-cell}
# Define geometric model
model = LeafGeometryGenerator(
    leaf_shape = "circular", 
    shape_param_distributions = {"area": {"powerlaw": {"low": 100.0, "high": 10000.0, "k": 1.5}}},
    image_shape = (512,512)
)

# Sample segmentation of canvas
leaf_table, segmentation_map = model.generate_segmentation()

# Define appearance sample
colormodel = LeafAppearanceSampler(leaf_table)

# Sample colors and textures
colormodel.sample_color(color_param_distributions = {"gray": {"normal": {"loc": 0.5, "scale": 0.2}}})
colormodel.sample_texture(texture_param_distributions = {"gray": {"normal": {"loc": 0, "scale": 0.05}}})

# Display the result
renderer = ImageRenderer(colormodel.leaf_table, segmentation_map)
renderer.render_image()
renderer.show()
```

```{note}
Since the rendering process simply adds the color (and texture) based on the partition to the canvas the resulting dead leaves image has a pixel-perfect rendering and a know segmentation map. 
```

## Next steps

See the documentation on [shape parameters](../user_guides/shapes.md).  
Learn more about [color models](../user_guides/colors.md).  
Learn more about [texture models](../user_guides/textures.md).  
Browse examples in the [Gallery](../gallery/index.rst) for example scripts.  
Consult the full list of available [distributions](../user_guides/distributions.md).  