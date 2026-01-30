from pathlib import Path
import torch
from dead_leaves import ImageRenderer
from sphinx_gallery.scrapers import figure_rst

from matplotlib.animation import Animation


def dead_leaves_scraper(block, block_vars: dict, gallery_conf: dict):
    """
    Sphinx-Gallery image & animation scraper for DeadLeaves.

    - Static images: Finds (ImageRenderer, torch.Tensor) pairs in the example globals
    and saves them using ImageRenderer.save().
    - Animations: Finds matplotlib Animations and saves them as GIF
    """
    image_path_iterator = block_vars["image_path_iterator"]
    example_globals = block_vars["example_globals"]

    rst = []

    # --------------------
    # 1. Handle animations
    # --------------------
    animations = [v for v in example_globals.values() if isinstance(v, Animation)]

    for ani in animations:
        image_path = Path(next(image_path_iterator)).with_suffix(".gif")
        # Save animation
        ani.save(image_path, writer="pillow", fps=10)
        rst.append(figure_rst([str(image_path)], gallery_conf["src_dir"]))

    # -----------------------
    # 2. Handle static images
    # -----------------------
    renderers = [v for v in example_globals.values() if isinstance(v, ImageRenderer)]

    if renderers:
        renderer = renderers[0]

        if "image" in example_globals and isinstance(
            example_globals["image"], torch.Tensor
        ):
            image = example_globals["image"]
        else:
            tensors = [
                v for v in example_globals.values() if isinstance(v, torch.Tensor)
            ]
            if tensors:
                image = tensors[-1]
            else:
                image = None

        if image is not None:
            image_path = Path(next(image_path_iterator))
            renderer.save(image, image_path)

            rst.append(figure_rst([str(image_path)], gallery_conf["src_dir"]))

    return "\n".join(rst)
