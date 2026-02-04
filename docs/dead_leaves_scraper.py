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
    animations = [v for v in example_globals.values() if isinstance(v, Animation)]
    renderers = [v for v in example_globals.values() if isinstance(v, ImageRenderer)]

    # --------------------
    # 1. Handle animations
    # --------------------

    if animations:
        ani = animations[0]
        image_path = Path(next(image_path_iterator)).with_suffix(".gif")
        # Save animation
        ani.save(image_path, writer="pillow", fps=10)
        rst.append(figure_rst([str(image_path)], gallery_conf["src_dir"]))

    # -----------------------
    # 2. Handle static images
    # -----------------------
    elif renderers:
        renderer = renderers[0]
        image_path = Path(next(image_path_iterator))

        if "image" in example_globals and isinstance(
            example_globals["image"], torch.Tensor
        ):
            image = example_globals["image"]
            renderer.save(image_path, image)
        else:
            renderer.save(image_path)

        rst.append(figure_rst([str(image_path)], gallery_conf["src_dir"]))

    return "\n".join(rst)
