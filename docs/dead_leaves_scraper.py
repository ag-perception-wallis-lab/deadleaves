from pathlib import Path
import torch
from dead_leaves import DeadLeavesImage
from sphinx_gallery.scrapers import figure_rst


def dead_leaves_scraper(block, block_vars: dict, gallery_conf: dict):
    """
    Sphinx-Gallery image scraper for DeadLeaves images.

    Finds (DeadLeavesImage, torch.Tensor) pairs in the example globals
    and saves them using DeadLeavesImage.save().
    """
    image_path_iterator = block_vars["image_path_iterator"]
    example_globals = block_vars["example_globals"]

    # Find all DeadLeavesImage instances
    models = [v for v in example_globals.values() if isinstance(v, DeadLeavesImage)]

    if not models:
        return ""

    model = models[0]

    # Find tensor named 'image'
    if "image" in example_globals and isinstance(
        example_globals["image"], torch.Tensor
    ):
        image = example_globals["image"]
    else:
        tensors = [v for v in example_globals.values() if isinstance(v, torch.Tensor)]
        if not tensors:
            return ""
        image = tensors[-1]

    # Pair them in order of appearance
    image_path = Path(next(image_path_iterator))
    model.save(image, image_path)

    return figure_rst([str(image_path)], gallery_conf["src_dir"])
