# DeadLeaves

<p align=center>
An open-source Python package for creating dead leaves images in a systematic, yet flexible way.
</p>

<p align=center>
<img src=docs/_static/logo_dead_leaves.png width=300>
</p>


## Core functionalities

- generating dead leaves images with properties (e.g. sizes, orientations, colors) drawn from a wide range of distributions (e.g. uniform, normal, Poisson, power-law, constant) or directly from an image.
- picking from various leaf shapes (circles, ellipsoids, rectangles, regular polygons).
- sampling in different color spaces (RGB, HSV, Gray-scale).
- applying different noise or image textures, either to the entire image or per-leaf.
- varying the image area covered by leaves, i.e. choosing between sparser or denser sampling and position mask.
- creating arbitrarily complex leaf configurations by adding dependencies between leaf features (e.g. space-dependent color gradients).

## Installation

You can install this package directly from gitlab through
```
pip install git+https://git.rwth-aachen.de/ag-perception-tuda/research/dead_leaves.git
```
This will prompt a username and password request.

This should work for Mac and Linux without any adjustments. For usage on Windows make sure you have the [**Git credential manager (GCM)**](github.com/git-ecosystem/git-credential-manager) setup as part of the git for Windows installation, otherwise the installation will fail without allowing you to put in credentials.

## Dependencies

We recommend using a Python version `3.12` or newer.
The dependencies should be automatically installed (at least using `pip`). `deadleaves`s required dependencies are: 

- [PyTorch](https://pytorch.org/)
- [Torchquad](https://torchquad.readthedocs.io/en/main/)
- [Torchvision](https://docs.pytorch.org/vision/stable/index.html)
- [Pandas](https://pandas.pydata.org/)
