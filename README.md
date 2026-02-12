# DeadLeaves

<p align=center>
An open-source Python package for creating dead leaves images in a systematic, yet flexible way.
</p>

[![Tests](https://github.com/ag-perception-wallis-lab/deadleaves/actions/workflows/test.yml/badge.svg)](https://github.com/ag-perception-wallis-lab/deadleaves/actions/workflows/test.yml)
[![Py versions](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/deadleaves/badge/?version=latest)](https://deadleaves.readthedocs.io/en/latest/?badge=latest)

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

![](docs/_static/figures/examples.png)

## Installation

You can install this package directly from github through

```
pip install git+https://github.com/ag-perception-wallis-lab/deadleaves.git
```

## Dependencies

We recommend using a Python version `3.12` or newer.
The dependencies should be automatically installed (at least using `pip`). `deadleaves`s required dependencies are: 

- [PyTorch](https://pytorch.org/)
- [Torchquad](https://torchquad.readthedocs.io/en/main/)
- [Torchvision](https://docs.pytorch.org/vision/stable/index.html)
- [Pandas](https://pandas.pydata.org/)
