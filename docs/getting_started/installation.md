# Installation

## Installing

You can install this package directly from gitlab through

```
pip install git+https://git.rwth-aachen.de/ag-perception-tuda/research/dead_leaves.git
```

This will prompt a username and password request.

This should work for Mac and Linux without any adjustments. For usage on Windows make sure you have the [**Git credential manager (GCM)**](https://github.com/git-ecosystem/git-credential-manager) setup as part of the git for Windows installation, otherwise the installation will fail without allowing you to put in credentials.

## Dependencies

The dependencies should be automatically installed (at least using `pip`). `dead_leaves`s required dependencies are: 

- [PyTorch](https://pytorch.org/)
- [Torchquad](https://torchquad.readthedocs.io/en/main/)
- [Torchvision](https://docs.pytorch.org/vision/stable/index.html)
- [Pandas](https://pandas.pydata.org/)

```{note}
If possible we advise to setup [CUDA](https://developer.nvidia.com/cuda-downloads) for GPU use. On CPU sampling a single high-resolution image may take multiple minutes.
```