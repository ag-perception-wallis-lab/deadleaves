# Installation

## Installing

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

```{note}
If possible we advise to setup [CUDA](https://developer.nvidia.com/cuda-downloads) for GPU use. On CPU sampling a single high-resolution image may take multiple minutes.
```