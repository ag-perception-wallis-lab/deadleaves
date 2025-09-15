# Dead Leaves

This package was build to generate dead leaves images in a simple fashion.
If a GPU is available and CUDA is set up, the tool will use the GPU for faster processing.

## Installation

You can install this package directly from gitlab through
```
pip install git+https://git.rwth-aachen.de/ag-perception-tuda/research/dead_leaves.git
```
This will prompt a username and password request.

This should work for Mac and Linux without any adjustments. For usage on Windows make sure you have the [**Git credential manager (GCM)**](github.com/git-ecosystem/git-credential-manager) setup as part of the git for Windows installation, otherwise the installation will fail without allowing you to put in credentials.

## Generating images

The package provides two main classes `DeadLeavesModel` and `DeadLeavesImage`. The first is used to define a model to sample a dead leaves partition from and the second employs a color and texture model on top of the partition.

Check out the demo in `examples` to see how it works.