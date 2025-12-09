# Partial sampling

The Dead Leaves Model allows controlling how many leaves are sampled and where they appear.

## Sparse sampling

Specifying the argument `n_sample` in the `DeadLeavesModel` limits the number of leaves to sample.
If the sampling stops before the entire image is filled, the resulting partition will contain empty pixels.

## Masking

Passing a `position_mask` to the `DeadLeavesModel` will exclude masked positions from the sampling process.
Any pixel where sampling is prohibited may remain empty in the partition.

## Background color

Empty pixels resulting from sparse sampling or masked positions are filled with a background color.
The background color can be specified explicitly via the `DeadLeavesImage` argument `background_color`, if not specified the background will be black.
