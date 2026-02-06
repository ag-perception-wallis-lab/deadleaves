import torch
from platform import system

__all__ = ["choose_compute_backend", "bounding_box"]


def choose_compute_backend() -> torch.device:
    """This function will automatically assign the available compute backend

    Returns:
        torch.device:
            A suitable PyTorch compute backend.
    """
    if system == "Darwin" and torch.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def bounding_box(partition: torch.Tensor, leaf_idx: int) -> tuple[int, int, int, int]:
    """Generate boundaries for bounding box for leaf index in partition

    Args:
        partition (torch.Tensor):
            Partition to use for bounding box
        leaf_idx (int):
            Index of relevant leaf

    Returns:
        tuple[int, int, int, int] | None:
            Top, left, bottom, right boundary indices of bounding box.
    """
    Y, X = (partition == leaf_idx).nonzero(as_tuple=True)

    if len(Y) == 0:
        raise ValueError(f"No elements match the required leaf index {leaf_idx}")

    top = int(Y.min().item())
    bottom = int(Y.max().item() + 1)
    left = int(X.min().item())
    right = int(X.max().item() + 1)

    return top, left, bottom, right
