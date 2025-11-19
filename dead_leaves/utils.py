import torch
from platform import system

__all__ = ["choose_compute_backend"]


def choose_compute_backend() -> torch.device:
    """This function will automatically assign the available compute backend

    Returns:
        torch.device: A suitable PyTorch compute backend.
    """
    if system == "Darwin" and torch.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def bounding_box(
    partition: torch.Tensor, leaf_idx: int
) -> tuple[int, int, int, int] | None:
    """Generate boundaries for bounding box for leaf index in partition

    Args:
        partition (torch.Tensor): Partition to use for bounding box
        leaf_idx (int): Index of relevant leaf

    Returns:
        tuple[int, int, int, int] | None: Boundary indices of bounding box.
    """
    Y, X = (partition == leaf_idx).nonzero(as_tuple=True)

    if len(Y) == 0:
        return None

    top = Y.min().item()
    bottom = Y.max().item() + 1
    left = X.min().item()
    right = X.max().item() + 1

    return top, left, bottom, right
