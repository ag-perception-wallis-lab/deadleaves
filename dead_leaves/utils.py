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
