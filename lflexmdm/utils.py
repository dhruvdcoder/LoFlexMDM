import torch
def bregman_divergence(b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Compute the Bregman divergence between two tensors."""
    eps = 1e-9
    a = torch.clamp(a, min=eps)
    b = torch.clamp(b, min=eps)
    return a - b + b * (torch.log(b) - torch.log(a))