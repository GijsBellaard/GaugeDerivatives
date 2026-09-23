import torch

from field import Field


def left_invariant_frame(shape: tuple[int, int, int], dx: float = 1.0) -> Field:
    """Left-invariant frame of position orientation space M2 = R^2 x S^1.
    Grid over (θ, y, x) with step 2π / shape[0] in θ, starting at θ = 0, and dx in y and x.

    A_1 = cos θ ∂_x + sin θ ∂_y (forward), 
    A_2 = -sin θ ∂_x + cos θ ∂_y (sideways), and
    A_3 = ∂_θ (turn), 

    Args:
        shape: Number of grid points along θ, y and x.
        dx: Grid step in y and x.

    Returns:
        Frame of type `sssul` in the grid basis, where [..., :, i] is A_{i+1}.
    """
    O, H, W = shape
    dtheta = 2 * torch.pi / O
    theta = torch.arange(O) * dtheta                             # [O]
    cos, sin, zero = theta.cos(), theta.sin(), torch.zeros(O)    # [O]
    forward = torch.stack([zero, sin / dx, cos / dx], dim=-1)    # [O, n]
    sideways = torch.stack([zero, cos / dx, -sin / dx], dim=-1)  # [O, n]
    turn = torch.stack([zero + 1 / dtheta, zero, zero], dim=-1)  # [O, n]
    data = torch.stack([forward, sideways, turn], dim=-1)        # [O, n, n]
    data = data[:, None, None].repeat(1, H, W, 1, 1)             # [O, H, W, n, n]
    return Field(data, "sssul")
