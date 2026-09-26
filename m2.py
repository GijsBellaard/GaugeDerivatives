import torch



def left_invariant_frame(orientations: int, dx: float = 1.0) -> torch.Tensor:
    O = orientations
    dtheta = 2 * torch.pi / O
    theta = torch.arange(O) * dtheta                             # [O]
    cos, sin, zero = theta.cos(), theta.sin(), torch.zeros(O)    # [O]
    forward = torch.stack([zero, sin / dx, cos / dx], dim=-1)    # [O, n]
    sideways = torch.stack([zero, cos / dx, -sin / dx], dim=-1)  # [O, n]
    turn = torch.stack([zero + 1 / dtheta, zero, zero], dim=-1)  # [O, n]
    data = torch.stack([forward, sideways, turn], dim=-1)        # [O, n, n]
    return data.reshape(O, 1, 1, 3, 3)                           # [O, 1, 1, n, n]
