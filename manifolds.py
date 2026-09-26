import torch


def m2_natural_frame(orientations: int, dx: float = 1.0) -> torch.Tensor:
    O = orientations
    dtheta = 2 * torch.pi / O
    theta = torch.arange(O) * dtheta                             # [O]
    cos, sin, zero = theta.cos(), theta.sin(), torch.zeros(O)    # [O]
    forward = torch.stack([zero, sin / dx, cos / dx], dim=-1)    # [O, n]
    sideways = torch.stack([zero, cos / dx, -sin / dx], dim=-1)  # [O, n]
    turn = torch.stack([zero + 1 / dtheta, zero, zero], dim=-1)  # [O, n]
    data = torch.stack([forward, sideways, turn], dim=-1)        # [O, n, n]
    return data.reshape(1, O, 1, 1, 3, 3)                        # [1, O, 1, 1, n, n]


def poincare_metric(size: int, max_radius: float = 0.98) -> torch.Tensor:
    step = 2 / (size - 1)
    y, x = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size),
                          indexing="ij")
    r2 = (x**2 + y**2).clamp(max=max_radius**2)
    scale = 2 * step / (1 - r2)                                        # [N, N]
    return scale.square().reshape(1, size, size, 1, 1) * torch.eye(2)  # [1, N, N, 2, 2]


def sphere_metric(n_theta: int, n_phi: int) -> torch.Tensor:
    theta = (torch.arange(n_theta) + 0.5) * torch.pi / n_theta
    step_theta, step_phi = torch.pi / n_theta, 2 * torch.pi / n_phi
    g_theta = torch.full((n_theta, n_phi), step_theta**2)                 # [Θ, Φ]
    g_phi = (theta.sin() * step_phi).square().reshape(n_theta, 1).expand(n_theta, n_phi)
    diagonal = torch.stack([g_theta, g_phi], dim=-1)                      # [Θ, Φ, 2]
    return torch.diag_embed(diagonal).reshape(1, n_theta, n_phi, 2, 2)    # [1, Θ, Φ, 2, 2]
