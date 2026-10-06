import torch

from geometry import covariant_derivative, grad
from grid import derivative, gaussian_blur


def _whiten(
    form: torch.Tensor, 
    metric: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.broadcast_to(metric, form.shape)
    basis = torch.linalg.inv(torch.linalg.cholesky(g)).mT
    return basis, basis.mT @ form @ basis


def eigenframe(
    field: torch.Tensor, 
    metric: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # A self-adjoint field A lowers to the symmetric g A. Solve in a g-orthonormal basis E,
    # then map back with v = E w.
    basis, F = _whiten(metric @ field, metric)
    values, w = torch.linalg.eigh((F + F.mT) / 2)
    return values, basis @ w


def adjoint(
    field: torch.Tensor,
    metric: torch.Tensor
) -> torch.Tensor:
    # A* = g^-1 A^T g, so that g(A X, Y) = g(X, A* Y).
    return torch.linalg.inv(metric) @ field.mT @ metric


def _regularize(
    field: torch.Tensor,
    metric: torch.Tensor,
    sigma: float,
    dims: list[int]
) -> torch.Tensor:
    # Blur the lowered components g A, which stay symmetric, so the field stays self-adjoint.
    return torch.linalg.inv(metric) @ gaussian_blur(metric @ field, sigma, dims)


def structure_field(
    field: torch.Tensor,
    metric: torch.Tensor,
    sigma: float = 1.0
) -> torch.Tensor:
    # S f = grad f ⊗ df, regularized.
    spatial_dims = list(range(1, field.ndim))
    df = derivative(field, spatial_dims)
    S = torch.einsum("...i,...j->...ij", grad(field, metric), df)
    return _regularize(S, metric, sigma, spatial_dims)


def hessian_field(
    field: torch.Tensor,
    connection: torch.Tensor,
    metric: torch.Tensor,
    sigma: float = 1.0,
    side: str = "right"
) -> torch.Tensor:
    # (H f)* (H f) for the right singular frame, (H f) (H f)* for the left, regularized,
    # with H f = ∇ grad f.
    spatial_dims = list(range(1, field.ndim))
    H = covariant_derivative(grad(field, metric), connection, "u")
    if side == "right":
        A = adjoint(H, metric) @ H
    elif side == "left":
        A = H @ adjoint(H, metric)
    else:
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")
    return _regularize(A, metric, sigma, spatial_dims)
