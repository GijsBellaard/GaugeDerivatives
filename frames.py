import torch

from geometry import covariant_derivative
from grid import derivative, gaussian_blur


def _whiten(
    form: torch.Tensor, 
    metric: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.broadcast_to(metric, form.shape)
    basis = torch.linalg.inv(torch.linalg.cholesky(g)).mT
    return basis, basis.mT @ form @ basis


def eigenframe(
    form: torch.Tensor, 
    metric: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Solve in a g-orthonormal basis E, then map back with v = E w.
    basis, F = _whiten(form, metric)
    values, w = torch.linalg.eigh((F + F.mT) / 2)
    return values, basis @ w


def squared_form(
    form: torch.Tensor,
    metric: torch.Tensor,
    slot: int = 0
) -> torch.Tensor:
    if slot == 1:
        form = form.mT
    elif slot != 0:
        raise ValueError(f"slot must be 0 or 1, got {slot!r}")
    return form.mT @ torch.linalg.inv(metric) @ form


def structure_tensor(
    field: torch.Tensor,
    sigma: float = 1.0
) -> torch.Tensor:
    spatial_dims = list(range(1, field.ndim))
    df = derivative(field, spatial_dims)
    form = torch.einsum("...i,...j->...ij", df, df)
    return gaussian_blur(form, sigma, spatial_dims)


def hessian_structure_tensor(
    field: torch.Tensor,
    connection: torch.Tensor,
    metric: torch.Tensor,
    sigma: float = 1.0,
    slot: int = 0
) -> torch.Tensor:
    spatial_dims = list(range(1, field.ndim))
    df = derivative(field, spatial_dims)
    hessian = covariant_derivative(df, connection, "l")
    form = squared_form(hessian, metric, slot)
    return gaussian_blur(form, sigma, spatial_dims)
