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


def singular_frames(
    form: torch.Tensor,
    metric: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Solve in a g-orthonormal basis E, then map back with v = E w.
    basis, F = _whiten(form, metric)
    u, sigma, vh = torch.linalg.svd(F)
    sigma = sigma.flip(-1)
    left = basis @ u.flip(-1)
    right = basis @ vh.mT.flip(-1)
    return sigma, left, right


def structure_tensor_frame(
    field: torch.Tensor, 
    sigma: float, 
    metric: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_dims = list(range(1, field.ndim))
    df = derivative(field, spatial_dims)
    structure_tensor = df[..., :, None] * df[..., None, :]
    structure_tensor = gaussian_blur(structure_tensor, sigma, spatial_dims)
    return eigenframe(structure_tensor, metric)


def hessian_frame(
    field: torch.Tensor, 
    difference_tensor: torch.Tensor, 
    metric: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_dims = list(range(1, field.ndim))
    df = derivative(field, spatial_dims)
    hessian = covariant_derivative(df, difference_tensor, "l")
    return eigenframe(hessian, metric)
