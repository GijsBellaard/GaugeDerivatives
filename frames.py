import torch

from field import change_basis, spatial_dims
from gaussian_blur import gaussian_blur
from geometry import covariant_derivative, differential


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
    metric: torch.Tensor,
    dim: tuple[int, ...] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    df = differential(field, dim)
    structure_tensor = df[..., :, None] * df[..., None, :]
    dims = spatial_dims(field, "", dim)
    structure_tensor = gaussian_blur(structure_tensor, sigma, dims)
    return eigenframe(structure_tensor, metric)


def hessian_frame(
    field: torch.Tensor, 
    difference_tensor: torch.Tensor, 
    metric: torch.Tensor,
    dim: tuple[int, ...] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    dims = spatial_dims(field, "", dim)
    hessian = covariant_derivative(differential(field, dims), difference_tensor, "l", dims)
    return eigenframe(hessian, metric)


def covariant_derivative_in_frame(
    field: torch.Tensor, 
    frame: torch.Tensor,
    difference_tensor: torch.Tensor, 
    order: int,
    indices: str = "",
    dim: tuple[int, ...] | None = None
) -> torch.Tensor:
    dims = spatial_dims(field, indices, dim)
    for _ in range(order):
        field = covariant_derivative(field, difference_tensor, indices, dims)
        indices += "l"
    for index, kind in enumerate(indices):
        field = change_basis(field, frame, index - len(indices), kind, dims)
    return field


def constant_metric_in_frame(
    metric: torch.Tensor, 
    frame: torch.Tensor
) -> torch.Tensor:
    inverse = torch.linalg.inv(frame)
    return inverse.mT @ metric @ inverse
