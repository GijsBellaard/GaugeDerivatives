import torch

from derivative import grad, hessian
from gaussian_blur import gaussian_blur
from utils import normalize_dims


def gauge_frame_hessian(
    field: torch.Tensor,
    dims: list[int] = None,
) -> torch.Tensor:
    """Compute the local gauge frame of field, the eigenvectors of its Hessian.
    Define n = len(dims).

    Args:
        field: Tensor of arbitrary shape [...].
        dims: List containing the indices of the spatial dimensions of field, defaults to all dimensions.

    Returns:
        Tensor of shape [..., n, n] with 2 additional dimensions of size n
        containing the n n-dimensional eigenvectors of the Hessian of field. 
        The [..., :, i] is the i-th frame vector.
    """
    H = hessian(field, dims=dims)
    H = (H + H.transpose(-2, -1)) / 2  # Force H symmetric (just in case)
    eigenvalues, eigenvectors = torch.linalg.eigh(H)
    return eigenvectors

def gauge_frame_structure_tensor(
    field: torch.Tensor,
    sigma: float,
    dims: list[int] = None,
) -> torch.Tensor:
    """Compute the local gauge frame of field, the eigenvectors of its structure tensor.
    Define n = len(dims).
    
    Args:
        field: Tensor of arbitrary shape [...].
        sigma: Standard deviation for the post blur
        dims: List containing the indices of the spatial dimensions of field, defaults to all dimensions.

    Returns:
        Tensor of shape [..., n, n] with 2 additional dimensions of size n
        containing the n n-dimensional eigenvectors of the structure tensor of field.
        The [..., :, i] is the i-th frame vector.
    """
    dims = normalize_dims(field, dims)
    G = grad(field, dims=dims)
    G2 = G[..., :, None] @ G[..., None, :]
    G2 = gaussian_blur(G2, sigma=sigma, dims=dims)
    eigenvalues, eigenvectors = torch.linalg.eigh(G2)
    return eigenvectors