import torch
import string
from functools import partial

def covariant_derivative(
    field: torch.Tensor,
    k: int = 1,
    dims: list[int] = None,
) -> torch.Tensor:
    """Take k covariant derivatives of field along its spatial dimensions.
    Define n = len(dims).

    Args:
        field: Tensor of arbitrary shape [...].
        k: Number of derivatives to take. Defaults to 1.
        dims: List containing the indices of the spatial dimensions of field, defaults to all dimensions.

    Returns:
        Tensor of shape [..., n, ..., n] with k additional dimensions of size n.
    """
    if dims is None:
        dims = list(range(field.ndim))
    dims = [d % field.ndim for d in dims]

    B = field
    for _ in range(k):
        B = torch.stack(torch.gradient(B, dim=dims), dim=-1)
    return B

grad = partial(covariant_derivative, k=1)
hessian = partial(covariant_derivative, k=2)

def gauge_derivative(
    field: torch.Tensor,
    frame: torch.Tensor,
    signature: list[int],
    dims: list[int] = None,
) -> torch.Tensor:
    """Take the gauge derivative of scalar field `field` with frame `frame` and signature `signature`.
    Define n = len(dims) and k = len(signature).

    Args:
        field: Tensor of arbitrary shape [...].
        frame: Tensor of shape [..., n, n] specifying the gauge frame.
        signature: Signature of the gauge derivative, a list of k integers where each integer is in the range [0, n-1].
        dims: List containing the indices of the spatial dimensions of field, defaults to all dimensions.

    Returns:
        Tensor of shape [...].
    """
    k = len(signature)
    deriv = covariant_derivative(field, k=k, dims=dims) 
    letters = string.ascii_letters[:k]                               
    equation = ",".join(["..." + letters] + ["..." + c for c in letters]) + "->..."
    # print(equation)
    return torch.einsum(equation, deriv, *(frame[..., :, i] for i in signature))