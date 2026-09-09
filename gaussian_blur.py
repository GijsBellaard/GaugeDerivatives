import torch

from utils import normalize_dims


def gaussian_blur(
    field: torch.Tensor,
    sigma: float,
    dims: list[int] | None = None
) -> torch.Tensor:
    """Blur a tensor with an isotropic Gaussian of standard deviation sigma samples, using fft.

    Args:
        field: Tensor of arbitrary shape [...].
        sigma: Standard deviation of the Gaussian
        dims: List containing the indices of the spatial dimensions of field, defaults to all dimensions.

    Returns:
        Tensor of the same shape as field.
    """
    dims = normalize_dims(field, dims)

    freq2 = torch.zeros(field.ndim * [1], dtype=field.dtype, device=field.device)
    for d in dims:
        shape = field.ndim * [1]
        shape[d] = field.shape[d]
        freq = torch.fft.fftfreq(field.shape[d], dtype=field.dtype, device=field.device)
        freq2 = freq2 + freq.reshape(shape).square()
    decay = torch.exp(-2 * (torch.pi * sigma) ** 2 * freq2)
    return torch.fft.ifftn(torch.fft.fftn(field, dim=dims) * decay, dim=dims).real
