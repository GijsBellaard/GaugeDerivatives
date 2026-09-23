import torch

from field import Field


def gaussian_blur(field: Field, sigma: float) -> Field:
    """Isotropic Gaussian blur of every component along the spatial dimensions, via FFT.

    Boundaries are periodic, so values within a few sigma of the edge are wrong.

    Args:
        field: Field of type `BSI`.
        sigma: Standard deviation in grid steps.

    Returns:
        Blurred field of type `BSI`.
    """
    data = field.data
    dims = field.spatial_dims

    # Transfer function exp(-sigma^2 |omega|^2 / 2) with omega = 2 pi f, f from fftfreq.
    frequency_squared = torch.zeros(data.ndim * [1], dtype=data.dtype, device=data.device)
    for d in dims:
        shape = data.ndim * [1]
        shape[d] = data.shape[d]
        axis = torch.fft.fftfreq(data.shape[d], dtype=data.dtype, device=data.device)
        frequency_squared = frequency_squared + axis.reshape(shape).square()
    decay = torch.exp(-2 * (torch.pi * sigma) ** 2 * frequency_squared)
    blurred = torch.fft.ifftn(torch.fft.fftn(data, dim=dims) * decay, dim=dims).real
    return Field(blurred, field.type)
