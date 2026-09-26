import torch



def gaussian_blur(field: torch.Tensor, sigma: float,
                  dim: tuple[int, ...] | None = None) -> torch.Tensor:
    dims = tuple(range(field.ndim)) if dim is None else dim

    # Transfer function exp(-sigma^2 |omega|^2 / 2) with omega = 2 pi f, f from fftfreq.
    frequency_squared = torch.zeros(field.ndim * [1], dtype=field.dtype, device=field.device)
    for d in dims:
        shape = field.ndim * [1]
        shape[d] = field.shape[d]
        axis = torch.fft.fftfreq(field.shape[d], dtype=field.dtype, device=field.device)
        frequency_squared = frequency_squared + axis.reshape(shape).square()
    decay = torch.exp(-2 * (torch.pi * sigma) ** 2 * frequency_squared)
    return torch.fft.ifftn(torch.fft.fftn(field, dim=dims) * decay, dim=dims).real
