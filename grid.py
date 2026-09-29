import torch


def derivative(field: torch.Tensor, dims: list[int]) -> torch.Tensor:
    parts = [torch.zeros_like(field) if field.shape[d] == 1
             else torch.gradient(field, dim=d)[0]
             for d in dims]
    return torch.stack(parts, dim=-1)


def gaussian_blur(
    field: torch.Tensor, 
    sigma: float,
    dims: list[int]
) -> torch.Tensor:
    # The real FFT halves the last dim, so that one gets rfftfreq.
    sizes = [field.shape[d] for d in dims]
    frequency_squared = torch.zeros(field.ndim * [1], dtype=field.dtype, device=field.device)
    for d in dims:
        fftfreq = torch.fft.rfftfreq if d == dims[-1] else torch.fft.fftfreq
        axis = fftfreq(field.shape[d], dtype=field.dtype, device=field.device)
        shape = field.ndim * [1]
        shape[d] = len(axis)
        frequency_squared = frequency_squared + axis.reshape(shape).square()
    decay = torch.exp(-2 * (torch.pi * sigma) ** 2 * frequency_squared)
    return torch.fft.irfftn(torch.fft.rfftn(field, dim=dims) * decay, s=sizes, dim=dims)
