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
    frequency_squared = torch.zeros(field.ndim * [1], dtype=field.dtype, device=field.device)
    for d in dims:
        shape = field.ndim * [1]
        shape[d] = field.shape[d]
        axis = torch.fft.fftfreq(field.shape[d], dtype=field.dtype, device=field.device)
        frequency_squared = frequency_squared + axis.reshape(shape).square()
    decay = torch.exp(-2 * (torch.pi * sigma) ** 2 * frequency_squared)
    return torch.fft.ifftn(torch.fft.fftn(field, dim=dims) * decay, dim=dims).real
