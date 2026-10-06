import math

import torch


def blurred_circular_step(
    angle: torch.Tensor,
    half_width: float,
    sigma: float,
    windings: int = 1
) -> torch.Tensor:
    # Consider the indicator function of (-half_width, half_width)
    # on the circle, i.e. periodic with period 2π.
    # Then run the heat equation on this domain, truncated to the nearest windings.
    scale = math.sqrt(2) * sigma  # erf integrates e^(-s²), a Gaussian of variance 1/2
    return sum(torch.erf((half_width - angle - 2 * torch.pi * n) / scale)
               + torch.erf((half_width + angle + 2 * torch.pi * n) / scale)
               for n in range(-windings, windings + 1)) / 2


def blurred_half_line_step(
    radius: torch.Tensor,
    cutoff: float,
    sigma: float
) -> torch.Tensor:
    # Consider the indicator function [0, cutoff)
    # on the positive half-line, with zero flux at the origin. 
    # Then run the heat equation on this domain
    scale = math.sqrt(2) * sigma  # erf integrates e^(-s²), a Gaussian of variance 1/2
    return (torch.erf((cutoff - radius) / scale) + torch.erf((cutoff + radius) / scale)) / 2


def cake_wavelets(
    size: int,
    orientations: int,
    angular_sigma: float,
    radial_sigma: float,
    cutoff: float = 0.4
) -> torch.Tensor:
    O = orientations
    frequency = torch.fft.fftfreq(size)
    fy, fx = torch.meshgrid(frequency, frequency, indexing="ij")
    angle = torch.atan2(fy, fx) - torch.pi / 2                                    # [N, N]
    theta = torch.arange(O).reshape(O, 1, 1) * 2 * torch.pi / O                   # [O, 1, 1]
    offset = torch.remainder(angle - theta + torch.pi, 2 * torch.pi) - torch.pi   # [O, N, N]
    cakes = blurred_circular_step(offset, torch.pi / O, angular_sigma)            # [O, N, N]
    cakes[:, 0, 0] = 1 / O  # Share the mean between all orientations.
    cakes = cakes * blurred_half_line_step(torch.hypot(fx, fy), cutoff, radial_sigma)
    return torch.fft.ifft2(cakes)                                                 # [O, N, N]


def lift(
    image: torch.Tensor, 
    wavelets: torch.Tensor
) -> torch.Tensor:
    return torch.fft.ifft2(torch.fft.fft2(image).unsqueeze(1) * torch.fft.fft2(wavelets))
