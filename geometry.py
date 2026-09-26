import torch

from field import spatial_dims


def partial_derivative(
    field: torch.Tensor,
    dim: tuple[int, ...] | None = None
) -> torch.Tensor:
    parts = [torch.zeros_like(field) if field.shape[d] == 1
             else torch.gradient(field, dim=d)[0]
             for d in (range(field.ndim) if dim is None else dim)]
    return torch.stack(parts, dim=-1)


def differential(
    field: torch.Tensor,
    dim: tuple[int, ...] | None = None
) -> torch.Tensor:
    return partial_derivative(field, dim)

def levi_civita_difference_tensor(metric: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(metric.ndim - 2))
    derivative = partial_derivative(metric, dims)
    term = (derivative.transpose(-1, -2)
            + derivative
            - derivative.movedim(-1, -3))
    inverse = torch.linalg.inv(metric)
    return 0.5 * torch.einsum("...ad,...dbc->...abc", inverse, term)


def weitzenbock_difference_tensor(frame: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(frame.ndim - 2))
    derivative = partial_derivative(frame, dims)
    coframe = torch.linalg.inv(frame)
    return -torch.einsum("...ilj,...lk->...ijk", derivative, coframe)


def covariant_derivative(
    field: torch.Tensor,
    difference_tensor: torch.Tensor,
    indices: str = "",
    dim: tuple[int, ...] | None = None
) -> torch.Tensor:
    # Label the indices of the field 0..p-1 and the derivative index p. The corrected
    # index is summed with the difference tensor over label p + 1.
    p = len(indices)
    data = partial_derivative(field, spatial_dims(field, indices, dim))
    for i, kind in enumerate(indices):
        labels = list(range(p))
        labels[i] = p + 1
        if kind == "u":
            data = data + torch.einsum(
                difference_tensor, [..., i, p, p + 1],
                field, [..., *labels],
                [..., *range(p + 1)]
            )  # + D^i_kl T^..l..
        else:
            data = data - torch.einsum(
                difference_tensor, [..., p + 1, p, i],
                field, [..., *labels],
                [..., *range(p + 1)]
            )  # - D^l_ki T_..l..
    return data
