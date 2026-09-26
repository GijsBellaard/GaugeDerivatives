import torch

from grid import derivative


def change_basis(
    field: torch.Tensor, 
    frame: torch.Tensor, 
    index: int, 
    kind: str
) -> torch.Tensor:
    N = field.ndim
    i = index % N
    labels = list(range(N))
    grid = list(range(frame.ndim - 2))
    out = labels.copy()
    out[i] = N
    if kind == "l":
        return torch.einsum(
            field, labels,
            frame, [*grid, i, N],
            out
        )  # T_..j.. = T_..i.. V^i_j
    elif kind == "u":
        return torch.einsum(
            torch.linalg.inv(frame), [*grid, N, i],
            field, labels,
            out
        )  # T^..j.. = (V^-1)^j_i T^..i..
    else:
        raise ValueError(f"kind must be 'l' or 'u', got {kind!r}")


def constant_metric_in_frame(
    metric: torch.Tensor, 
    frame: torch.Tensor
) -> torch.Tensor:
    inverse = torch.linalg.inv(frame)
    return inverse.mT @ metric @ inverse


def levi_civita_connection(metric: torch.Tensor) -> torch.Tensor:
    spatial_dims = list(range(1, metric.ndim - 2))
    dg = derivative(metric, spatial_dims)
    term = (dg.transpose(-1, -2)
            + dg
            - dg.movedim(-1, -3))
    inverse = torch.linalg.inv(metric)
    return 0.5 * torch.einsum("...ad,...dbc->...abc", inverse, term)


def weitzenbock_connection(frame: torch.Tensor) -> torch.Tensor:
    spatial_dims = list(range(1, frame.ndim - 2))
    dV = derivative(frame, spatial_dims)
    coframe = torch.linalg.inv(frame)
    return -torch.einsum("...ilj,...lk->...ijk", dV, coframe)


def covariant_derivative(
    field: torch.Tensor,
    connection: torch.Tensor,
    indices: str = ""
) -> torch.Tensor:
    p = len(indices)
    spatial_dims = list(range(1, field.ndim - p))
    data = derivative(field, spatial_dims)
    for i, kind in enumerate(indices):
        labels = list(range(p))
        labels[i] = p + 1
        if kind == "u":
            data = data + torch.einsum(
                connection, [..., i, p, p + 1],
                field, [..., *labels],
                [..., *range(p + 1)]
            )  # + D^i_kl T^..l..
        else:
            data = data - torch.einsum(
                connection, [..., p + 1, p, i],
                field, [..., *labels],
                [..., *range(p + 1)]
            )  # - D^l_ki T_..l..
    return data


def covariant_derivative_in_frame(
    field: torch.Tensor, 
    frame: torch.Tensor,
    connection: torch.Tensor, 
    order: int,
    indices: str = ""
) -> torch.Tensor:
    for _ in range(order):
        field = covariant_derivative(field, connection, indices)
        indices += "l"
    for index, kind in enumerate(indices):
        field = change_basis(field, frame, index - len(indices), kind)
    return field
