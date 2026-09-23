import torch

from field import Field


def partial_derivative(field: Field) -> Field:
    """Partial derivatives of the components: the flat connection ∇^flat of the grid.

    Central differences, one-sided at the boundary, one grid step per unit.

    Args:
        field: Field of type `BSI`.

    Returns:
        Field of type `BSIl`, where out.data[..., a] is ∂_a of field.data.
    """
    parts = torch.gradient(field.data, dim=field.spatial_dims)
    data = torch.stack(parts, dim=-1)
    type = field.type + "l"
    return Field(data, type)


def levi_civita_difference_tensor(metric: Field) -> Field:
    """Levi-Civita connection of a metric, as its difference tensor D = ∇ - ∇^flat.

    D^a_bc = 1/2 g^ad (∂_b g_dc + ∂_c g_db - ∂_d g_bc).

    Args:
        metric: Metric of type `Sll`.

    Returns:
        Difference tensor of type `Sull`.
    """
    if metric.indices_type != "ll":
        raise ValueError(f"a metric has two lower indices, got {metric.indices_type!r}")
    derivative = partial_derivative(metric).data  # [..., i, j, k] is ∂_k g_ij
    term = (derivative.transpose(-1, -2)          # ∂_b g_dc
            + derivative                          # ∂_c g_db
            - derivative.movedim(-1, -3))         # ∂_d g_bc
    inverse = torch.linalg.inv(metric.data)
    data = 0.5 * torch.einsum("...ad,...dbc->...abc", inverse, term)
    type = metric.prefix_type + "ull"
    return Field(data, type)


def covariant_derivative(field: Field, difference_tensor: Field) -> Field:
    """Covariant derivative of the connection ∇ = ∇^flat + D.

    ∇_z T^a_b = ∂_z T^a_b + D^a_zm T^m_b - D^m_zb T^a_m, and likewise for any number of
    indices.

    Args:
        field: Field of type `BSI`.
        difference_tensor: D = ∇ - ∇^flat of type `Sull`, with [..., a, z, m] = D^a_zm.

    Returns:
        Field of type `BSIl`, where out.data[..., z] is ∇_z of the field.
    """
    # Label the indices of the field 0..p-1 and the derivative index p. The corrected
    # index is summed with the difference tensor over label p + 1.
    p = len(field.indices_type)
    data = partial_derivative(field).data
    for i, kind in enumerate(field.indices_type):
        labels = list(range(p))
        labels[i] = p + 1
        if kind == "u":
            data += torch.einsum(
                difference_tensor.data, [..., i, p, p + 1],
                field.data, [..., *labels],
                [..., *range(p + 1)]
            )  # + D^a_zm T^..m..
        else:
            data -= torch.einsum(
                difference_tensor.data, [..., p + 1, p, i],
                field.data, [..., *labels],
                [..., *range(p + 1)]
            )  # - D^m_za T_..m..
    type = field.type + "l"
    return Field(data, type)
