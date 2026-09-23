import torch

from field import Field


def partial_derivative(field: Field) -> Field:
    """Partial derivatives along the grid by central differences, one grid step per unit.

    Args:
        field: Field of type BSI.

    Returns:
        Field of type BSIl, where out.data[..., a] is the derivative of field.data along
        spatial dimension a.
    """
    parts = torch.gradient(field.data, dim=field.spatial_dims)
    data = torch.stack(parts, dim=-1)
    type = field.type + "l"
    return Field(data, type)


def connection(metric: Field) -> Field:
    """Levi-Civita connection components of a metric.

    G^a_bc = 1/2 g^ad (d_b g_dc + d_c g_db - d_d g_bc), with [..., a, b, c] = G^a_bc.

    The components do not transform as a tensor; the type string only describes the layout.

    Args:
        metric: Metric of type Sll.

    Returns:
        Connection components of type Sull.
    """
    if metric.indices_type != "ll":
        raise ValueError(f"a metric has two lower indices, got {metric.indices_type!r}")
    derivative = partial_derivative(metric).data # [..., i, j, k] is d_k g_ij
    term = (derivative.transpose(-1, -2)         # d_b g_dc
            + derivative                         # d_c g_db
            - derivative.movedim(-1, -3))        # d_d g_bc
    inverse = torch.linalg.inv(metric.data)
    data = 0.5 * torch.einsum("...ad,...dbc->...abc", inverse, term)
    type = metric.prefix_type + "ull"
    return Field(data, type)


def covariant_derivative(field: Field, connection: Field) -> Field:
    """Covariant derivative.

    nabla_z T^a_b = d_z T^a_b + G^a_zm T^m_b - G^m_zb T^a_m,
    and likewise for any number of indices.

    Args:
        field: Field of type BSI.
        connection: Connection components of type Sull.

    Returns:
        Field of type BSIl, where out.data[..., z] is nabla_z of the field.
    """
    # Label the indices of the field 0..p-1 and the derivative index p. The corrected
    # index is summed with the connection over label p + 1.
    p = len(field.indices_type)
    data = partial_derivative(field).data
    for i, kind in enumerate(field.indices_type):
        labels = list(range(p))
        labels[i] = p + 1
        if kind == "u":
            data += torch.einsum(
                connection.data, [..., i, p, p + 1],
                field.data, [..., *labels],
                [..., *range(p + 1)]
            ) # + G^a_zm T^..m..
        else:
            data -= torch.einsum(
                connection.data, [..., p + 1, p, i],
                field.data, [..., *labels],
                [..., *range(p + 1)]
            ) # - G^m_za T_..m..
    type = field.type + "l"
    return Field(data, type)
