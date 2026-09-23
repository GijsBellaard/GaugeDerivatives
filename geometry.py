import torch

from field import Field, contract


def partial_derivative(field: Field) -> Field:
    """Partial derivatives of the components: the flat connection ∇^flat of the grid.

    Central differences, one-sided at the boundary, one grid step per unit.

    Args:
        field: Field of type `BSI` in the grid basis.

    Returns:
        Field of type `BSIl` in the grid basis, where out.data[..., a] is ∂_a of field.data.
    """
    parts = torch.gradient(field.data, dim=field.spatial_dims)
    data = torch.stack(parts, dim=-1)
    type = field.type + "l"
    return Field(data, type)


def differential(field: Field) -> Field:
    """Differential df of a scalar field, ∇_a f = ∂_a f for every connection.

    Args:
        field: Scalar field of type `BS`.

    Returns:
        Field of type `BSl` in the grid basis, where out.data[..., a] is ∂_a f.
    """
    if field.indices_type:
        raise ValueError(f"expected a scalar field, got indices {field.indices_type!r}")
    return partial_derivative(field)


def levi_civita_difference_tensor(metric: Field) -> Field:
    """Levi-Civita connection of a metric, as its difference tensor D = ∇ - ∇^flat.

    D^a_bc = 1/2 g^ad (∂_b g_dc + ∂_c g_db - ∂_d g_bc).

    Args:
        metric: Metric of type `Sll` in the grid basis.

    Returns:
        Difference tensor of type `Sull` in the grid basis.
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


def weitzenbock_difference_tensor(frame: Field) -> Field:
    """Weitzenböck connection of a frame, the connection for which the frame is parallel.

    ∇F_i = 0, so it is flat, with torsion T(F_i, F_j) = -[F_i, F_j]. As difference tensor
    D = ∇ - ∇^flat, D^a_zm = -∂_z F^a_i (F^-1)^i_m.

    Args:
        frame: Frame of type `Sul` in the grid basis, where [..., :, i] is F_i.

    Returns:
        Difference tensor of type `Sull` in the grid basis.
    """
    if frame.indices_type != "ul":
        raise ValueError(f"a frame has an upper and a lower index, got {frame.indices_type!r}")
    derivative = partial_derivative(frame)                      # [..., a, i, z] is ∂_z F^a_i
    coframe = Field(torch.linalg.inv(frame.data), frame.type)   # [..., i, m] is (F^-1)^i_m
    difference = contract(derivative, 1, coframe, 0)            # [..., a, z, m]
    return Field(-difference.data, difference.type)


def covariant_derivative(field: Field, difference_tensor: Field) -> Field:
    """Covariant derivative of field using the connection ∇ = ∇^flat + D.

    ∇_z T^a_b = ∂_z T^a_b + D^a_zm T^m_b - D^m_zb T^a_m, and likewise for any number of
    indices.

    Args:
        field: Field of type `BSI` in the grid basis.
        difference_tensor: D = ∇ - ∇^flat of type `Sull` in the grid basis, with
            [..., a, z, m] = D^a_zm.

    Returns:
        Field of type `BSIl` in the grid basis, where out.data[..., z] is ∇_z of the field.
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
