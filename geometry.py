import torch

from field import Field, contract


def partial_derivative(field: Field) -> Field:
    """Partial derivatives of the components: the flat connection ∇^flat of the grid.

    Central differences, one-sided at the boundary, one grid step per unit. Zero along
    spatial dimensions of size 1, as the field is constant along them.

    Args:
        field: Tensor field T of type `BSI`, components in e_i and e^i.

    Returns:
        ∇^flat T of type `BSIl`, components in e_i and e^i, where [..., k] is e_k of the
        components of T.
    """
    parts = [torch.zeros_like(field.data) if field.data.shape[d] == 1
             else torch.gradient(field.data, dim=d)[0]
             for d in field.spatial_dims]
    data = torch.stack(parts, dim=-1)
    type = field.type + "l"
    return Field(data, type)


def differential(field: Field) -> Field:
    """Differential of a scalar field.

    Args:
        field: Scalar field f.

    Returns:
        df = (e_i f) e^i.
    """
    if field.indices_type:
        raise ValueError(f"expected a scalar field, got indices {field.indices_type!r}")
    return partial_derivative(field)


def gradient(field: Field, metric: Field) -> Field:
    """Gradient of a scalar field.

    Args:
        field: Scalar field f.
        metric: Metric g = g_ij e^i ⊗ e^j, without batch dimensions.

    Returns:
        grad f = g^ij (e_j f) e_i.
    """
    if metric.indices_type != "ll":
        raise ValueError(f"a metric has two lower indices, got {metric.indices_type!r}")
    inverse = Field(torch.linalg.inv(metric.data), metric.prefix_type + "uu")
    return contract(inverse, 1, differential(field), 0)


def inner_product(a: Field, b: Field, metric: Field) -> Field:
    """Inner product of two vector fields or two covector fields.

    Args:
        a: Vector field a = a^i E_i, or covector field a = a_i E^i, in any basis E_i.
        b: Field of the same kind as a, in the same basis.
        metric: Metric g = g_ij E^i ⊗ E^j, without batch dimensions, in the same basis.

    Returns:
        g(a, b) = g_ij a^i b^j, or g^ij a_i b_j for covectors.
    """
    if a.indices_type not in ("u", "l") or b.indices_type != a.indices_type:
        raise ValueError(f"expected two vector or two covector fields, got {a.type!r} "
                         f"and {b.type!r}")
    if metric.indices_type != "ll":
        raise ValueError(f"a metric has two lower indices, got {metric.indices_type!r}")
    if a.indices_type == "l":
        metric = Field(torch.linalg.inv(metric.data), metric.prefix_type + "uu")
    return contract(contract(metric, 0, a, 0), 0, b, 0)


def norm2(field: Field, metric: Field) -> Field:
    """Squared norm of a vector or covector field, inner_product(field, field, metric).

    Args:
        field: Vector field a^i E_i, or covector field a_i E^i, in any basis E_i.
        metric: Metric g = g_ij E^i ⊗ E^j, without batch dimensions, in the same basis.

    Returns:
        g(a, a).
    """
    return inner_product(field, field, metric)


def levi_civita_difference_tensor(metric: Field) -> Field:
    """Levi-Civita connection of a metric, as its difference tensor D = ∇ - ∇^flat.

    D^i_jk = 1/2 g^il ((e_j g_lk) + (e_k g_lj) - (e_l g_jk)).

    Args:
        metric: Metric g = g_ij e^i ⊗ e^j, without batch dimensions.

    Returns:
        D = D^i_jk e_i ⊗ e^j ⊗ e^k, without batch dimensions.
    """
    if metric.indices_type != "ll":
        raise ValueError(f"a metric has two lower indices, got {metric.indices_type!r}")
    derivative = partial_derivative(metric).data  # [..., l, j, k] is (e_k g_lj)
    term = (derivative.transpose(-1, -2)          # (e_j g_lk)
            + derivative                          # (e_k g_lj)
            - derivative.movedim(-1, -3))         # (e_l g_jk)
    inverse = torch.linalg.inv(metric.data)
    data = 0.5 * torch.einsum("...ad,...dbc->...abc", inverse, term)
    type = metric.prefix_type + "ull"
    return Field(data, type)


def weitzenbock_difference_tensor(frame: Field) -> Field:
    """Weitzenböck connection of a frame, the connection for which the frame is parallel.

    Let v_l = V^i_l e_i be the frame. The difference tensor D = ∇ - ∇^flat has components
    ∇_e_j e_k = D^i_jk e_i, as ∇^flat_e_j e_k = 0. Then:
        1. The frame is parallel, ∇_e_j v_l = 0 for every l.
        2. By the product rule, ∇_e_j v_l = ∇_e_j (V^m_l e_m) = ((e_j V^i_l) + D^i_jm V^m_l) e_i,
           so (e_j V^i_l) + D^i_jm V^m_l = 0.
        3. Multiply by (V^-1)^l_k and sum over l, with V^m_l (V^-1)^l_k = δ^m_k:
           (e_j V^i_l) (V^-1)^l_k + D^i_jk = 0.
        4. So D^i_jk = -(e_j V^i_l) (V^-1)^l_k.

    Args:
        frame: Frame v_j = V^i_j e_i, without batch dimensions.

    Returns:
        D = D^i_jk e_i ⊗ e^j ⊗ e^k, without batch dimensions.
    """
    if frame.indices_type != "ul":
        raise ValueError(f"a frame has an upper and a lower index, got {frame.indices_type!r}")
    derivative = partial_derivative(frame)                      # [..., i, l, j] is (e_j V^i_l)
    coframe = Field(torch.linalg.inv(frame.data), frame.type)   # [..., l, k] is (V^-1)^l_k
    difference = contract(derivative, 1, coframe, 0)            # [..., i, j, k]
    return Field(-difference.data, difference.type)


def covariant_derivative(field: Field, difference_tensor: Field) -> Field:
    """Covariant derivative of field using the connection ∇ = ∇^flat + D.

    ∇_k T^i_j = (e_k T^i_j) + D^i_kl T^l_j - D^l_kj T^i_l, and likewise for any number of
    indices.

    Args:
        field: Tensor field T of type `BSI`, components in e_i and e^i.
        difference_tensor: D = D^i_jk e_i ⊗ e^j ⊗ e^k, without batch dimensions.

    Returns:
        ∇T of type `BSIl`, components in e_i and e^i, where [..., k] is ∇_e_k T.
    """
    # Label the indices of the field 0..p-1 and the derivative index p. The corrected
    # index is summed with the difference tensor over label p + 1.
    p = len(field.indices_type)
    data = partial_derivative(field).data
    for i, kind in enumerate(field.indices_type):
        labels = list(range(p))
        labels[i] = p + 1
        if kind == "u":
            data = data + torch.einsum(
                difference_tensor.data, [..., i, p, p + 1],
                field.data, [..., *labels],
                [..., *range(p + 1)]
            )  # + D^i_kl T^..l..
        else:
            data = data - torch.einsum(
                difference_tensor.data, [..., p + 1, p, i],
                field.data, [..., *labels],
                [..., *range(p + 1)]
            )  # - D^l_ki T_..l..
    type = field.type + "l"
    return Field(data, type)
