import torch

from field import Field, change_basis, tensor_product
from gaussian_blur import gaussian_blur
from geometry import covariant_derivative, differential


def _whiten(form: Field, metric: Field) -> tuple[torch.Tensor, torch.Tensor]:
    """Change to a basis that is orthonormal in g.

    The functions below look for stationary points of F(v, v), |F(v, .)|^2 and |F(., v)|^2
    subject to g(v, v) = 1. For g = I these are eigh and svd: subject to |w| = 1, w^T A w is
    stationary at the eigenvectors of (A + A^T)/2, and |A^T w|^2 and |A w|^2 at the left
    and right singular vectors of A.

    For general g, factor g = L L^T. The columns of E = L^-T are orthonormal in g,
    E^T g E = I, so substituting v = E w turns g(v, v) = 1 into |w| = 1 and F into
    A = E^T F E. Solve for w, then v = E w.

    Returns:
        E and A = E^T F E.
    """
    if form.indices_type != "ll" or metric.indices_type != "ll":
        raise ValueError(f"expected two lower indices, got {form.indices_type!r} "
                         f"and {metric.indices_type!r}")
    g = torch.broadcast_to(metric.data, form.data.shape)
    basis = torch.linalg.inv(torch.linalg.cholesky(g)).mT
    return basis, basis.mT @ form.data @ basis


def eigenframe(form: Field, metric: Field) -> tuple[Field, Field]:
    """Directions v where F(v, v) is stationary subject to g(v, v) = 1.

    Solves (F + F^T)/2 v = λ g v, with values λ = F(v, v).

    Args:
        form: Bilinear form F = F_ij E^i ⊗ E^j, in any basis E_i.
        metric: Metric g = g_ij E^i ⊗ E^j, without batch dimensions, in the same basis.

    Returns:
        Values λ_j, ascending, and frame v_j = V^i_j E_i.
    """
    # Solve in a g-orthonormal basis E, then map back with v = E w.
    basis, F = _whiten(form, metric)
    values, w = torch.linalg.eigh((F + F.mT) / 2)
    frame = basis @ w
    values_type = form.prefix_type + "l"
    frame_type = form.prefix_type + "ul"
    return Field(values, values_type), Field(frame, frame_type)


def singular_frames(form: Field, metric: Field) -> tuple[Field, Field, Field]:
    """Singular value decomposition of F with respect to g.

    The left frame u and right frame v hold the stationary points of |F(u, .)|^2 and
    |F(., v)|^2 subject to unit length in g, paired so that F(u_i, v_j) = σ_i δ_ij. Only
    differs from eigenframe for a non-symmetric F.

    Args:
        form: Bilinear form F = F_ij E^i ⊗ E^j, in any basis E_i.
        metric: Metric g = g_ij E^i ⊗ E^j, without batch dimensions, in the same basis.

    Returns:
        Singular values σ_j, ascending, and the left frame u_j = U^i_j E_i and
        right frame v_j = V^i_j E_i.
    """
    # Solve in a g-orthonormal basis E, then map back with v = E w.
    basis, F = _whiten(form, metric)
    u, sigma, vh = torch.linalg.svd(F)
    sigma = sigma.flip(-1)
    left = basis @ u.flip(-1)
    right = basis @ vh.mT.flip(-1)
    values_type = form.prefix_type + "l"
    frame_type = form.prefix_type + "ul"
    return Field(sigma, values_type), Field(left, frame_type), Field(right, frame_type)


def structure_tensor_frame(field: Field, sigma: float, metric: Field) -> tuple[Field, Field]:
    """Eigenframe of the structure tensor, df ⊗ df blurred.

    Args:
        field: Scalar field f.
        sigma: Standard deviation of the blur in grid steps.
        metric: Metric g = g_ij e^i ⊗ e^j, without batch dimensions.

    Returns:
        Values λ_j, ascending, and frame v_j = V^i_j e_i.
    """
    df = differential(field)
    structure_tensor = gaussian_blur(tensor_product(df, df), sigma)
    return eigenframe(structure_tensor, metric)


def hessian_frame(field: Field, difference_tensor: Field, metric: Field) -> tuple[Field, Field]:
    """Eigenframe of the Hessian ∇∇f.

    Args:
        field: Scalar field f.
        difference_tensor: D = D^i_jk e_i ⊗ e^j ⊗ e^k of the connection
            ∇ = ∇^flat + D, without batch dimensions.
        metric: Metric g = g_ij e^i ⊗ e^j, without batch dimensions.

    Returns:
        Values λ_j, ascending, and frame v_j = V^i_j e_i.
    """
    hessian = covariant_derivative(differential(field), difference_tensor)
    return eigenframe(hessian, metric)


def covariant_derivative_in_frame(field: Field, frame: Field, difference_tensor: Field,
                                  order: int) -> Field:
    """Covariant derivative of the given order, in the frame basis.

    For a scalar field f, the gauge derivatives
    ∇^m f = (∇^m f)(v_i_1, ..., v_i_m) v^i_1 ⊗ ... ⊗ v^i_m.

    Args:
        field: Tensor field T of type `BSI`, components in e_i and e^i.
        frame: Frame v_j = V^i_j e_i.
        difference_tensor: D = D^i_jk e_i ⊗ e^j ⊗ e^k of the connection
            ∇ = ∇^flat + D, without batch dimensions.
        order: Number m of covariant derivatives.

    Returns:
        ∇^m T of type `BSI` followed by m `l`, components in v_i and v^i.
    """
    for _ in range(order):
        field = covariant_derivative(field, difference_tensor)
    for index in range(len(field.indices_type)):
        field = change_basis(field, frame, index)
    return field


def constant_metric_in_frame(metric: torch.Tensor, frame: Field) -> Field:
    """Metric whose components in a frame are constant, g(v_i, v_j) = G_ij.

    g = G_kl v^k ⊗ v^l = g_ij e^i ⊗ e^j with g_ij = (V^-1)^k_i G_kl (V^-1)^l_j.

    Args:
        metric: Tensor G of shape [n, n], the components in the frame.
        frame: Frame v_j = V^i_j e_i, without batch dimensions.

    Returns:
        g = g_ij e^i ⊗ e^j, without batch dimensions.
    """
    inverse = torch.linalg.inv(frame.data)
    data = inverse.mT @ metric @ inverse
    type = frame.prefix_type + "ll"
    return Field(data, type)
