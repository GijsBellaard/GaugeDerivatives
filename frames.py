import torch

from field import Field


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
        form: Field of type `BSll`.
        metric: Metric of type `Sll`.

    Returns:
        Values of type `BSl`, ascending, and frame of type `BSul`, where [..., :, i] is v_i.
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
        form: Field of type `BSll`.
        metric: Metric of type `Sll`.

    Returns:
        Singular values σ of type `BSl`, ascending, and the left and right frames of type
        `BSul`, where [..., :, i] is u_i and v_i.
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
