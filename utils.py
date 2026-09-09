import torch


def normalize_dims(
    field: torch.Tensor,
    dims: list[int] | None,
) -> list[int]:
    """Resolve a dims argument into a list of non-negative dimension indices of field.

    Args:
        field: Tensor whose dimensions dims refers to.
        dims: List containing the indices of the dimensions, possibly negative,
            defaults to all dimensions of field.

    Returns:
        List of non-negative indices into field.shape.
    """
    if dims is None:
        return list(range(field.ndim))
    return [d % field.ndim for d in dims]
