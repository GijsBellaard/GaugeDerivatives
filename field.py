import torch


def spatial_dims(
    field: torch.Tensor, 
    indices: str,
    dim: tuple[int, ...] | None
) -> tuple[int, ...]:
    if dim is None:
        return tuple(range(field.ndim - len(indices)))
    return tuple(d % field.ndim for d in dim)


def change_basis(
    field: torch.Tensor, 
    frame: torch.Tensor, 
    index: int, 
    kind: str,
    dim: tuple[int, ...] | None = None
) -> torch.Tensor:
    last = frame.ndim - 3 if dim is None else max(d % field.ndim for d in dim)
    frame = frame.reshape(*frame.shape[:-2], *[1] * (field.ndim - last - 2), *frame.shape[-2:])
    field = field.movedim(index, -1)
    if kind == "l":
        field = torch.einsum("...i,...ij->...j", field, frame)
    else:
        field = torch.einsum("...ji,...i->...j", torch.linalg.inv(frame), field)
    return field.movedim(-1, index)
