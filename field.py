from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Field:
    """Tensor field on a grid, stored as its components and a type string.

    `type` has one character per dimension of `data`: `b` batch, `s` spatial, `u` upper
    index, `l` lower index. Batch dimensions come first, then spatial, then indices, with
    upper and lower in any order. n is the number of spatial dimensions and every index has
    size n. The indices are components some basis e_a and its dual e^a. Keeping track of 
    which indices are in which frame is up to the caller.

    In docstrings, `B` is any number of batch dimensions, `S` the spatial dimensions, and
    `I`, `J` any string of `u` and `l`. E.g. `BSIl` is a field of type `BSI` with a lower
    index appended.
    """
    data: torch.Tensor
    type: str

    def __post_init__(self):
        if len(self.type) != self.data.ndim:
            raise ValueError(f"type {self.type!r} has {len(self.type)} characters "
                             f"for {self.data.ndim} dimensions")
        if set(self.type) - set("bsul"):
            raise ValueError(f"type {self.type!r} uses characters outside 'bsul'")
        ordered = ("b" * self.type.count("b") + "s" * self.type.count("s")
                   + "".join(c for c in self.type if c in "ul"))
        if self.type != ordered:
            raise ValueError(f"type {self.type!r} is out of order, expected {ordered!r}: "
                             f"batch dimensions, then spatial, then indices")
        for d in self.index_dims:
            if self.data.shape[d] != self.n:
                raise ValueError(f"dimension {d} is an index so it must have size n={self.n}, "
                                 f"got {self.data.shape[d]}")

    def __repr__(self) -> str:
        return (f"Field(type={self.type!r}, shape={tuple(self.data.shape)}, "
                f"dtype={self.data.dtype}, device={self.data.device})")

    @property
    def n(self) -> int:
        """Number of spatial dimensions."""
        return self.type.count("s")

    @property
    def indices_type(self) -> str:
        """Index characters of type, e.g. `ul`."""
        return "".join(c for c in self.type if c in "ul")

    @property
    def prefix_type(self) -> str:
        """Batch and spatial characters of type, e.g. `bss`."""
        return "".join(c for c in self.type if c in "bs")

    @property
    def spatial_dims(self) -> list[int]:
        """Positions of the spatial dimensions in data."""
        return [d for d, c in enumerate(self.type) if c == "s"]

    @property
    def index_dims(self) -> list[int]:
        """Positions of the index dimensions in data."""
        return [d for d, c in enumerate(self.type) if c in "ul"]


def tensor_product(a: Field, b: Field) -> Field:
    """Tensor product a ⊗ b, with (a ⊗ b)[..., I, J] = a[..., I] b[..., J].

    Args:
        a: Field of type `BSI`.
        b: Field of type `BSJ`.

    Returns:
        Field of type `BSIJ`.
    """
    if a.n != b.n:
        raise ValueError(f"fields disagree on n: {a.n} and {b.n}")

    # Label the indices of a 0..p-1 and those of b p..p+q-1.
    p, q = len(a.indices_type), len(b.indices_type)
    data = torch.einsum(a.data, [..., *range(p)], b.data, [..., *range(p, p + q)],
                        [..., *range(p + q)])
    type = max(a.prefix_type, b.prefix_type, key=len) + a.indices_type + b.indices_type
    return Field(data, type)


def contract(a: Field, index_a: int, b: Field, index_b: int) -> Field:
    """Contract an index of a with an index of b, one upper and one lower.

    Args:
        a: Field of type `BSI`.
        index_a: Position of the contracted index in `I`.
        b: Field of type `BSJ`.
        index_b: Position of the contracted index in `J`.

    Returns:
        Field of type `BSIJ` without the contracted pair.
    """
    if a.n != b.n:
        raise ValueError(f"fields disagree on n: {a.n} and {b.n}")
    if a.indices_type[index_a] == b.indices_type[index_b]:
        raise ValueError(f"index {index_a} of {a.type!r} and index {index_b} of {b.type!r} "
                         f"are both {a.indices_type[index_a]!r}")

    # Label the indices of a 0..p-1 and those of b p..p+q-1; the contracted pair shares one.
    p, q = len(a.indices_type), len(b.indices_type)
    labels = list(range(p + q))
    labels[p + index_b] = index_a
    kept = [i for i in range(p + q) if i not in (index_a, p + index_b)]
    data = torch.einsum(a.data, [..., *labels[:p]], b.data, [..., *labels[p:]], [..., *kept])
    kinds = "".join((a.indices_type + b.indices_type)[i] for i in kept)
    type = max(a.prefix_type, b.prefix_type, key=len) + kinds
    return Field(data, type)


def trace(field: Field, index_i: int, index_j: int) -> Field:
    """Contract two indices of a field, one upper and one lower: T^m_m.

    Args:
        field: Field of type `BSI`.
        index_i: Position of the first index in `I`.
        index_j: Position of the second index in `I`.

    Returns:
        Field of type `BSI` without the two indices.
    """
    if field.indices_type[index_i] == field.indices_type[index_j]:
        raise ValueError(f"indices {index_i} and {index_j} of {field.type!r} "
                         f"are both {field.indices_type[index_i]!r}")
    # Label the indices 0..p-1; the contracted pair shares one.
    p = len(field.indices_type)
    labels = list(range(p))
    labels[index_j] = index_i
    kept = [i for i in range(p) if i not in (index_i, index_j)]
    data = torch.einsum(field.data, [..., *labels], [..., *kept])
    kinds = "".join(field.indices_type[i] for i in kept)
    type = field.prefix_type + kinds
    return Field(data, type)


def change_basis(field: Field, frame: Field, index: int) -> Field:
    """Express one index of a field in a frame F_i = F^a_i e_a.

    A lower index becomes T_i = T_a F^a_i and an upper index T^i = (F^-1)^i_a T^a. 

    Args:
        field: Field of type `BSI`.
        frame: Frame of type `BSul`, where [..., :, i] is the i-th frame vector.
        index: Position of the index in `I`.

    Returns:
        Field of type `BSI` with the index in the given frame
    """
    if frame.indices_type != "ul":
        raise ValueError(f"a frame has an upper and a lower index, got {frame.indices_type!r}")

    # The index is summed with the frame over the spare label p, and the frame hands back
    # its frame index at the same position.
    p = len(field.indices_type)
    labels = list(range(p))
    labels[index] = p
    if field.indices_type[index] == "l":
        data = torch.einsum(
            field.data, [..., *labels], 
            frame.data, [..., p, index],
            [..., *range(p)]
        )
    else:
        inverse = torch.linalg.inv(frame.data)
        data = torch.einsum(
            inverse, [..., index, p], 
            field.data, [..., *labels],
            [..., *range(p)]
        )
    type = max(field.prefix_type, frame.prefix_type, key=len) + field.indices_type
    return Field(data, type)
