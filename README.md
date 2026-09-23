# Gauge Derivatives

Derivatives of a field in a local frame derived from the field itself, e.g. the
eigenvectors of its structure tensor or Hessian. The frame rotates with the field, so the
derivatives do not depend on its orientation, up to the signs of the frame vectors.

A *signature* `[i0, i1, ...]` lists the frame vector for each successive derivative:
`[1, 1]` is the second derivative along the second frame vector.

## Tensor fields

A `Field` is a `torch.Tensor` with a type string, one character per dimension:

| character | meaning |
|-----------|---------|
| `b` | batch |
| `s` | spatial |
| `u` | upper index |
| `l` | lower index |

Batch dimensions come first, then spatial, then indices, with upper and lower in any
order. E.g. `"bssul"` is a batch of (1,1)-tensor fields on a 2D grid. `n` is the number of
spatial dimensions and every index has size `n`. All fields in a computation live on the
same grid.

In the docstrings, `B` is any number of batch dimensions, `S` the spatial dimensions, and
`I`, `J` any string of `u` and `l`. E.g. `covariant_derivative` takes a field of type `BSI`
and returns one of type `BSIl`.

Derivatives are per grid step. Grid spacing and curvature go into the metric.

## Common fields

On a 2D grid:

| type | object | examples |
|------|--------|----------|
| `ss` | scalar field | image `f` |
| `bss` | batch of scalar fields | several images |
| `ssu` | vector field | gradient `g^ab ∂_b f` |
| `ssl` | covector field | differential `df` |
| `ssll` | (0,2)-tensor field | metric `g_ab`, Hessian `∇_a∇_b f`, structure tensor |
| `ssuu` | (2,0)-tensor field | inverse metric `g^ab` |
| `ssul` | (1,1)-tensor field | frame `F^a_i`, identity `δ^a_b` |
| `ssl…l` | k lower indices | k-th covariant derivative of `f` |
| `ssull` | connection components | `Γ^a_bc`, not a tensor field |

On a 3D grid add an `s`: `sss` is a scalar field on a volume and every index has size 3.

A frame `F^a_i` holds the frame vectors as columns: `a` is the coordinate component, `i`
says which frame vector. Eigenvalues from `eigenframe` are `ssl`, one per frame vector.

The type says how components transform, not only what they are. On a Euclidean grid,
`g_ab`, `g^ab` and `δ^a_b` all hold the identity matrix but are `ssll`, `ssuu` and `ssul`.

## Example

```python
import torch
from field import Field, change_basis, tensor_product
from frames import eigenframe
from gaussian_blur import gaussian_blur
from geometry import connection, covariant_derivative

field = gaussian_blur(Field(torch.randn(64, 64), "ss"), sigma=2.0)

metric = Field(torch.eye(2).repeat(64, 64, 1, 1), "ssll")  # Euclidean
levi_civita = connection(metric)  # "ssull", zero here

df = covariant_derivative(field, levi_civita)  # "ssl"
outer = tensor_product(df, df)  # "ssll"
eigenvalues, frame = eigenframe(gaussian_blur(outer, sigma=1.0), metric)  # "ssl", "ssul"

third = field
for _ in range(3):
    third = covariant_derivative(third, levi_civita)  # "sslll"
gauge = change_basis(third, frame)  # "sslll", indices in the frame
gauge.data[..., 0, 1, 1]  # signature [0, 1, 1]
```

With a non-constant metric the same code computes covariant derivatives in that geometry.

## Gauge frame

![gauge frame](images/gauge_frame.svg)

Structure tensor and Hessian frames on a crop of a photograph. Blue is the first frame
vector, orange the second. For the structure tensor, blue runs along edges and orange
across them.

## Gauge derivatives

![gauge derivatives](images/gauge_derivatives.svg)

## Modules

| file | contents |
|------|----------|
| `field.py` | `Field`, `tensor_product`, `contract`, `trace`, `change_basis` |
| `geometry.py` | `partial_derivative`, `connection`, `covariant_derivative` |
| `frames.py` | `eigenframe`, `singular_frames` |
| `gaussian_blur.py` | `gaussian_blur` |
| `make_figures.py` | regenerates the figures |
| `test_geometry.py` | tests, `python test_geometry.py` |
