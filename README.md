# Gauge Derivatives

With this library you can calculate Gauge derivatives of tensor fields on a manifold endowed with a metric tensor field and a connection.

A _field_  is a tensor-valued signal on a manifold. 
A _Gauge frame_ is a basis of vector fields derived from a scalar field, e.g. from its structure tensor or Hessian. 
A _Gauge derivative_ of a field is a derivative in a Gauge frame direction.

## Gauge frame

![gauge frame](images/gauge_frame.svg)

Structure tensor and Hessian frames on a crop of a photograph. 
Blue is the first frame vector, orange the second. 
For the structure tensor, blue runs along edges and orange across them.

## Gauge derivatives

![gauge derivatives](images/gauge_derivatives.svg)

## Tensor fields

A `Field` is a `torch.Tensor`, that being a array with a _shape_, with an additional _type string_ with `len(shape)=len(type)`.

| character | meaning |
|-----------|---------|
| `b` | batch |
| `s` | spatial |
| `u` | upper index |
| `l` | lower index |

Batch dimensions come first, then spatial, then indices, with upper and lower in any
order. 
The number of `s`'s in the type is called `n` and is the number of spatial dimensions.
Every index has size `n`.
For example, a Field with shape `[B,H,W,2,2]` and type `"bssul"` is a batch of (1,1)-tensor fields on a 2D `H x W` grid . 

### Common fields

On a 2D grid of `H×W` points, with a batch of `B`:

| shape | type | object | examples |
|-------|------|--------|----------|
| `[H,W]` | `ss` | scalar field | image `f` |
| `[B,H,W]` | `bss` | batch of scalar fields | several images |
| `[H,W,2]` | `ssu` | vector field | gradient `g^ab ∂_b f` |
| `[H,W,2]` | `ssl` | covector field | differential `∂_a f` |
| `[H,W,2,2]` | `ssll` | (0,2)-tensor field | metric `g_ab`, Hessian `(∇ ∇ f)_ab`, structure tensor `(S f)_ab` |
| `[H,W,2,2]` | `ssuu` | (2,0)-tensor field | inverse metric `g^ab` |
| `[H,W,2,2]` | `ssul` | (1,1)-tensor field | frame `F^a_i`, linear mappings `A^a_b` |
| `[H,W,2,…,2]` | `ssl…l` | k lower indices | k-th covariant derivative of `f` |
| `[H,W,2,2,2]` | `ssull` | connection components | Christoffel symbols `Γ^a_bc` of a Levi-Civita connection |

## Example

```python
import torch
from field import Field, change_basis, tensor_product
from frames import eigenframe
from gaussian_blur import gaussian_blur
from geometry import covariant_derivative, levi_civita_connection

field = gaussian_blur(Field(torch.randn(64, 64), "ss"), sigma=2.0)

metric = Field(torch.eye(2).repeat(64, 64, 1, 1), "ssll")  # Standard Euclidean metric
connection = levi_civita_connection(metric)  # "ssull", zero in this case

df = covariant_derivative(field, connection)  # "ssl"
outer = tensor_product(df, df)  # "ssll"
eigenvalues, frame = eigenframe(gaussian_blur(outer, sigma=1.0), metric)  # "ssl", "ssul"

third = field
for _ in range(3):
    third = covariant_derivative(third, connection)  # "sslll"
gauge = change_basis(third, frame)  # "sslll", indices in the frame
gauge.data[..., 0, 1, 1]  # signature [0, 1, 1]
```

## Functions

`B` is any number of batch dimensions, `S` the spatial dimensions, and
`I`, `J` any string of `u` and `l`. 

### `field.py`

`Field(data, type)`, and:

| function | takes | returns | computes |
|----------|-------|---------|----------|
| `tensor_product(a, b)` | `a: BSI`, `b: BSJ` | `BSIJ` | `a ⊗ b` |
| `contract(a, i, b, j)` | `a: BSI`, `b: BSJ` | `BSIJ` without the pair | sum of index `i` of `a` with index `j` of `b` |
| `trace(field, i, j)` | `field: BSI` | `BSI` without the pair | sum of indices `i` and `j` of `field` |
| `change_basis(field, frame)` | `field: BSI`, `frame: BSul` | `BSI` | every index expressed in the frame |

### `geometry.py`

| function | takes | returns | computes |
|----------|-------|---------|----------|
| `partial_derivative(field)` | `field: BSI` | `BSIl` | `∂_z T` |
| `levi_civita_connection(metric)` | `metric: Sll` | `Sull` | `Γ^a_bc` of the metric |
| `covariant_derivative(field, connection)` | `field: BSI`, `connection: Sull` | `BSIl` | `∇_z T`, for any connection |

### `frames.py`

| function | takes | returns | computes |
|----------|-------|---------|----------|
| `eigenframe(form, metric)` | `form: BSll`, `metric: Sll` | `BSl`, `BSul` | stationary points of `F(v,v)` with `g(v,v) = 1` |
| `singular_frames(form, metric)` | `form: BSll`, `metric: Sll` | `BSl`, `BSul`, `BSul` | `σ` and the left and right singular frames of `F` in `g` |

### `gaussian_blur.py`

| function | takes | returns | computes |
|----------|-------|---------|----------|
| `gaussian_blur(field, sigma)` | `field: BSI` | `BSI` | Gaussian blur along `S`, periodic boundaries |

`make_figures.py` regenerates the figures.