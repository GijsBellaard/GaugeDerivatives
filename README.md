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

Indices are components in the grid basis unless converted with `change_basis`, and the
type string does not record which. `partial_derivative`, `covariant_derivative`,
`levi_civita_difference_tensor` and `gaussian_blur` need every index in the grid basis.
`contract` and `trace` need the two summed indices in the same basis.

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
| `[H,W,2,2,2]` | `ssull` | (1,2)-tensor field  | difference tensor `D = ∇ − ∇^flat` from the grid's flat connection `∇^flat` |

## Example

```python
import torch
from field import Field, change_basis, tensor_product
from frames import eigenframe
from gaussian_blur import gaussian_blur
from geometry import covariant_derivative, differential, levi_civita_difference_tensor

field = gaussian_blur(Field(torch.randn(64, 64), "ss"), sigma=2.0)

metric = Field(torch.eye(2).repeat(64, 64, 1, 1), "ssll")  # Standard Euclidean metric
difference_tensor = levi_civita_difference_tensor(metric)  # "ssull", zero in this case

df = differential(field)  # "ssl"
outer = tensor_product(df, df)  # "ssll"
eigenvalues, frame = eigenframe(gaussian_blur(outer, sigma=1.0), metric)  # "ssl", "ssul"

third = field
for _ in range(3):
    third = covariant_derivative(third, difference_tensor)  # "sslll"
gauge = third
for index in range(3):
    gauge = change_basis(gauge, frame, index)  # "sslll", indices in the frame
gauge.data[..., 0, 1, 1]  # signature [0, 1, 1]
```

## Example on M2

Derivatives of `f = x y + θ²` on position orientation space `M2 = R² × S¹`, in the frame
`A_1 = cos θ ∂_x + sin θ ∂_y`, `A_2 = −sin θ ∂_x + cos θ ∂_y`, `A_3 = ∂_θ`, compared with
the exact answer.

```python
import torch
from field import Field, change_basis
from geometry import covariant_derivative, differential, weitzenbock_difference_tensor
from m2 import left_invariant_frame

torch.set_default_dtype(torch.float64)
O, H, W = 64, 16, 16
A = left_invariant_frame((O, H, W))  # "sssul", the frame A_1, A_2, A_3
D = weitzenbock_difference_tensor(A)  # "sssull", the connection with ∇A_i = 0

θ, y, x = torch.meshgrid(2 * torch.pi * torch.arange(O) / O, torch.arange(H) * 1.0,
                         torch.arange(W) * 1.0, indexing="ij")
cosθ, sinθ = θ.cos(), θ.sin()

f = Field(x * y + θ**2, "sss")

# Exact analytical derivatives
A1f = y * cosθ + x * sinθ
A2f = x * cosθ - y * sinθ
A3f = 2 * θ
A3A1f = x * cosθ - y * sinθ
A1A3f = 0 * θ

# First and second derivatives numerically in grid basis
first = differential(f)
second = covariant_derivative(first, D)

# Change basis to Ai frame
firstA = change_basis(first, A, 0)  # [..., i] = A_i f
secondA = second
for index in range(2):
    secondA = change_basis(secondA, A, index)  # [..., i, j] = A_j A_i f

# Check error between exact analytical derivatives and the numerical ones
inner = (slice(2, -2),) * 3  # away from the edges of the grid
def error(computed, exact):
    return (computed - exact)[inner].abs().max().item()

error(firstA.data[..., 0], A1f)  # 1e-14
error(firstA.data[..., 1], A2f)  # 1e-14
error(firstA.data[..., 2], A3f)  # 1e-13
error(secondA.data[..., 0, 2], A3A1f)  # 3e-2
error(secondA.data[..., 2, 0], A1A3f)  # 1e-13
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
| `change_basis(field, frame, index)` | `field: BSI`, `frame: BSul` | `BSI` | index `index` from the grid basis to the frame |

### `geometry.py`

| function | takes | returns | computes |
|----------|-------|---------|----------|
| `partial_derivative(field)` | `field: BSI` | `BSIl` | `∇^flat_z T = ∂_z T` |
| `differential(field)` | `field: BS` | `BSl` | `df`, the same for every connection |
| `levi_civita_difference_tensor(metric)` | `metric: Sll` | `Sull` | `D` of the Levi-Civita connection |
| `weitzenbock_difference_tensor(frame)` | `frame: Sul` | `Sull` | `D` of the connection for which the frame is parallel |
| `covariant_derivative(field, difference_tensor)` | `field: BSI`, `difference_tensor: Sull` | `BSIl` | `∇_z T` with `∇ = ∇^flat + D`, for any connection |

### `frames.py`

| function | takes | returns | computes |
|----------|-------|---------|----------|
| `eigenframe(form, metric)` | `form: BSll`, `metric: Sll` | `BSl`, `BSul` | stationary points of `F(v,v)` with `g(v,v) = 1` |
| `singular_frames(form, metric)` | `form: BSll`, `metric: Sll` | `BSl`, `BSul`, `BSul` | `σ` and the left and right singular frames of `F` in `g` |

### `m2.py`

| function | takes | returns | computes |
|----------|-------|---------|----------|
| `left_invariant_frame(shape, dx)` | grid shape `(θ, y, x)` | `sssul` | forward, sideways and turn frame `A_1, A_2, A_3` of `M2 = R² × S¹` |

### `gaussian_blur.py`

| function | takes | returns | computes |
|----------|-------|---------|----------|
| `gaussian_blur(field, sigma)` | `field: BSI` | `BSI` | Gaussian blur along `S`, periodic boundaries |

`make_figures.py` regenerates the figures.