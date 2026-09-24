# Gauge Derivatives

With this library you can calculate gauge derivatives of tensor fields on a manifold endowed with a metric tensor field and a connection.

A _field_ is a tensor-valued signal on a manifold.
A _gauge frame_ is a basis of vector fields derived from a scalar field, e.g. from its structure tensor or Hessian.
A _gauge derivative_ of a field is a derivative in a gauge frame direction.

More formally, let $M$ be an $n$-dimensional manifold, $\nabla : \Gamma(T^{(p,q)}M) \to \Gamma(T^{(p,q+1)} M)$ the total covariant derivative on tensor fields (induced by some connection), and $v_0, \dots, v_{n-1} \in \Gamma(T M)$ a gauge frame obtained from a scalar field $f : M \to \mathbb{R}$.
The gauge derivative of signature $(i_1, \dots i_m)$ is defined as
$$
    (\delta f)_{(i_1, \dots i_m)} := (\nabla^m f)(v_{i_1}, \dots, v_{i_m})
$$

## Gauge Derivatives on R2

![gauge frame and derivatives on R2](images/r2.svg)

```python
import torch
from field import Field
from frames import covariant_derivative_in_frame, structure_tensor_frame
from gaussian_blur import gaussian_blur
from geometry import levi_civita_difference_tensor

H, W = 64, 64
signal = Field(torch.randn(H, W), "ss")  # [64, 64] ss
signal = gaussian_blur(signal, sigma=2.0)  # [64, 64] ss
metric = Field(torch.eye(2).reshape(1, 1, 2, 2), "ssll")  # [1, 1, 2, 2] ssll
difference_tensor = levi_civita_difference_tensor(metric)  # [1, 1, 2, 2, 2] ssull
eigenvalues, gauge_frame = structure_tensor_frame(signal, 1.0, metric)  # [64, 64, 2] ssl, [64, 64, 2, 2] ssul
gauge_derivatives = covariant_derivative_in_frame(signal, gauge_frame, difference_tensor, 2)  # [64, 64, 2, 2] ssll
```

## Gauge Derivatives on  M2

![gauge frame and derivatives on M2](images/m2.webp)

```python
import torch
from field import Field
from frames import constant_metric_in_frame, covariant_derivative_in_frame, hessian_frame
from gaussian_blur import gaussian_blur
from geometry import weitzenbock_difference_tensor
from m2 import left_invariant_frame

O, H, W = 64, 64, 64
signal = Field(torch.randn(O, H, W), "sss")  # [64, 64, 64] sss
signal = gaussian_blur(signal, sigma=2.0)  # [64, 64, 64] sss
li_frame = left_invariant_frame(O)  # [64, 1, 1, 3, 3] sssul
difference_tensor = weitzenbock_difference_tensor(li_frame)  # [64, 1, 1, 3, 3, 3] sssull
G = torch.diag(torch.tensor([1.0, 4.0, 0.5]))
metric = constant_metric_in_frame(G, li_frame)  # [64, 1, 1, 3, 3] sssll
values, gauge_frame = hessian_frame(signal, difference_tensor, metric)  # [64, 64, 64, 3] sssl, [64, 64, 64, 3, 3] sssul
gauge_derivatives = covariant_derivative_in_frame(signal, gauge_frame, difference_tensor, 2)  # [64, 64, 64, 3, 3] sssll
```

## Tensor fields

A `Field` is a `torch.Tensor`, that being an array with a _shape_, with an additional _type string_ with `len(shape)=len(type)`.

| character | meaning |
|-----------|---------|
| `b` | batch |
| `s` | spatial |
| `u` | upper index |
| `l` | lower index |

Batch dimensions come first, then spatial, then indices, with upper and lower in any order.\
The number of `s`'s in the type is called `n` and is the number of spatial dimensions. \
Every index has size `n`.

Indices are components in some basis and the type string does not record which.\
This means the user has to keep track of which basis is used where.

### Common fields

On a 2D grid of `H×W` points, with a batch of `B`:

| shape | type | object | examples |
|-------|------|--------|----------|
| `[H,W]` | `ss` | (0,0)-tensor field, i.e. scalar field | image `f` |
| `[H,W,2]` | `ssu` | (1,0)-tensor field, i.e. vector field | gradient `g^ij (e_j f) e_i ` |
| `[H,W,2]` | `ssl` | (0,1)-tensor field, i.e. covector field | differential `e_i(f) e^i` |
| `[H,W,2,2]` | `ssll` | (0,2)-tensor field, i.e. bilinear form field | metric `g_ij e^i ⊗ e^j`, Hessian `(∇ ∇ f)_ij e^i ⊗ e^j`, structure tensor `(S f)_ij e^i ⊗ e^j` |
| `[H,W,2,2]` | `ssuu` | (2,0)-tensor field | inverse metric `g^ij e_i ⊗ e_j` |
| `[H,W,2,2]` | `ssul` | (1,1)-tensor field, i.e. linear map field | frame `F^i_j e_i ⊗ f^j`, linear mappings `A^i_j e_i ⊗ e^j` |
| `[H,W,2,…,2]` | `ssl…l` | (0,k)-tensor field | k-th covariant derivative of a scalar field |
| `[H,W,2,2,2]` | `ssull` | (1,2)-tensor field  | difference tensor `D^i_kl e_i ⊗ e^k ⊗ e^l`|

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
| `partial_derivative(field)` | `field: BSI` | `BSIl` | `∇^flat T` |
| `differential(field)` | `field: BS` | `BSl` | `df` |
| `gradient(field, metric)` | `field: BS`, `metric: Sll` | `BSu` | `grad f = g^ij e_j(f) e_i` |
| `inner_product(a, b, metric)` | `a, b: BSu` or `BSl`, `metric: Sll` | `BS` | `g_ij a^i b^j` or `g^ij a_i b_j` |
| `norm2(field, metric)` | `field: BSu` or `BSl`, `metric: Sll` | `BS` | `inner_product(field, field, metric)` |
| `levi_civita_difference_tensor(metric)` | `metric: Sll` | `Sull` | `D` of the Levi-Civita connection |
| `weitzenbock_difference_tensor(frame)` | `frame: Sul` | `Sull` | `D` of the connection for which the frame is parallel |
| `covariant_derivative(field, difference_tensor)` | `field: BSI`, `difference_tensor: Sull` | `BSIl` | `∇ T` with `∇ = ∇^flat + D` |

### `frames.py`

| function | takes | returns | computes |
|----------|-------|---------|----------|
| `eigenframe(form, metric)` | `form: BSll`, `metric: Sll` | `BSl`, `BSul` | stationary points of `F(v,v)` with `g(v,v) = 1` |
| `singular_frames(form, metric)` | `form: BSll`, `metric: Sll` | `BSl`, `BSul`, `BSul` | `σ` and the left and right singular frames of `F` in `g` |
| `structure_tensor_frame(field, sigma, metric)` | `field: BS`, `metric: Sll` | `BSl`, `BSul` | eigenframe of `df ⊗ df` blurred with `sigma` |
| `hessian_frame(field, difference_tensor, metric)` | `field: BS`, `difference_tensor: Sull`, `metric: Sll` | `BSl`, `BSul` | eigenframe of `∇∇f` |
| `constant_metric_in_frame(metric, frame)` | `metric: [n, n]`, `frame: Sul` | `Sll` | metric with components `G` in the frame |
| `covariant_derivative_in_frame(field, frame, difference_tensor, order)` | `field: BSI`, `frame: BSul`, `difference_tensor: Sull` | `BSI` + `order` × `l` | `order`-th covariant derivative, in the frame basis |

### `m2.py`

| function | takes | returns | computes |
|----------|-------|---------|----------|
| `left_invariant_frame(orientations, dx)` | number of orientations, grid step `dx` in y and x | `sssul` | forward, sideways and turn frame `A_1, A_2, A_3` of `M2 = R² × S¹` |

### `gaussian_blur.py`

| function | takes | returns | computes |
|----------|-------|---------|----------|
| `gaussian_blur(field, sigma)` | `field: BSI` | `BSI` | Gaussian blur along `S`, periodic boundaries |

`make_figures.py` regenerates the figures.