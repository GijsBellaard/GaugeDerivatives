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
| `[H,W]` | `ss` | (0,0)-tensor field, i.e. scalar field | image $f$ |
| `[H,W,2]` | `ssu` | (1,0)-tensor field, i.e. vector field | gradient $g^{ij} (e_j f)\, e_i$ |
| `[H,W,2]` | `ssl` | (0,1)-tensor field, i.e. covector field | differential $(e_i f)\, e^i$ |
| `[H,W,2,2]` | `ssll` | (0,2)-tensor field, i.e. bilinear form field | metric $g_{ij}\, e^i \otimes e^j$, Hessian $(\nabla\nabla f)_{ij}\, e^i \otimes e^j$, structure tensor $(Sf)_{ij}\, e^i \otimes e^j$ |
| `[H,W,2,2]` | `ssuu` | (2,0)-tensor field | inverse metric $g^{ij}\, e_i \otimes e_j$ |
| `[H,W,2,2]` | `ssul` | (1,1)-tensor field, i.e. linear map field | frame $V^i{}_j\, e_i \otimes v^j$, linear mappings $A^i{}_j\, e_i \otimes e^j$ |
| `[H,W,2,…,2]` | `ssl…l` | (0,k)-tensor field | $k$-th covariant derivative $(\nabla^k f)_{i_1 \dots i_k}\, e^{i_1} \otimes \dots \otimes e^{i_k}$ of a scalar field |
| `[H,W,2,2,2]` | `ssull` | (1,2)-tensor field | difference tensor $D^i{}_{kl}\, e_i \otimes e^k \otimes e^l$ |

## Functions

$e_i$ and $e^i$ are the grid basis and its dual, $E_i$ and $E^i$ any basis, and a frame is
$v_j = V^i{}_j\, e_i$. For generic tensor fields, `B` is any number of batch dimensions, `S`
the spatial dimensions, and `I`, `J` any string of `u` and `l`.

### `field.py`

`Field(data, type)`, and:

| function | takes | returns |
|----------|-------|---------|
| `tensor_product(a, b)` | $a$ of type `BSI`, $b$ of type `BSJ`, in the same basis | $a \otimes b$ of type `BSIJ`, e.g. $a_i b_j\, E^i \otimes E^j$ |
| `contract(a, i, b, j)` | $a$ of type `BSI`, $b$ of type `BSJ`, index `i` of $a$ and `j` of $b$ in the same basis | type `BSIJ` without the summed pair, e.g. $a^{ij} b_j\, E_i$ |
| `trace(field, i, j)` | $T$ of type `BSI`, indices `i` and `j` in the same basis | type `BSI` without the summed pair, e.g. $T^i{}_i$ |
| `change_basis(field, frame, index)` | $T$ of type `BSI` with index `index` in $E_i$ or $E^i$, frame $v_j = V^i{}_j\, E_i$ | type `BSI` with that index in $v_j$ or $v^j$ |

### `geometry.py`

| function | takes | returns |
|----------|-------|---------|
| `partial_derivative(field)` | $T$ of type `BSI` in $e_i$ and $e^i$ | $\nabla^\flat T$ of type `BSIl`, derivative index last |
| `differential(field)` | scalar field $f$ | $df = (e_i f)\, e^i$ |
| `gradient(field, metric)` | scalar field $f$, metric $g_{ij}\, e^i \otimes e^j$ | $\operatorname{grad} f = g^{ij} (e_j f)\, e_i$ |
| `inner_product(a, b, metric)` | $a^i E_i$ and $b^i E_i$ (or $a_i E^i$ and $b_i E^i$), metric $g_{ij}\, E^i \otimes E^j$ | $g_{ij} a^i b^j$ (or $g^{ij} a_i b_j$) |
| `norm2(field, metric)` | $a^i E_i$ (or $a_i E^i$), metric $g_{ij}\, E^i \otimes E^j$ | $g_{ij} a^i a^j$ (or $g^{ij} a_i a_j$) |
| `levi_civita_difference_tensor(metric)` | metric $g_{ij}\, e^i \otimes e^j$ | $D^i{}_{jk}\, e_i \otimes e^j \otimes e^k$ of the Levi-Civita connection |
| `weitzenbock_difference_tensor(frame)` | frame $v_j = V^i{}_j\, e_i$ | $D^i{}_{jk}\, e_i \otimes e^j \otimes e^k$ of the connection for which the frame is parallel |
| `covariant_derivative(field, difference_tensor)` | $T$ of type `BSI` in $e_i$ and $e^i$, $D^i{}_{jk}\, e_i \otimes e^j \otimes e^k$ | $\nabla T$ of type `BSIl` with $\nabla = \nabla^\flat + D$, derivative index last |

### `frames.py`

| function | takes | returns |
|----------|-------|---------|
| `eigenframe(form, metric)` | form $F_{ij}\, E^i \otimes E^j$, metric $g_{ij}\, E^i \otimes E^j$ | values $\lambda_j$ and frame $v_j = V^i{}_j\, E_i$, the stationary points of $F(v,v)$ with $g(v,v) = 1$ |
| `singular_frames(form, metric)` | form $F_{ij}\, E^i \otimes E^j$, metric $g_{ij}\, E^i \otimes E^j$ | singular values $\sigma_j$, left frame $u_j = U^i{}_j\, E_i$ and right frame $v_j = V^i{}_j\, E_i$ |
| `structure_tensor_frame(field, sigma, metric)` | scalar field $f$, blur `sigma`, metric $g_{ij}\, e^i \otimes e^j$ | values $\lambda_j$ and frame $v_j = V^i{}_j\, e_i$ of $df \otimes df$ blurred |
| `hessian_frame(field, difference_tensor, metric)` | scalar field $f$, $D^i{}_{jk}\, e_i \otimes e^j \otimes e^k$, metric $g_{ij}\, e^i \otimes e^j$ | values $\lambda_j$ and frame $v_j = V^i{}_j\, e_i$ of $\nabla\nabla f$ |
| `constant_metric_in_frame(metric, frame)` | `[n, n]` tensor $G$, frame $v_j = V^i{}_j\, e_i$ | metric $g_{ij}\, e^i \otimes e^j = G_{ij}\, v^i \otimes v^j$ |
| `covariant_derivative_in_frame(field, frame, difference_tensor, order)` | $T$ of type `BSI` in $e_i$ and $e^i$, frame $v_j = V^i{}_j\, e_i$, $D^i{}_{jk}\, e_i \otimes e^j \otimes e^k$ | $\nabla^m T$ of type `BSI` + $m$ × `l` in $v_i$ and $v^i$, for $f$ the gauge derivatives $(\nabla^m f)(v_{i_1}, \dots, v_{i_m})$ |

### `m2.py`

| function | takes | returns |
|----------|-------|---------|
| `left_invariant_frame(orientations, dx)` | number of orientations, grid step `dx` in $y$ and $x$ | frame $A_j = V^i{}_j\, e_i$ of $M_2 = \mathbb{R}^2 \times S^1$, forward $A_0$, sideways $A_1$ and turn $A_2$ |

### `gaussian_blur.py`

| function | takes | returns |
|----------|-------|---------|
| `gaussian_blur(field, sigma)` | $T$ of type `BSI` in $e_i$ and $e^i$, standard deviation `sigma` in grid steps | type `BSI`, blurred along `S` with periodic boundaries |

`make_figures.py` regenerates the figures.