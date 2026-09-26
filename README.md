# Gauge Derivatives

With this library you can calculate gauge derivatives of tensor fields on a manifold endowed with a metric tensor field and a connection.

A _field_ is a tensor-valued signal on a manifold.
A _gauge frame_ is a basis of vector fields derived from a scalar field, e.g. from its structure tensor or Hessian.
A _gauge derivative_ of a field is a derivative in a gauge frame direction.

More formally, let $`M`$ be an $`n`$-dimensional manifold, $`\nabla : \Gamma(T^{(p,q)}M) \to \Gamma(T^{(p,q+1)} M)`$ the total covariant derivative on tensor fields (induced by some connection), and $`v_0, \dots, v_{n-1} \in \Gamma(T M)`$ a gauge frame obtained from a scalar field $`f : M \to \mathbb{R}`$.
The gauge derivative of signature $`(i_1, \dots i_m)`$ is defined as
$$
    (\delta f)_{(i_1, \dots i_m)} := (\nabla^m f)(v_{i_1}, \dots, v_{i_m})
$$

## Gauge Derivatives on R2

![gauge frame and derivatives on R2](images/r2.svg)

```python
import torch
from frames import covariant_derivative_in_frame, structure_tensor_frame
from gaussian_blur import gaussian_blur
from geometry import levi_civita_difference_tensor

H, W = 64, 64
signal = torch.randn(H, W)                                 # [64, 64]
signal = gaussian_blur(signal, sigma=2.0)                  # [64, 64]
metric = torch.eye(2).reshape(1, 1, 2, 2)                  # [1, 1, 2, 2]
difference_tensor = levi_civita_difference_tensor(metric)  # [1, 1, 2, 2, 2]
eigenvalues, gauge_frame = structure_tensor_frame(signal, 1.0, metric)  # [64, 64, 2], [64, 64, 2, 2]
gauge_derivatives = covariant_derivative_in_frame(signal, gauge_frame, difference_tensor, 2)  # [64, 64, 2, 2]
```

## Gauge Derivatives on  M2

![gauge frame and derivatives on M2](images/m2.webp)

```python
import torch
from frames import constant_metric_in_frame, covariant_derivative_in_frame, hessian_frame
from gaussian_blur import gaussian_blur
from geometry import weitzenbock_difference_tensor
from m2 import left_invariant_frame

O, H, W = 64, 64, 64
signal = torch.randn(O, H, W)                                # [64, 64, 64]
signal = gaussian_blur(signal, sigma=2.0)                    # [64, 64, 64]
li_frame = left_invariant_frame(O)                           # [64, 1, 1, 3, 3]
difference_tensor = weitzenbock_difference_tensor(li_frame)  # [64, 1, 1, 3, 3, 3]
G = torch.diag(torch.tensor([1.0, 4.0, 0.5]))
metric = constant_metric_in_frame(G, li_frame)               # [64, 1, 1, 3, 3]
values, gauge_frame = hessian_frame(signal, difference_tensor, metric)  # [64, 64, 64, 3], [64, 64, 64, 3, 3]
gauge_derivatives = covariant_derivative_in_frame(signal, gauge_frame, difference_tensor, 2)  # [64, 64, 64, 3, 3]
```