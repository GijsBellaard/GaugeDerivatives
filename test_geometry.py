"""Tests. Run with `python test_geometry.py`."""
import sys

import torch

from field import Field, change_basis, contract, tensor_product, trace
from frames import eigenframe, singular_frames
from gaussian_blur import gaussian_blur
from geometry import levi_civita_connection, covariant_derivative, partial_derivative

torch.set_default_dtype(torch.float64)

FAILURES = []
INTERIOR = (slice(4, -4), slice(4, -4))


def check(name, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {name:54} {detail}")
    if not condition:
        FAILURES.append(name)


def inverse(metric):
    """Inverse metric g^ab."""
    return Field(torch.linalg.inv(metric.data), metric.prefix_type + "uu")


def laplacian(field, metric, connection):
    """g^ab nabla_a nabla_b f of a scalar field."""
    hessian = covariant_derivative(covariant_derivative(field, connection), connection)
    return trace(contract(hessian, 0, inverse(metric), 0), 0, 1)


print("polar chart of flat 2D space")
# Derivatives are per grid step, so the step sizes go into the metric.
N = 300
r = torch.linspace(1.0, 3.0, N)
theta = torch.linspace(0.2, 2.0, N)
radius, angle = torch.meshgrid(r, theta, indexing="ij")
dr, dtheta = float(r[1] - r[0]), float(theta[1] - theta[0])
data = torch.zeros(N, N, 2, 2)
data[..., 0, 0] = dr**2
data[..., 1, 1] = (radius * dtheta)**2
metric = Field(data, "ssll")
levi_civita = levi_civita_connection(metric)

error = covariant_derivative(metric, levi_civita).data[INTERIOR].abs().max()
check("metric compatibility, nabla g = 0", error < 1e-12, f"{error:.2e}")

identity = contract(metric, 0, inverse(metric), 0)
error = (identity.data - torch.eye(2)).abs().max()
check("g contracted with its inverse is the identity", error < 1e-12, f"{error:.2e}")

exact = torch.zeros(N, N, 2, 2, 2)
exact[..., 0, 1, 1] = -radius * dtheta**2 / dr
exact[..., 1, 0, 1] = exact[..., 1, 1, 0] = dr / radius
error = (levi_civita.data - exact)[INTERIOR].abs().max()
check("connection matches the analytic polar values", error < 1e-9, f"{error:.2e}")

harmonic = Field(radius**2 * torch.cos(2 * angle), "ss")
hessian = covariant_derivative(covariant_derivative(harmonic, levi_civita), levi_civita)
error = (hessian.data - hessian.data.transpose(-1, -2))[INTERIOR].abs().max()
check("nabla nabla f is symmetric (no torsion)", error < 1e-9, f"{error:.2e}")

for name, values in (("r^2 cos 2th", radius**2 * torch.cos(2 * angle)),
                     ("log r", torch.log(radius)),
                     ("r^3 cos 3th", radius**3 * torch.cos(3 * angle))):
    field = Field(values, "ss")
    residual = laplacian(field, metric, levi_civita).data[INTERIOR]
    error = residual.abs().max() / values[INTERIOR].abs().max()
    check(f"laplacian of {name} is zero", error < 1e-3, f"{error:.2e} relative")

print("\ntype string")
flat = Field(torch.eye(2).repeat(8, 8, 1, 1), "ssll")
check("metric shape",
      flat.n == 2 and flat.indices_type == "ll", f"{tuple(flat.data.shape)}")
check("connection of a flat metric is zero",
      bool((levi_civita_connection(flat).data == 0).all()), f"{tuple(levi_civita_connection(flat).data.shape)}")

for bad, why in ((("ssu", (8, 8, 3)), "index of the wrong size"),
                 (("suls", (8, 2, 2, 8)), "dimensions out of order"),
                 (("ss", (8, 8, 2)), "type of the wrong length")):
    try:
        Field(torch.randn(*bad[1]), bad[0])
        check(f"rejects {why}", False)
    except ValueError as error:
        check(f"rejects {why}", True, str(error)[:38])

try:
    contract(flat, 0, flat, 1)
    check("rejects contracting two lower indices", False)
except ValueError:
    check("rejects contracting two lower indices", True)

check("derivative appends a lower index",
      partial_derivative(flat).type == "sslll", partial_derivative(flat).type)

print("\ngauge derivatives, flat plane")
field = gaussian_blur(Field(torch.randn(64, 80), "ss"), sigma=2.0)
flat = Field(torch.eye(2).repeat(64, 80, 1, 1), "ssll")
levi_civita = levi_civita_connection(flat)
df = covariant_derivative(field, levi_civita)
outer = tensor_product(df, df)
eigenvalues, frame = eigenframe(gaussian_blur(outer, sigma=1.0), flat)
check("eigenframe output types",
      eigenvalues.type == "ssl" and frame.type == "ssul", f"{eigenvalues.type!r} {frame.type!r}")

orthonormal = torch.einsum("...ai,...aj->...ij", frame.data, frame.data)
error = (orthonormal - torch.eye(2)).abs().max()
check("frame is orthonormal in g", error < 1e-10, f"{error:.2e}")

first = change_basis(df, frame)
along = torch.einsum("...a,...a->...", df.data, frame.data[..., 0])
error = (first.data[..., 0] - along).abs().max()
check("first order gauge derivative = df(frame vector)", error < 1e-14, f"{error:.2e}")

skewed = Field(frame.data + 0.5 * frame.data.flip(-1), "ssul")    # not orthonormal
covector, vector = Field(torch.randn(64, 80, 2), "ssl"), Field(torch.randn(64, 80, 2), "ssu")
before = contract(covector, 0, vector, 0).data
after = contract(change_basis(covector, skewed), 0, change_basis(vector, skewed), 0).data
error = (before - after).abs().max()
check("change of basis leaves contractions unchanged", error < 1e-12, f"{error:.2e}")

print("\nconnection with torsion")
# nabla_b nabla_a f - nabla_a nabla_b f = -T^m_ba d_m f, with T^m_ba = G^m_ba - G^m_ab.
field = Field(torch.randn(40, 50).cumsum(0).cumsum(1), "ss")
general = Field(torch.randn(40, 50, 2, 2, 2), "ssull")  # random G, with torsion
hessian = covariant_derivative(covariant_derivative(field, general), general)  # [..., a, b]
T = general.data - general.data.transpose(-1, -2)  # [..., m, b, a]
d = partial_derivative(field).data
expected = -torch.einsum("...mba,...m->...ab", T, d)
error = (hessian.data - hessian.data.mT - expected).abs().max()
check("Hessian antisymmetric part is -T^m_ba d_m f", error < 1e-12, f"{error:.2e}")

print("\nframes, curved metric, non-symmetric F")
g = torch.randn(9, 11, 2, 2)
g = g @ g.mT + 0.5 * torch.eye(2)
F = torch.randn(9, 11, 2, 2)
g_inv = torch.linalg.inv(g)
values, frame = eigenframe(Field(F, "ssll"), Field(g, "ssll"))
sigma, left, right = singular_frames(Field(F, "ssll"), Field(g, "ssll"))
for name, Q, v, lam in (
    ("eigenframe", (F + F.mT) / 2, frame.data, values.data),
    ("singular_frames, left", F @ g_inv @ F.mT, left.data, sigma.data.square()),
    ("singular_frames, right", F.mT @ g_inv @ F, right.data, sigma.data.square()),
):
    stationary = (Q @ v - g @ v * lam[..., None, :]).abs().max()
    orthonormal = (v.mT @ g @ v - torch.eye(2)).abs().max()
    error = max(stationary, orthonormal)
    check(f"{name}: Q v = lambda g v, v^T g v = I", error < 1e-12, f"{error:.2e}")
error = (left.data.mT @ F @ right.data - torch.diag_embed(sigma.data)).abs().max()
check("singular_frames: F(u_i, v_j) = sigma_i delta_ij", error < 1e-12, f"{error:.2e}")

print()
if FAILURES:
    print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
    sys.exit(1)
print("all checks passed")
