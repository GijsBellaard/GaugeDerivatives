from pathlib import Path

import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from field import Field, change_basis, tensor_product
from frames import eigenframe
from gaussian_blur import gaussian_blur
from geometry import levi_civita_difference_tensor, covariant_derivative

IMAGES = Path(__file__).parent / "images"

SIGMA = 2.0                  # Blur of the photograph
FRAME_SIGMA = 1.0            # Blur of the structure tensor
CROP = (120, 350, 160, 360)  # Region shown in the frame figure
STEP = 8                     # Draw a frame every STEP-th pixel
DERIVATIVE_SCALE = 2         # Downsampling of the photograph for the derivative figure

FRAME_COLORS = ("#2a78d6", "#eb6834")
ORDERS = [
    ([0], [1]),
    ([0, 0], [0, 1], [1, 1]),
    ([0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]),
]


def load_image(scale: int = 1) -> Field:
    """Grayscale sample photograph in [0, 1], downsampled by scale."""
    with cbook.get_sample_data("grace_hopper.jpg") as f:
        image = TF.pil_to_tensor(Image.open(f).convert("L"))[0] / 255
    h, w = image.shape
    image = TF.resize(image[None], [h // scale, w // scale], antialias=True)[0]
    return Field(image, "ss")


def euclidean(shape: tuple[int, ...]) -> Field:
    """Euclidean metric on a grid of the given shape."""
    n = len(shape)
    return Field(torch.eye(n).repeat(*shape, 1, 1), "s" * n + "ll")


def structure_tensor(field: Field, sigma: float, difference_tensor: Field) -> Field:
    """df ⊗ df, blurred."""
    df = covariant_derivative(field, difference_tensor)
    return gaussian_blur(tensor_product(df, df), sigma)


def hessian(field: Field, difference_tensor: Field) -> Field:
    """∇∇f."""
    return covariant_derivative(covariant_derivative(field, difference_tensor), difference_tensor)


def show_gauge_frame(ax, image, frame, r0, r1, c0, c1, step):
    ax.imshow(image.data[r0:r1, c0:c1], cmap="gray", extent=(c0, c1, r1, r0))
    ys, xs = torch.meshgrid(torch.arange(r0, r1, step), torch.arange(c0, c1, step), indexing="ij")
    for i, color in enumerate(FRAME_COLORS):
        v = frame.data[r0:r1:step, c0:c1:step, :, i]
        ax.quiver(
            xs, ys,
            v[..., 1], v[..., 0],
            color=color, pivot="mid", angles="xy",
            headwidth=0, headlength=0, headaxislength=0,
            scale=40, width=0.004
        )


def show_gauge_derivative(ax, component, signature, quantile=0.99):
    limit = component.abs().quantile(quantile).item()
    im = ax.imshow(component, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_title(f"signature {signature}", fontsize=10)
    bar = ax.figure.colorbar(im, ax=ax, shrink=0.5)
    bar.outline.set_visible(False)


def make_frame_figure(path: Path) -> None:
    blurred = gaussian_blur(load_image(), sigma=SIGMA)
    metric = euclidean(blurred.data.shape)
    levi_civita = levi_civita_difference_tensor(metric)
    frames = (
        ("Structure tensor",
         eigenframe(structure_tensor(blurred, FRAME_SIGMA, levi_civita), metric)[1]),
        ("Hessian", eigenframe(hessian(blurred, levi_civita), metric)[1])
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    fig.suptitle("Gauge frames from eigenvectors", fontsize=11)
    for ax, (title, frame) in zip(axes, frames):
        ax.set_title(title, fontsize=10)
        show_gauge_frame(ax, blurred, frame, *CROP, STEP)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_derivative_figure(path: Path) -> None:
    blurred = gaussian_blur(load_image(DERIVATIVE_SCALE), sigma=SIGMA / DERIVATIVE_SCALE)
    metric = euclidean(blurred.data.shape)
    levi_civita = levi_civita_difference_tensor(metric)
    _, frame = eigenframe(structure_tensor(blurred, FRAME_SIGMA, levi_civita), metric)

    fig, axes = plt.subplots(len(ORDERS), max(len(row) for row in ORDERS), figsize=(16, 13))
    derivative = blurred
    for row, signatures in zip(axes, ORDERS):  # row i holds order i + 1
        derivative = covariant_derivative(derivative, levi_civita)
        in_frame = derivative
        for index in range(len(derivative.indices_type)):
            in_frame = change_basis(in_frame, frame, index)
        for ax, signature in zip(row, signatures):
            show_gauge_derivative(ax, in_frame.data[..., *signature], signature)
        for ax in row[len(signatures):]:
            ax.set_axis_off()
    fig.suptitle("First-, second- and third-order gauge derivatives", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)

    make_frame_figure(IMAGES / "gauge_frame.svg")
    make_derivative_figure(IMAGES / "gauge_derivatives.svg")
