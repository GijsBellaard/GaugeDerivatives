from pathlib import Path

import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image

from derivative import gauge_derivative
from frames import gauge_frame_hessian, gauge_frame_structure_tensor
from gaussian_blur import gaussian_blur

IMAGES = Path(__file__).parent / "images"

SIGMA = 2.0                 # Pre blur applied to the photograph, as in main.ipynb
FRAME_SIGMA = 1.0           # Post blur of the structure tensor
CROP = (120, 350, 160, 360) # Region of the photograph the frame figure zooms in on
STEP = 8                    # Draw a frame every STEP-th pixel
DERIVATIVE_SCALE = 2        # The derivative figure uses a photograph this many times smaller

FRAME_COLORS = ("#2a78d6", "#eb6834")
ORDERS = [
    ([0], [1]),
    ([0, 0], [0, 1], [1, 1]),
    ([0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]),
]

def load_image(scale: int = 1) -> torch.Tensor:
    with cbook.get_sample_data("grace_hopper.jpg") as f:
        image = torchvision.transforms.functional.pil_to_tensor(Image.open(f).convert("L"))[0] / 255
    h, w = image.shape
    return torchvision.transforms.functional.resize(image[None], [h // scale, w // scale], antialias=True)[0]


def show_gauge_frame(ax, image, frame, r0, r1, c0, c1, step):
    ax.imshow(image[r0:r1, c0:c1], cmap="gray", extent=(c0, c1, r1, r0))
    ys, xs = torch.meshgrid(torch.arange(r0, r1, step), torch.arange(c0, c1, step), indexing="ij")
    for i, color in enumerate(FRAME_COLORS):
        v = frame[r0:r1:step, c0:c1:step, :, i]
        ax.quiver(
            xs, ys,
            v[..., 1], v[..., 0],
            color=color, pivot="mid", angles="xy",
            headwidth=0, headlength=0, headaxislength=0,
            scale=40, width=0.004
        )

def show_gauge_derivative(ax, image, frame, signature, quantile=0.99):
    field = gauge_derivative(image, frame, signature)
    v = field.abs().quantile(quantile).item()
    im = ax.imshow(field, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_title(f"signature {signature}", fontsize=10)
    bar = ax.figure.colorbar(im, ax=ax, shrink=0.5)
    bar.outline.set_visible(False)


def make_frame_figure(path: Path) -> None:
    blurred = gaussian_blur(load_image(), sigma=SIGMA)
    frames = (
        ("Structure tensor", gauge_frame_structure_tensor(blurred, sigma=FRAME_SIGMA)),
        ("Hessian", gauge_frame_hessian(blurred))
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    fig.suptitle("Gauge frames obtained by taking eigenbasis of...", fontsize=11)
    for ax, (title, frame) in zip(axes, frames):
        ax.set_title(title, fontsize=10)
        show_gauge_frame(ax, blurred, frame, *CROP, STEP)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_derivative_figure(path: Path) -> None:
    blurred = gaussian_blur(load_image(DERIVATIVE_SCALE), sigma=SIGMA / DERIVATIVE_SCALE)
    frame = gauge_frame_structure_tensor(blurred, sigma=FRAME_SIGMA)

    fig, axes = plt.subplots(len(ORDERS), max(len(row) for row in ORDERS), figsize=(16, 13))
    for row, signatures in zip(axes, ORDERS):
        for ax, signature in zip(row, signatures):
            show_gauge_derivative(ax, blurred, frame, signature)
        for ax in row[len(signatures):]:
            ax.set_axis_off()
    fig.suptitle("first-, second- and third-order gauge derivatives", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)

    make_frame_figure(IMAGES / "gauge_frame.svg")
    make_derivative_figure(IMAGES / "gauge_derivatives.svg")
