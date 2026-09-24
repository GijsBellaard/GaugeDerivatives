from pathlib import Path

import matplotlib
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
import numpy as np
import pyvista as pv
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from field import Field
from frames import (constant_metric_in_frame, covariant_derivative_in_frame, hessian_frame,
                    structure_tensor_frame)
from gaussian_blur import gaussian_blur
from geometry import levi_civita_difference_tensor, weitzenbock_difference_tensor
from m2 import left_invariant_frame

IMAGES = Path(__file__).parent / "images"
FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"  # has θ and π

SIGMA = 2.0                  # Blur of the photograph
FRAME_SIGMA = 1.0            # Blur of the structure tensor
CROP = (120, 350, 160, 360)  # Region shown in the frame figure
STEP = 8                     # Draw a frame every STEP-th pixel
DERIVATIVE_SCALE = 2         # Downsampling of the photograph for the derivative figure

ORIENTATIONS = 128           # Grid of the ribbon on M2, [ORIENTATIONS, SIZE, SIZE]
SIZE = 88
SPACING = 0.5                # Grid step in y and x
RADIUS = 12.0                # Radius of the ribbon's core circle
WIDTH = 4.0                  # Standard deviations of the ribbon across and through it
THICKNESS = 1.5
XI = 4.0
METRIC = torch.diag(torch.tensor([1.0, 1.0, XI**2]))  # in the frame A
MARGIN = 16                  # Orientations at either end left out, as θ is not periodic
THETA_STEP = 8               # Draw a frame every THETA_STEP-th orientation
RIBBON_SIGMA = 8.0           # Blur of the structure tensor of the ribbon, in grid steps
SPACE_TICKS = (-20, -10, 0, 10, 20)
THETA_TICKS = {0: "0", 0.5: "π/2", 1: "π", 1.5: "3π/2", 2: "2π"}  # θ / π and its label

FRAME_COLORS = ("#2a78d6", "#eb6834")
RIBBON_COLORS = (*FRAME_COLORS, "#3aa655")
RIBBON_DIRECTIONS = (1, 2)   # Across and through, as (δf)_0 along the ribbon is zero


def load_image(scale: int = 1) -> Field:
    """Grayscale sample photograph in [0, 1], downsampled by scale."""
    with cbook.get_sample_data("grace_hopper.jpg") as f:
        image = TF.pil_to_tensor(Image.open(f).convert("L"))[0] / 255
    h, w = image.shape
    image = TF.resize(image[None], [h // scale, w // scale], antialias=True)[0]
    return Field(image, "ss")


def euclidean(n: int) -> Field:
    """Euclidean metric on an n-dimensional grid, constant so of shape [1, ..., 1, n, n]."""
    return Field(torch.eye(n).reshape(*[1] * n, n, n), "s" * n + "ll")


def ribbon_directions() -> tuple[torch.Tensor, torch.Tensor]:
    """Unit vectors across and through the ribbon, in the frame A.

    Across is A_2. Through is orthogonal to it and to the core's tangent T = R A_1 + A_3 in
    METRIC: the covector T × A_2 = (-1, 0, R) annihilates both, raised with the metric.
    """
    across = torch.tensor([0.0, 1.0, 0.0])
    through = torch.linalg.solve(METRIC, torch.tensor([-1.0, 0.0, RADIUS]))
    return across, through / (through @ METRIC @ through).sqrt()


def helical_ribbon(theta: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> Field:
    """Ribbon on M2 around the lifted circle (x, y, θ) = (R cos t, R sin t, t + π/2).

    Gaussian with standard deviations WIDTH across it and THICKNESS through it, see
    ribbon_directions.
    """
    phi = torch.atan2(y, x)
    wrapped = torch.remainder(theta - phi + torch.pi / 2, 2 * torch.pi) - torch.pi
    radial = RADIUS - torch.sqrt(x**2 + y**2)
    offset = torch.stack([torch.zeros_like(phi), radial, wrapped], dim=-1)  # in A at t = phi
    across, through = ribbon_directions()
    u = torch.einsum("...i,ij,j->...", offset, METRIC, across)
    v = torch.einsum("...i,ij,j->...", offset, METRIC, through)
    return Field(torch.exp(-u**2 / (2 * WIDTH**2) - v**2 / (2 * THICKNESS**2)), "sss")


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


def show_gauge_derivative(ax, component, title, cmap, quantile=0.99):
    """Image of a gauge derivative, symmetric around 0 if it takes negative values."""
    limit = component.abs().quantile(quantile).item()
    vmin = -limit if component.min() < 0 else 0
    im = ax.imshow(component, cmap=cmap, vmin=vmin, vmax=limit)
    ax.set_title(title, fontsize=12)
    bar = ax.figure.colorbar(im, ax=ax, shrink=0.5)
    bar.outline.set_visible(False)


def make_frame_figure(path: Path) -> None:
    blurred = gaussian_blur(load_image(), sigma=SIGMA)
    metric = euclidean(blurred.n)
    levi_civita = levi_civita_difference_tensor(metric)
    frames = (
        ("Structure Tensor", structure_tensor_frame(blurred, FRAME_SIGMA, metric)[1]),
        ("Hessian", hessian_frame(blurred, levi_civita, metric)[1])
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    fig.suptitle("Gauge frame", fontsize=11)
    for ax, (title, frame) in zip(axes, frames):
        ax.set_title(title, fontsize=10)
        show_gauge_frame(ax, blurred, frame, *CROP, STEP)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_derivative_figure(path: Path) -> None:
    blurred = gaussian_blur(load_image(DERIVATIVE_SCALE), sigma=SIGMA / DERIVATIVE_SCALE)
    metric = euclidean(blurred.n)
    levi_civita = levi_civita_difference_tensor(metric)
    _, frame = structure_tensor_frame(blurred, FRAME_SIGMA, metric)

    # The sign of v_1 is arbitrary, so that of (δf)_1 is too, but not that of (δf)_11.
    first = covariant_derivative_in_frame(blurred, frame, levi_civita, 1).data[..., 1].abs()
    second = covariant_derivative_in_frame(blurred, frame, levi_civita, 2).data[..., 1, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    show_gauge_derivative(ax1, first, r"$|(\delta f)_1|$",
                          derivative_cmap("first", FRAME_COLORS[1]))
    show_gauge_derivative(ax2, second, r"$(\delta f)_{11}$", "RdBu_r")
    fig.suptitle("Gauge derivatives", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def ribbon() -> tuple[Field, Field, Field]:
    """Helical ribbon on its grid, its structure tensor frame and the Weitzenböck difference
    tensor.

    With a blur as large as the ribbon, the frame is along, across and through it over its
    whole cross-section, unlike the Hessian frame, which is only ordered so near its core.
    """
    A = left_invariant_frame(ORIENTATIONS, SPACING)
    f = helical_ribbon(*ribbon_coordinates())
    _, frame = structure_tensor_frame(f, RIBBON_SIGMA, constant_metric_in_frame(METRIC, A))
    return f, frame, weitzenbock_difference_tensor(A)


def in_plot(vector: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """A vector with components in the frame A at orientation θ, in (x, y, XI θ)."""
    c, s = theta.cos(), theta.sin()
    return torch.stack([vector[0] * c - vector[1] * s, vector[0] * s + vector[1] * c,
                        XI * vector[2]])


def ribbon_coordinates() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """θ, y and x on the ribbon's grid, centred in y and x."""
    space = (torch.arange(SIZE) - (SIZE - 1) / 2) * SPACING
    return torch.meshgrid(torch.arange(ORIENTATIONS) * 2 * torch.pi / ORIENTATIONS,
                          space, space,
                          indexing="ij")


def ribbon_volume(**arrays: torch.Tensor) -> pv.ImageData:
    """Arrays on the ribbon's grid as a volume in (x, y, XI θ).

    A_1, A_2 and A_3 / XI are orthonormal in the metric, so in these coordinates it looks
    Euclidean.
    """
    center = (SIZE - 1) / 2 * SPACING
    volume = pv.ImageData(dimensions=(SIZE, SIZE, ORIENTATIONS),
                          spacing=(SPACING, SPACING, XI * 2 * torch.pi / ORIENTATIONS),
                          origin=(-center, -center, 0))
    for name, data in arrays.items():
        volume.point_data[name] = data.flatten().numpy()
    return volume


def segments(pairs: list) -> pv.PolyData:
    return pv.line_segments_from_points(torch.tensor(pairs).reshape(-1, 3).numpy())


def show_ribbon_axes(plotter: pv.Plotter) -> None:
    """Axes x and y along the front edges of the floor and θ up its left corner, a grid on the
    floor and the two back walls, and the camera."""
    h, top = (SIZE - 1) / 2 * SPACING, XI * 2 * torch.pi
    heights = [XI * torch.pi * t for t in THETA_TICKS]
    grid = []
    for t in SPACE_TICKS:
        grid += [((t, -h, 0), (t, h, 0)), ((-h, t, 0), (h, t, 0)),
                 ((t, h, 0), (t, h, top)), ((-h, t, 0), (-h, t, top))]
    for z in heights:
        grid += [((-h, -h, z), (-h, h, z)), ((-h, h, z), (h, h, z))]
    tick, pad = 1.5, 4.5
    axes = [((-h, -h, 0), (h, -h, 0)), ((h, -h, 0), (h, h, 0)), ((-h, -h, 0), (-h, -h, top))]
    axes += [((t, -h, 0), (t, -h - tick, 0)) for t in SPACE_TICKS]
    axes += [((h, t, 0), (h + tick, t, 0)) for t in SPACE_TICKS]
    axes += [((-h, -h, z), (-h - tick, -h - tick, z)) for z in heights]
    plotter.add_mesh(segments(grid), color="#d0d0d0", line_width=1)
    plotter.add_mesh(segments(axes), color="black", line_width=2)

    points = ([(t, -h - pad, 0) for t in SPACE_TICKS] + [(h + pad, t, 0) for t in SPACE_TICKS]
              + [(-h - pad, -h - pad, z) for z in heights]
              + [(0, -h - 2.5 * pad, 0), (h + 2.5 * pad, 0, 0), (-h, -h, top + 1.5 * pad)])
    labels = ([f"{t}".replace("-", "−") for t in 2 * SPACE_TICKS] + list(THETA_TICKS.values())
              + ["x", "y", "θ"])
    plotter.add_point_labels(points, labels, font_file=str(FONT), font_size=24, bold=False,
                             text_color="black", show_points=False, shape=None,
                             always_visible=True)
    plotter.camera_position = [(75, -75, 55), (0, 0, XI * torch.pi), (0, 0, 1)]
    plotter.reset_camera()
    plotter.camera.zoom(0.9)


def add_density(plotter: pv.Plotter, volume: pv.ImageData, name: str, limit: float,
                cmap: str | Colormap = "magma", opacity: str | list = "linear") -> None:
    """Volume rendering of an array of volume, opacity from 0 to limit."""
    density = plotter.add_volume(volume, scalars=name, cmap=cmap, opacity=opacity,
                                 clim=(0, limit), show_scalar_bar=False)
    density.prop.interpolation_type = "linear"


def ribbon_plotter() -> pv.Plotter:
    return pv.Plotter(off_screen=True, window_size=(1400, 1100))


def screenshots(*plotters: pv.Plotter, margin: int = 10) -> list[np.ndarray]:
    """Screenshots of the plotters, which are closed, cropped alike to what is drawn."""
    images = [plotter.screenshot(return_img=True) for plotter in plotters]
    for plotter in plotters:
        plotter.close()
    drawn = np.any([(image < 250).any(-1) for image in images], axis=0)
    rows, cols = np.nonzero(drawn.any(1))[0], np.nonzero(drawn.any(0))[0]
    r0, c0 = max(rows[0] - margin, 0), max(cols[0] - margin, 0)
    r1, c1 = rows[-1] + margin + 1, cols[-1] + margin + 1
    return [image[r0:r1, c0:c1] for image in images]


def show_rendering(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(image)
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()


def add_colorbar(ax, cmap: str | Colormap, limit: float) -> None:
    bar = ax.figure.colorbar(ScalarMappable(Normalize(0, limit), cmap), ax=ax, shrink=0.5)
    bar.outline.set_visible(False)


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def make_ribbon_signal_figure(path: Path, f: Field) -> None:
    plotter = ribbon_plotter()
    add_density(plotter, ribbon_volume(f=f.data), "f", 1.0, opacity="sigmoid_8")
    show_ribbon_axes(plotter)
    [image] = screenshots(plotter)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    show_rendering(ax, image, "Ribbon signal $f$ on M2")
    add_colorbar(ax, "magma", 1.0)
    save(fig, path)


def make_ribbon_frame_figure(path: Path, f: Field, frame: Field) -> None:
    theta, y, x = ribbon_coordinates()
    plotter = ribbon_plotter()
    add_density(plotter, ribbon_volume(f=f.data), "f", 1.0, cmap="Greys", opacity=[0, 0.2])
    steps = torch.tensor([XI * 2 * torch.pi / ORIENTATIONS, SPACING, SPACING])  # of e_θ, e_y, e_x
    for k in range(MARGIN, ORIENTATIONS - MARGIN, THETA_STEP):
        # Grid point nearest to the core at t = θ - π/2
        i = round(-RADIUS * theta[k, 0, 0].cos().item() / SPACING + (SIZE - 1) / 2)
        j = round(RADIUS * theta[k, 0, 0].sin().item() / SPACING + (SIZE - 1) / 2)
        point = torch.stack([x[k, i, j], y[k, i, j], XI * theta[k, i, j]])
        for n, color in enumerate(RIBBON_COLORS):
            v = 2.0 * (frame.data[k, i, j, :, n] * steps).flip(0)  # (x, y, XI θ)
            tube = pv.Tube(pointa=(point - v).tolist(), pointb=(point + v).tolist(), radius=0.25)
            plotter.add_mesh(tube, color=color)
    show_ribbon_axes(plotter)
    plotter.enable_depth_peeling()
    [image] = screenshots(plotter)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    show_rendering(ax, image, "Structure tensor frame")
    save(fig, path)


def ribbon_derivatives(f: Field, frame: Field,
                       difference_tensor: Field) -> tuple[pv.ImageData, dict[str, str], float]:
    """Volume with f and the first gauge derivatives |(δf)_i| for i in RIBBON_DIRECTIONS,
    their names with their colors, and their maximum.

    The sign of each frame vector is arbitrary, hence |(δf)_i|.
    """
    first = covariant_derivative_in_frame(f, frame, difference_tensor, 1).data.abs()
    names = {rf"$|(\delta f)_{i}|$": i for i in RIBBON_DIRECTIONS}
    volume = ribbon_volume(f=f.data, **{name: first[..., i] for name, i in names.items()})
    colors = {name: RIBBON_COLORS[i] for name, i in names.items()}
    return volume, colors, first[..., RIBBON_DIRECTIONS].max().item()


def derivative_cmap(name: str, color: str) -> Colormap:
    return LinearSegmentedColormap.from_list(name, ["white", color, "black"])


def make_ribbon_derivative_figure(path: Path, volume: pv.ImageData, colors: dict[str, str],
                                  limit: float) -> None:
    outline = volume.contour([0.5], scalars="f")
    plotters = []
    for name, color in colors.items():
        plotter = ribbon_plotter()
        add_density(plotter, volume, name, limit, cmap=derivative_cmap(name, color))
        plotter.add_mesh(outline, color="#888888", opacity=0.12)
        show_ribbon_axes(plotter)
        plotters.append(plotter)
    images = screenshots(*plotters)

    fig, axes = plt.subplots(1, len(colors), figsize=(6.5 * len(colors), 4.4))
    for ax, image, (name, color) in zip(axes, images, colors.items()):
        show_rendering(ax, image, name)
        add_colorbar(ax, derivative_cmap(name, color), limit)
    fig.suptitle("Gauge derivatives", fontsize=13)
    save(fig, path)


def make_ribbon_cross_section_figure(path: Path, volume: pv.ImageData, colors: dict[str, str],
                                     limit: float) -> None:
    # Cross-section perpendicular to the core at θ = π, sampled in (x, y, XI θ) along the unit
    # vectors of ribbon_directions.
    extent = 7.0
    across, through = torch.meshgrid(torch.linspace(-extent, extent, 141),
                                     torch.linspace(-extent, extent, 141), indexing="xy")
    theta = torch.tensor(torch.pi)
    t = theta - torch.pi / 2
    center = torch.stack([RADIUS * t.cos(), RADIUS * t.sin(), XI * theta])
    a2, n = (in_plot(direction, theta) for direction in ribbon_directions())
    points = center + across[..., None] * a2 + through[..., None] * n
    sampled = pv.PolyData(points.reshape(-1, 3).numpy()).sample(volume)

    panels = [("f", "$f$", "Greys", 1.0)] + [(name, name, derivative_cmap(name, color), limit)
                                             for name, color in colors.items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.2))
    for ax, (name, title, cmap, top) in zip(axes, panels):
        ax.imshow(sampled[name].reshape(across.shape), origin="lower", cmap=cmap, vmin=0,
                  vmax=top, extent=(-extent, extent, -extent, extent))
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("across")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_ylabel("through")
    fig.suptitle("Cross-section at θ = π", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)

    make_frame_figure(IMAGES / "gauge_frame.svg")
    make_derivative_figure(IMAGES / "gauge_derivatives.svg")

    f, frame, difference_tensor = ribbon()
    derivatives = ribbon_derivatives(f, frame, difference_tensor)
    make_ribbon_signal_figure(IMAGES / "ribbon_signal.png", f)
    make_ribbon_frame_figure(IMAGES / "ribbon_frame.png", f, frame)
    make_ribbon_derivative_figure(IMAGES / "ribbon_derivatives.png", *derivatives)
    make_ribbon_cross_section_figure(IMAGES / "ribbon_cross_section.svg", *derivatives)
