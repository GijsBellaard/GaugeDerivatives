from pathlib import Path

import matplotlib
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap
import pyvista as pv
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from field import Field
from frames import constant_metric_in_frame, gauge_jet, hessian_frame, structure_tensor_frame
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


def euclidean(n: int) -> Field:
    """Euclidean metric on an n-dimensional grid, constant so of shape [1, ..., 1, n, n]."""
    return Field(torch.eye(n).reshape(*[1] * n, n, n), "s" * n + "ll")


def helical_ribbon(theta: torch.Tensor, y: torch.Tensor, x: torch.Tensor,
                   metric: torch.Tensor) -> Field:
    """Ribbon on M2 around the lifted circle (x, y, θ) = (R cos t, R sin t, t + π/2).

    Gaussian with standard deviations WIDTH across it, along A_2, and THICKNESS through it,
    along N. Both are orthonormal in the metric and orthogonal to the core's tangent
    T = R A_1 + A_3. On the core the Hessian frame is N, A_2, T.
    """
    phi = torch.atan2(y, x)
    wrapped = torch.remainder(theta - phi + torch.pi / 2, 2 * torch.pi) - torch.pi
    radial = RADIUS - torch.sqrt(x**2 + y**2)
    offset = torch.stack([torch.zeros_like(phi), radial, wrapped], dim=-1)  # in A at t = phi
    across = torch.tensor([0.0, 1.0, 0.0])
    through = torch.tensor([-XI**2, 0.0, RADIUS]) / (XI * (XI**2 + RADIUS**2) ** 0.5)
    u = torch.einsum("...i,ij,j->...", offset, metric, across)
    v = torch.einsum("...i,ij,j->...", offset, metric, through)
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


def show_gauge_derivative(ax, component, signature, quantile=0.99):
    limit = component.abs().quantile(quantile).item()
    im = ax.imshow(component, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_title(f"signature {signature}", fontsize=10)
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

    fig, axes = plt.subplots(len(ORDERS), max(len(row) for row in ORDERS), figsize=(16, 13))
    for order, (row, signatures) in enumerate(zip(axes, ORDERS), start=1):
        in_frame = gauge_jet(blurred, frame, levi_civita, order)
        for ax, signature in zip(row, signatures):
            show_gauge_derivative(ax, in_frame.data[..., *signature], signature)
        for ax in row[len(signatures):]:
            ax.set_axis_off()
    fig.suptitle("First-, second- and third-order gauge derivatives", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def ribbon() -> tuple[Field, Field, Field]:
    """Helical ribbon on its grid, its Hessian frame and the Weitzenböck difference tensor."""
    theta, y, x = ribbon_coordinates()
    A = left_invariant_frame(ORIENTATIONS, SPACING)
    difference_tensor = weitzenbock_difference_tensor(A)
    f = helical_ribbon(theta, y, x, METRIC)
    _, frame = hessian_frame(f, difference_tensor, constant_metric_in_frame(METRIC, A))
    return f, frame, difference_tensor


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
    plotter.add_point_labels(points, labels, font_file=str(FONT), font_size=18, bold=False,
                             text_color="black", show_points=False, shape=None,
                             always_visible=True)
    plotter.camera_position = [(75, -75, 55), (0, 0, XI * torch.pi), (0, 0, 1)]
    plotter.reset_camera()
    plotter.camera.zoom(0.9)


def add_density(plotter: pv.Plotter, volume: pv.ImageData, name: str, limit: float,
                cmap: str | Colormap = "magma", opacity: str | list = "linear") -> None:
    """Volume rendering of an array of volume, opacity from 0 to limit."""
    density = plotter.add_volume(volume, scalars=name, cmap=cmap, opacity=opacity,
                                 clim=(0, limit), scalar_bar_args={"title": name, "n_labels": 3})
    density.prop.interpolation_type = "linear"


def make_ribbon_signal_figure(path: Path) -> None:
    f, _, _ = ribbon()
    plotter = pv.Plotter(off_screen=True, window_size=(1400, 1100))
    add_density(plotter, ribbon_volume(f=f.data), "f", 1.0, opacity="sigmoid_8")
    show_ribbon_axes(plotter)
    plotter.screenshot(path)
    plotter.close()


def make_ribbon_frame_figure(path: Path) -> None:
    f, frame, _ = ribbon()
    theta, y, x = ribbon_coordinates()
    plotter = pv.Plotter(off_screen=True, window_size=(1400, 1100))
    add_density(plotter, ribbon_volume(f=f.data), "f", 1.0, cmap="Greys", opacity=[0, 0.2])
    plotter.remove_scalar_bar()
    steps = torch.tensor([XI * 2 * torch.pi / ORIENTATIONS, SPACING, SPACING])  # of e_θ, e_y, e_x
    for k in range(MARGIN, ORIENTATIONS - MARGIN, THETA_STEP):
        # Grid point nearest to the core at t = θ - π/2
        i = round(-RADIUS * theta[k, 0, 0].cos().item() / SPACING + (SIZE - 1) / 2)
        j = round(RADIUS * theta[k, 0, 0].sin().item() / SPACING + (SIZE - 1) / 2)
        point = torch.stack([x[k, i, j], y[k, i, j], XI * theta[k, i, j]])
        for n, color in enumerate(RIBBON_COLORS):
            v = 3.5 * (frame.data[k, i, j, :, n] * steps).flip(0)  # (x, y, XI θ)
            tube = pv.Tube(pointa=(point - v).tolist(), pointb=(point + v).tolist(), radius=0.25)
            plotter.add_mesh(tube, color=color)
    show_ribbon_axes(plotter)
    plotter.enable_depth_peeling()
    plotter.screenshot(path)
    plotter.close()


def ribbon_derivatives() -> tuple[pv.ImageData, list[str], float]:
    """Volume with f and the first gauge derivatives |v_i f| of the ribbon in its structure
    tensor frame, their names and their maximum.

    With a blur as large as the ribbon, the frame is through, across and along it over its
    whole cross-section, unlike the Hessian frame. The sign of each frame vector is
    arbitrary, hence |v_i f|.
    """
    f, _, difference_tensor = ribbon()
    metric = constant_metric_in_frame(METRIC, left_invariant_frame(ORIENTATIONS, SPACING))
    _, frame = structure_tensor_frame(f, RIBBON_SIGMA, metric)
    frame = Field(frame.data.flip(-1), frame.type)  # descending, so v_1 is through
    first = gauge_jet(f, frame, difference_tensor, 1).data.abs()
    names = [f"\u2223v{chr(0x2080 + i + 1)}f\u2223" for i in range(3)]
    volume = ribbon_volume(f=f.data, **{name: first[..., i] for i, name in enumerate(names)})
    return volume, names, first.max().item()


def derivative_cmap(name: str, color: str) -> Colormap:
    return LinearSegmentedColormap.from_list(name, ["white", color, "black"])


def make_ribbon_derivative_figure(path: Path) -> None:
    volume, names, limit = ribbon_derivatives()
    outline = volume.contour([0.5], scalars="f")
    plotter = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(2700, 800), border=False)
    for i, (name, color) in enumerate(zip(names, RIBBON_COLORS)):
        plotter.subplot(0, i)
        add_density(plotter, volume, name, limit, cmap=derivative_cmap(name, color))
        plotter.add_mesh(outline, color="#888888", opacity=0.12)
        plotter.add_text(name, position="upper_edge", font_size=20, color="black",
                         font_file=str(FONT))
        show_ribbon_axes(plotter)
    for bar in plotter.scalar_bars.values():
        bar.SetTitle("")  # the panel title says it
    plotter.screenshot(path)
    plotter.close()


def make_ribbon_cross_section_figure(path: Path) -> None:
    volume, names, limit = ribbon_derivatives()

    # Cross-section perpendicular to the core at θ = π, sampled in (x, y, XI θ) along the unit
    # vectors A_2 and N of helical_ribbon, orthonormal in the metric.
    extent = 7.0
    across, through = torch.meshgrid(torch.linspace(-extent, extent, 141),
                                     torch.linspace(-extent, extent, 141), indexing="xy")
    theta = torch.tensor(torch.pi)
    t = theta - torch.pi / 2
    center = torch.stack([RADIUS * t.cos(), RADIUS * t.sin(), XI * theta])
    a2 = torch.stack([-t.cos(), -t.sin(), torch.tensor(0.0)])
    n = torch.stack([XI * t.sin(), -XI * t.cos(), torch.tensor(RADIUS)])
    n = n / (XI**2 + RADIUS**2) ** 0.5
    points = center + across[..., None] * a2 + through[..., None] * n
    sampled = pv.PolyData(points.reshape(-1, 3).numpy()).sample(volume)

    panels = [("f", "Greys", 1.0)] + [(name, derivative_cmap(name, color), limit)
                                      for name, color in zip(names, RIBBON_COLORS)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.2))
    for ax, (name, cmap, top) in zip(axes, panels):
        ax.imshow(sampled[name].reshape(across.shape), origin="lower", cmap=cmap, vmin=0,
                  vmax=top, extent=(-extent, extent, -extent, extent))
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("across")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_ylabel("through")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)

    make_frame_figure(IMAGES / "gauge_frame.svg")
    make_derivative_figure(IMAGES / "gauge_derivatives.svg")
    make_ribbon_signal_figure(IMAGES / "ribbon_signals.png")
    make_ribbon_frame_figure(IMAGES / "ribbon_frame.png")
    make_ribbon_derivative_figure(IMAGES / "ribbon_derivatives.png")
    make_ribbon_cross_section_figure(IMAGES / "ribbon_cross_section.svg")
