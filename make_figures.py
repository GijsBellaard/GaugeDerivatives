from io import BytesIO
from pathlib import Path

import matplotlib
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pyvista as pv
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from field import Field
from frames import constant_metric_in_frame, covariant_derivative_in_frame, structure_tensor_frame
from gaussian_blur import gaussian_blur
from geometry import levi_civita_difference_tensor, weitzenbock_difference_tensor
from m2 import left_invariant_frame

IMAGES = Path(__file__).parent / "images"
FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"  # has θ and π

SIGMA = 2.0                  # Blur of the photograph
FRAME_SIGMA = 1.0            # Blur of the structure tensor
CROP = (120, 350, 160, 360)  # Region shown in the frame panel, rows then columns
STEP = 8                     # Draw a frame every STEP-th pixel

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
TURN_FRAMES = 120            # Frames of one turn of the camera around the ribbon
FRAME_DURATION = 50          # Milliseconds per frame


def load_image() -> Field:
    """Grayscale sample photograph in [0, 1]."""
    with cbook.get_sample_data("grace_hopper.jpg") as f:
        image = TF.pil_to_tensor(Image.open(f).convert("L"))[0] / 255
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


def add_colorbar(ax, cmap: str | Colormap, vmin: float, vmax: float) -> Colorbar:
    bar = ax.figure.colorbar(ScalarMappable(Normalize(vmin, vmax), cmap), ax=ax, shrink=0.5)
    bar.outline.set_visible(False)
    return bar


def add_frame_legend(ax, labels: list[str]) -> None:
    """Legend of the frame vectors in RIBBON_COLORS, where a colorbar would be, so that the
    panels line up."""
    legend_ax = add_colorbar(ax, "gray", 0, 1).ax
    legend_ax.clear()
    legend_ax.set_axis_off()
    handles = [Line2D([], [], color=color, linewidth=3, label=label)
               for color, label in zip(RIBBON_COLORS, labels)]
    legend_ax.legend(handles=handles, loc="center left", frameon=False)


def show_gauge_derivative(ax, component, title, cmap, quantile=0.99):
    """Image of a gauge derivative, symmetric around 0 if it takes negative values."""
    limit = component.abs().quantile(quantile).item()
    vmin = -limit if component.min() < 0 else 0
    ax.imshow(component, cmap=cmap, vmin=vmin, vmax=limit)
    ax.set_title(title, fontsize=12)
    add_colorbar(ax, cmap, vmin, limit)


def derivative_cmap(name: str, color: str) -> Colormap:
    return LinearSegmentedColormap.from_list(name, ["white", color, "black"])


def make_r2_figure(path: Path) -> None:
    """The photograph, its structure tensor frame on CROP, and its gauge derivatives across
    edges."""
    blurred = gaussian_blur(load_image(), sigma=SIGMA)
    metric = euclidean(blurred.n)
    levi_civita = levi_civita_difference_tensor(metric)
    _, frame = structure_tensor_frame(blurred, FRAME_SIGMA, metric)
    # The sign of v_1 is arbitrary, so that of (δf)_1 is too, but not that of (δf)_11.
    first = covariant_derivative_in_frame(blurred, frame, levi_civita, 1).data[..., 1].abs()
    second = covariant_derivative_in_frame(blurred, frame, levi_civita, 2).data[..., 1, 1]

    fig, axes = plt.subplots(2, 2, figsize=(9, 8.2))
    (signal_ax, frame_ax), (first_ax, second_ax) = axes
    signal_ax.imshow(blurred.data, cmap="gray", vmin=0, vmax=1)
    r0, r1, c0, c1 = CROP
    signal_ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, edgecolor="white",
                                  linewidth=1.5))
    signal_ax.set_title("$f$", fontsize=12)
    add_colorbar(signal_ax, "gray", 0, 1)
    show_gauge_frame(frame_ax, blurred, frame, *CROP, STEP)
    frame_ax.set_title("Structure tensor frame", fontsize=12)
    add_frame_legend(frame_ax, ["$v_0$ along", "$v_1$ across"])
    show_gauge_derivative(first_ax, first, r"$|(\delta f)_1|$",
                          derivative_cmap("first", FRAME_COLORS[1]))
    show_gauge_derivative(second_ax, second, r"$(\delta f)_{11}$", "RdBu_r")
    for ax in axes.flat:
        ax.set_axis_off()
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


def set_ribbon_camera(plotter: pv.Plotter) -> None:
    plotter.camera_position = [(75, -75, 55), (0, 0, XI * torch.pi), (0, 0, 1)]
    plotter.reset_camera()
    plotter.camera.zoom(0.9)


def show_ribbon_axes(plotter: pv.Plotter) -> None:
    """Axes x and y along the front edges of the floor and θ up a corner, and a grid on the
    floor and the two back walls, for the current camera.

    Like the axes of matplotlib, they move to other edges and walls as the camera turns.
    """
    h, top = (SIZE - 1) / 2 * SPACING, XI * 2 * torch.pi
    focus, position = plotter.camera.focal_point, plotter.camera.position
    sx = 1 if position[0] >= focus[0] else -1   # Side of the camera in x and y
    sy = 1 if position[1] >= focus[1] else -1
    back_x, back_y, front_x, front_y = -sx * h, -sy * h, sx * h, sy * h
    heights = [XI * torch.pi * t for t in THETA_TICKS]
    grid = []
    for t in SPACE_TICKS:
        grid += [((t, -h, 0), (t, h, 0)), ((-h, t, 0), (h, t, 0)),
                 ((t, back_y, 0), (t, back_y, top)), ((back_x, t, 0), (back_x, t, top))]
    for z in heights:
        grid += [((back_x, -h, z), (back_x, h, z)), ((-h, back_y, z), (h, back_y, z))]
    tick, pad = 1.5, 4.5
    axes = [((-h, front_y, 0), (h, front_y, 0)), ((front_x, -h, 0), (front_x, h, 0)),
            ((back_x, front_y, 0), (back_x, front_y, top))]
    axes += [((t, front_y, 0), (t, front_y + sy * tick, 0)) for t in SPACE_TICKS]
    axes += [((front_x, t, 0), (front_x + sx * tick, t, 0)) for t in SPACE_TICKS]
    axes += [((back_x, front_y, z), (back_x - sx * tick, front_y + sy * tick, z))
             for z in heights]
    plotter.add_mesh(segments(grid), color="#d0d0d0", line_width=1, name="grid")
    plotter.add_mesh(segments(axes), color="black", line_width=2, name="axes")

    points = ([(t, front_y + sy * pad, 0) for t in SPACE_TICKS]
              + [(front_x + sx * pad, t, 0) for t in SPACE_TICKS]
              + [(back_x - sx * pad, front_y + sy * pad, z) for z in heights]
              + [(0, front_y + sy * 2.5 * pad, 0), (front_x + sx * 2.5 * pad, 0, 0),
                 (back_x, front_y, top + 1.5 * pad)])
    labels = ([f"{t}".replace("-", "−") for t in 2 * SPACE_TICKS] + list(THETA_TICKS.values())
              + ["x", "y", "θ"])
    plotter.add_point_labels(points, labels, font_file=str(FONT), font_size=28, bold=False,
                             text_color="black", show_points=False, shape=None,
                             always_visible=True, name="labels")


def ribbon_plotter() -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True, window_size=(1000, 800))
    set_ribbon_camera(plotter)
    return plotter


def add_density(plotter: pv.Plotter, volume: pv.ImageData, name: str, limit: float,
                cmap: str | Colormap = "magma", opacity: str | list = "linear") -> None:
    """Volume rendering of an array of volume, opacity from 0 to limit."""
    density = plotter.add_volume(volume, scalars=name, cmap=cmap, opacity=opacity,
                                 clim=(0, limit), show_scalar_bar=False)
    density.prop.interpolation_type = "linear"


def add_ribbon_frame(plotter: pv.Plotter, frame: Field) -> None:
    """The frame as tubes, on the core of the ribbon at every THETA_STEP-th orientation."""
    theta, y, x = ribbon_coordinates()
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
    plotter.enable_depth_peeling()


def trim(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Image cropped to what is drawn, and the row and column where the crop starts."""
    drawn = (image < 250).any(-1)
    rows, cols = np.nonzero(drawn.any(1))[0], np.nonzero(drawn.any(0))[0]
    return image[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], rows[0], cols[0]


def align(trimmed: list[tuple[np.ndarray, int, int]], margin: int = 10) -> list[np.ndarray]:
    """Trimmed images back in place on white canvases of one size, just large enough for all."""
    r0 = min(r for _, r, _ in trimmed)
    c0 = min(c for _, _, c in trimmed)
    r1 = max(r + image.shape[0] for image, r, _ in trimmed)
    c1 = max(c + image.shape[1] for image, _, c in trimmed)
    canvases = []
    for image, r, c in trimmed:
        canvas = np.full((r1 - r0 + 2 * margin, c1 - c0 + 2 * margin, 3), 255, dtype=np.uint8)
        r, c = r - r0 + margin, c - c0 + margin
        canvas[r:r + image.shape[0], c:c + image.shape[1]] = image
        canvases.append(canvas)
    return canvases


def turntable(plotters: list[pv.Plotter], frames: int) -> list[list[np.ndarray]]:
    """Screenshots of the plotters, which are closed, as their cameras turn once around the θ
    axis, all cropped alike to what is drawn. [i][k] is plotter i at frame k."""
    trimmed = []
    for _ in range(frames):
        for plotter in plotters:
            show_ribbon_axes(plotter)
            trimmed.append(trim(plotter.screenshot(return_img=True)))
            plotter.camera.Azimuth(360 / frames)
    for plotter in plotters:
        plotter.close()
    images = align(trimmed)
    return [images[i::len(plotters)] for i in range(len(plotters))]


def make_m2_figure(path: Path) -> None:
    """The ribbon, its structure tensor frame and its first gauge derivatives across and
    through it, as the camera turns around it."""
    f, frame, difference_tensor = ribbon()
    # The sign of each frame vector is arbitrary, hence |(δf)_i|.
    first = covariant_derivative_in_frame(f, frame, difference_tensor, 1).data.abs()
    limit = first[..., RIBBON_DIRECTIONS].max().item()
    names = [f"d{i}" for i in RIBBON_DIRECTIONS]
    volume = ribbon_volume(f=f.data, **{name: first[..., i]
                                        for name, i in zip(names, RIBBON_DIRECTIONS)})
    outline = volume.contour([0.5], scalars="f")

    signal_plotter = ribbon_plotter()
    add_density(signal_plotter, volume, "f", 1.0, opacity="sigmoid_8")
    frame_plotter = ribbon_plotter()
    add_density(frame_plotter, volume, "f", 1.0, cmap="Greys", opacity=[0, 0.2])
    add_ribbon_frame(frame_plotter, frame)
    derivative_plotters, cmaps = [], []
    for name, i in zip(names, RIBBON_DIRECTIONS):
        cmaps.append(derivative_cmap(name, RIBBON_COLORS[i]))
        plotter = ribbon_plotter()
        add_density(plotter, volume, name, limit, cmap=cmaps[-1])
        plotter.add_mesh(outline, color="#888888", opacity=0.12)
        derivative_plotters.append(plotter)
    images = turntable([signal_plotter, frame_plotter, *derivative_plotters], TURN_FRAMES)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    shown = [ax.imshow(panel[0]) for ax, panel in zip(axes.flat, images)]
    (signal_ax, frame_ax), derivative_axes = axes
    signal_ax.set_title("$f$", fontsize=12)
    add_colorbar(signal_ax, "magma", 0, 1)
    frame_ax.set_title("Structure tensor frame", fontsize=12)
    add_frame_legend(frame_ax, ["$v_0$ along", "$v_1$ across", "$v_2$ through"])
    for ax, i, cmap in zip(derivative_axes, RIBBON_DIRECTIONS, cmaps):
        ax.set_title(rf"$|(\delta f)_{i}|$", fontsize=12)
        add_colorbar(ax, cmap, 0, limit)
    for ax in axes.flat:
        ax.set_axis_off()
    fig.tight_layout()

    frames = []
    for k in range(TURN_FRAMES):
        for image, panel in zip(shown, images):
            image.set_data(panel[k])
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
        frames.append(Image.open(buffer).convert("RGB"))
    plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=FRAME_DURATION,
                   loop=0, quality=80, method=6)


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)
    make_r2_figure(IMAGES / "r2.svg")
    make_m2_figure(IMAGES / "m2.webp")
