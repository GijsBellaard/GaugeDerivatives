import cmath
from io import BytesIO
import math
from pathlib import Path

import matplotlib
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pyvista as pv
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from frames import structure_tensor_frame
from geometry import (constant_metric_in_frame, covariant_derivative_in_frame,
                      levi_civita_difference_tensor, weitzenbock_difference_tensor)
from grid import gaussian_blur
from manifolds import left_invariant_frame, poincare_metric, sphere_metric

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
FRAME_DURATION = 100         # Milliseconds per frame

TILING = (4, 4, 4)           # Angles π/p, π/q and π/r of the triangles tiling the Poincaré disk
EDGE_WIDTH = 0.15            # Hyperbolic width of the edges between the triangles
DISK_SIZE = 768              # Grid of the Poincaré disk, [DISK_SIZE, DISK_SIZE] over [-1, 1]^2
DISK_RADIUS = 0.9            # Radius shown, as the grid can't resolve the tiling beyond it
DISK_STEP = 32               # Draw a frame every DISK_STEP-th pixel

SPHERE_TILING = (2, 3, 5)    # Dihedral angles π/p, π/q and π/r of the mirrors tiling the sphere
SPHERE_EDGE_WIDTH = 0.05     # Width of the edges between the triangles, in radians
SPHERE_GRID = (256, 512)     # Grid of the sphere over (θ, φ)
SPHERE_PAD = 8               # Grid steps added beyond the poles and around φ while computing
SPHERE_STEP = 16             # Draw a frame every SPHERE_STEP-th row and column


def load_image() -> torch.Tensor:
    with cbook.get_sample_data("grace_hopper.jpg") as f:
        image = TF.pil_to_tensor(Image.open(f).convert("L"))[0] / 255
    return image


def euclidean(n: int) -> torch.Tensor:
    return torch.eye(n).reshape(1, *[1] * n, n, n)


def ribbon_directions() -> tuple[torch.Tensor, torch.Tensor]:
    across = torch.tensor([0.0, 1.0, 0.0])
    through = torch.linalg.solve(METRIC, torch.tensor([-1.0, 0.0, RADIUS]))
    return across, through / (through @ METRIC @ through).sqrt()


def helical_ribbon(theta: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    phi = torch.atan2(y, x)
    wrapped = torch.remainder(theta - phi + torch.pi / 2, 2 * torch.pi) - torch.pi
    radial = RADIUS - torch.sqrt(x**2 + y**2)
    offset = torch.stack([torch.zeros_like(phi), radial, wrapped], dim=-1)  # in A at t = phi
    across, through = ribbon_directions()
    u = torch.einsum("...i,ij,j->...", offset, METRIC, across)
    v = torch.einsum("...i,ij,j->...", offset, METRIC, through)
    return torch.exp(-u**2 / (2 * WIDTH**2) - v**2 / (2 * THICKNESS**2))


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


def add_colorbar(ax, cmap: str | Colormap, vmin: float, vmax: float) -> Colorbar:
    bar = ax.figure.colorbar(ScalarMappable(Normalize(vmin, vmax), cmap), ax=ax, shrink=0.5)
    bar.outline.set_visible(False)
    return bar


def add_frame_legend(ax, labels: list[str]) -> None:
    legend_ax = add_colorbar(ax, "gray", 0, 1).ax
    legend_ax.clear()
    legend_ax.set_axis_off()
    handles = [Line2D([], [], color=color, linewidth=3, label=label)
               for color, label in zip(RIBBON_COLORS, labels)]
    legend_ax.legend(handles=handles, loc="center left", frameon=False)


def show_gauge_derivative(ax, component, title, cmap, quantile=0.99):
    # NaN marks pixels that are not shown.
    limit = component.abs().nanquantile(quantile).item()
    vmin = -limit if (component < 0).any() else 0
    ax.imshow(component, cmap=cmap, vmin=vmin, vmax=limit)
    ax.set_title(title, fontsize=12)
    add_colorbar(ax, cmap, vmin, limit)


def derivative_cmap(name: str, color: str) -> Colormap:
    return LinearSegmentedColormap.from_list(name, ["white", color, "black"])


def make_r2_figure(path: Path) -> None:
    blurred = gaussian_blur(load_image().unsqueeze(0), SIGMA, [1, 2])
    metric = euclidean(2)
    levi_civita = levi_civita_difference_tensor(metric)
    _, frame = structure_tensor_frame(blurred, FRAME_SIGMA, metric)
    # The sign of v_1 is arbitrary, so that of (δf)_1 is too, but not that of (δf)_11.
    first = covariant_derivative_in_frame(blurred, frame, levi_civita, 1)[0, ..., 1].abs()
    second = covariant_derivative_in_frame(blurred, frame, levi_civita, 2)[0, ..., 1, 1]
    blurred, frame = blurred[0], frame[0]

    fig, axes = plt.subplots(2, 2, figsize=(9, 8.2))
    (signal_ax, frame_ax), (first_ax, second_ax) = axes
    signal_ax.imshow(blurred, cmap="gray", vmin=0, vmax=1)
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


def hyperbolic_triangle(p: int, q: int, r: int) -> tuple[float, complex, float]:
    # Triangle with angles π/p at the origin, π/q at B on the real axis and π/r at C on the
    # ray at angle π/p. Its sides follow from the angles by the hyperbolic law of cosines,
    # and BC is the circle |z - c| = ρ through B and C orthogonal to the unit circle,
    # Re(conj(z) c) = (1 + |z|^2) / 2 for z = B, C.
    a, b, g = math.pi / p, math.pi / q, math.pi / r
    ob = math.acosh((math.cos(g) + math.cos(a) * math.cos(b)) / (math.sin(a) * math.sin(b)))
    oc = math.acosh((math.cos(b) + math.cos(a) * math.cos(g)) / (math.sin(a) * math.sin(g)))
    B = math.tanh(ob / 2)
    C = math.tanh(oc / 2) * cmath.exp(1j * a)
    lhs = torch.tensor([[B, 0.0], [C.real, C.imag]], dtype=torch.float64)
    rhs = torch.tensor([(1 + B**2) / 2, (1 + abs(C)**2) / 2], dtype=torch.float64)
    cx, cy = torch.linalg.solve(lhs, rhs).tolist()
    return a, complex(cx, cy), math.sqrt(cx**2 + cy**2 - 1)


def disk_coordinates() -> torch.Tensor:
    y, x = torch.meshgrid(torch.linspace(-1, 1, DISK_SIZE, dtype=torch.float64),
                          torch.linspace(-1, 1, DISK_SIZE, dtype=torch.float64), indexing="ij")
    return torch.complex(x, y)


def hyperbolic_tiling() -> torch.Tensor:
    # Triangles of TILING, black and white by the parity of the reflections reaching them,
    # with edges EDGE_WIDTH wide in the hyperbolic metric.
    a, c, rho = hyperbolic_triangle(*TILING)
    z = disk_coordinates()
    outside = z.abs() > 0.99
    z = torch.where(outside, 0, z)
    parity = torch.ones(z.shape, dtype=torch.float64)

    # Fold every point into the triangle. Rotating by 2a is two reflections, so it keeps
    # the parity, and each reflection in a side flips it.
    turns = torch.floor(torch.remainder(z.angle(), 2 * math.pi) / (2 * a))
    z = z * torch.exp(-2j * a * turns)
    ray = cmath.exp(-1j * a)
    for _ in range(300):
        for flip, reflected in (
            (z.imag < 0, z.conj()),                                   # real axis
            ((z * ray).imag > 0, z.conj() / ray**2),                  # ray at angle a
            ((z - c).abs() < rho, c + rho**2 / (z - c).conj()),       # circle through B, C
        ):
            z = torch.where(flip, reflected, z)
            parity = torch.where(flip, -parity, parity)

    # Hyperbolic distances to the sides: sinh d = 2 |Im(z e^{-iφ})| / (1 - |z|^2) to a line
    # through the origin at angle φ, and ||z - c|^2 - ρ^2| / (ρ (1 - |z|^2)) to the circle.
    # A product rather than the nearest side, so that f is smooth.
    scale = 1 - z.abs().square()
    distances = torch.stack([
        torch.asinh(2 * z.imag.abs() / scale),
        torch.asinh(2 * (z * ray).imag.abs() / scale),
        torch.asinh(((z - c).abs().square() - rho**2).abs() / (rho * scale)),
    ])
    f = 0.5 + 0.5 * parity * torch.tanh(distances / EDGE_WIDTH).prod(dim=0)
    return torch.where(outside, 0.5, f).float()


def make_poincare_figure(path: Path) -> None:
    f = hyperbolic_tiling().unsqueeze(0)
    metric = poincare_metric(DISK_SIZE)
    levi_civita = levi_civita_difference_tensor(metric)
    _, frame = structure_tensor_frame(f, FRAME_SIGMA, metric)
    # The sign of v_1 is arbitrary, so that of (δf)_1 is too, but not that of (δf)_11.
    first = covariant_derivative_in_frame(f, frame, levi_civita, 1)[0, ..., 1].abs()
    second = covariant_derivative_in_frame(f, frame, levi_civita, 2)[0, ..., 1, 1]
    f, frame = f[0], frame[0]
    outside = disk_coordinates().abs() > DISK_RADIUS
    shown = lambda image: image.masked_fill(outside, torch.nan)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8.2))
    (signal_ax, frame_ax), (first_ax, second_ax) = axes
    signal_ax.imshow(shown(f), cmap="gray", vmin=0, vmax=1)
    signal_ax.set_title("$f$", fontsize=12)
    add_colorbar(signal_ax, "gray", 0, 1)

    # Frame vectors at their length in the grid, so that they shrink towards the rim, where
    # a unit of hyperbolic length is ever fewer pixels. At the centre it is 1 / (2 step).
    frame_ax.imshow(shown(f), cmap="gray", vmin=0, vmax=1, alpha=0.5)
    rows, cols = torch.meshgrid(torch.arange(0, DISK_SIZE, DISK_STEP),
                                torch.arange(0, DISK_SIZE, DISK_STEP), indexing="ij")
    inside = ~outside[rows, cols]
    rows, cols = rows[inside], cols[inside]
    centre_length = (DISK_SIZE - 1) / 4
    for i, color in enumerate(FRAME_COLORS):
        v = frame[rows, cols, :, i]
        frame_ax.quiver(cols, rows, v[:, 1], v[:, 0], color=color, pivot="mid",
                        angles="xy", scale_units="xy", scale=centre_length / (0.8 * DISK_STEP),
                        headwidth=0, headlength=0, headaxislength=0, width=0.004)
    frame_ax.set_title("Structure tensor frame", fontsize=12)
    add_frame_legend(frame_ax, ["$v_0$ along", "$v_1$ across"])

    show_gauge_derivative(first_ax, shown(first), r"$|(\delta f)_1|$",
                          derivative_cmap("first", FRAME_COLORS[1]))
    show_gauge_derivative(second_ax, shown(second), r"$(\delta f)_{11}$", "RdBu_r")
    centre = (DISK_SIZE - 1) / 2
    for ax in axes.flat:
        ax.add_patch(Circle((centre, centre), DISK_RADIUS * centre, fill=False,
                            edgecolor="#888888", linewidth=0.8))
        ax.set_xlim(-0.5, DISK_SIZE - 0.5)
        ax.set_ylim(DISK_SIZE - 0.5, -0.5)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def ribbon() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A = left_invariant_frame(ORIENTATIONS, SPACING)
    f = helical_ribbon(*ribbon_coordinates()).unsqueeze(0)
    _, frame = structure_tensor_frame(f, RIBBON_SIGMA, constant_metric_in_frame(METRIC, A))
    return f, frame, weitzenbock_difference_tensor(A)


def ribbon_coordinates() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    space = (torch.arange(SIZE) - (SIZE - 1) / 2) * SPACING
    return torch.meshgrid(torch.arange(ORIENTATIONS) * 2 * torch.pi / ORIENTATIONS,
                          space, space,
                          indexing="ij")


def ribbon_volume(**arrays: torch.Tensor) -> pv.ImageData:
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
    density = plotter.add_volume(volume, scalars=name, cmap=cmap, opacity=opacity,
                                 clim=(0, limit), show_scalar_bar=False)
    density.prop.interpolation_type = "linear"


def add_ribbon_frame(plotter: pv.Plotter, frame: torch.Tensor) -> None:
    theta, y, x = ribbon_coordinates()
    steps = torch.tensor([XI * 2 * torch.pi / ORIENTATIONS, SPACING, SPACING])  # of e_θ, e_y, e_x
    for k in range(MARGIN, ORIENTATIONS - MARGIN, THETA_STEP):
        # Grid point nearest to the core at t = θ - π/2
        i = round(-RADIUS * theta[k, 0, 0].cos().item() / SPACING + (SIZE - 1) / 2)
        j = round(RADIUS * theta[k, 0, 0].sin().item() / SPACING + (SIZE - 1) / 2)
        point = torch.stack([x[k, i, j], y[k, i, j], XI * theta[k, i, j]])
        for n, color in enumerate(RIBBON_COLORS):
            v = 3.5 * (frame[k, i, j, :, n] * steps).flip(0)  # (x, y, XI θ)
            tube = pv.Tube(pointa=(point - v).tolist(), pointb=(point + v).tolist(), radius=0.25)
            plotter.add_mesh(tube, color=color)
    plotter.enable_depth_peeling()


def trim(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    drawn = (image < 250).any(-1)
    rows, cols = np.nonzero(drawn.any(1))[0], np.nonzero(drawn.any(0))[0]
    return image[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], rows[0], cols[0]


def align(trimmed: list[tuple[np.ndarray, int, int]], margin: int = 10) -> list[np.ndarray]:
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


def turntable(plotters: list[pv.Plotter], frames: int, each_frame=lambda plotter: None,
              degrees: float = 360) -> list[list[np.ndarray]]:
    trimmed = []
    for _ in range(frames):
        for plotter in plotters:
            each_frame(plotter)
            plotter.render()
            trimmed.append(trim(plotter.screenshot(return_img=True)))
            plotter.camera.Azimuth(degrees / frames)
    for plotter in plotters:
        plotter.close()
    images = align(trimmed)
    return [images[i::len(plotters)] for i in range(len(plotters))]


def make_m2_figure(path: Path) -> None:
    f, frame, difference_tensor = ribbon()
    # The sign of each frame vector is arbitrary, hence |(δf)_i|.
    first = covariant_derivative_in_frame(f, frame, difference_tensor, 1)[0].abs()
    f, frame = f[0], frame[0]
    limit = first[..., RIBBON_DIRECTIONS].max().item()
    names = [f"d{i}" for i in RIBBON_DIRECTIONS]
    volume = ribbon_volume(f=f, **{name: first[..., i]
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
    images = turntable([signal_plotter, frame_plotter, *derivative_plotters], TURN_FRAMES,
                       show_ribbon_axes)

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
    save_animation(path, fig, shown, images)


def save_animation(path: Path, fig, shown: list, images: list[list[np.ndarray]]) -> None:
    # Every frame of the turntable images in the panels shown of fig, as an animated WebP.
    fig.tight_layout()
    frames = []
    for k in range(len(images[0])):
        for image, panel in zip(shown, images):
            image.set_data(panel[k])
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
        frames.append(Image.open(buffer).convert("RGB"))
    plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=FRAME_DURATION,
                   loop=0, quality=80, method=6)


def sphere_mirrors() -> torch.Tensor:
    # Unit normals n_k of three planes through the origin, at dihedral angles π/p between
    # n_0 and n_1, π/q between n_1 and n_2, and π/r between n_0 and n_2. The triangle they
    # bound is where every n_k · x > 0.
    a, b, c = (math.pi / k for k in SPHERE_TILING)
    y = -math.cos(c)
    x = (-math.cos(b) + math.cos(a) * y) / math.sin(a)
    return torch.tensor([[0.0, 1.0, 0.0],
                         [math.sin(a), -math.cos(a), 0.0],
                         [x, y, math.sqrt(1 - x**2 - y**2)]], dtype=torch.float64)


def sphere_coordinates() -> tuple[torch.Tensor, torch.Tensor]:
    n_theta, n_phi = SPHERE_GRID
    theta = (torch.arange(n_theta, dtype=torch.float64) + 0.5) * torch.pi / n_theta
    phi = torch.arange(n_phi, dtype=torch.float64) * 2 * torch.pi / n_phi
    return torch.meshgrid(theta, phi, indexing="ij")


def sphere_points(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    return torch.stack([theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()], dim=-1)


def spherical_tiling() -> torch.Tensor:
    # Triangles between the mirrors of SPHERE_TILING, black and white by the parity of the
    # reflections reaching them, with edges SPHERE_EDGE_WIDTH wide. Like hyperbolic_tiling,
    # but the mirrors are planes through the origin, and the distance to one is asin(n · x).
    normals = sphere_mirrors()
    x = sphere_points(*sphere_coordinates())
    parity = torch.ones(x.shape[:-1], dtype=torch.float64)
    for _ in range(60):
        for n in normals:
            along = x @ n
            flip = along < 0
            x = torch.where(flip.unsqueeze(-1), x - 2 * along.unsqueeze(-1) * n, x)
            parity = torch.where(flip, -parity, parity)
    distances = torch.asin((x @ normals.T).clamp(-1, 1))
    f = 0.5 + 0.5 * parity * torch.tanh(distances / SPHERE_EDGE_WIDTH).prod(dim=-1)
    return f.float()


def pad_sphere(field: torch.Tensor) -> torch.Tensor:
    # SPHERE_PAD more rows beyond each pole and columns around φ, so that the derivatives
    # need no one-sided differences. Beyond a pole, θ becomes -θ and φ becomes φ + π, which
    # keeps scalar fields and the diagonal metric as they are.
    n_phi, pad = SPHERE_GRID[1], SPHERE_PAD
    north = field[:, :pad].flip(1).roll(n_phi // 2, dims=2)
    south = field[:, -pad:].flip(1).roll(n_phi // 2, dims=2)
    field = torch.cat([north, field, south], dim=1)
    return torch.cat([field[:, :, -pad:], field, field[:, :, :pad]], dim=2)


def crop_sphere(field: torch.Tensor) -> torch.Tensor:
    pad = SPHERE_PAD
    return field[:, pad:-pad, pad:-pad]


def sphere_mesh(**arrays: torch.Tensor) -> pv.StructuredGrid:
    # The grid as a surface, with its first column repeated to close it around φ.
    close = lambda array: torch.cat([array, array[:, :1]], dim=1)
    theta, phi = sphere_coordinates()
    points = close(sphere_points(theta, phi))
    mesh = pv.StructuredGrid(*(points[..., k].numpy() for k in range(3)))
    for name, array in arrays.items():
        mesh.point_data[name] = close(array).numpy().ravel(order="F")
    return mesh


def sphere_plotter() -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
    plotter.camera_position = [(4, -4, 2.5), (0, 0, 0), (0, 0, 1)]
    plotter.reset_camera()
    return plotter


def add_sphere_frame(plotter: pv.Plotter, frame: torch.Tensor) -> None:
    # The frame as tubes, at every SPHERE_STEP-th row and column of the grid.
    n_theta, n_phi = SPHERE_GRID
    step_theta, step_phi = torch.pi / n_theta, 2 * torch.pi / n_phi
    i, j = torch.meshgrid(torch.arange(SPHERE_STEP // 2, n_theta, SPHERE_STEP),
                          torch.arange(0, n_phi, SPHERE_STEP), indexing="ij")
    i, j = i.flatten(), j.flatten()
    theta, phi = (i + 0.5) * step_theta, j * step_phi
    d_theta = torch.stack([theta.cos() * phi.cos(), theta.cos() * phi.sin(), -theta.sin()], -1)
    d_phi = torch.stack([-phi.sin(), phi.cos(), torch.zeros_like(phi)], -1) * theta.sin().unsqueeze(-1)
    points = 1.005 * sphere_points(theta, phi)
    for n, color in enumerate(FRAME_COLORS):
        v = frame[i, j, :, n].double()
        v = (v[:, 0] * step_theta).unsqueeze(-1) * d_theta + (v[:, 1] * step_phi).unsqueeze(-1) * d_phi
        pairs = torch.stack([points - 0.05 * v, points + 0.05 * v], dim=1)
        plotter.add_mesh(segments(pairs.tolist()).tube(radius=0.006), color=color)


def make_s2_figure(path: Path) -> None:
    f = spherical_tiling().unsqueeze(0)
    metric = sphere_metric(*SPHERE_GRID)
    f_padded, metric_padded = pad_sphere(f), pad_sphere(metric)
    levi_civita = levi_civita_difference_tensor(metric_padded)
    _, frame = structure_tensor_frame(f_padded, FRAME_SIGMA, metric_padded)
    # The sign of v_1 is arbitrary, so that of (δf)_1 is too, but not that of (δf)_11.
    first = covariant_derivative_in_frame(f_padded, frame, levi_civita, 1)[..., 1]
    second = covariant_derivative_in_frame(f_padded, frame, levi_civita, 2)[..., 1, 1]
    first, second = crop_sphere(first)[0].abs(), crop_sphere(second)[0]
    f, frame = f[0], crop_sphere(frame)[0]
    first_limit = first.quantile(0.99).item()
    second_limit = second.abs().quantile(0.99).item()
    first_cmap = derivative_cmap("first", FRAME_COLORS[1])

    mesh = sphere_mesh(f=f, first=first, second=second)
    panels = [("f", "gray", (0, 1)), ("f", "gray", (-1, 1.6)),
              ("first", first_cmap, (0, first_limit)),
              ("second", "RdBu_r", (-second_limit, second_limit))]
    plotters = []
    for name, cmap, clim in panels:
        plotter = sphere_plotter()
        plotter.add_mesh(mesh, scalars=name, cmap=cmap, clim=clim, show_scalar_bar=False,
                         smooth_shading=True, ambient=0.45, diffuse=0.6, specular=0.0)
        plotters.append(plotter)
    add_sphere_frame(plotters[1], frame)
    # The tiling is the same after half a turn about the poles, so that loops.
    images = turntable(plotters, TURN_FRAMES // 2, degrees=180)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    shown = [ax.imshow(panel[0]) for ax, panel in zip(axes.flat, images)]
    (signal_ax, frame_ax), (first_ax, second_ax) = axes
    signal_ax.set_title("$f$", fontsize=12)
    add_colorbar(signal_ax, "gray", 0, 1)
    frame_ax.set_title("Structure tensor frame", fontsize=12)
    add_frame_legend(frame_ax, ["$v_0$ along", "$v_1$ across"])
    first_ax.set_title(r"$|(\delta f)_1|$", fontsize=12)
    add_colorbar(first_ax, first_cmap, 0, first_limit)
    second_ax.set_title(r"$(\delta f)_{11}$", fontsize=12)
    add_colorbar(second_ax, "RdBu_r", -second_limit, second_limit)
    for ax in axes.flat:
        ax.set_axis_off()
    save_animation(path, fig, shown, images)


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)
    make_r2_figure(IMAGES / "r2.svg")
    make_poincare_figure(IMAGES / "poincare.svg")
    make_m2_figure(IMAGES / "m2.webp")
    make_s2_figure(IMAGES / "s2.webp")
