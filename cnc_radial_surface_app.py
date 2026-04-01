"""
4th-Axis Oblong Pocket Generator

Interactive application that:
1. Builds wrapped 4th-axis oblong pocket paths on a cylindrical shaft
2. Displays shaft and pocket toolpath in 3D
3. Exports absolute metric G-code with X/A/Z moves
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox


@dataclass
class CNCParameters:
    shaft_diameter: float = 37.0
    pocket_length: float = 25.0
    pocket_end_radius: float = 4.0
    pocket_depth: float = 0.2
    tool_diameter: float = 2.0
    step_over: float = 0.4
    pattern_count: int = 6
    pattern_start_angle: float = 0.0
    axial_center: float = 0.0
    oblong_angle_deg: float = 0.0
    feed_rate: float = 450.0
    spindle_speed: int = 12000
    safe_height: float = 2.0
    path_point_spacing: float = 0.35
    mesh_axial_samples: int = 140
    mesh_angular_samples: int = 160
    show_toolpath: bool = True
    show_boundaries: bool = True
    output_filename: str = "4th_axis_oblong_pockets.nc"


@dataclass
class AppState:
    params: CNCParameters
    fig: plt.Figure
    ax3d: plt.Axes
    sliders: Dict[str, Slider]
    file_box: TextBox
    toggle: CheckButtons
    status_text: plt.Text
    redraw_timer: object
    redraw_pending: bool


def _build_spiral_levels(max_radius: float, step_over: float) -> np.ndarray:
    if max_radius <= 1e-9:
        return np.array([0.0], dtype=float)

    step = max(0.02, float(step_over))
    levels = np.arange(0.0, max_radius + 0.5 * step, step, dtype=float)
    if levels.size == 0:
        levels = np.array([0.0, max_radius], dtype=float)
    if levels[-1] < max_radius - 1e-9:
        levels = np.append(levels, max_radius)
    levels[0] = 0.0
    levels[-1] = max_radius
    return np.unique(np.round(levels, 9))


def _sample_linear_move(
    x_start: float,
    s_start: float,
    x_end: float,
    s_end: float,
    z_cut: float,
    spacing: float,
) -> np.ndarray:
    distance = float(np.hypot(x_end - x_start, s_end - s_start))
    ds = max(0.05, float(spacing))
    n = max(2, int(np.ceil(distance / ds)) + 1)
    x_vals = np.linspace(x_start, x_end, n)
    s_vals = np.linspace(s_start, s_end, n)
    z_vals = np.full_like(x_vals, z_cut)
    return np.column_stack((x_vals, s_vals, z_vals))


def _sample_centerline(
    straight_half_len: float, z_cut: float, spacing: float
) -> np.ndarray:
    if straight_half_len <= 1e-9:
        return np.array([[0.0, 0.0, z_cut]], dtype=float)
    return _sample_linear_move(
        -straight_half_len, 0.0, straight_half_len, 0.0, z_cut, spacing
    )


def _sample_obround_loop(
    level_radius: float, straight_half_len: float, z_cut: float, spacing: float
) -> np.ndarray:
    if level_radius <= 1e-9:
        return _sample_centerline(straight_half_len, z_cut, spacing)

    ds = max(0.05, float(spacing))
    if straight_half_len <= 1e-9:
        n_circle = max(28, int(np.ceil((2.0 * np.pi * level_radius) / ds)) + 1)
        theta = np.linspace(-0.5 * np.pi, 1.5 * np.pi, n_circle)
        x_vals = level_radius * np.cos(theta)
        s_vals = level_radius * np.sin(theta)
        loop = np.column_stack((x_vals, s_vals, np.full_like(x_vals, z_cut)))
        if not np.allclose(loop[0, :2], loop[-1, :2]):
            loop = np.vstack([loop, loop[0]])
        return loop

    n_line = max(2, int(np.ceil((2.0 * straight_half_len) / ds)) + 1)
    n_arc = max(16, int(np.ceil((np.pi * level_radius) / ds)) + 1)

    bottom_x = np.linspace(straight_half_len, -straight_half_len, n_line)
    bottom_s = np.full_like(bottom_x, -level_radius)

    left_theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_arc)
    left_x = -straight_half_len - level_radius * np.cos(left_theta)
    left_s = level_radius * np.sin(left_theta)

    top_x = np.linspace(-straight_half_len, straight_half_len, n_line)
    top_s = np.full_like(top_x, level_radius)

    right_theta = np.linspace(0.5 * np.pi, -0.5 * np.pi, n_arc)
    right_x = straight_half_len + level_radius * np.cos(right_theta)
    right_s = level_radius * np.sin(right_theta)

    x_vals = np.concatenate([bottom_x, left_x[1:], top_x[1:], right_x[1:]])
    s_vals = np.concatenate([bottom_s, left_s[1:], top_s[1:], right_s[1:]])
    loop = np.column_stack((x_vals, s_vals, np.full_like(x_vals, z_cut)))
    if not np.allclose(loop[0, :2], loop[-1, :2]):
        loop = np.vstack([loop, loop[0]])
    return loop


def _rotate_xs_path(path: np.ndarray, angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(float(angle_deg))
    if abs(angle_rad) <= 1e-12:
        return path.copy()

    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    out = path.copy()
    x = path[:, 0]
    sv = path[:, 1]
    out[:, 0] = c * x - s * sv
    out[:, 1] = s * x + c * sv
    return out


def _rotate_xs_boundary(boundary: np.ndarray, angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(float(angle_deg))
    if abs(angle_rad) <= 1e-12:
        return boundary.copy()

    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    out = boundary.copy()
    x = boundary[:, 0]
    sv = boundary[:, 1]
    out[:, 0] = c * x - s * sv
    out[:, 1] = s * x + c * sv
    return out


def _estimate_max_circumferential_span(params: CNCParameters) -> float:
    """
    Estimate the maximum pocket width along the circumferential unwrapped axis S.

    This uses the actual rotated obround (capsule) geometry instead of a loose
    bounding-box estimate, avoiding false overlap rejections for moderate angles.
    """
    radius = float(params.pocket_end_radius)
    straight_half_len = max(0.0, 0.5 * params.pocket_length - radius)
    angle_rad = np.deg2rad(float(params.oblong_angle_deg))
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))

    if straight_half_len <= 1e-12:
        return 2.0 * radius

    # Segment centers for the capsule's swept circles.
    t_samples = max(220, int(np.ceil(straight_half_len * 44.0)))
    t = np.linspace(-straight_half_len, straight_half_len, t_samples)
    x_center = t * c
    s_center = t * s

    x_min = float(np.min(x_center) - radius)
    x_max = float(np.max(x_center) + radius)
    x_samples = max(320, int(np.ceil((x_max - x_min) / 0.03)))
    x_vals = np.linspace(x_min, x_max, x_samples)

    dx = x_vals[:, None] - x_center[None, :]
    inside = np.abs(dx) <= radius
    if not np.any(inside):
        return 2.0 * radius

    root = np.zeros_like(dx)
    root[inside] = np.sqrt(np.maximum(radius * radius - dx[inside] * dx[inside], 0.0))

    low = np.where(inside, s_center[None, :] - root, np.inf)
    high = np.where(inside, s_center[None, :] + root, -np.inf)
    width = np.max(high, axis=1) - np.min(low, axis=1)
    return float(np.max(width))


def _validate_geometry(params: CNCParameters) -> None:
    tool_radius = 0.5 * params.tool_diameter
    if params.shaft_diameter <= 0.0:
        raise ValueError("Shaft diameter must be positive.")
    if params.pocket_depth <= 0.0:
        raise ValueError("Pocket depth must be positive.")
    if params.pocket_end_radius <= tool_radius + 1e-6:
        raise ValueError("Pocket end radius must be larger than tool radius.")
    if params.pocket_length <= 2.0 * tool_radius + 1e-6:
        raise ValueError("Pocket length is too short for this tool diameter.")
    if params.pocket_length + 1e-9 < 2.0 * params.pocket_end_radius:
        raise ValueError(
            "Pocket length must be at least 2x end radius for an oblong shape."
        )

    count = max(1, int(round(params.pattern_count)))
    if count > 1:
        circumference = np.pi * params.shaft_diameter
        pitch = circumference / count
        pocket_width = _estimate_max_circumferential_span(params)
        if pitch <= pocket_width + 0.05:
            raise ValueError(
                "Pattern count too high: pockets overlap around the circumference."
            )


def generate_single_pocket_toolpath(params: CNCParameters) -> np.ndarray:
    """Generate one continuous spiral-style oblong pocket in (X, S, Z)."""
    _validate_geometry(params)

    tool_radius = 0.5 * params.tool_diameter
    straight_half_len = max(0.0, 0.5 * params.pocket_length - params.pocket_end_radius)
    inner_radius = params.pocket_end_radius - tool_radius
    z_cut = -abs(params.pocket_depth)
    levels = _build_spiral_levels(inner_radius, params.step_over)

    segment = _sample_centerline(straight_half_len, z_cut, params.path_point_spacing)
    for level in levels[1:]:
        start_x = straight_half_len
        start_s = -float(level)
        connector = _sample_linear_move(
            float(segment[-1, 0]),
            float(segment[-1, 1]),
            start_x,
            start_s,
            z_cut,
            params.path_point_spacing,
        )
        loop = _sample_obround_loop(
            float(level), straight_half_len, z_cut, params.path_point_spacing
        )
        segment = np.vstack([segment, connector[1:], loop[1:]])

    if segment.shape[0] < 2:
        raise ValueError("No valid pocket spiral generated.")

    segment = _rotate_xs_path(segment, params.oblong_angle_deg)
    segment[:, 0] += float(params.axial_center)
    return segment


def generate_toolpath(params: CNCParameters) -> np.ndarray:
    """Generate NaN-separated multi-pocket toolpath in unwrapped coordinates."""
    base_segment = generate_single_pocket_toolpath(params)

    shaft_radius = 0.5 * params.shaft_diameter
    count = max(1, int(round(params.pattern_count)))
    segments: List[np.ndarray] = []

    for idx in range(count):
        angle_deg = params.pattern_start_angle + (360.0 * idx / count)
        s_offset = shaft_radius * np.deg2rad(angle_deg)
        seg = base_segment.copy()
        seg[:, 1] += s_offset
        segments.append(seg)

    result = segments[0]
    for seg in segments[1:]:
        result = np.vstack([result, np.array([[np.nan, np.nan, np.nan]]), seg])

    if result.shape[0] > 300000:
        raise ValueError("Toolpath too dense. Increase point spacing or step-over.")
    return result


def _split_toolpath_segments(toolpath: np.ndarray) -> List[np.ndarray]:
    if toolpath.ndim != 2 or toolpath.shape[1] != 3:
        raise ValueError("Toolpath must be an N x 3 array.")

    is_break = np.any(np.isnan(toolpath), axis=1)
    segments: List[np.ndarray] = []
    start = 0
    for i, broken in enumerate(is_break):
        if broken:
            if i > start:
                segments.append(toolpath[start:i])
            start = i + 1
    if start < toolpath.shape[0]:
        segments.append(toolpath[start:])
    return [seg for seg in segments if seg.shape[0] >= 2]


def _fmt_num(value: float, decimals: int = 4) -> str:
    if abs(value) < 1e-10:
        value = 0.0
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0")
        if text.endswith("."):
            return text
    return text


def _fmt_axis(axis: str, value: float, decimals: int = 4) -> str:
    return f"{axis}{_fmt_num(value, decimals=decimals)}"


def _fmt_feed(value: float) -> str:
    return f"F{_fmt_num(value, decimals=1)}"


def export_gcode(file_path: str | Path, toolpath: np.ndarray, params: CNCParameters) -> None:
    """Export unwrapped (X,S,Z) path to 4th-axis G-code with X/A/Z moves."""
    segments = _split_toolpath_segments(toolpath)
    if not segments:
        raise ValueError("Toolpath must contain at least one valid segment.")

    path = Path(file_path)
    shaft_radius = 0.5 * params.shaft_diameter
    safe_z = float(params.safe_height)
    feed = float(params.feed_rate)
    spindle = int(params.spindle_speed)
    plunge_feed = min(feed, 80.0)

    lines = [
        f"({path.stem})",
        "(Generated by cnc_radial_surface_app.py)",
        f"(Shaft D={params.shaft_diameter:.3f} | Pocket L={params.pocket_length:.3f} R={params.pocket_end_radius:.3f} | Depth={params.pocket_depth:.3f} | Angle={params.oblong_angle_deg:.3f}deg)",
        "G90 G94",
        "G21",
        "M6 T1",
        f"S{spindle} M3",
        "G54",
    ]

    for segment in segments:
        cut_z = float(segment[0, 2])
        if not np.allclose(segment[:, 2], cut_z):
            raise ValueError("Each machining segment must have constant Z depth.")

        x_vals = segment[:, 0]
        a_vals = np.rad2deg(segment[:, 1] / shaft_radius)

        lines.append(
            f"G0 {_fmt_axis('X', float(x_vals[0]))} {_fmt_axis('A', float(a_vals[0]), decimals=5)}"
        )
        lines.append(f"G0 {_fmt_axis('Z', safe_z)}")
        lines.append(f"G1 {_fmt_axis('Z', cut_z)} {_fmt_feed(plunge_feed)}")

        if segment.shape[0] > 1:
            lines.append(
                f"G1 {_fmt_axis('X', float(x_vals[1]))} {_fmt_axis('A', float(a_vals[1]), decimals=5)} {_fmt_feed(feed)}"
            )
            for x, a in zip(x_vals[2:], a_vals[2:]):
                lines.append(
                    f"G1 {_fmt_axis('X', float(x))} {_fmt_axis('A', float(a), decimals=5)}"
                )

        lines.append(f"G0 {_fmt_axis('Z', safe_z)}")

    lines.extend(["M5", "G28", "M30"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _single_pocket_boundary(params: CNCParameters, arc_samples: int = 80) -> np.ndarray:
    radius = float(params.pocket_end_radius)
    straight_half_len = max(0.0, 0.5 * params.pocket_length - radius)
    perimeter = max(1e-6, 4.0 * straight_half_len + 2.0 * np.pi * radius)
    spacing = max(0.05, perimeter / max(40, int(arc_samples) * 2))

    loop = _sample_obround_loop(radius, straight_half_len, 0.0, spacing)
    boundary = loop[:, :2]
    boundary = _rotate_xs_boundary(boundary, params.oblong_angle_deg)
    boundary[:, 0] += float(params.axial_center)
    return boundary


def _pattern_boundaries(params: CNCParameters) -> List[np.ndarray]:
    _validate_geometry(params)
    base = _single_pocket_boundary(params)
    shaft_radius = 0.5 * params.shaft_diameter
    count = max(1, int(round(params.pattern_count)))

    boundaries: List[np.ndarray] = []
    for idx in range(count):
        angle_deg = params.pattern_start_angle + (360.0 * idx / count)
        s_offset = shaft_radius * np.deg2rad(angle_deg)
        boundary = base.copy()
        boundary[:, 1] += s_offset
        boundaries.append(boundary)
    return boundaries


def _spiral_level_count(params: CNCParameters) -> int:
    tool_radius = 0.5 * params.tool_diameter
    inner_radius = max(0.0, params.pocket_end_radius - tool_radius)
    return int(_build_spiral_levels(inner_radius, params.step_over).size)


def _unwrap_to_cyl(points: np.ndarray, shaft_radius: float) -> np.ndarray:
    x = points[:, 0]
    s = points[:, 1]
    z = points[:, 2]
    theta = s / shaft_radius
    radius = shaft_radius + z
    y = radius * np.cos(theta)
    zc = radius * np.sin(theta)
    return np.column_stack((x, y, zc))


def _boundary_to_cyl(boundary: np.ndarray, shaft_radius: float) -> np.ndarray:
    x = boundary[:, 0]
    s = boundary[:, 1]
    theta = s / shaft_radius
    y = shaft_radius * np.cos(theta)
    z = shaft_radius * np.sin(theta)
    return np.column_stack((x, y, z))


def _generate_shaft_mesh(
    params: CNCParameters,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    shaft_radius = 0.5 * params.shaft_diameter
    x_margin = max(6.0, 0.35 * params.pocket_length)
    x_min = params.axial_center - 0.5 * params.pocket_length - x_margin
    x_max = params.axial_center + 0.5 * params.pocket_length + x_margin

    nx = max(50, int(params.mesh_axial_samples))
    nt = max(80, int(params.mesh_angular_samples))
    x = np.linspace(x_min, x_max, nx)
    theta = np.linspace(0.0, 2.0 * np.pi, nt)
    xx, tt = np.meshgrid(x, theta, indexing="xy")

    yy = shaft_radius * np.cos(tt)
    zz = shaft_radius * np.sin(tt)
    return xx, yy, zz, x_min, x_max


def update_visualization(state: AppState) -> None:
    params = state.params
    state.ax3d.cla()

    toolpath = generate_toolpath(params)
    segments = _split_toolpath_segments(toolpath)
    boundaries = _pattern_boundaries(params)

    shaft_radius = 0.5 * params.shaft_diameter
    xx, yy, zz, x_min, x_max = _generate_shaft_mesh(params)
    state.ax3d.plot_surface(
        xx,
        yy,
        zz,
        color="#d0d6dc",
        linewidth=0.0,
        antialiased=True,
        alpha=0.5,
        rcount=min(120, zz.shape[0]),
        ccount=min(160, zz.shape[1]),
    )

    if params.show_boundaries:
        for boundary in boundaries:
            cyl = _boundary_to_cyl(boundary, shaft_radius)
            state.ax3d.plot(
                cyl[:, 0],
                cyl[:, 1],
                cyl[:, 2],
                color="#1f77b4",
                linewidth=1.4,
                alpha=0.95,
            )

    if params.show_toolpath:
        for seg in segments:
            cyl = _unwrap_to_cyl(seg, shaft_radius)
            state.ax3d.plot(
                cyl[:, 0],
                cyl[:, 1],
                cyl[:, 2],
                color="#d62728",
                linewidth=1.15,
                alpha=0.95,
            )

    r_plot = shaft_radius + max(2.2, params.safe_height + 0.4)
    state.ax3d.set_xlim(x_min, x_max)
    state.ax3d.set_ylim(-r_plot, r_plot)
    state.ax3d.set_zlim(-r_plot, r_plot)
    state.ax3d.set_box_aspect((x_max - x_min, 2.0 * r_plot, 2.0 * r_plot))
    state.ax3d.view_init(elev=20.0, azim=-58.0)
    state.ax3d.set_xlabel("X (mm)")
    state.ax3d.set_ylabel("Y (mm)")
    state.ax3d.set_zlabel("Z (mm)")
    state.ax3d.set_title("4th-Axis Spiral Oblong Pocket Pattern on Shaft")

    points = int(sum(seg.shape[0] for seg in segments))
    loops_per_pocket = max(0, _spiral_level_count(params) - 1)
    state.status_text.set_text(
        f"Pockets: {len(segments)} | Spiral Loops/Pocket: {loops_per_pocket} | Points: {points} | "
        f"Depth: {params.pocket_depth:.3f} mm | Angle: {params.oblong_angle_deg:.1f} deg"
    )
    state.fig.canvas.draw_idle()


def _build_slider(
    fig: plt.Figure,
    pos: Tuple[float, float, float, float],
    label: str,
    vmin: float,
    vmax: float,
    vinit: float,
    valstep: float | None = None,
) -> Slider:
    ax = fig.add_axes(pos)
    return Slider(
        ax=ax,
        label=label,
        valmin=vmin,
        valmax=vmax,
        valinit=vinit,
        valstep=valstep,
    )


def main() -> None:
    params = CNCParameters()
    defaults = CNCParameters()

    fig = plt.figure(figsize=(15, 9))
    ax3d = fig.add_axes([0.05, 0.08, 0.66, 0.86], projection="3d")

    slider_y = np.linspace(0.90, 0.335, 16)
    sliders = {
        "shaft_diameter": _build_slider(
            fig,
            (0.75, slider_y[0], 0.22, 0.022),
            "Shaft Dia (mm)",
            8.0,
            180.0,
            params.shaft_diameter,
        ),
        "pocket_length": _build_slider(
            fig,
            (0.75, slider_y[1], 0.22, 0.022),
            "Pocket Length (mm)",
            2.0,
            120.0,
            params.pocket_length,
        ),
        "pocket_end_radius": _build_slider(
            fig,
            (0.75, slider_y[2], 0.22, 0.022),
            "Pocket End R (mm)",
            0.8,
            30.0,
            params.pocket_end_radius,
        ),
        "pocket_depth": _build_slider(
            fig,
            (0.75, slider_y[3], 0.22, 0.022),
            "Pocket Depth (mm)",
            0.01,
            5.0,
            params.pocket_depth,
        ),
        "tool_diameter": _build_slider(
            fig,
            (0.75, slider_y[4], 0.22, 0.022),
            "Tool Dia (mm)",
            0.4,
            20.0,
            params.tool_diameter,
        ),
        "step_over": _build_slider(
            fig,
            (0.75, slider_y[5], 0.22, 0.022),
            "Step-over (mm)",
            0.02,
            4.0,
            params.step_over,
        ),
        "pattern_count": _build_slider(
            fig,
            (0.75, slider_y[6], 0.22, 0.022),
            "Pattern Count",
            1,
            48,
            params.pattern_count,
            1,
        ),
        "pattern_start_angle": _build_slider(
            fig,
            (0.75, slider_y[7], 0.22, 0.022),
            "Pattern Start (deg)",
            0.0,
            360.0,
            params.pattern_start_angle,
        ),
        "axial_center": _build_slider(
            fig,
            (0.75, slider_y[8], 0.22, 0.022),
            "Axial Center X (mm)",
            -120.0,
            120.0,
            params.axial_center,
        ),
        "oblong_angle_deg": _build_slider(
            fig,
            (0.75, slider_y[9], 0.22, 0.022),
            "Oblong Angle (deg)",
            -90.0,
            90.0,
            params.oblong_angle_deg,
        ),
        "feed_rate": _build_slider(
            fig,
            (0.75, slider_y[10], 0.22, 0.022),
            "Feed (mm/min)",
            20.0,
            10000.0,
            params.feed_rate,
        ),
        "spindle_speed": _build_slider(
            fig,
            (0.75, slider_y[11], 0.22, 0.022),
            "Spindle (RPM)",
            1000,
            40000,
            params.spindle_speed,
            100,
        ),
        "safe_height": _build_slider(
            fig,
            (0.75, slider_y[12], 0.22, 0.022),
            "Safe Z (mm)",
            0.1,
            30.0,
            params.safe_height,
        ),
        "path_point_spacing": _build_slider(
            fig,
            (0.75, slider_y[13], 0.22, 0.022),
            "Path Spacing (mm)",
            0.05,
            5.0,
            params.path_point_spacing,
        ),
        "mesh_axial_samples": _build_slider(
            fig,
            (0.75, slider_y[14], 0.22, 0.022),
            "Mesh Axial",
            50,
            400,
            params.mesh_axial_samples,
            1,
        ),
        "mesh_angular_samples": _build_slider(
            fig,
            (0.75, slider_y[15], 0.22, 0.022),
            "Mesh Angular",
            80,
            600,
            params.mesh_angular_samples,
            1,
        ),
    }

    toggle_ax = fig.add_axes([0.75, 0.245, 0.22, 0.06])
    toggle = CheckButtons(
        toggle_ax,
        ["Show Toolpath", "Show Pockets"],
        [params.show_toolpath, params.show_boundaries],
    )

    file_box_ax = fig.add_axes([0.75, 0.195, 0.22, 0.04])
    file_box = TextBox(file_box_ax, "Output File", initial=params.output_filename)

    export_ax = fig.add_axes([0.75, 0.14, 0.105, 0.04])
    reset_ax = fig.add_axes([0.865, 0.14, 0.105, 0.04])
    export_btn = Button(export_ax, "Export NC")
    reset_btn = Button(reset_ax, "Reset")

    status_text = fig.text(0.75, 0.008, "", fontsize=9)

    state = AppState(
        params=params,
        fig=fig,
        ax3d=ax3d,
        sliders=sliders,
        file_box=file_box,
        toggle=toggle,
        status_text=status_text,
        redraw_timer=None,
        redraw_pending=False,
    )

    geometry_slider_keys = {
        "shaft_diameter",
        "pocket_length",
        "pocket_end_radius",
        "pocket_depth",
        "tool_diameter",
        "step_over",
        "pattern_count",
        "pattern_start_angle",
        "axial_center",
        "oblong_angle_deg",
        "path_point_spacing",
        "mesh_axial_samples",
        "mesh_angular_samples",
    }

    def pull_params_from_widgets() -> None:
        p = state.params
        p.shaft_diameter = float(sliders["shaft_diameter"].val)
        p.pocket_length = float(sliders["pocket_length"].val)
        p.pocket_end_radius = float(sliders["pocket_end_radius"].val)
        p.pocket_depth = float(sliders["pocket_depth"].val)
        p.tool_diameter = float(sliders["tool_diameter"].val)
        p.step_over = float(sliders["step_over"].val)
        p.pattern_count = int(round(sliders["pattern_count"].val))
        p.pattern_start_angle = float(sliders["pattern_start_angle"].val)
        p.axial_center = float(sliders["axial_center"].val)
        p.oblong_angle_deg = float(sliders["oblong_angle_deg"].val)
        p.feed_rate = float(sliders["feed_rate"].val)
        p.spindle_speed = int(round(sliders["spindle_speed"].val))
        p.safe_height = float(sliders["safe_height"].val)
        p.path_point_spacing = float(sliders["path_point_spacing"].val)
        p.mesh_axial_samples = int(round(sliders["mesh_axial_samples"].val))
        p.mesh_angular_samples = int(round(sliders["mesh_angular_samples"].val))
        p.output_filename = state.file_box.text.strip() or defaults.output_filename

    def perform_redraw() -> None:
        if not state.redraw_pending:
            return
        state.redraw_pending = False
        pull_params_from_widgets()
        try:
            update_visualization(state)
        except ValueError as exc:
            state.status_text.set_text(f"Parameter issue: {exc}")
            state.fig.canvas.draw_idle()

    def schedule_redraw() -> None:
        state.redraw_pending = True
        state.redraw_timer.stop()
        state.redraw_timer.start()

    def on_slider_change(slider_key: str) -> None:
        pull_params_from_widgets()
        if slider_key in geometry_slider_keys:
            schedule_redraw()
            return
        state.status_text.set_text("Updated spindle/feed settings")
        state.fig.canvas.draw_idle()

    def on_toggle(_: str) -> None:
        status = state.toggle.get_status()
        state.params.show_toolpath = bool(status[0])
        state.params.show_boundaries = bool(status[1])
        schedule_redraw()

    def on_file_submit(text: str) -> None:
        state.params.output_filename = text.strip() or defaults.output_filename

    def on_export(_: object) -> None:
        pull_params_from_widgets()
        try:
            toolpath = generate_toolpath(state.params)
            export_gcode(state.params.output_filename, toolpath, state.params)
            state.status_text.set_text(
                f"Exported: {Path(state.params.output_filename).resolve()}"
            )
        except Exception as exc:  # pragma: no cover - runtime UI path
            state.status_text.set_text(f"Export failed: {exc}")
        state.fig.canvas.draw_idle()

    def on_reset(_: object) -> None:
        for key, slider in sliders.items():
            slider.set_val(getattr(defaults, key))

        current = state.toggle.get_status()
        if current[0] != defaults.show_toolpath:
            state.toggle.set_active(0)
        current = state.toggle.get_status()
        if current[1] != defaults.show_boundaries:
            state.toggle.set_active(1)

        state.params.show_toolpath = bool(state.toggle.get_status()[0])
        state.params.show_boundaries = bool(state.toggle.get_status()[1])
        state.params.output_filename = defaults.output_filename
        state.file_box.set_val(defaults.output_filename)
        state.redraw_pending = True
        perform_redraw()

    state.redraw_timer = fig.canvas.new_timer(interval=90)
    state.redraw_timer.add_callback(perform_redraw)

    for key, slider in sliders.items():
        slider.on_changed(lambda _value, slider_key=key: on_slider_change(slider_key))
    toggle.on_clicked(on_toggle)
    file_box.on_submit(on_file_submit)
    export_btn.on_clicked(on_export)
    reset_btn.on_clicked(on_reset)

    pull_params_from_widgets()
    state.redraw_pending = True
    perform_redraw()
    plt.show()


if __name__ == "__main__":
    main()
