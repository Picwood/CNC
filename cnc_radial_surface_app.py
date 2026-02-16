"""
Flat Polar Engraving Generator

Interactive application that:
1. Builds a planar periodic polar contour: r(theta) = R + A * sin(n * theta + k)
2. Displays flat stock and contour toolpath in 3D
3. Exports industrial-style 3-axis G-code for constant-depth engraving
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox

try:
    import cnc_toolpath_accel as _accel  # type: ignore
except Exception:
    _accel = None


@dataclass
class CNCParameters:
    radius: float = 35.0
    amplitude: float = 6.0
    angular_frequency: int = 6
    phase_shift: float = 0.0
    z_scale: float = 1.0
    tool_diameter: float = 3.0
    step_over: float = 0.4
    pass_count: int = 1
    feed_rate: float = 700.0
    spindle_speed: int = 12000
    safe_height: float = 5.0
    depth_offset: float = 0.6
    mesh_radial_samples: int = 120
    mesh_angular_samples: int = 260
    path_point_spacing: float = 0.35
    show_toolpath: bool = True
    output_filename: str = "polar_engraving.nc"


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


def polar_radius(
    theta: np.ndarray | float, params: CNCParameters, radial_offset: float = 0.0
) -> np.ndarray | float:
    """Periodic sinusoidal radius function in polar coordinates."""
    return (
        params.radius
        + radial_offset
        + params.amplitude * np.sin(params.angular_frequency * theta + params.phase_shift)
    )


def generate_surface(
    params: CNCParameters,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a flat circular stock surface for visualization."""
    stock_radius = params.radius + abs(params.amplitude) + params.tool_diameter + 3.0
    nr = max(30, int(params.mesh_radial_samples))
    nth = max(80, int(params.mesh_angular_samples))

    r = np.linspace(0.0, stock_radius, nr)
    theta = np.linspace(0.0, 2.0 * np.pi, nth, endpoint=True)
    rr, tt = np.meshgrid(r, theta, indexing="xy")

    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    z = np.zeros_like(x)
    return x, y, z


def _build_theta_samples(params: CNCParameters, radial_offset: float) -> np.ndarray:
    """Adaptive theta sampling for near-constant XY point spacing."""
    spacing = max(0.05, params.path_point_spacing)
    theta_end = 2.0 * np.pi

    theta_values = [0.0]
    theta = 0.0
    while theta < theta_end:
        r = float(polar_radius(theta, params, radial_offset))
        dr_dtheta = (
            params.amplitude
            * params.angular_frequency
            * np.cos(params.angular_frequency * theta + params.phase_shift)
        )
        ds_dtheta = np.sqrt(r * r + dr_dtheta * dr_dtheta)
        dtheta = np.clip(spacing / max(ds_dtheta, 1e-8), 0.0005, 0.2)
        theta = min(theta + dtheta, theta_end)
        theta_values.append(theta)

    if theta_values[-1] < theta_end:
        theta_values.append(theta_end)
    return np.asarray(theta_values, dtype=float)


def _build_radial_offsets(params: CNCParameters) -> np.ndarray:
    """
    Build offsets around centerline contour from explicit pass count.

    `step_over` controls radial spacing between adjacent contours and
    `pass_count` controls how many total contour passes are generated.
    """
    count = max(1, int(round(params.pass_count)))
    pitch = max(0.02, params.step_over)
    if count == 1:
        return np.array([0.0], dtype=float)

    # Symmetric offsets around centerline. Even counts omit exact center.
    return (np.arange(count, dtype=float) - 0.5 * (count - 1)) * pitch


def _split_toolpath_segments(toolpath: np.ndarray) -> List[np.ndarray]:
    """Split NaN-separated toolpath into contiguous machining segments."""
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


def _generate_toolpath_python(
    params: CNCParameters, offsets: np.ndarray | None = None
) -> np.ndarray:
    """
    Generate constant-depth engraving contour(s) from periodic polar radius.

    Returns:
        Nx3 array with NaN separator rows between contour segments.
    """
    tool_radius = max(0.05, 0.5 * params.tool_diameter)
    if offsets is None:
        offsets = _build_radial_offsets(params)
    cut_z = -abs(params.depth_offset) * max(0.01, params.z_scale)

    loops: List[np.ndarray] = []
    for offset in offsets:
        theta = _build_theta_samples(params, float(offset))
        r = polar_radius(theta, params, float(offset))
        r_min = float(np.min(r))
        if r_min <= tool_radius:
            raise ValueError(
                "Contour collapses near center. Increase Radius, reduce Amplitude, "
                "or reduce tool diameter."
            )

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.full_like(x, cut_z)
        loop = np.column_stack((x, y, z))

        # Ensure each contour is explicitly closed for engraving.
        if not np.allclose(loop[0, :2], loop[-1, :2]):
            loop = np.vstack([loop, loop[0]])
        loops.append(loop)

    if not loops:
        raise ValueError("No valid contour passes generated.")

    result = loops[0]
    for loop in loops[1:]:
        result = np.vstack([result, np.array([[np.nan, np.nan, np.nan]]), loop])

    if result.shape[0] > 220000:
        raise ValueError(
            "Toolpath too dense (>220k points). Increase point spacing or step-over."
        )
    return result


def _toolpath_backend_name() -> str:
    return "C++" if _accel is not None else "Python"


def generate_toolpath(params: CNCParameters) -> np.ndarray:
    """
    Generate constant-depth engraving contour(s) from periodic polar radius.

    Returns:
        Nx3 array with NaN separator rows between contour segments.
    """
    offsets = _build_radial_offsets(params)

    if _accel is None:
        return _generate_toolpath_python(params, offsets=offsets)

    try:
        path = _accel.generate_toolpath(
            float(params.radius),
            float(params.amplitude),
            int(params.angular_frequency),
            float(params.phase_shift),
            float(params.tool_diameter),
            float(params.depth_offset),
            float(params.z_scale),
            float(params.path_point_spacing),
            np.asarray(offsets, dtype=np.float64),
        )
        path = np.asarray(path, dtype=np.float64)
    except Exception:
        path = _generate_toolpath_python(params, offsets=offsets)

    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("Generated toolpath has an unexpected shape.")
    if path.shape[0] > 220000:
        raise ValueError(
            "Toolpath too dense (>220k points). Increase point spacing or step-over."
        )
    return path


def export_gcode(
    file_path: str | Path,
    toolpath: np.ndarray,
    params: CNCParameters,
) -> None:
    """Export NaN-separated contour segments as absolute metric G-code."""
    segments = _split_toolpath_segments(toolpath)
    if not segments:
        raise ValueError("Toolpath must contain at least one valid segment.")

    path = Path(file_path)
    safe_z = float(params.safe_height)
    feed = float(params.feed_rate)
    spindle = int(params.spindle_speed)
    plunge_feed = min(feed, 250.0)

    lines = [
        "%",
        "(Planar sinusoidal polar engraving)",
        "G21",
        "G90",
        "G17",
        "G94",
        f"G0 Z{safe_z:.4f}",
        f"M3 S{spindle}",
    ]

    for segment in segments:
        first = segment[0]
        lines.append(f"G0 X{first[0]:.4f} Y{first[1]:.4f}")
        lines.append(f"G1 Z{first[2]:.4f} F{plunge_feed:.2f}")
        lines.append(f"F{feed:.2f}")
        for point in segment[1:]:
            lines.append(f"G1 X{point[0]:.4f} Y{point[1]:.4f} Z{point[2]:.4f}")
        lines.append(f"G0 Z{safe_z:.4f}")

    lines.extend(
        [
            "G0 X0.0000 Y0.0000",
            "M5",
            "M30",
            "%",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _radial_span(params: CNCParameters) -> Tuple[float, float]:
    offsets = _build_radial_offsets(params)
    theta = np.linspace(0.0, 2.0 * np.pi, 2000, endpoint=True)
    mins = []
    maxs = []
    for offset in offsets:
        r = polar_radius(theta, params, float(offset))
        mins.append(float(np.min(r)))
        maxs.append(float(np.max(r)))
    return min(mins), max(maxs)


def _decimate_for_plot(segment: np.ndarray, max_points: int = 1600) -> np.ndarray:
    """Reduce display point count to keep interactive redraw fast."""
    if segment.shape[0] <= max_points:
        return segment
    step = max(1, int(np.ceil(segment.shape[0] / max_points)))
    sampled = segment[::step]
    if not np.allclose(sampled[-1, :2], segment[-1, :2]):
        sampled = np.vstack([sampled, segment[-1]])
    return sampled


def update_visualization(state: AppState) -> None:
    """Refresh flat stock mesh and optional contour engraving toolpath."""
    params = state.params
    state.ax3d.cla()

    x, y, z = generate_surface(params)
    state.ax3d.plot_surface(
        x,
        y,
        z,
        color="#d3d8dc",
        linewidth=0.0,
        antialiased=True,
        alpha=0.45,
        rcount=min(80, z.shape[0]),
        ccount=min(100, z.shape[1]),
    )

    path = generate_toolpath(params)
    segments = _split_toolpath_segments(path)
    if params.show_toolpath:
        for segment in segments:
            display_segment = _decimate_for_plot(segment)
            state.ax3d.plot(
                display_segment[:, 0],
                display_segment[:, 1],
                display_segment[:, 2],
                color="black",
                linewidth=1.2,
                alpha=0.95,
            )

    r_min, r_max = _radial_span(params)
    span = max(10.0, r_max + params.tool_diameter + 2.0)
    cut_z = float(np.nanmin(path[:, 2]))
    z_top = max(1.0, abs(cut_z) + 0.8)
    z_bottom = min(cut_z - 0.5, -0.1)

    state.ax3d.set_xlim(-span, span)
    state.ax3d.set_ylim(-span, span)
    state.ax3d.set_zlim(z_bottom, z_top)
    state.ax3d.set_box_aspect((1.0, 1.0, 0.12))
    state.ax3d.view_init(elev=74.0, azim=-90.0)
    state.ax3d.set_xlabel("X (mm)")
    state.ax3d.set_ylabel("Y (mm)")
    state.ax3d.set_zlabel("Z (mm)")
    state.ax3d.set_title("Planar Sinusoidal Polar Engraving")

    points = sum(seg.shape[0] for seg in segments)
    state.status_text.set_text(
        f"Passes: {len(segments)} | Points: {points} | r_min: {r_min:.2f} mm | "
        f"r_max: {r_max:.2f} mm | Backend: {_toolpath_backend_name()}"
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
        ax=ax, label=label, valmin=vmin, valmax=vmax, valinit=vinit, valstep=valstep
    )


def main() -> None:
    params = CNCParameters()
    fig = plt.figure(figsize=(15, 9))
    ax3d = fig.add_axes([0.05, 0.08, 0.66, 0.86], projection="3d")

    slider_y = np.linspace(0.89, 0.40, 13)
    sliders = {
        "radius": _build_slider(
            fig, (0.75, slider_y[0], 0.22, 0.022), "Mean Radius R (mm)", 5.0, 180.0, params.radius
        ),
        "amplitude": _build_slider(
            fig, (0.75, slider_y[1], 0.22, 0.022), "Radial Amp A (mm)", 0.0, 50.0, params.amplitude
        ),
        "angular_frequency": _build_slider(
            fig, (0.75, slider_y[2], 0.22, 0.022), "Lobes n", 1, 32, params.angular_frequency, 1
        ),
        "phase_shift": _build_slider(
            fig, (0.75, slider_y[3], 0.22, 0.022), "Phase k (rad)", 0.0, 2.0 * np.pi, params.phase_shift
        ),
        "z_scale": _build_slider(
            fig, (0.75, slider_y[4], 0.22, 0.022), "Z Scale", 0.1, 4.0, params.z_scale
        ),
        "tool_diameter": _build_slider(
            fig, (0.75, slider_y[5], 0.22, 0.022), "Tool Dia (mm)", 0.2, 20.0, params.tool_diameter
        ),
        "step_over": _build_slider(
            fig, (0.75, slider_y[6], 0.22, 0.022), "Step-over (mm)", 0.02, 5.0, params.step_over
        ),
        "pass_count": _build_slider(
            fig, (0.75, slider_y[7], 0.22, 0.022), "Passes", 1, 50, params.pass_count, 1
        ),
        "feed_rate": _build_slider(
            fig, (0.75, slider_y[8], 0.22, 0.022), "Feed (mm/min)", 50.0, 8000.0, params.feed_rate
        ),
        "spindle_speed": _build_slider(
            fig, (0.75, slider_y[9], 0.22, 0.022), "Spindle (RPM)", 1000, 30000, params.spindle_speed, 100
        ),
        "safe_height": _build_slider(
            fig, (0.75, slider_y[10], 0.22, 0.022), "Safe Z (mm)", 1.0, 60.0, params.safe_height
        ),
        "depth_offset": _build_slider(
            fig, (0.75, slider_y[11], 0.22, 0.022), "Engrave Depth (mm)", 0.02, 12.0, params.depth_offset
        ),
        "path_point_spacing": _build_slider(
            fig, (0.75, slider_y[12], 0.22, 0.022), "Point Spacing (mm)", 0.05, 5.0, params.path_point_spacing
        ),
    }

    mesh_r = _build_slider(
        fig, (0.75, 0.37, 0.22, 0.022), "Mesh Radial", 30, 320, params.mesh_radial_samples, 1
    )
    mesh_t = _build_slider(
        fig, (0.75, 0.34, 0.22, 0.022), "Mesh Angular", 80, 560, params.mesh_angular_samples, 1
    )
    sliders["mesh_radial_samples"] = mesh_r
    sliders["mesh_angular_samples"] = mesh_t

    toggle_ax = fig.add_axes([0.75, 0.27, 0.22, 0.05])
    toggle = CheckButtons(toggle_ax, ["Show Toolpath"], [params.show_toolpath])

    file_box_ax = fig.add_axes([0.75, 0.21, 0.22, 0.04])
    file_box = TextBox(file_box_ax, "Output File", initial=params.output_filename)

    export_ax = fig.add_axes([0.75, 0.15, 0.105, 0.045])
    reset_ax = fig.add_axes([0.865, 0.15, 0.105, 0.045])
    export_btn = Button(export_ax, "Export G-code")
    reset_btn = Button(reset_ax, "Reset")

    status_text = fig.text(0.75, 0.1, "", fontsize=9)

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
    defaults = CNCParameters()
    geometry_slider_keys = {
        "radius",
        "amplitude",
        "angular_frequency",
        "phase_shift",
        "z_scale",
        "tool_diameter",
        "step_over",
        "pass_count",
        "depth_offset",
        "path_point_spacing",
        "mesh_radial_samples",
        "mesh_angular_samples",
    }

    def pull_params_from_widgets() -> None:
        p = state.params
        p.radius = float(sliders["radius"].val)
        p.amplitude = float(sliders["amplitude"].val)
        p.angular_frequency = int(round(sliders["angular_frequency"].val))
        p.phase_shift = float(sliders["phase_shift"].val)
        p.z_scale = float(sliders["z_scale"].val)
        p.tool_diameter = float(sliders["tool_diameter"].val)
        p.step_over = float(sliders["step_over"].val)
        p.pass_count = int(round(sliders["pass_count"].val))
        p.feed_rate = float(sliders["feed_rate"].val)
        p.spindle_speed = int(round(sliders["spindle_speed"].val))
        p.safe_height = float(sliders["safe_height"].val)
        p.depth_offset = float(sliders["depth_offset"].val)
        p.path_point_spacing = float(sliders["path_point_spacing"].val)
        p.mesh_radial_samples = int(round(sliders["mesh_radial_samples"].val))
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
        state.status_text.set_text(
            f"Updated cutting settings | Backend: {_toolpath_backend_name()}"
        )
        state.fig.canvas.draw_idle()

    def on_toggle(_: str) -> None:
        state.params.show_toolpath = bool(state.toggle.get_status()[0])
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
        if state.toggle.get_status()[0] != defaults.show_toolpath:
            state.toggle.set_active(0)
        state.params.show_toolpath = bool(state.toggle.get_status()[0])
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
