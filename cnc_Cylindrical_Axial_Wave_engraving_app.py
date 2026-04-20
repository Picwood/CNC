"""
4th-Axis Axial Wave Engraving Generator

Interactive application that:
1. Builds wrapped 4th-axis sinusoidal wave paths on a cylindrical shaft
2. Displays shaft and engraving toolpaths in 3D
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
    axial_amplitude: float = 5.0
    wave_count: int = 6
    phase_shift_deg: float = 0.0
    pattern_count: int = 4
    pattern_start_angle: float = 0.0
    pattern_phase_delta_deg: float = 0.0
    axial_center: float = 0.0
    engrave_depth: float = 0.2
    feed_rate: float = 450.0
    spindle_speed: int = 12000
    safe_height: float = 2.0
    path_point_spacing: float = 0.35
    mesh_axial_samples: int = 140
    mesh_angular_samples: int = 160
    show_toolpath: bool = True
    show_reference: bool = True
    output_filename: str = "4th_axis_axial_wave_engraving.nc"


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


def _validate_geometry(params: CNCParameters) -> None:
    if params.shaft_diameter <= 0.0:
        raise ValueError("Shaft diameter must be positive.")
    if params.engrave_depth <= 0.0:
        raise ValueError("Engraving depth must be positive.")
    shaft_radius = 0.5 * params.shaft_diameter
    if params.engrave_depth >= shaft_radius - 1e-6:
        raise ValueError("Engraving depth must be less than shaft radius.")
    if params.wave_count < 0:
        raise ValueError("Wave count must be zero or positive.")
    if int(round(params.pattern_count)) < 1:
        raise ValueError("Pattern count must be at least 1.")


def _build_theta_samples(
    params: CNCParameters,
    phase_shift_rad: float,
) -> np.ndarray:
    spacing = max(0.05, float(params.path_point_spacing))
    theta_end = 2.0 * np.pi
    theta_values = [0.0]
    theta = 0.0

    shaft_radius = 0.5 * params.shaft_diameter
    amplitude = float(params.axial_amplitude)
    wave_count = float(params.wave_count)

    while theta < theta_end:
        dx_dtheta = amplitude * wave_count * np.cos(wave_count * theta + phase_shift_rad)
        ds_dtheta = shaft_radius
        ds_path = float(np.hypot(dx_dtheta, ds_dtheta))
        dtheta = np.clip(spacing / max(ds_path, 1e-9), 0.0005, 0.25)
        theta = min(theta + float(dtheta), theta_end)
        theta_values.append(theta)

    if theta_values[-1] < theta_end:
        theta_values.append(theta_end)
    return np.asarray(theta_values, dtype=float)


def _generate_wave_segment(
    params: CNCParameters,
    s_offset: float,
    phase_shift_rad: float,
    z_value: float,
) -> np.ndarray:
    theta = _build_theta_samples(params, phase_shift_rad)
    shaft_radius = 0.5 * params.shaft_diameter
    wave_count = float(params.wave_count)

    x = params.axial_center + params.axial_amplitude * np.sin(
        wave_count * theta + phase_shift_rad
    )
    s = shaft_radius * theta + s_offset
    z = np.full_like(theta, float(z_value))
    return np.column_stack((x, s, z))


def generate_toolpath(params: CNCParameters) -> np.ndarray:
    """Generate NaN-separated multi-wave toolpath in unwrapped coordinates."""
    _validate_geometry(params)

    shaft_radius = 0.5 * params.shaft_diameter
    count = max(1, int(round(params.pattern_count)))
    phase_base = float(np.deg2rad(params.phase_shift_deg))
    phase_delta = float(np.deg2rad(params.pattern_phase_delta_deg))
    z_cut = -abs(float(params.engrave_depth))
    segments: List[np.ndarray] = []

    for idx in range(count):
        angle_deg = params.pattern_start_angle + (360.0 * idx / count)
        s_offset = shaft_radius * np.deg2rad(angle_deg)
        phase_shift = phase_base + idx * phase_delta
        seg = _generate_wave_segment(params, s_offset, phase_shift, z_cut)
        segments.append(seg)

    result = segments[0]
    for seg in segments[1:]:
        result = np.vstack([result, np.array([[np.nan, np.nan, np.nan]]), seg])

    if result.shape[0] > 300000:
        raise ValueError("Toolpath too dense. Increase path spacing or reduce patterns.")
    return result


def generate_reference_paths(params: CNCParameters) -> np.ndarray:
    """Generate NaN-separated reference wave paths at shaft surface (Z=0)."""
    _validate_geometry(params)

    shaft_radius = 0.5 * params.shaft_diameter
    count = max(1, int(round(params.pattern_count)))
    phase_base = float(np.deg2rad(params.phase_shift_deg))
    phase_delta = float(np.deg2rad(params.pattern_phase_delta_deg))
    segments: List[np.ndarray] = []

    for idx in range(count):
        angle_deg = params.pattern_start_angle + (360.0 * idx / count)
        s_offset = shaft_radius * np.deg2rad(angle_deg)
        phase_shift = phase_base + idx * phase_delta
        seg = _generate_wave_segment(params, s_offset, phase_shift, z_value=0.0)
        segments.append(seg)

    result = segments[0]
    for seg in segments[1:]:
        result = np.vstack([result, np.array([[np.nan, np.nan, np.nan]]), seg])
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
        "(Generated by cnc_Cylindrical_Axial_Wave_engraving_app.py)",
        (
            f"(Shaft D={params.shaft_diameter:.3f} | "
            f"Axial Amp={params.axial_amplitude:.3f} | "
            f"Waves/Rev={int(round(params.wave_count))} | "
            f"Depth={params.engrave_depth:.3f})"
        ),
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


def _unwrap_to_cyl(points: np.ndarray, shaft_radius: float) -> np.ndarray:
    x = points[:, 0]
    s = points[:, 1]
    z = points[:, 2]
    theta = s / shaft_radius
    radius = shaft_radius + z
    y = radius * np.cos(theta)
    zc = radius * np.sin(theta)
    return np.column_stack((x, y, zc))


def _generate_shaft_mesh(
    params: CNCParameters,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    shaft_radius = 0.5 * params.shaft_diameter
    amp = abs(float(params.axial_amplitude))
    x_margin = max(6.0, 0.8 * amp + 4.0)
    x_min = params.axial_center - amp - x_margin
    x_max = params.axial_center + amp + x_margin

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
    reference = generate_reference_paths(params)
    tool_segments = _split_toolpath_segments(toolpath)
    reference_segments = _split_toolpath_segments(reference)

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

    if params.show_reference:
        for seg in reference_segments:
            cyl = _unwrap_to_cyl(seg, shaft_radius)
            state.ax3d.plot(
                cyl[:, 0],
                cyl[:, 1],
                cyl[:, 2],
                color="#1f77b4",
                linewidth=1.0,
                alpha=0.9,
            )

    if params.show_toolpath:
        for seg in tool_segments:
            cyl = _unwrap_to_cyl(seg, shaft_radius)
            state.ax3d.plot(
                cyl[:, 0],
                cyl[:, 1],
                cyl[:, 2],
                color="#d62728",
                linewidth=1.25,
                alpha=0.95,
            )

    r_plot = shaft_radius + max(2.0, params.safe_height + 0.4)
    state.ax3d.set_xlim(x_min, x_max)
    state.ax3d.set_ylim(-r_plot, r_plot)
    state.ax3d.set_zlim(-r_plot, r_plot)
    state.ax3d.set_box_aspect((x_max - x_min, 2.0 * r_plot, 2.0 * r_plot))
    state.ax3d.view_init(elev=20.0, azim=-58.0)
    state.ax3d.set_xlabel("X (mm)")
    state.ax3d.set_ylabel("Y (mm)")
    state.ax3d.set_zlabel("Z (mm)")
    state.ax3d.set_title("4th-Axis Axial Wave Engraving on Shaft")

    points = int(sum(seg.shape[0] for seg in tool_segments))
    state.status_text.set_text(
        f"Patterns: {len(tool_segments)} | Waves/Rev: {int(round(params.wave_count))} | "
        f"Points: {points} | Depth: {params.engrave_depth:.3f} mm"
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

    slider_y = np.linspace(0.90, 0.34, 15)
    sliders = {
        "shaft_diameter": _build_slider(
            fig,
            (0.75, slider_y[0], 0.22, 0.022),
            "Shaft Dia (mm)",
            8.0,
            180.0,
            params.shaft_diameter,
        ),
        "axial_amplitude": _build_slider(
            fig,
            (0.75, slider_y[1], 0.22, 0.022),
            "Axial Amp (mm)",
            0.0,
            80.0,
            params.axial_amplitude,
        ),
        "wave_count": _build_slider(
            fig,
            (0.75, slider_y[2], 0.22, 0.022),
            "Waves / Rev",
            0,
            60,
            params.wave_count,
            1,
        ),
        "phase_shift_deg": _build_slider(
            fig,
            (0.75, slider_y[3], 0.22, 0.022),
            "Phase Shift (deg)",
            -180.0,
            180.0,
            params.phase_shift_deg,
        ),
        "pattern_count": _build_slider(
            fig,
            (0.75, slider_y[4], 0.22, 0.022),
            "Pattern Count",
            1,
            48,
            params.pattern_count,
            1,
        ),
        "pattern_start_angle": _build_slider(
            fig,
            (0.75, slider_y[5], 0.22, 0.022),
            "Pattern Start (deg)",
            0.0,
            360.0,
            params.pattern_start_angle,
        ),
        "pattern_phase_delta_deg": _build_slider(
            fig,
            (0.75, slider_y[6], 0.22, 0.022),
            "Phase Delta/Pattern",
            -180.0,
            180.0,
            params.pattern_phase_delta_deg,
        ),
        "axial_center": _build_slider(
            fig,
            (0.75, slider_y[7], 0.22, 0.022),
            "Axial Center X (mm)",
            -140.0,
            140.0,
            params.axial_center,
        ),
        "engrave_depth": _build_slider(
            fig,
            (0.75, slider_y[8], 0.22, 0.022),
            "Engrave Depth (mm)",
            0.01,
            6.0,
            params.engrave_depth,
        ),
        "feed_rate": _build_slider(
            fig,
            (0.75, slider_y[9], 0.22, 0.022),
            "Feed (mm/min)",
            20.0,
            12000.0,
            params.feed_rate,
        ),
        "spindle_speed": _build_slider(
            fig,
            (0.75, slider_y[10], 0.22, 0.022),
            "Spindle (RPM)",
            1000,
            40000,
            params.spindle_speed,
            100,
        ),
        "safe_height": _build_slider(
            fig,
            (0.75, slider_y[11], 0.22, 0.022),
            "Safe Z (mm)",
            0.1,
            30.0,
            params.safe_height,
        ),
        "path_point_spacing": _build_slider(
            fig,
            (0.75, slider_y[12], 0.22, 0.022),
            "Path Spacing (mm)",
            0.05,
            5.0,
            params.path_point_spacing,
        ),
        "mesh_axial_samples": _build_slider(
            fig,
            (0.75, slider_y[13], 0.22, 0.022),
            "Mesh Axial",
            50,
            400,
            params.mesh_axial_samples,
            1,
        ),
        "mesh_angular_samples": _build_slider(
            fig,
            (0.75, slider_y[14], 0.22, 0.022),
            "Mesh Angular",
            80,
            600,
            params.mesh_angular_samples,
            1,
        ),
    }

    toggle_ax = fig.add_axes([0.75, 0.255, 0.22, 0.06])
    toggle = CheckButtons(
        toggle_ax,
        ["Show Toolpath", "Show Ref Wave"],
        [params.show_toolpath, params.show_reference],
    )

    file_box_ax = fig.add_axes([0.75, 0.205, 0.22, 0.04])
    file_box = TextBox(file_box_ax, "Output File", initial=params.output_filename)

    export_ax = fig.add_axes([0.75, 0.15, 0.105, 0.04])
    reset_ax = fig.add_axes([0.865, 0.15, 0.105, 0.04])
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
        "axial_amplitude",
        "wave_count",
        "phase_shift_deg",
        "pattern_count",
        "pattern_start_angle",
        "pattern_phase_delta_deg",
        "axial_center",
        "engrave_depth",
        "safe_height",
        "path_point_spacing",
        "mesh_axial_samples",
        "mesh_angular_samples",
    }

    def pull_params_from_widgets() -> None:
        p = state.params
        p.shaft_diameter = float(sliders["shaft_diameter"].val)
        p.axial_amplitude = float(sliders["axial_amplitude"].val)
        p.wave_count = int(round(sliders["wave_count"].val))
        p.phase_shift_deg = float(sliders["phase_shift_deg"].val)
        p.pattern_count = int(round(sliders["pattern_count"].val))
        p.pattern_start_angle = float(sliders["pattern_start_angle"].val)
        p.pattern_phase_delta_deg = float(sliders["pattern_phase_delta_deg"].val)
        p.axial_center = float(sliders["axial_center"].val)
        p.engrave_depth = float(sliders["engrave_depth"].val)
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
        state.params.show_reference = bool(status[1])
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
        if current[1] != defaults.show_reference:
            state.toggle.set_active(1)

        state.params.show_toolpath = bool(state.toggle.get_status()[0])
        state.params.show_reference = bool(state.toggle.get_status()[1])
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
