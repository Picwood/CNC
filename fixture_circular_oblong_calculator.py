"""
Circular oblong slot coverage calculator for a radial clamp fixture.

Coverage is computed from drilled-hole center positions only.
Hole diameter is intentionally excluded from the coverage logic so each
diameter case can be evaluated separately.

Assumptions:
- Four curved obround slots start every 90 degrees: 0, 90, 180, 270.
- Opposite slots are used together as clamp pairs:
  - pair A: starts at 0 and 180 degrees
  - pair B: starts at 90 and 270 degrees
- The user defines the two alternating slot spreads A/B.
- The user defines the two drilled-hole angles from a common reference.
- A fixture rotation is valid if the two drilled holes simultaneously fall
  inside one opposite slot pair, in either assignment order.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider


FULL_ROTATION_DEG = 360.0


@dataclass(frozen=True)
class SlotInput:
    spread_a_deg: float = 75.0
    spread_b_deg: float = 75.0
    b_start_offset_deg: float = -10.0
    hole_1_angle_deg: float = 0.0
    hole_2_angle_deg: float = 180.0
    plot: bool = True


@dataclass(frozen=True)
class SlotMetrics:
    start_angles_deg: Tuple[float, float, float, float]
    spreads_deg: Tuple[float, float, float, float]
    b_start_offset_deg: float
    hole_angles_deg: Tuple[float, float]
    a_hole_1_intervals_deg: Tuple[Tuple[float, float], ...]
    a_hole_2_intervals_deg: Tuple[Tuple[float, float], ...]
    b_hole_1_intervals_deg: Tuple[Tuple[float, float], ...]
    b_hole_2_intervals_deg: Tuple[Tuple[float, float], ...]
    valid_rotation_intervals_deg: Tuple[Tuple[float, float], ...]
    total_coverage_deg: float
    coverage_percent: float


def normalize_angle_deg(angle_deg: float) -> float:
    return angle_deg % FULL_ROTATION_DEG


def normalize_interval(start_deg: float, span_deg: float) -> List[Tuple[float, float]]:
    if span_deg <= 0.0:
        return []
    if span_deg >= FULL_ROTATION_DEG - 1e-9:
        return [(0.0, FULL_ROTATION_DEG)]

    start = normalize_angle_deg(start_deg)
    end = start + span_deg
    if end <= FULL_ROTATION_DEG:
        return [(start, end)]
    return [(0.0, end - FULL_ROTATION_DEG), (start, FULL_ROTATION_DEG)]


def merge_intervals(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    ordered = sorted((start, end) for start, end in intervals if end > start + 1e-9)
    if not ordered:
        return []

    merged: List[List[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def intersect_intervals(
    left: Sequence[Tuple[float, float]], right: Sequence[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    intersections: List[Tuple[float, float]] = []
    for left_start, left_end in left:
        for right_start, right_end in right:
            start = max(left_start, right_start)
            end = min(left_end, right_end)
            if end > start + 1e-9:
                intersections.append((start, end))
    return merge_intervals(intersections)


def total_interval_length(intervals: Sequence[Tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def rotation_intervals_for_hole_and_slot(
    hole_angle_deg: float, slot_start_deg: float, slot_spread_deg: float
) -> List[Tuple[float, float]]:
    """
    Valid fixture rotations phi such that:
    hole_angle is inside [phi + slot_start, phi + slot_start + slot_spread].
    """
    return normalize_interval(
        hole_angle_deg - slot_start_deg - slot_spread_deg,
        slot_spread_deg,
    )


def rotation_intervals_for_hole_and_slot_family(
    hole_angle_deg: float,
    slot_starts_deg: Sequence[float],
    slot_spread_deg: float,
) -> List[Tuple[float, float]]:
    return merge_intervals(
        interval
        for slot_start_deg in slot_starts_deg
        for interval in rotation_intervals_for_hole_and_slot(
            hole_angle_deg, slot_start_deg, slot_spread_deg
        )
    )


def calculate_metrics(slot_input: SlotInput) -> SlotMetrics:
    starts = (
        0.0,
        normalize_angle_deg(90.0 + slot_input.b_start_offset_deg),
        180.0,
        normalize_angle_deg(270.0 + slot_input.b_start_offset_deg),
    )
    spreads = (
        max(0.0, slot_input.spread_a_deg),
        max(0.0, slot_input.spread_b_deg),
        max(0.0, slot_input.spread_a_deg),
        max(0.0, slot_input.spread_b_deg),
    )
    hole_angles = (
        normalize_angle_deg(slot_input.hole_1_angle_deg),
        normalize_angle_deg(slot_input.hole_2_angle_deg),
    )

    a_hole_1 = rotation_intervals_for_hole_and_slot_family(
        hole_angles[0], (starts[0], starts[2]), spreads[0]
    )
    a_hole_2 = rotation_intervals_for_hole_and_slot_family(
        hole_angles[1], (starts[0], starts[2]), spreads[0]
    )
    b_hole_1 = rotation_intervals_for_hole_and_slot_family(
        hole_angles[0], (starts[1], starts[3]), spreads[1]
    )
    b_hole_2 = rotation_intervals_for_hole_and_slot_family(
        hole_angles[1], (starts[1], starts[3]), spreads[1]
    )

    valid_intervals = merge_intervals(
        [
            *a_hole_1,
            *a_hole_2,
            *b_hole_1,
            *b_hole_2,
        ]
    )
    total_coverage = total_interval_length(valid_intervals)

    return SlotMetrics(
        start_angles_deg=starts,
        spreads_deg=spreads,
        b_start_offset_deg=slot_input.b_start_offset_deg,
        hole_angles_deg=hole_angles,
        a_hole_1_intervals_deg=tuple(a_hole_1),
        a_hole_2_intervals_deg=tuple(a_hole_2),
        b_hole_1_intervals_deg=tuple(b_hole_1),
        b_hole_2_intervals_deg=tuple(b_hole_2),
        valid_rotation_intervals_deg=tuple(valid_intervals),
        total_coverage_deg=total_coverage,
        coverage_percent=100.0 * total_coverage / FULL_ROTATION_DEG,
    )


def _format_intervals(intervals: Sequence[Tuple[float, float]]) -> str:
    if not intervals:
        return "none"
    return ", ".join(f"[{start:.3f}, {end:.3f}]" for start, end in intervals)


def _summary_lines(metrics: SlotMetrics) -> List[str]:
    lines = [
        "Circular oblong fixture summary",
        f"Spread A / B:            {metrics.spreads_deg[0]:.3f} / {metrics.spreads_deg[1]:.3f} deg",
        f"B start offset:          {metrics.b_start_offset_deg:.3f} deg from nominal",
        f"Hole 1 / 2 angles:       {metrics.hole_angles_deg[0]:.3f} / {metrics.hole_angles_deg[1]:.3f} deg",
        f"A / Hole 1:             {_format_intervals(metrics.a_hole_1_intervals_deg)}",
        f"A / Hole 2:             {_format_intervals(metrics.a_hole_2_intervals_deg)}",
        f"B / Hole 1:             {_format_intervals(metrics.b_hole_1_intervals_deg)}",
        f"B / Hole 2:             {_format_intervals(metrics.b_hole_2_intervals_deg)}",
        f"Union:                  {_format_intervals(metrics.valid_rotation_intervals_deg)}",
        f"Total coverage:          {metrics.total_coverage_deg:.3f} deg",
        f"Coverage ratio:          {metrics.coverage_percent:.2f} %",
    ]
    return lines


def print_summary(metrics: SlotMetrics) -> None:
    for line in _summary_lines(metrics):
        print(line)


def _draw_slot_arc(
    ax: plt.Axes,
    radius: float,
    start_deg: float,
    spread_deg: float,
    color: str,
    linewidth: float,
) -> None:
    if spread_deg <= 0.0:
        return
    for seg_start, seg_end in normalize_interval(start_deg, spread_deg):
        ax.add_patch(
            patches.Arc(
                (0.0, 0.0),
                2.0 * radius,
                2.0 * radius,
                angle=0.0,
                theta1=seg_start,
                theta2=seg_end,
                lw=linewidth,
                color=color,
            )
        )


def _draw_geometry(ax_geom: plt.Axes, ax_cov: plt.Axes, metrics: SlotMetrics) -> None:
    ax_geom.clear()
    ax_cov.clear()

    slot_radius = 1.0
    hole_radius = 0.88

    ax_geom.add_patch(plt.Circle((0.0, 0.0), 1.12, fill=False, linestyle="--", color="0.75"))
    ax_geom.add_patch(plt.Circle((0.0, 0.0), slot_radius, fill=False, linestyle=":", color="0.7"))
    ax_geom.plot(0.0, 0.0, "ko", markersize=4)

    colors = ("tab:blue", "tab:orange", "tab:blue", "tab:orange")
    for start_deg, spread_deg, color in zip(
        metrics.start_angles_deg,
        metrics.spreads_deg,
        colors,
    ):
        _draw_slot_arc(ax_geom, slot_radius, start_deg, spread_deg, color, linewidth=3.0)
        end_angle = math.radians(start_deg + spread_deg)
        ax_geom.text(
            1.18 * math.cos(end_angle),
            1.18 * math.sin(end_angle),
            f"{spread_deg:.1f}",
            ha="center",
            va="center",
            fontsize=9,
            color=color,
        )

    hole_colors = ("tab:red", "tab:green")
    for idx, hole_angle_deg in enumerate(metrics.hole_angles_deg, start=1):
        theta = math.radians(hole_angle_deg)
        x = hole_radius * math.cos(theta)
        y = hole_radius * math.sin(theta)
        ax_geom.plot([0.0, x], [0.0, y], linestyle="--", color=hole_colors[idx - 1], alpha=0.6)
        ax_geom.plot(x, y, "o", color=hole_colors[idx - 1], markersize=8)
        ax_geom.text(
            1.05 * x,
            1.05 * y,
            f"H{idx} {hole_angle_deg:.1f}",
            color=hole_colors[idx - 1],
            fontsize=9,
            ha="center",
            va="center",
        )

    ax_geom.set_aspect("equal", adjustable="box")
    ax_geom.set_title("Reference Layout")
    ax_geom.set_xlabel("X")
    ax_geom.set_ylabel("Y")
    ax_geom.set_xlim(-1.35, 1.35)
    ax_geom.set_ylim(-1.35, 1.35)
    ax_geom.grid(True, alpha=0.25)

    coverage_rows = [
        ("A / Hole 1", metrics.a_hole_1_intervals_deg, "tab:blue"),
        ("A / Hole 2", metrics.a_hole_2_intervals_deg, "tab:cyan"),
        ("B / Hole 1", metrics.b_hole_1_intervals_deg, "tab:orange"),
        ("B / Hole 2", metrics.b_hole_2_intervals_deg, "tab:brown"),
        ("Union", metrics.valid_rotation_intervals_deg, "tab:green"),
    ]

    for label, intervals, color in coverage_rows:
        for start_deg, end_deg in intervals:
            ax_cov.barh(
                y=label,
                width=end_deg - start_deg,
                left=start_deg,
                height=0.65,
                color=color,
                alpha=0.80,
            )

    ax_cov.set_title("Valid Fixture Rotation")
    ax_cov.set_xlabel("Fixture rotation from reference [deg]")
    ax_cov.set_xlim(0.0, FULL_ROTATION_DEG)
    ax_cov.grid(True, axis="x", alpha=0.25)


def plot_geometry(slot_input: SlotInput) -> None:
    fig, (ax_geom, ax_cov) = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle("Circular Oblong Hole-Based Coverage")
    plt.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.32, wspace=0.22)

    initial_metrics = calculate_metrics(slot_input)
    summary_text = fig.text(
        0.07,
        0.235,
        "\n".join(_summary_lines(initial_metrics)),
        va="top",
        family="monospace",
        fontsize=9,
    )

    spread_a_ax = fig.add_axes([0.07, 0.15, 0.34, 0.03])
    spread_b_ax = fig.add_axes([0.07, 0.10, 0.34, 0.03])
    b_offset_ax = fig.add_axes([0.07, 0.05, 0.34, 0.03])
    hole_1_ax = fig.add_axes([0.52, 0.15, 0.34, 0.03])
    hole_2_ax = fig.add_axes([0.52, 0.10, 0.34, 0.03])
    reset_ax = fig.add_axes([0.88, 0.10, 0.08, 0.08])

    spread_a_slider = Slider(
        spread_a_ax,
        "Spread A",
        0.0,
        FULL_ROTATION_DEG,
        valinit=slot_input.spread_a_deg,
        valstep=0.1,
    )
    spread_b_slider = Slider(
        spread_b_ax,
        "Spread B",
        0.0,
        FULL_ROTATION_DEG,
        valinit=slot_input.spread_b_deg,
        valstep=0.1,
    )
    b_offset_slider = Slider(
        b_offset_ax,
        "B offset",
        -90.0,
        90.0,
        valinit=slot_input.b_start_offset_deg,
        valstep=0.1,
    )
    hole_1_slider = Slider(
        hole_1_ax,
        "Hole 1",
        0.0,
        FULL_ROTATION_DEG,
        valinit=slot_input.hole_1_angle_deg,
        valstep=0.1,
    )
    hole_2_slider = Slider(
        hole_2_ax,
        "Hole 2",
        0.0,
        FULL_ROTATION_DEG,
        valinit=slot_input.hole_2_angle_deg,
        valstep=0.1,
    )
    reset_button = Button(reset_ax, "Reset")

    def build_current_input() -> SlotInput:
        return SlotInput(
            spread_a_deg=float(spread_a_slider.val),
            spread_b_deg=float(spread_b_slider.val),
            b_start_offset_deg=float(b_offset_slider.val),
            hole_1_angle_deg=float(hole_1_slider.val),
            hole_2_angle_deg=float(hole_2_slider.val),
            plot=True,
        )

    def redraw() -> None:
        metrics = calculate_metrics(build_current_input())
        _draw_geometry(ax_geom, ax_cov, metrics)
        summary_text.set_text("\n".join(_summary_lines(metrics)))
        fig.canvas.draw_idle()

    def on_slider_change(_value: float) -> None:
        redraw()

    def on_reset(_event: object) -> None:
        spread_a_slider.reset()
        spread_b_slider.reset()
        b_offset_slider.reset()
        hole_1_slider.reset()
        hole_2_slider.reset()

    spread_a_slider.on_changed(on_slider_change)
    spread_b_slider.on_changed(on_slider_change)
    b_offset_slider.on_changed(on_slider_change)
    hole_1_slider.on_changed(on_slider_change)
    hole_2_slider.on_changed(on_slider_change)
    reset_button.on_clicked(on_reset)

    _draw_geometry(ax_geom, ax_cov, initial_metrics)
    plt.show()


def parse_args() -> SlotInput:
    parser = argparse.ArgumentParser(
        description="Calculate valid fixture rotation from slot spreads and drilled-hole angles."
    )
    parser.add_argument(
        "--spread-a",
        type=float,
        default=75.0,
        help="Slot spread for starts at 0 and 180 deg.",
    )
    parser.add_argument(
        "--spread-b",
        type=float,
        default=75.0,
        help="Slot spread for starts at 90 and 270 deg.",
    )
    parser.add_argument(
        "--b-start-offset",
        type=float,
        default=-10.0,
        help="Offset applied to the B family starts relative to nominal 90 and 270 deg.",
    )
    parser.add_argument(
        "--hole-1-angle",
        type=float,
        default=0.0,
        help="Reference angle of drilled hole 1.",
    )
    parser.add_argument(
        "--hole-2-angle",
        type=float,
        default=180.0,
        help="Reference angle of drilled hole 2.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip the matplotlib visualization.")
    args = parser.parse_args()

    return SlotInput(
        spread_a_deg=args.spread_a,
        spread_b_deg=args.spread_b,
        b_start_offset_deg=args.b_start_offset,
        hole_1_angle_deg=args.hole_1_angle,
        hole_2_angle_deg=args.hole_2_angle,
        plot=not args.no_plot,
    )


def main() -> None:
    slot_input = parse_args()
    metrics = calculate_metrics(slot_input)
    print_summary(metrics)
    if slot_input.plot:
        plot_geometry(slot_input)


if __name__ == "__main__":
    main()
