"""
Flat Polar Engraving Generator

Interactive application that:
1. Builds a planar periodic polar contour: r(theta) = R + A * sin(n * theta + k)
2. Displays flat stock and contour toolpath in 3D
3. Exports industrial-style 3-axis G-code and DXF/DWG contour geometry
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
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

_ACTIVE_BACKEND = "Python"


@dataclass
class CNCParameters:
    radius: float = 35.0
    amplitude: float = 6.0
    amplitude_attenuation: float = 0.0
    angular_frequency: int = 6
    phase_shift: float = 0.0
    pass_phase_delta: float = 0.0
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
    theta: np.ndarray | float,
    params: CNCParameters,
    radial_offset: float = 0.0,
    phase_shift_override: float | None = None,
) -> np.ndarray | float:
    """Periodic sinusoidal radius function in polar coordinates."""
    amplitude_eff = _effective_amplitude(params, radial_offset)
    phase = params.phase_shift if phase_shift_override is None else phase_shift_override
    return (
        params.radius
        + radial_offset
        + amplitude_eff * np.sin(params.angular_frequency * theta + phase)
    )


def _effective_amplitude(params: CNCParameters, radial_offset: float) -> float:
    """
    Reduce lobe amplitude on inner passes to avoid tight/overlapping center regions.

    attenuation=0 keeps full amplitude.
    attenuation>0 scales amplitude by ((radius+offset)/radius)^attenuation, clamped to [0,1].
    """
    attenuation = max(0.0, float(params.amplitude_attenuation))
    if attenuation <= 0.0 or params.radius <= 1e-9:
        return float(params.amplitude)

    ratio = (params.radius + radial_offset) / params.radius
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return float(params.amplitude) * (ratio**attenuation)


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


def _build_theta_samples(
    params: CNCParameters, radial_offset: float, phase_shift: float
) -> np.ndarray:
    """Adaptive theta sampling for near-constant XY point spacing."""
    spacing = max(0.05, params.path_point_spacing)
    theta_end = 2.0 * np.pi
    amplitude_eff = _effective_amplitude(params, radial_offset)

    theta_values = [0.0]
    theta = 0.0
    while theta < theta_end:
        r = float(
            polar_radius(
                theta, params, radial_offset, phase_shift_override=float(phase_shift)
            )
        )
        dr_dtheta = (
            amplitude_eff
            * params.angular_frequency
            * np.cos(params.angular_frequency * theta + phase_shift)
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
    for pass_idx, offset in enumerate(offsets):
        phase_local = params.phase_shift + pass_idx * params.pass_phase_delta
        theta = _build_theta_samples(params, float(offset), float(phase_local))
        r = polar_radius(
            theta, params, float(offset), phase_shift_override=float(phase_local)
        )
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
    return _ACTIVE_BACKEND


def generate_toolpath(params: CNCParameters) -> np.ndarray:
    """
    Generate constant-depth engraving contour(s) from periodic polar radius.

    Returns:
        Nx3 array with NaN separator rows between contour segments.
    """
    global _ACTIVE_BACKEND, _accel
    offsets = _build_radial_offsets(params)

    if _accel is None:
        _ACTIVE_BACKEND = "Python"
        return _generate_toolpath_python(params, offsets=offsets)

    try:
        path = _accel.generate_toolpath(
            float(params.radius),
            float(params.amplitude),
            float(params.amplitude_attenuation),
            int(params.angular_frequency),
            float(params.phase_shift),
            float(params.pass_phase_delta),
            float(params.tool_diameter),
            float(params.depth_offset),
            float(params.z_scale),
            float(params.path_point_spacing),
            np.asarray(offsets, dtype=np.float64),
        )
        path = np.asarray(path, dtype=np.float64)
        _ACTIVE_BACKEND = "C++"
    except Exception:
        # Disable acceleration after a runtime/signature mismatch to avoid repeated exceptions.
        _accel = None
        path = _generate_toolpath_python(params, offsets=offsets)
        _ACTIVE_BACKEND = "Python"

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


def _export_dxf_r12_polyline(file_path: str | Path, toolpath: np.ndarray) -> None:
    """Export contour segments to robust R12 ASCII DXF (POLYLINE/VERTEX)."""
    segments = _split_toolpath_segments(toolpath)
    if not segments:
        raise ValueError("Toolpath must contain at least one valid segment.")

    path = Path(file_path)
    if path.suffix.lower() != ".dxf":
        path = path.with_suffix(".dxf")

    lines = [
        "0",
        "SECTION",
        "2",
        "HEADER",
        "9",
        "$ACADVER",
        "1",
        "AC1009",
        "0",
        "ENDSEC",
        "0",
        "SECTION",
        "2",
        "TABLES",
        "0",
        "TABLE",
        "2",
        "LTYPE",
        "70",
        "1",
        "0",
        "LTYPE",
        "2",
        "CONTINUOUS",
        "70",
        "64",
        "3",
        "Solid line",
        "72",
        "65",
        "73",
        "0",
        "40",
        "0.0",
        "0",
        "ENDTAB",
        "0",
        "TABLE",
        "2",
        "LAYER",
        "70",
        "1",
        "0",
        "LAYER",
        "2",
        "0",
        "70",
        "0",
        "62",
        "7",
        "6",
        "CONTINUOUS",
        "0",
        "ENDTAB",
        "0",
        "ENDSEC",
        "0",
        "SECTION",
        "2",
        "ENTITIES",
    ]

    for seg in segments:
        xy = seg[:, :2]
        closed = bool(np.allclose(xy[0], xy[-1]))
        if closed:
            xy = xy[:-1]

        lines.extend(
            [
                "0",
                "POLYLINE",
                "8",
                "0",
                "66",
                "1",
                "70",
                "1" if closed else "0",
                "10",
                "0.0",
                "20",
                "0.0",
                "30",
                "0.0",
            ]
        )

        for x, y in xy:
            lines.extend(
                [
                    "0",
                    "VERTEX",
                    "8",
                    "0",
                    "10",
                    f"{float(x):.6f}",
                    "20",
                    f"{float(y):.6f}",
                    "30",
                    "0.0",
                ]
            )
        lines.extend(["0", "SEQEND"])

    lines.extend(["0", "ENDSEC", "0", "EOF"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _export_dxf_spline(file_path: str | Path, toolpath: np.ndarray) -> None:
    """Export contour segments to ASCII DXF using SPLINE entities."""
    segments = _split_toolpath_segments(toolpath)
    if not segments:
        raise ValueError("Toolpath must contain at least one valid segment.")

    path = Path(file_path)
    if path.suffix.lower() != ".dxf":
        path = path.with_suffix(".dxf")

    lines = [
        "0",
        "SECTION",
        "2",
        "HEADER",
        "9",
        "$ACADVER",
        "1",
        "AC1015",
        "9",
        "$INSUNITS",
        "70",
        "4",
        "0",
        "ENDSEC",
        "0",
        "SECTION",
        "2",
        "TABLES",
        "0",
        "TABLE",
        "2",
        "LTYPE",
        "70",
        "1",
        "0",
        "LTYPE",
        "2",
        "CONTINUOUS",
        "70",
        "64",
        "3",
        "Solid line",
        "72",
        "65",
        "73",
        "0",
        "40",
        "0.0",
        "0",
        "ENDTAB",
        "0",
        "TABLE",
        "2",
        "LAYER",
        "70",
        "1",
        "0",
        "LAYER",
        "2",
        "0",
        "70",
        "0",
        "62",
        "7",
        "6",
        "CONTINUOUS",
        "0",
        "ENDTAB",
        "0",
        "ENDSEC",
        "0",
        "SECTION",
        "2",
        "ENTITIES",
    ]

    for seg in segments:
        xy = seg[:, :2]
        closed = bool(np.allclose(xy[0], xy[-1]))
        if closed:
            xy = xy[:-1]

        if xy.shape[0] < 4:
            continue

        flags = 8 + (1 if closed else 0)
        lines.extend(
            [
                "0",
                "SPLINE",
                "8",
                "0",
                "100",
                "AcDbEntity",
                "100",
                "AcDbSpline",
                "70",
                str(flags),
                "71",
                "3",
                "72",
                "0",
                "73",
                "0",
                "74",
                str(xy.shape[0]),
                "42",
                "0.0000001",
                "43",
                "0.0000001",
                "44",
                "0.0000001",
                "210",
                "0.0",
                "220",
                "0.0",
                "230",
                "1.0",
            ]
        )
        for x, y in xy:
            lines.extend(["11", f"{float(x):.6f}", "21", f"{float(y):.6f}", "31", "0.0"])

    lines.extend(["0", "ENDSEC", "0", "EOF"])
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def export_dxf(file_path: str | Path, toolpath: np.ndarray, flavor: str = "catia") -> None:
    """
    Export contour segments to DXF.

    `flavor="catia"`: R12 POLYLINE/VERTEX for broad CATIA compatibility.
    `flavor="spline"`: SPLINE entities (may not import in CATIA).
    """
    flavor_norm = flavor.strip().lower()
    if flavor_norm == "spline":
        _export_dxf_spline(file_path, toolpath)
        return
    _export_dxf_r12_polyline(file_path, toolpath)


def export_catia_parameters(file_path: str | Path, params: CNCParameters) -> Path:
    """Export CATIA macro parameters as a key=value text file."""
    path = Path(file_path)
    if path.suffix.lower() != ".txt":
        path = path.with_suffix(".txt")

    lines = [
        "# CATIA spline sketch import parameters (units: mm)",
        f"radius={params.radius:.9f}",
        f"amplitude={params.amplitude:.9f}",
        f"amplitude_attenuation={params.amplitude_attenuation:.9f}",
        f"angular_frequency={int(params.angular_frequency)}",
        f"phase_shift={params.phase_shift:.12f}",
        f"pass_phase_delta={params.pass_phase_delta:.12f}",
        f"step_over={params.step_over:.9f}",
        f"pass_count={int(params.pass_count)}",
        f"path_point_spacing={params.path_point_spacing:.9f}",
        "sketch_name=PolarSpline",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _build_catia_macro_text(default_param_path: Path) -> str:
    """Create CATScript content that reads parameters and builds sketch splines."""
    param_path = str(default_param_path.resolve()).replace('"', '""')
    return f"""Option Explicit

Sub CATMain()
    Dim defaultParamFile
    defaultParamFile = "{param_path}"

    Dim paramFile
    paramFile = defaultParamFile
    If Not FileExists(paramFile) Then
        paramFile = InputBox("Parameter file not found. Enter full path:", "CATIA Polar Spline Import", defaultParamFile)
        If Len(Trim(paramFile)) = 0 Then
            MsgBox "No parameter file provided."
            Exit Sub
        End If
        If Not FileExists(paramFile) Then
            MsgBox "Parameter file still not found: " & paramFile
            Exit Sub
        End If
    End If

    Dim params
    Set params = LoadParams(paramFile)
    If params Is Nothing Then
        MsgBox "Unable to load parameters from: " & paramFile
        Exit Sub
    End If

    If CATIA.Documents.Count = 0 Then
        MsgBox "Open a CATPart document before running this macro."
        Exit Sub
    End If

    Dim partDoc
    Set partDoc = CATIA.ActiveDocument
    If InStr(1, partDoc.Name, ".CATPart", 1) = 0 Then
        MsgBox "Active document is not a CATPart."
        Exit Sub
    End If

    Dim part
    Set part = partDoc.Part
    Dim body
    Set body = part.MainBody

    Dim originElements
    Set originElements = part.OriginElements
    Dim refPlane
    Set refPlane = originElements.PlaneXY

    Dim sketches
    Set sketches = body.Sketches
    Dim sketch
    Set sketch = sketches.Add(refPlane)
    sketch.Name = GetParamString(params, "sketch_name", "PolarSpline")

    Dim factory2D
    Set factory2D = sketch.OpenEdition

    Dim radius, amplitude, ampAtten, lobes, phaseShift, passPhaseDelta, stepOver, passCount, pointSpacing
    radius = GetParamDouble(params, "radius", 35.0)
    amplitude = GetParamDouble(params, "amplitude", 6.0)
    ampAtten = GetParamDouble(params, "amplitude_attenuation", 0.0)
    lobes = CLng(GetParamDouble(params, "angular_frequency", 6.0))
    phaseShift = GetParamDouble(params, "phase_shift", 0.0)
    passPhaseDelta = GetParamDouble(params, "pass_phase_delta", 0.0)
    stepOver = GetParamDouble(params, "step_over", 0.4)
    passCount = CLng(GetParamDouble(params, "pass_count", 1.0))
    pointSpacing = GetParamDouble(params, "path_point_spacing", 0.35)

    If passCount < 1 Then passCount = 1
    If pointSpacing < 0.01 Then pointSpacing = 0.01
    If lobes < 1 Then lobes = 1

    Dim pi
    pi = 3.14159265358979

    Dim i, j
    For i = 0 To passCount - 1
        Dim offset
        offset = (CDbl(i) - 0.5 * CDbl(passCount - 1)) * stepOver

        Dim ampEff
        ampEff = EffectiveAmplitude(radius, amplitude, offset, ampAtten)
        Dim phaseForPass
        phaseForPass = phaseShift + passPhaseDelta * CDbl(i)

        Dim approxRadius
        approxRadius = radius + Abs(ampEff) + Abs(offset)
        If approxRadius < 1.0 Then approxRadius = 1.0

        Dim nPts
        nPts = CLng((2 * pi * approxRadius) / pointSpacing)
        If nPts < 80 Then nPts = 80
        If nPts > 2500 Then nPts = 2500

        Dim poles
        poles = BuildPoleArray(factory2D, radius, ampEff, lobes, phaseForPass, offset, nPts, pi)

        Dim spline
        Set spline = factory2D.CreateSpline(poles)
    Next

    sketch.CloseEdition
    part.InWorkObject = sketch
    part.Update

    MsgBox "CATIA splines created from: " & paramFile
End Sub

Function FileExists(path)
    Dim fso
    Set fso = CreateObject("Scripting.FileSystemObject")
    FileExists = fso.FileExists(path)
End Function

Function LoadParams(path)
    On Error Resume Next
    Dim fso, ts
    Set fso = CreateObject("Scripting.FileSystemObject")
    Set ts = fso.OpenTextFile(path, 1, False)
    If Err.Number <> 0 Then
        Set LoadParams = Nothing
        Exit Function
    End If

    Dim dict
    Set dict = CreateObject("Scripting.Dictionary")

    Do Until ts.AtEndOfStream
        Dim line, eqPos, key, value
        line = Trim(ts.ReadLine)
        If Len(line) > 0 Then
            If Left(line, 1) <> "#" Then
                eqPos = InStr(line, "=")
                If eqPos > 1 Then
                    key = LCase(Trim(Left(line, eqPos - 1)))
                    value = Trim(Mid(line, eqPos + 1))
                    If dict.Exists(key) Then
                        dict.Item(key) = value
                    Else
                        dict.Add key, value
                    End If
                End If
            End If
        End If
    Loop
    ts.Close
    Set LoadParams = dict
End Function

Function GetParamDouble(dict, key, defaultVal)
    Dim k
    k = LCase(key)
    If dict.Exists(k) Then
        On Error Resume Next
        GetParamDouble = CDbl(dict.Item(k))
        If Err.Number <> 0 Then
            Err.Clear
            GetParamDouble = defaultVal
        End If
        On Error GoTo 0
    Else
        GetParamDouble = defaultVal
    End If
End Function

Function GetParamString(dict, key, defaultVal)
    Dim k
    k = LCase(key)
    If dict.Exists(k) Then
        GetParamString = CStr(dict.Item(k))
    Else
        GetParamString = defaultVal
    End If
End Function

Function EffectiveAmplitude(radius, amplitude, offset, attenuation)
    If attenuation <= 0 Or Abs(radius) < 0.000000001 Then
        EffectiveAmplitude = amplitude
        Exit Function
    End If

    Dim ratio
    ratio = (radius + offset) / radius
    If ratio < 0 Then ratio = 0
    If ratio > 1 Then ratio = 1
    EffectiveAmplitude = amplitude * (ratio ^ attenuation)
End Function

Function BuildPoleArray(factory2D, radius, amplitudeEff, lobes, phaseForPass, offset, nPts, pi)
    Dim poles()
    ReDim poles(nPts)

    Dim j
    For j = 0 To nPts
        Dim theta, r, x, y
        theta = (2 * pi * CDbl(j)) / CDbl(nPts)
        r = radius + offset + amplitudeEff * Sin(CDbl(lobes) * theta + phaseForPass)
        x = r * Cos(theta)
        y = r * Sin(theta)
        Set poles(j) = factory2D.CreatePoint(x, y)
    Next

    BuildPoleArray = poles
End Function
"""


def export_catia_macro(file_path: str | Path, param_file_path: str | Path) -> Path:
    """Export CATIA CATScript macro that draws splines in a sketch."""
    path = Path(file_path)
    if path.suffix.lower() != ".catscript":
        path = path.with_suffix(".CATScript")
    content = _build_catia_macro_text(Path(param_file_path))
    path.write_text(content, encoding="ascii")
    return path


def export_catia_bundle(base_path: str | Path, params: CNCParameters) -> Tuple[Path, Path]:
    """Export both CATIA parameter file and CATScript macro."""
    base = Path(base_path)
    stem = base.stem if base.suffix else base.name
    folder = base.parent if str(base.parent) not in ("", ".") else Path(".")
    param_path = folder / f"{stem}_catia_params.txt"
    macro_path = folder / f"{stem}_catia_import.CATScript"

    exported_param = export_catia_parameters(param_path, params)
    exported_macro = export_catia_macro(macro_path, exported_param)
    return exported_param, exported_macro


def _find_oda_converter() -> Path | None:
    """Locate ODA File Converter executable for DXF->DWG conversion."""
    env_path = os.environ.get("ODA_FILE_CONVERTER")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    candidates = [
        Path(r"C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe"),
        Path(r"C:\Program Files\ODA\ODAFileConverter 25.12.0\ODAFileConverter.exe"),
        Path(r"C:\Program Files\ODA\ODAFileConverter 25.11.0\ODAFileConverter.exe"),
    ]
    for c in candidates:
        if c.exists():
            return c

    oda_root = Path(r"C:\Program Files\ODA")
    if oda_root.exists():
        matches = sorted(oda_root.rglob("ODAFileConverter.exe"))
        if matches:
            return matches[-1]
    return None


def export_dwg(file_path: str | Path, toolpath: np.ndarray) -> None:
    """
    Export DWG by generating DXF first, then converting with ODA File Converter.

    Requires ODA File Converter installed locally, or env var:
    ODA_FILE_CONVERTER=C:\\path\\to\\ODAFileConverter.exe
    """
    converter = _find_oda_converter()
    if converter is None:
        raise RuntimeError(
            "ODA File Converter not found. Install it or set ODA_FILE_CONVERTER."
        )

    target = Path(file_path)
    if target.suffix.lower() != ".dwg":
        target = target.with_suffix(".dwg")

    with tempfile.TemporaryDirectory(prefix="cnc_dwg_") as tmp:
        in_dir = Path(tmp) / "in"
        out_dir = Path(tmp) / "out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        dxf_name = f"{target.stem}.dxf"
        temp_dxf = in_dir / dxf_name
        export_dxf(temp_dxf, toolpath)

        cmd = [
            str(converter),
            str(in_dir),
            str(out_dir),
            "ACAD2018",
            "DWG",
            "0",
            "1",
            dxf_name,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"DWG conversion failed (code {proc.returncode}). "
                f"{proc.stdout.strip()} {proc.stderr.strip()}".strip()
            )

        produced = out_dir / target.name
        if not produced.exists():
            found = list(out_dir.rglob(target.name))
            if not found:
                raise RuntimeError("DWG conversion completed but output file was not found.")
            produced = found[0]
        shutil.copyfile(produced, target)


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

    slider_y = np.linspace(0.89, 0.385, 15)
    sliders = {
        "radius": _build_slider(
            fig, (0.75, slider_y[0], 0.22, 0.022), "Mean Radius R (mm)", 5.0, 180.0, params.radius
        ),
        "amplitude": _build_slider(
            fig, (0.75, slider_y[1], 0.22, 0.022), "Radial Amp A (mm)", 0.0, 50.0, params.amplitude
        ),
        "amplitude_attenuation": _build_slider(
            fig, (0.75, slider_y[2], 0.22, 0.022), "Amp Atten", 0.0, 4.0, params.amplitude_attenuation
        ),
        "angular_frequency": _build_slider(
            fig, (0.75, slider_y[3], 0.22, 0.022), "Lobes n", 1, 32, params.angular_frequency, 1
        ),
        "phase_shift": _build_slider(
            fig, (0.75, slider_y[4], 0.22, 0.022), "Phase k (rad)", 0.0, 2.0 * np.pi, params.phase_shift
        ),
        "pass_phase_delta": _build_slider(
            fig, (0.75, slider_y[5], 0.22, 0.022), "Phase/Pass (rad)", -2.0 * np.pi, 2.0 * np.pi, params.pass_phase_delta
        ),
        "z_scale": _build_slider(
            fig, (0.75, slider_y[6], 0.22, 0.022), "Z Scale", 0.1, 4.0, params.z_scale
        ),
        "tool_diameter": _build_slider(
            fig, (0.75, slider_y[7], 0.22, 0.022), "Tool Dia (mm)", 0.2, 20.0, params.tool_diameter
        ),
        "step_over": _build_slider(
            fig, (0.75, slider_y[8], 0.22, 0.022), "Step-over (mm)", 0.02, 5.0, params.step_over
        ),
        "pass_count": _build_slider(
            fig, (0.75, slider_y[9], 0.22, 0.022), "Passes", 1, 50, params.pass_count, 1
        ),
        "feed_rate": _build_slider(
            fig, (0.75, slider_y[10], 0.22, 0.022), "Feed (mm/min)", 50.0, 8000.0, params.feed_rate
        ),
        "spindle_speed": _build_slider(
            fig, (0.75, slider_y[11], 0.22, 0.022), "Spindle (RPM)", 1000, 30000, params.spindle_speed, 100
        ),
        "safe_height": _build_slider(
            fig, (0.75, slider_y[12], 0.22, 0.022), "Safe Z (mm)", 1.0, 60.0, params.safe_height
        ),
        "depth_offset": _build_slider(
            fig, (0.75, slider_y[13], 0.22, 0.022), "Engrave Depth (mm)", 0.02, 12.0, params.depth_offset
        ),
        "path_point_spacing": _build_slider(
            fig, (0.75, slider_y[14], 0.22, 0.022), "Point Spacing (mm)", 0.05, 5.0, params.path_point_spacing
        ),
    }

    mesh_r = _build_slider(
        fig, (0.75, 0.335, 0.22, 0.022), "Mesh Radial", 30, 320, params.mesh_radial_samples, 1
    )
    mesh_t = _build_slider(
        fig, (0.75, 0.305, 0.22, 0.022), "Mesh Angular", 80, 560, params.mesh_angular_samples, 1
    )
    sliders["mesh_radial_samples"] = mesh_r
    sliders["mesh_angular_samples"] = mesh_t

    toggle_ax = fig.add_axes([0.75, 0.245, 0.22, 0.05])
    toggle = CheckButtons(toggle_ax, ["Show Toolpath"], [params.show_toolpath])

    file_box_ax = fig.add_axes([0.75, 0.195, 0.22, 0.04])
    file_box = TextBox(file_box_ax, "Output File", initial=params.output_filename)

    export_ax = fig.add_axes([0.75, 0.14, 0.105, 0.04])
    export_dxf_ax = fig.add_axes([0.865, 0.14, 0.105, 0.04])
    export_dwg_ax = fig.add_axes([0.75, 0.092, 0.105, 0.04])
    export_catia_ax = fig.add_axes([0.865, 0.092, 0.105, 0.04])
    reset_ax = fig.add_axes([0.75, 0.044, 0.22, 0.04])
    export_btn = Button(export_ax, "Export NC")
    export_dxf_btn = Button(export_dxf_ax, "Export DXF")
    export_dwg_btn = Button(export_dwg_ax, "Export DWG")
    export_catia_btn = Button(export_catia_ax, "Export CATIA")
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
    defaults = CNCParameters()
    geometry_slider_keys = {
        "radius",
        "amplitude",
        "amplitude_attenuation",
        "angular_frequency",
        "phase_shift",
        "pass_phase_delta",
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
        p.amplitude_attenuation = float(sliders["amplitude_attenuation"].val)
        p.angular_frequency = int(round(sliders["angular_frequency"].val))
        p.phase_shift = float(sliders["phase_shift"].val)
        p.pass_phase_delta = float(sliders["pass_phase_delta"].val)
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

    def on_export_dxf(_: object) -> None:
        pull_params_from_widgets()
        try:
            toolpath = generate_toolpath(state.params)
            dxf_path = Path(state.params.output_filename)
            if dxf_path.suffix.lower() != ".dxf":
                dxf_path = dxf_path.with_suffix(".dxf")
            export_dxf(dxf_path, toolpath)
            state.status_text.set_text(f"Exported: {dxf_path.resolve()}")
        except Exception as exc:  # pragma: no cover - runtime UI path
            state.status_text.set_text(f"DXF export failed: {exc}")
        state.fig.canvas.draw_idle()

    def on_export_dwg(_: object) -> None:
        pull_params_from_widgets()
        try:
            toolpath = generate_toolpath(state.params)
            dwg_path = Path(state.params.output_filename)
            if dwg_path.suffix.lower() != ".dwg":
                dwg_path = dwg_path.with_suffix(".dwg")
            export_dwg(dwg_path, toolpath)
            state.status_text.set_text(f"Exported: {dwg_path.resolve()}")
        except Exception as exc:  # pragma: no cover - runtime UI path
            state.status_text.set_text(f"DWG export failed: {exc}")
        state.fig.canvas.draw_idle()

    def on_export_catia(_: object) -> None:
        pull_params_from_widgets()
        try:
            param_path, macro_path = export_catia_bundle(
                state.params.output_filename, state.params
            )
            state.status_text.set_text(
                f"Exported: {param_path.resolve()} | {macro_path.resolve()}"
            )
        except Exception as exc:  # pragma: no cover - runtime UI path
            state.status_text.set_text(f"CATIA export failed: {exc}")
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
    export_dxf_btn.on_clicked(on_export_dxf)
    export_dwg_btn.on_clicked(on_export_dwg)
    export_catia_btn.on_clicked(on_export_catia)
    reset_btn.on_clicked(on_reset)

    pull_params_from_widgets()
    state.redraw_pending = True
    perform_redraw()
    plt.show()


if __name__ == "__main__":
    main()
