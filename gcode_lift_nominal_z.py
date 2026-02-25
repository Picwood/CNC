import argparse
import math
import re
from pathlib import Path
from typing import Optional


Z_TOKEN_RE = re.compile(r"([Zz])([+-]?(?:\d+(?:\.\d*)?|\.\d+))")
X_TOKEN_RE = re.compile(r"([Xx])([+-]?(?:\d+(?:\.\d*)?|\.\d+))")
SPINDLE_STOP_RE = re.compile(r"^\s*M0?5\b", re.IGNORECASE)


def split_inline_comment(line: str) -> tuple[str, str]:
    comment_index = line.find(";")
    if comment_index == -1:
        return line, ""
    return line[:comment_index], line[comment_index:]


def format_like_original(original_number: str, new_value: float) -> str:
    original_body = original_number[1:] if original_number[:1] in "+-" else original_number
    force_plus = original_number.startswith("+")
    sign = "-" if new_value < 0 else ("+" if force_plus else "")
    absolute_value = abs(new_value)

    if "." in original_body:
        decimals = len(original_body.split(".", 1)[1])
        formatted_body = f"{absolute_value:.{decimals}f}"
    elif math.isclose(absolute_value, round(absolute_value), abs_tol=1e-12):
        formatted_body = str(int(round(absolute_value)))
    else:
        formatted_body = f"{absolute_value:.6f}".rstrip("0").rstrip(".")

    return f"{sign}{formatted_body}"


def process_line(line: str, target_z: float, lift: float, tolerance: float) -> tuple[str, int]:
    code_part, comment_part = split_inline_comment(line)
    replacements = 0

    def replace_if_target(match: re.Match[str]) -> str:
        nonlocal replacements
        original_number = match.group(2)
        z_value = float(original_number)

        if math.isclose(z_value, target_z, abs_tol=tolerance):
            replacements += 1
            adjusted = z_value + lift
            return f"{match.group(1)}{format_like_original(original_number, adjusted)}"

        return match.group(0)

    updated_code = Z_TOKEN_RE.sub(replace_if_target, code_part)
    return f"{updated_code}{comment_part}", replacements


def extract_last_axis_value(line: str, axis_re: re.Pattern[str]) -> Optional[float]:
    code_part, _ = split_inline_comment(line)
    matches = axis_re.findall(code_part)
    if not matches:
        return None
    return float(matches[-1][1])


def find_large_x_tail_start(
    lines: list[str],
    x_step_threshold: float,
    min_large_x_steps: int,
    max_x_gap_lines: int,
) -> Optional[int]:
    events: list[int] = []
    previous_x: Optional[float] = None

    for idx, line in enumerate(lines):
        x_value = extract_last_axis_value(line, X_TOKEN_RE)
        if x_value is None:
            continue

        if previous_x is not None and abs(x_value - previous_x) >= x_step_threshold:
            events.append(idx)

        previous_x = x_value

    if not events:
        return None

    run_start = 0
    selected_start: Optional[int] = None
    for i in range(1, len(events) + 1):
        run_ended = i == len(events) or (events[i] - events[i - 1]) > max_x_gap_lines
        if run_ended:
            run_length = i - run_start
            if run_length >= min_large_x_steps:
                selected_start = events[run_start]
            run_start = i

    return selected_start


def trim_tail_keep_spindle_stop(lines: list[str], trim_start: int) -> tuple[list[str], bool]:
    removed_tail = lines[trim_start:]
    spindle_stop_line = None
    for line in removed_tail:
        code_part, _ = split_inline_comment(line)
        if SPINDLE_STOP_RE.search(code_part):
            spindle_stop_line = line if line.endswith("\n") else f"{line}\n"
            break

    kept_lines = lines[:trim_start]
    if spindle_stop_line:
        kept_lines.append(spindle_stop_line)

    return kept_lines, spindle_stop_line is not None


def process_file(
    input_path: Path,
    output_path: Path,
    target_z: float,
    lift: float,
    tolerance: float,
    trim_large_x_tail: bool,
    x_step_threshold: float,
    min_large_x_steps: int,
    max_x_gap_lines: int,
) -> tuple[int, Optional[int], bool]:
    changes = 0
    with input_path.open("r", encoding="utf-8") as source:
        lines = source.readlines()

    output_lines = []
    for line in lines:
        updated_line, line_changes = process_line(line, target_z, lift, tolerance)
        output_lines.append(updated_line)
        changes += line_changes

    trim_start = None
    spindle_stop_kept = False
    if trim_large_x_tail:
        trim_start = find_large_x_tail_start(
            output_lines,
            x_step_threshold=x_step_threshold,
            min_large_x_steps=min_large_x_steps,
            max_x_gap_lines=max_x_gap_lines,
        )
        if trim_start is not None:
            output_lines, spindle_stop_kept = trim_tail_keep_spindle_stop(output_lines, trim_start)

    with output_path.open("w", encoding="utf-8") as target:
        target.writelines(output_lines)

    return changes, trim_start, spindle_stop_kept


def build_default_output_path(input_path: Path) -> Path:
    suffix = input_path.suffix or ".nc"
    return input_path.with_name(f"{input_path.stem}_pocketing-only{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lift nominal stock-radius moves by increasing a target Z value in a G-code file."
    )
    parser.add_argument("input", nargs="?", type=Path, default=Path("4th_machine.nc"), help="Input G-code file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output G-code file. Default: <input>_pocketing-only.nc",
    )
    parser.add_argument("--target-z", type=float, default=17.0, help="Z value to detect and lift. Default: 17.0")
    parser.add_argument("--lift", type=float, default=1.0, help="Amount to add to matching Z moves. Default: 1.0")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2e-1,
        help="Absolute tolerance for matching the target Z value. Default: 1e-6",
    )
    parser.add_argument(
        "--trim-large-x-tail",
        action="store_true",
        help=(
            "Trim from the last run of large X jumps and keep only spindle stop command (M05) "
            "from the removed tail."
        ),
    )
    parser.add_argument(
        "--x-step-threshold",
        type=float,
        default=1.0,
        help="Minimum |dX| considered a large X travel. Default: 1.0",
    )
    parser.add_argument(
        "--min-large-x-steps",
        type=int,
        default=3,
        help="Minimum consecutive large X travels required to detect the removable tail. Default: 3",
    )
    parser.add_argument(
        "--max-x-gap-lines",
        type=int,
        default=5,
        help="Maximum line gap between large X travels to consider them consecutive. Default: 5",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or build_default_output_path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    changes, trim_start, spindle_stop_kept = process_file(
        input_path,
        output_path,
        args.target_z,
        args.lift,
        args.tolerance,
        args.trim_large_x_tail,
        args.x_step_threshold,
        args.min_large_x_steps,
        args.max_x_gap_lines,
    )

    summary = f"Wrote '{output_path}' with {changes} adjusted Z move(s) (target {args.target_z:g}, lift +{args.lift:g})."
    if args.trim_large_x_tail:
        if trim_start is not None:
            summary += f" Trimmed tail from line {trim_start + 1}"
            summary += " and kept M05." if spindle_stop_kept else " but found no M05 to keep."
        else:
            summary += " No large-X tail detected; no trim applied."
    print(summary)


if __name__ == "__main__":
    main()
