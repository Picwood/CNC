#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

double clamp_value(double value, double low, double high) {
    return std::max(low, std::min(value, high));
}

py::array_t<double> generate_toolpath(
    double radius,
    double amplitude,
    double amplitude_attenuation,
    int angular_frequency,
    double phase_shift,
    double pass_phase_delta,
    double tool_diameter,
    double depth_offset,
    double z_scale,
    double path_point_spacing,
    py::array_t<double, py::array::c_style | py::array::forcecast> offsets_arr) {
    if (angular_frequency < 1) {
        throw std::runtime_error("angular_frequency must be >= 1");
    }

    const double tool_radius = std::max(0.05, 0.5 * tool_diameter);
    const double cut_z = -std::abs(depth_offset) * std::max(0.01, z_scale);
    const double spacing = std::max(0.05, path_point_spacing);
    const auto offsets = offsets_arr.unchecked<1>();
    if (offsets.shape(0) < 1) {
        throw std::runtime_error("No radial offsets provided.");
    }

    std::vector<double> out_xyz;
    out_xyz.reserve(30000);

    for (py::ssize_t oi = 0; oi < offsets.shape(0); ++oi) {
        const double offset = offsets(oi);
        const double phase_local = phase_shift + static_cast<double>(oi) * pass_phase_delta;
        double amp_eff = amplitude;
        if (amplitude_attenuation > 0.0 && std::abs(radius) > 1e-9) {
            double ratio = (radius + offset) / radius;
            ratio = clamp_value(ratio, 0.0, 1.0);
            amp_eff = amplitude * std::pow(ratio, amplitude_attenuation);
        }

        std::vector<double> theta_values;
        theta_values.reserve(4096);
        theta_values.push_back(0.0);

        double theta = 0.0;
        while (theta < kTwoPi) {
            const double phase = static_cast<double>(angular_frequency) * theta + phase_local;
            const double r = radius + offset + amp_eff * std::sin(phase);
            const double dr_dtheta =
                amp_eff * static_cast<double>(angular_frequency) * std::cos(phase);
            const double ds_dtheta = std::sqrt(r * r + dr_dtheta * dr_dtheta);
            const double dtheta =
                clamp_value(spacing / std::max(ds_dtheta, 1e-8), 0.0005, 0.2);
            theta = std::min(theta + dtheta, kTwoPi);
            theta_values.push_back(theta);
        }
        if (theta_values.back() < kTwoPi) {
            theta_values.push_back(kTwoPi);
        }

        double min_r = std::numeric_limits<double>::infinity();
        double first_x = 0.0;
        double first_y = 0.0;
        double last_x = 0.0;
        double last_y = 0.0;

        for (std::size_t ti = 0; ti < theta_values.size(); ++ti) {
            const double t = theta_values[ti];
            const double r = radius + offset +
                             amp_eff *
                                 std::sin(static_cast<double>(angular_frequency) * t + phase_local);
            min_r = std::min(min_r, r);

            const double x = r * std::cos(t);
            const double y = r * std::sin(t);
            if (ti == 0) {
                first_x = x;
                first_y = y;
            }
            last_x = x;
            last_y = y;
            out_xyz.push_back(x);
            out_xyz.push_back(y);
            out_xyz.push_back(cut_z);
        }

        if (min_r <= tool_radius) {
            throw std::runtime_error(
                "Contour collapses near center. Increase Radius, reduce Amplitude, "
                "or reduce tool diameter.");
        }

        const bool already_closed =
            std::abs(last_x - first_x) < 1e-9 && std::abs(last_y - first_y) < 1e-9;
        if (!already_closed) {
            out_xyz.push_back(first_x);
            out_xyz.push_back(first_y);
            out_xyz.push_back(cut_z);
        }

        if (oi + 1 < offsets.shape(0)) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            out_xyz.push_back(nan);
            out_xyz.push_back(nan);
            out_xyz.push_back(nan);
        }
    }

    const std::size_t rows = out_xyz.size() / 3;
    if (rows > 220000) {
        throw std::runtime_error(
            "Toolpath too dense (>220k points). Increase point spacing or step-over.");
    }

    py::array_t<double> result({rows, static_cast<std::size_t>(3)});
    auto result_mut = result.mutable_unchecked<2>();
    for (std::size_t i = 0; i < rows; ++i) {
        result_mut(i, 0) = out_xyz[3 * i];
        result_mut(i, 1) = out_xyz[3 * i + 1];
        result_mut(i, 2) = out_xyz[3 * i + 2];
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(cnc_toolpath_accel, m) {
    m.doc() = "Accelerated toolpath generation for planar sinusoidal polar engraving";
    m.def(
        "generate_toolpath",
        &generate_toolpath,
        py::arg("radius"),
        py::arg("amplitude"),
        py::arg("amplitude_attenuation"),
        py::arg("angular_frequency"),
        py::arg("phase_shift"),
        py::arg("pass_phase_delta"),
        py::arg("tool_diameter"),
        py::arg("depth_offset"),
        py::arg("z_scale"),
        py::arg("path_point_spacing"),
        py::arg("offsets"));
}
