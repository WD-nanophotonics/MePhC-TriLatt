"""Diagnostic-only periodic-supercell single-band Berry plaquette harness.

The harness deliberately keeps the result in memory. It is a bounded R6.6B
diagnostic, not a production Berry workflow or a persistent record writer.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

import supercell_config
from r5_deformation import finite_patch_preview


def _positive_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real number.")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a finite positive real number.")
    return normalized


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1.")
    return int(value)


def _validate_q_point(q_point):
    values = np.asarray(q_point, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("generic_fractional_supercell must be a finite 2D point.")
    return values


def _validate_band_inputs(target_band_index, num_bands):
    num_bands = _positive_integer(num_bands, "num_bands")
    if (
        isinstance(target_band_index, bool)
        or not isinstance(target_band_index, Integral)
        or target_band_index < 0
        or target_band_index >= num_bands
    ):
        raise ValueError("target_band_index must be a valid Python 0-based band index.")
    target_band_index = int(target_band_index)
    if num_bands < target_band_index + 2:
        raise ValueError("num_bands must include at least one band above the target band.")
    return target_band_index, num_bands


def _field_replication(field):
    matrix = np.asarray(field.supercell.matrix, dtype=int)
    if matrix.shape != (2, 2) or not np.array_equal(matrix, np.diag(np.diag(matrix))):
        raise ValueError("diagnostic requires a verified diagonal periodic-supercell field.")
    replication = tuple(int(value) for value in np.diag(matrix))
    if any(value < 1 for value in replication):
        raise ValueError("field supercell replication must be positive.")
    return replication


def _cartesian_corners(field, q_point, cartesian_step):
    field.require_verified()
    basis = np.asarray(field.reciprocal_basis, dtype=float)
    if basis.shape != (2, 2) or not np.all(np.isfinite(basis)):
        raise ValueError("verified field reciprocal_basis must be finite 2D data.")
    lower_left = q_point @ basis.T
    return [
        lower_left,
        lower_left + np.array([cartesian_step, 0.0]),
        lower_left + np.array([cartesian_step, cartesian_step]),
        lower_left + np.array([0.0, cartesian_step]),
    ]


def _reciprocal_xy(point):
    """Convert a public Meep vector result to serializable x/y coordinates."""
    if hasattr(point, "x") and hasattr(point, "y"):
        values = np.array([float(point.x), float(point.y)], dtype=float)
    else:
        values = np.asarray(point, dtype=float).reshape(-1)[:2]
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("calculator returned non-finite 2D reciprocal coordinates.")
    return values


def _frequency_matrix(solver, *, point_count, num_bands):
    frequencies = getattr(solver, "all_freqs", None)
    if frequencies is None:
        frequencies = getattr(solver, "freqs", None)
    values = np.asarray(frequencies, dtype=float)
    expected_shape = (point_count, num_bands)
    if values.shape != expected_shape:
        raise ValueError(
            f"supercell isolation frequencies must have shape {expected_shape}, got {values.shape}."
        )
    return values


def _isolation_gaps(frequencies, target_band_index):
    target = frequencies[:, target_band_index]
    other_indices = [index for index in range(frequencies.shape[1]) if index != target_band_index]
    gaps = np.min(np.abs(frequencies[:, other_indices] - target[:, None]), axis=1)
    return target, gaps


def run_supercell_berry_diagnostic(
    config_module=supercell_config,
    *,
    field=None,
    generic_fractional_supercell=(0.073, 0.041),
    cartesian_step=0.001,
    target_band_index=0,
    num_bands=2,
    resolution=8,
    min_isolation_gap=1e-8,
    overlap_tol=1e-14,
):
    """Run one generic periodic-supercell Berry plaquette diagnostically."""
    q_point = _validate_q_point(generic_fractional_supercell)
    cartesian_step = _positive_real(cartesian_step, "cartesian_step")
    resolution = _positive_integer(resolution, "resolution")
    min_isolation_gap = _positive_real(min_isolation_gap, "min_isolation_gap")
    overlap_tol = _positive_real(overlap_tol, "overlap_tol")
    target_band_index, num_bands = _validate_band_inputs(target_band_index, num_bands)

    if field is None:
        field = config_module.make_verified_field()
    field.require_verified()
    replication = _field_replication(field)
    pattern = finite_patch_preview(config_module, field, replication=replication)
    band = config_module.make_band(resolution=resolution)
    calculator = band.build_supercell_berry_calculator(
        pattern,
        field,
        num_bands=num_bands,
        resolution=resolution,
        overlap_tol=overlap_tol,
        polarization="TE",
    )

    cartesian_corners = _cartesian_corners(field, q_point, cartesian_step)
    fractional_corners = [
        _reciprocal_xy(calculator.cartesian_to_reciprocal(corner))
        for corner in cartesian_corners
    ]
    solver = band.run_supercell(
        pattern,
        field,
        q_points=[tuple(point) for point in fractional_corners],
        num_bands=num_bands,
        resolution=resolution,
        polarization="TE",
    )

    result = {
        "diagnostic": "R6.6B_periodic_supercell_single_band_plaquette",
        "q_coordinate": "generic_fractional_supercell",
        "generic_fractional_supercell": q_point.tolist(),
        "cartesian_corners": [point.tolist() for point in cartesian_corners],
        "fractional_corners": [point.tolist() for point in fractional_corners],
        "target_band_index": target_band_index,
        "num_bands": num_bands,
        "resolution": resolution,
        "cartesian_step": cartesian_step,
        "min_isolation_gap_threshold": min_isolation_gap,
        "overlap_tol": overlap_tol,
        "berry_curvature": None,
    }

    frequencies = _frequency_matrix(
        solver,
        point_count=len(cartesian_corners),
        num_bands=num_bands,
    )
    if not np.all(np.isfinite(frequencies)):
        result.update(
            {
                "status": "rejected_nonfinite_frequency",
                "target_frequencies": None,
                "isolation_gaps": None,
                "minimum_isolation_gap": None,
            }
        )
        return result

    target_frequencies, gaps = _isolation_gaps(frequencies, target_band_index)
    result.update(
        {
            "target_frequencies": target_frequencies.tolist(),
            "isolation_gaps": gaps.tolist(),
            "minimum_isolation_gap": float(np.min(gaps)),
        }
    )
    if not np.all(np.isfinite(gaps)):
        result["status"] = "rejected_nonfinite_isolation_gap"
        return result
    if np.any(gaps < min_isolation_gap):
        result["status"] = "rejected_band_isolation"
        return result

    try:
        berry_value = calculator.calculate(
            cartesian_corners[0],
            step=cartesian_step,
            band_index=target_band_index,
        )
    except (FloatingPointError, RuntimeError, ValueError) as error:
        result.update(
            {
                "status": "rejected_berry_calculation",
                "berry_error": f"{type(error).__name__}: {error}",
            }
        )
        return result
    berry_value = float(berry_value)
    if not np.isfinite(berry_value):
        result["status"] = "rejected_nonfinite_berry"
        return result
    result.update({"status": "accepted_isolated_single_band", "berry_curvature": berry_value})
    return result


def main():
    result = run_supercell_berry_diagnostic()
    print(result)
    return result


if __name__ == "__main__":
    main()
