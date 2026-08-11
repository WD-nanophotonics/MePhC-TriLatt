"""Shared record and triangular reciprocal-space helpers for TriLatt."""

from pathlib import Path

import numpy as np

from mephc.kspace import TriangularKSpace
from mephc.records import (
    canonical_record_path,
    find_matching_record,
    load_record,
)


PROJECT_ROOT = Path(__file__).resolve().parent
HBZ_DOMAIN = "k_centered_hbz"


def compute_parameters(config_module, resolution):
    """Return numerical metadata shared by band, Berry, and EFS records."""
    return {
        "resolution": int(resolution),
        "polarization": "TE",
        "geometry": config_module.geometry_parameters(),
    }


def resolve_existing_record(
    config_module,
    *,
    kind,
    task_params,
    compute_params,
    run_mode,
    record_path,
    reuse_requires_compute_match,
):
    """Apply TriLatt's explicit-path and auto/compute/plot-only rules."""
    if record_path is not None:
        path = Path(record_path)
        return load_record(path), path
    if run_mode not in {"auto", "compute", "plot_only"}:
        raise ValueError("run_mode must be 'auto', 'compute', or 'plot_only'.")
    if run_mode in {"auto", "plot_only"}:
        record, path = find_matching_record(
            PROJECT_ROOT,
            config_module.geometry_id(),
            kind,
            task_params=task_params,
            compute_params=compute_params,
            require_compute_match=reuse_requires_compute_match,
        )
        if record is not None:
            return record, path
        if run_mode == "plot_only":
            expected = canonical_record_path(
                PROJECT_ROOT,
                config_module.geometry_id(),
                kind,
                task_params,
            )
            raise FileNotFoundError(
                f"No matching {kind!r} record found. "
                f"Expected canonical path: {expected}"
            )
    return None, None


def has_exact_c3_geometry(config_module):
    """Return whether every active regular polygon has exact 120-degree symmetry."""
    config_module.validate_geometry()
    active_sides = [config_module.n1]
    if config_module.r2 is not None:
        active_sides.append(config_module.n2)
    return all(int(sides) % 3 == 0 for sides in active_sides)


def resolve_symmetry_mode(config_module, symmetry_mode):
    """Resolve ``auto`` to an exact C3 reduction or direct full-HBZ sampling."""
    if symmetry_mode not in {"auto", "c3", "raw_hbz"}:
        raise ValueError("symmetry_mode must be 'auto', 'c3', or 'raw_hbz'.")
    exact_c3 = has_exact_c3_geometry(config_module)
    if symmetry_mode == "auto":
        return "c3" if exact_c3 else "raw_hbz"
    if symmetry_mode == "c3" and not exact_c3:
        raise ValueError(
            "C3 reduction requires every active polygon side count to be "
            "divisible by 3. Use symmetry_mode='auto' or 'raw_hbz'."
        )
    return symmetry_mode


def hbz_sampling(config_module, *, grid_n, shrinking, symmetry_mode):
    """Return the triangular k-space object, actual mode, and raw sample points."""
    if not isinstance(grid_n, int) or isinstance(grid_n, bool) or grid_n < 1:
        raise ValueError("grid_n must be an integer >= 1.")
    if not 0 <= shrinking < 2.0 / 3.0:
        raise ValueError("shrinking must satisfy 0 <= shrinking < 2/3.")
    actual_mode = resolve_symmetry_mode(config_module, symmetry_mode)
    kspace = TriangularKSpace(
        N=grid_n,
        shrinking=shrinking,
        lattice_model=config_module.canonical_lattice(),
    )
    points = kspace.mini_space() if actual_mode == "c3" else kspace.shrunken_hbz()
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        raise ValueError(
            "The selected HBZ grid contains no k-points; increase grid_n "
            "or reduce shrinking."
        )
    return kspace, actual_mode, points


def c3_expand_arrays(kspace, raw_k_points, *raw_arrays):
    """Expand several arrays with one shared C3 point ordering."""
    arrays = [np.asarray(values) for values in raw_arrays]
    if not arrays:
        raise ValueError("At least one value array is required.")
    column_counts = []
    matrices = []
    for values in arrays:
        if values.shape[0] != len(raw_k_points):
            raise ValueError("Every value array must match raw_k_points.")
        matrix = values[:, None] if values.ndim == 1 else values
        if matrix.ndim != 2:
            raise ValueError("Expanded values must be one- or two-dimensional.")
        matrices.append(matrix)
        column_counts.append(matrix.shape[1])
    combined = np.concatenate(matrices, axis=1)
    expanded_points, expanded = kspace.c3_expand(raw_k_points, combined)
    outputs = []
    start = 0
    for original, columns in zip(arrays, column_counts):
        values = expanded[:, start : start + columns]
        outputs.append(values[:, 0] if original.ndim == 1 else values)
        start += columns
    return expanded_points, outputs
