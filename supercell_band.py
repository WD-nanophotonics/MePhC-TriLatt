"""Bounded user-facing R6 periodic-supercell band runner.

User field and run parameters live in ``supercell_config.py``. This runner
accepts a caller-supplied verified ``PeriodicSupercellField`` directly and
does not interpret or reconstruct its displacement formula.
"""

from numbers import Integral
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import supercell_config
from mephc.deformation import PeriodicSupercellField
from r5_deformation import build_supercell_solver


def _positive_integer(value, name):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1.")
    return int(value)


def _validate_q_points(value):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 2:
        raise ValueError("q_points must contain at least two finite 2D points.")
    if not np.all(np.isfinite(array)):
        raise ValueError("q_points must contain only finite values.")
    return tuple(tuple(float(component) for component in point) for point in array)


def _field_replication(field):
    if not isinstance(field, PeriodicSupercellField):
        raise TypeError("R6 runner requires a PeriodicSupercellField.")
    field.require_verified()
    matrix = np.asarray(field.supercell.matrix, dtype=int)
    if matrix.shape != (2, 2) or not np.array_equal(matrix, np.diag(np.diag(matrix))):
        raise ValueError("R6 runner currently requires diagonal replication.")
    values = tuple(int(value) for value in np.diag(matrix))
    if any(value < 1 for value in values):
        raise ValueError("field supercell replication must be positive.")
    return list(values)


def _frequency_arrays(solver, band, *, point_count, requested_bands):
    raw_frequencies = getattr(solver, "all_freqs", None)
    if raw_frequencies is None:
        raw_frequencies = solver.freqs
    normalized = np.asarray(raw_frequencies, dtype=float)
    expected_shape = (point_count, requested_bands)
    if normalized.shape != expected_shape:
        raise ValueError(
            f"R6 solver frequencies must have shape {expected_shape}, got {normalized.shape}."
        )
    if not np.all(np.isfinite(normalized)):
        raise ValueError("R6 solver returned non-finite normalized frequencies.")
    actual = np.asarray(band.calculate_actual_freqs(normalized), dtype=float)
    if actual.shape != expected_shape:
        raise ValueError(
            f"R6 THz frequencies must have shape {expected_shape}, got {actual.shape}."
        )
    if not np.all(np.isfinite(actual)):
        raise ValueError("R6 solver returned non-finite THz frequencies.")
    return normalized, actual


def compute_supercell_band(
    config_module=supercell_config,
    *,
    field=None,
    q_points=None,
    resolution=None,
    num_bands=None,
):
    """Run R6 with a verified caller field and generic supercell q-points."""
    q_points = _validate_q_points(config_module.q_points if q_points is None else q_points)
    resolution = _positive_integer(
        config_module.resolution if resolution is None else resolution,
        "resolution",
    )
    num_bands = _positive_integer(
        config_module.num_bands if num_bands is None else num_bands,
        "num_bands",
    )
    if field is None:
        field = config_module.make_verified_field()
    replication = _field_replication(field)
    solver, metadata = build_supercell_solver(
        config_module,
        field,
        q_points=q_points,
        resolution=resolution,
        num_bands=num_bands,
    )
    band = metadata.get("band")
    if band is None or not hasattr(band, "calculate_actual_freqs"):
        raise ValueError("R6 adapter metadata must contain the band unit converter.")
    normalized, actual = _frequency_arrays(
        solver,
        band,
        point_count=len(q_points),
        requested_bands=num_bands,
    )
    return {
        "q_points": np.asarray(q_points, dtype=float),
        "q_point_coordinate": "generic_fractional_supercell",
        "sample_coordinate": np.arange(len(q_points), dtype=float),
        "freqs": normalized,
        "actual_freqs": actual,
        "replication": replication,
        "resolution": resolution,
        "num_bands": num_bands,
        "field": field,
        "field_metadata": field.metadata(),
        "solver": solver,
        "metadata": metadata,
    }


def plot_supercell_band(result, *, use_actual=True, save=False, show=False, image_path=None):
    """Plot frequency versus generic q-point sample index."""
    values = np.asarray(result["actual_freqs"] if use_actual else result["freqs"], dtype=float)
    x_values = np.asarray(result["sample_coordinate"], dtype=float)
    fig, axis = plt.subplots(figsize=(6.0, 4.5))
    for band_index in range(values.shape[1]):
        axis.plot(
            x_values,
            values[:, band_index],
            marker="o",
            linewidth=1.4,
            label=f"band {band_index + 1}",
        )
    axis.set_xlabel("Generic q-point sample index")
    axis.set_ylabel("Frequency (THz)" if use_actual else "Normalized frequency")
    axis.set_title("TriLatt R6.2 periodic-supercell band demonstration")
    axis.set_xticks(x_values)
    axis.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.75)
    axis.legend()
    output_path = None
    if save:
        output_path = Path(image_path or supercell_config.plot_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axis, output_path


def main():
    result = compute_supercell_band(supercell_config)
    figure, _, image_path = plot_supercell_band(
        result,
        use_actual=supercell_config.use_actual,
        save=supercell_config.save_plot,
        show=supercell_config.show_plot,
    )
    if not supercell_config.show_plot:
        plt.close(figure)
    print("replication:", result["replication"])
    print("q-point coordinate:", result["q_point_coordinate"])
    print("normalized frequency shape:", result["freqs"].shape)
    print("actual frequency shape:", result["actual_freqs"].shape)
    print("finite normalized frequencies:", bool(np.all(np.isfinite(result["freqs"]))))
    print("finite actual frequencies:", bool(np.all(np.isfinite(result["actual_freqs"]))))
    print("image:", image_path)
    return result


if __name__ == "__main__":
    main()
