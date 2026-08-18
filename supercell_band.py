"""Bounded user-facing R6 periodic-supercell band demonstration.

This entry point is intentionally separate from the primitive-cell workflows.
It uses a deterministic periodic demonstration field, generic fractional
supercell q-points, and the existing R6 adapter. It does not create persistent
records or claim support for other response workflows.
"""

from numbers import Integral, Real
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config

from mephc.deformation import AnalyticDeformationField, periodic_supercell_field
from r5_deformation import build_supercell_solver


# Editable demonstration parameters. The displacement field is explicitly a
# demonstration, not a physical material preset.
replication = (2, 2)
q_points = (
    (0.0, 0.0),
    (0.25, 0.0),
    (0.25, 0.25),
)
resolution = 16
num_bands = 2
demo_amplitude = 0.02

# Plot controls. No persistent record/cache is created by this entry point.
use_actual = True
save_plot = False
show_plot = False
plot_path = project_root / "image" / "supercell_band.png"


def _positive_integer(value, name):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1.")
    return int(value)


def _validate_replication(value):
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != 2:
        raise ValueError("replication must contain exactly two positive integers.")
    result = tuple(_positive_integer(item, "replication values") for item in value)
    return result


def _validate_q_points(value):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 2:
        raise ValueError("q_points must contain at least two finite 2D points.")
    if not np.all(np.isfinite(array)):
        raise ValueError("q_points must contain only finite values.")
    return tuple(tuple(float(component) for component in point) for point in array)


def _periodic_demonstration_field(lattice, replication, amplitude):
    if not isinstance(amplitude, Real) or isinstance(amplitude, bool) or not np.isfinite(float(amplitude)):
        raise ValueError("demo_amplitude must be a finite real number.")
    basis = np.asarray(lattice.direct_basis, dtype=float)
    inverse_basis = np.linalg.inv(basis)
    periods = np.asarray(replication, dtype=float)
    amplitude = float(amplitude)

    def displacement(points):
        values = np.asarray(points, dtype=float)
        fractional = values @ inverse_basis.T
        phase = 2.0 * np.pi * fractional / periods
        return amplitude * np.column_stack(
            (
                np.sin(phase[:, 0]),
                np.cos(phase[:, 1]) - 1.0,
            )
        )

    return AnalyticDeformationField(
        displacement,
        stable_id=f"trilatt-r6-demonstration-{replication[0]}x{replication[1]}",
        parameters={
            "kind": "periodic-demonstration",
            "replication": list(replication),
            "amplitude": amplitude,
        },
    )


def make_verified_field(config_module=config, *, replication=replication, amplitude=demo_amplitude):
    """Construct the verified periodic field used by the R6 adapter."""
    replication = _validate_replication(replication)
    lattice = config_module.canonical_lattice()
    base = _periodic_demonstration_field(lattice, replication, amplitude)
    return periodic_supercell_field(base, lattice, replication)


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
    config_module=config,
    *,
    replication=replication,
    q_points=q_points,
    resolution=resolution,
    num_bands=num_bands,
    amplitude=demo_amplitude,
):
    """Run the bounded R6 adapter and return generic-q frequency arrays."""
    replication = _validate_replication(replication)
    q_points = _validate_q_points(q_points)
    resolution = _positive_integer(resolution, "resolution")
    num_bands = _positive_integer(num_bands, "num_bands")
    field = make_verified_field(
        config_module,
        replication=replication,
        amplitude=amplitude,
    )
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
        "replication": list(replication),
        "resolution": resolution,
        "num_bands": num_bands,
        "field": field,
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
    axis.set_title("TriLatt R6 periodic-supercell band demonstration")
    axis.set_xticks(x_values)
    axis.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.75)
    axis.legend()
    output_path = None
    if save:
        output_path = Path(image_path or plot_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axis, output_path


def main():
    result = compute_supercell_band(
        config,
        replication=replication,
        q_points=q_points,
        resolution=resolution,
        num_bands=num_bands,
        amplitude=demo_amplitude,
    )
    figure, _, image_path = plot_supercell_band(
        result,
        use_actual=use_actual,
        save=save_plot,
        show=show_plot,
    )
    if not show_plot:
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
