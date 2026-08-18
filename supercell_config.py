"""User-editable configuration and periodic field for bounded R6/R6.2/R6.3."""

from numbers import Integral, Real
from pathlib import Path

import numpy as np

import config as trilatt_config
from mephc.deformation import AnalyticDeformationField, periodic_supercell_field


# Editable R6.2 parameters. q_points are generic fractional supercell points.
replication = (2, 2)
q_points = ((0.0, 0.0), (0.25, 0.0), (0.25, 0.25))
resolution = 16
num_bands = 2
demo_amplitude = 0.02

# R6.3 record controls. Plot settings do not participate in identity matching.
run_mode = "auto"
archive_record = False
record_path = None
reuse_requires_compute_match = True
save_tmp = True
use_actual = True
save_plot = False
show_plot = False
plot_path = Path(__file__).resolve().parent / "image" / "supercell_band.png"


def _positive_integer(value, name):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1.")
    return int(value)


def validate_replication(value):
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != 2:
        raise ValueError("replication must contain exactly two positive integers.")
    return tuple(_positive_integer(item, "replication values") for item in value)


def _periodic_demonstration_field(lattice, replication_value, amplitude):
    if not isinstance(amplitude, Real) or isinstance(amplitude, bool) or not np.isfinite(float(amplitude)):
        raise ValueError("demo_amplitude must be a finite real number.")
    basis = np.asarray(lattice.direct_basis, dtype=float)
    inverse_basis = np.linalg.inv(basis)
    periods = np.asarray(replication_value, dtype=float)
    amplitude = float(amplitude)

    def displacement(points):
        values = np.asarray(points, dtype=float)
        fractional = values @ inverse_basis.T
        phase = 2.0 * np.pi * fractional / periods
        return amplitude * np.column_stack((np.sin(phase[:, 0]), np.cos(phase[:, 1]) - 1.0))

    return AnalyticDeformationField(
        displacement,
        stable_id=f"trilatt-r6.2-demonstration-{replication_value[0]}x{replication_value[1]}",
        parameters={
            "kind": "periodic-demonstration",
            "status": "example-only-not-physical-preset",
            "replication": list(replication_value),
            "amplitude": amplitude,
        },
    )


def make_verified_field(*, replication=None, amplitude=None):
    """Return the user-defined field after MePhC periodicity verification."""
    replication_value = validate_replication(replication if replication is not None else replication_default())
    amplitude_value = demo_amplitude if amplitude is None else amplitude
    lattice = trilatt_config.canonical_lattice()
    base_field = _periodic_demonstration_field(lattice, replication_value, amplitude_value)
    return periodic_supercell_field(base_field, lattice, replication_value)


def replication_default():
    return replication


# Keep the existing TriLatt geometry/solver authority available to the R6
# adapter while leaving the user-editable field definition in this module.
def canonical_lattice():
    return trilatt_config.canonical_lattice()


def geometry_id():
    return trilatt_config.geometry_id()


def geometry_parameters():
    return trilatt_config.geometry_parameters()


def build_pattern():
    return trilatt_config.build_pattern()


def make_band(*args, **kwargs):
    return trilatt_config.make_band(*args, **kwargs)
