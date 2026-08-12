from numbers import Integral, Real
import numpy as np
from mephc.band import Band
from mephc.affine import AffineTransform2D
from mephc.bravais import BravaisLattice2D
from mephc.bz import tracked_landmark
from mephc.kspace import generic_bz_path, triangular_gkm_path


# All physical lengths below use nm. ``a`` is the real-space lattice period;
# MPB geometry is normalized by this value, and THz conversion also uses it.
a = 400

# r1 is the circumradius of the first polygon.
r1 = 120
# r2 alone selects the unit-cell rule:
#   None  -> one polygon: create_unitcell(n1, theta1)
#   value -> two polygons: create_unitcell(n1, theta1, n2, theta2)
# Set only r2=None to switch to the single-polygon triangular case. n2 and
# theta2 may remain populated because they are ignored while r2 is None.
r2 = 110

# n is the number of polygon sides. n=3 is a triangle; n=16 closely
# approximates a circle. theta is passed directly to MePhC in degrees.
n1 = 3
theta1 = 0
n2 = 3
theta2 = 60

# Effective background refractive index; MPB uses epsilon = n_eff**2.
n_eff = 2.7
# Meep object height in nm. This 2D workflow only needs it to span the cell.
height = 100

# Global direct-space uniaxial deformation.  The axis angle is measured
# counter-clockwise from +x in Cartesian coordinates.  At factor == 1 the
# angle is intentionally canonicalized away and legacy IDs remain unchanged.
stretch_factor = 1.0
stretch_angle_degrees = 0.0


def _compact_number(value):
    if isinstance(value, Integral):
        text = str(int(value))
    else:
        text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _validate_polygon(sides, angle, label):
    if not isinstance(sides, Integral) or isinstance(sides, bool) or sides < 3:
        raise ValueError(f"{label} must be an integer >= 3.")
    if not isinstance(angle, Real) or isinstance(angle, bool):
        raise ValueError(f"theta{label[-1]} must be a real number.")


def validate_geometry():
    """Validate only the parameters used by the selected r2 rule."""
    if not isinstance(a, Real) or isinstance(a, bool) or a <= 0:
        raise ValueError("a must be a positive number.")
    if not isinstance(r1, Real) or isinstance(r1, bool) or r1 <= 0:
        raise ValueError("r1 must be a positive number.")
    if not isinstance(n_eff, Real) or isinstance(n_eff, bool) or n_eff <= 0:
        raise ValueError("n_eff must be a positive number.")
    if not isinstance(height, Real) or isinstance(height, bool) or height <= 0:
        raise ValueError("height must be a positive number.")
    if not isinstance(stretch_factor, Real) or isinstance(stretch_factor, bool) or not float(stretch_factor) > 0:
        raise ValueError("stretch_factor must be a positive finite number.")
    if not np.isfinite(float(stretch_factor)):
        raise ValueError("stretch_factor must be finite.")
    if not isinstance(stretch_angle_degrees, Real) or isinstance(stretch_angle_degrees, bool) or not np.isfinite(float(stretch_angle_degrees)):
        raise ValueError("stretch_angle_degrees must be finite.")
    AffineTransform2D.uniaxial(float(stretch_factor), float(stretch_angle_degrees))
    _validate_polygon(n1, theta1, "n1")

    if r2 is None:
        return
    if not isinstance(r2, Real) or isinstance(r2, bool) or r2 <= 0:
        raise ValueError("r2 must be None or a positive number.")
    if n2 is None or theta2 is None:
        raise ValueError("n2 and theta2 are required when r2 is not None.")
    _validate_polygon(n2, theta2, "n2")


def geometry_id():
    """Return a compact ID made only from the active physical parameters."""
    validate_geometry()
    if r2 is None:
        result = (
            f"a{_compact_number(a)}"
            f"_r{_compact_number(r1)}"
            f"_n{_compact_number(n1)}"
            f"_t{_compact_number(theta1)}"
            f"_neff{_compact_number(n_eff)}"
            f"_h{_compact_number(height)}"
        )
    else:
        result = (
        f"a{_compact_number(a)}"
        f"_r{_compact_number(r1)}-{_compact_number(r2)}"
        f"_n{_compact_number(n1)}-{_compact_number(n2)}"
        f"_t{_compact_number(theta1)}-{_compact_number(theta2)}"
        f"_neff{_compact_number(n_eff)}"
        f"_h{_compact_number(height)}"
        )
    if float(stretch_factor) != 1.0:
        result += f"_s{_compact_number(stretch_factor)}_ang{_compact_number(stretch_angle_degrees)}"
    return result


def geometry_parameters():
    """Return the active parameters stored in calculation metadata."""
    validate_geometry()
    parameters = {
        "a": a,
        "r1": r1,
        "r2": r2,
        "n1": n1,
        "theta1": theta1,
        "n_eff": n_eff,
        "height": height,
    }
    if r2 is not None:
        parameters.update({"n2": n2, "theta2": theta2})
    if float(stretch_factor) != 1.0:
        parameters.update({"stretch_factor": float(stretch_factor), "stretch_angle_degrees": float(stretch_angle_degrees)})
    return parameters


def canonical_lattice():
    """Return a fresh canonical undeformed triangular lattice model.

    The factory is intentionally geometry-only: R2 does not expose stretch
    parameters.  Band, real-space placement, and k-space helpers consume this
    same model instance within each workflow call.
    """

    base = BravaisLattice2D.triangular()
    transform = AffineTransform2D.uniaxial(float(stretch_factor), float(stretch_angle_degrees))
    return base.transformed(transform)


def make_band(*, resolution):
    """Build the MePhC solver; its triangular geometry lattice is the default."""
    validate_geometry()
    return Band(
        a=a,
        r1=r1,
        r2=r2,
        n_eff=n_eff,
        h=height,
        resolution=resolution,
        lattice_model=canonical_lattice(),
    )


def build_pattern():
    """Build one or two polygons using MePhC's r2-based unit-cell rule."""
    band = make_band(resolution=1)
    if r2 is None:
        return band.create_unitcell(n1, theta1, show=False)
    return band.create_unitcell(n1, theta1, n2, theta2, show=False)


def reciprocal_landmark():
    """Return identity K or current-BZ ``tracked_K1`` metadata."""
    lattice = canonical_lattice()
    landmark = tracked_landmark(lattice)
    selected = np.asarray(landmark["cartesian"], dtype=float)
    result = dict(landmark)
    result.update(
        {
            "cartesian": selected,
            "solver_reciprocal_fractional": lattice.cartesian_to_reciprocal(selected),
            "deformation_factor": float(stretch_factor),
            "deformation_angle_degrees": float(stretch_angle_degrees)
            if float(stretch_factor) != 1.0
            else 0.0,
            "lattice": lattice.metadata(),
            "bz_vertices": np.asarray(landmark["bz"].vertices, dtype=float),
        }
    )
    result.pop("bz", None)
    return result


def k_point():
    """Return legacy K or the explicit current tracked landmark."""
    return tuple(np.asarray(reciprocal_landmark()["cartesian"], dtype=float))


def band_path():
    """Return the Gamma-K-M-Gamma path in Cartesian reciprocal coordinates."""
    lattice = canonical_lattice()
    return triangular_gkm_path() if lattice.supports_legacy("gkm") else generic_bz_path(lattice)
