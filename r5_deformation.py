"""TriLatt integration through the shared MePhC R5 field authority."""

from __future__ import annotations

from mephc.affine import AffineTransform2D
from mephc.deformation import canonicalize_field, periodic_supercell_field
from mephc.deformation_geometry import replicated_rigid_pattern
from mephc.patterns import convert_to_one_layer_pattern_list
from mephc.r5 import primitive_guard, record_identity, supercell_metadata


def global_field(config_module):
    return canonicalize_field(
        AffineTransform2D.uniaxial(
            float(config_module.stretch_factor),
            float(config_module.stretch_angle_degrees),
        )
    )


def _pattern_data(config_module):
    pattern = config_module.build_pattern()
    return convert_to_one_layer_pattern_list(getattr(pattern, "pattern", pattern))


def finite_patch_preview(config_module, field, replication=(3, 3), pattern=None):
    """Build a local/aperiodic preview; no reciprocal-space meaning is attached."""
    return replicated_rigid_pattern(
        _pattern_data(config_module) if pattern is None else pattern,
        config_module.canonical_lattice(),
        replication=replication,
        field=field,
    )


def periodic_supercell_preview(config_module, field, replication=(2, 2)):
    field = periodic_supercell_field(field, config_module.canonical_lattice(), replication)
    return {
        "pattern": finite_patch_preview(config_module, field, replication=replication),
        "supercell": supercell_metadata(field),
        "record_identity": record_identity(field, reference_lattice=config_module.canonical_lattice(), replication=replication),
    }


__all__ = ["finite_patch_preview", "global_field", "periodic_supercell_preview", "primitive_guard"]
