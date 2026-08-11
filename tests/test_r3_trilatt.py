"""TriLatt-facing deformation and identity compatibility checks."""

from __future__ import annotations

import unittest

import config
import workflow


class TriLattR3Tests(unittest.TestCase):
    def setUp(self):
        self.factor = config.stretch_factor
        self.angle = config.stretch_angle_degrees

    def tearDown(self):
        config.stretch_factor = self.factor
        config.stretch_angle_degrees = self.angle

    def test_identity_angle_keeps_legacy_geometry_id_and_c3(self):
        config.stretch_factor = 1.0
        config.stretch_angle_degrees = 73.0
        identity_id = config.geometry_id()
        config.stretch_angle_degrees = 0.0
        self.assertEqual(identity_id, config.geometry_id())
        self.assertEqual(workflow.resolve_symmetry_mode(config, "auto"), "c3")

    def test_nonidentity_has_distinct_id_and_full_bz_policy(self):
        identity_id = config.geometry_id()
        config.stretch_factor = 1.1
        config.stretch_angle_degrees = 30.0
        self.assertNotEqual(identity_id, config.geometry_id())
        self.assertEqual(workflow.resolve_symmetry_mode(config, "auto"), "full_bz")
        with self.assertRaises(ValueError):
            workflow.resolve_symmetry_mode(config, "c3")
        with self.assertRaises(ValueError):
            workflow.resolve_symmetry_mode(config, "raw_hbz")

    def test_nonidentity_pattern_centers_follow_basis_without_shape_change(self):
        config.stretch_factor = 1.2
        config.stretch_angle_degrees = 0.0
        pattern = config.build_pattern()
        polygon = pattern.pattern[0][0]
        self.assertEqual(len(polygon), config.n1)
        self.assertEqual(pattern.outer_instance.lattice_model.current_symmetry, "generic_affine")
        self.assertEqual(pattern.outer_instance.lattice_model.reference_family, "triangular")


if __name__ == "__main__":
    unittest.main()
