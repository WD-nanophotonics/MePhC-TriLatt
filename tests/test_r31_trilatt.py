import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_config():
    spec = importlib.util.spec_from_file_location("r31_trilatt_config", ROOT / "config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = load_config()


class R31GeometryTests(unittest.TestCase):
    def test_all_required_cases_move_centers_and_keep_local_features_rigid(self):
        original = (config.stretch_factor, config.stretch_angle_degrees)
        try:
            config.stretch_factor = 1.0
            config.stretch_angle_degrees = 0.0
            reference = config.build_pattern()
            reference_centers = np.asarray([np.mean(layer[0], axis=0) for layer in reference.pattern])
            reference_offsets = [layer[0] - reference_centers[index] for index, layer in enumerate(reference.pattern)]
            reference_edges = [np.roll(vertices, -1, axis=0) - vertices for vertices in reference_offsets]
            reference_areas = [
                0.5 * np.sum(offset[:, 0] * np.roll(offset[:, 1], -1) - offset[:, 1] * np.roll(offset[:, 0], -1))
                for offset in reference_offsets
            ]

            for factor, angle in ((1.0, 0.0), (1.1, 0.0), (0.9, 30.0), (1.08, 17.0)):
                config.stretch_factor = factor
                config.stretch_angle_degrees = angle
                current = config.build_pattern()
                self.assertEqual([len(layer) for layer in current.pattern], [1, 1])
                current_centers = np.asarray([np.mean(layer[0], axis=0) for layer in current.pattern])
                transform = config.canonical_lattice().deformation_matrix
                np.testing.assert_allclose(current_centers, reference_centers @ transform.T, atol=1e-12)
                np.testing.assert_allclose(
                    current_centers[1] - current_centers[0],
                    (reference_centers[1] - reference_centers[0]) @ transform.T,
                    atol=1e-12,
                )
                for index, layer in enumerate(current.pattern):
                    offsets = layer[0] - current_centers[index]
                    np.testing.assert_allclose(offsets, reference_offsets[index], atol=1e-12)
                    np.testing.assert_allclose(
                        np.roll(offsets, -1, axis=0) - offsets,
                        reference_edges[index],
                        atol=1e-12,
                    )
                    area = 0.5 * np.sum(offsets[:, 0] * np.roll(offsets[:, 1], -1) - offsets[:, 1] * np.roll(offsets[:, 0], -1))
                    self.assertAlmostEqual(float(area), float(reference_areas[index]), places=12)
        finally:
            config.stretch_factor, config.stretch_angle_degrees = original

    def test_nonidentity_landmark_metadata_is_explicit(self):
        original = (config.stretch_factor, config.stretch_angle_degrees)
        try:
            config.stretch_factor = 1.08
            config.stretch_angle_degrees = 17.0
            landmark = config.reciprocal_landmark()
            self.assertEqual(landmark["landmark_kind"], "tracked_K1")
            self.assertEqual(landmark["display_label"], "tracked_K1")
            self.assertEqual(landmark["selection_strategy"], "nearest_current_bz_vertex_to_F_inverse_transpose_reference_plus_x_K")
            self.assertNotEqual(tuple(config.k_point()), (2.0 / 3.0, 0.0))
            self.assertIn("solver_reciprocal_fractional", landmark)
            self.assertIn("bz_vertices", landmark)
        finally:
            config.stretch_factor, config.stretch_angle_degrees = original


if __name__ == "__main__":
    unittest.main()
