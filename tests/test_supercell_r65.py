import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import supercell_band
import supercell_config


class FakeBand:
    def calculate_actual_freqs(self, values):
        return np.asarray(values, dtype=float) * 10.0


class FakeSolver:
    all_freqs = np.asarray([[0.1], [0.2]], dtype=float)


class SupercellR65Tests(unittest.TestCase):
    def setUp(self):
        self.field = supercell_config.make_verified_field(replication=(2, 2), amplitude=0.02)

    def test_builder_count_shared_vertices_scalar_and_per_segment(self):
        scalar = supercell_config.build_q_path(
            anchors=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
            subdivisions=2,
        )
        self.assertEqual(len(scalar["q_points"]), 1 + 2 + 2)
        np.testing.assert_allclose(scalar["q_points"][2], [1.0, 0.0])
        self.assertEqual(scalar["q_path_anchor_indices"], [0, 2, 4])
        per_segment = supercell_config.build_q_path(
            anchors=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
            subdivisions=(1, 3),
        )
        self.assertEqual(len(per_segment["q_points"]), 1 + 1 + 3)
        self.assertEqual(per_segment["q_path_subdivisions"], [1, 3])
        self.assertEqual(per_segment["q_path_anchor_indices"], [0, 1, 4])
        self.assertEqual(per_segment["q_path_anchor_labels"], ["q0", "q1", "q2"])

    def test_invalid_anchors_and_subdivisions_fail_before_adapter(self):
        invalid_anchors = [((0.0, 0.0),), ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), ((0.0, np.nan), (1.0, 0.0))]
        for anchors in invalid_anchors:
            with self.subTest(anchors=anchors):
                with self.assertRaises(ValueError):
                    supercell_config.build_q_path(anchors=anchors, subdivisions=1)
        invalid_subdivisions = [0, -1, True, (1,), (1.5, 1), (1, False)]
        for subdivisions in invalid_subdivisions:
            with self.subTest(subdivisions=subdivisions):
                with self.assertRaises(ValueError):
                    supercell_config.build_q_path(subdivisions=subdivisions)

    def test_explicit_q_points_bypass_config_builder(self):
        class Config:
            q_points = None
            resolution = 4
            num_bands = 1
            canonical_lattice = staticmethod(supercell_config.canonical_lattice)

            @staticmethod
            def build_q_path():
                raise AssertionError("explicit q_points must bypass config builder")

        explicit = ((0.0, 0.0), (0.125, 0.25))
        with patch.object(
            supercell_band,
            "build_supercell_solver",
            return_value=(FakeSolver(), {"band": FakeBand()}),
        ) as adapter:
            result = supercell_band.compute_supercell_band(
                Config,
                field=self.field,
                q_points=explicit,
                resolution=4,
                num_bands=1,
            )
        self.assertEqual(adapter.call_args.kwargs["q_points"], explicit)
        self.assertIsNone(result["q_path_metadata"])

    def test_expanded_points_remain_task_identity_authority(self):
        path = supercell_config.build_q_path(
            anchors=((0.0, 0.0), (0.5, 0.0)), subdivisions=2
        )
        explicit = tuple(tuple(point) for point in path["q_points"])
        first = supercell_band._record_parameters(supercell_config, self.field, explicit, 4, 1)
        second = supercell_band._record_parameters(supercell_config, self.field, explicit, 4, 1)
        self.assertEqual(first[1], second[1])
        equivalent = supercell_config.build_q_path(
            anchors=((0.0, 0.0), (0.25, 0.0), (0.5, 0.0)), subdivisions=1
        )
        third = supercell_band._record_parameters(
            supercell_config, self.field, tuple(tuple(point) for point in equivalent["q_points"]), 4, 1
        )
        self.assertEqual(first[1]["q_point_digest"], third[1]["q_point_digest"])

    def test_reciprocal_steps_are_uniform_for_equal_segment_subdivisions(self):
        path = supercell_config.build_q_path(
            anchors=((0.0, 0.0), (0.5, 0.25)), subdivisions=4
        )
        derived = supercell_band._reciprocal_path_data(self.field, path["q_points"])
        steps = np.diff(derived["reciprocal_path_distance"])
        np.testing.assert_allclose(steps, steps[0])
        self.assertEqual(len(path["q_points"]), 5)

    def test_record_persists_anchor_provenance_without_runtime_objects(self):
        path = supercell_config.build_q_path(subdivisions=(2, 2))
        raw = {
            "q_points": np.asarray(path["q_points"]),
            "sample_coordinate": np.arange(len(path["q_points"]), dtype=float),
            "freqs": np.ones((len(path["q_points"]), 1)),
            "actual_freqs": np.ones((len(path["q_points"]), 1)),
            "replication": [2, 2],
            "resolution": 4,
            "num_bands": 1,
            "field": self.field,
            "field_metadata": self.field.metadata(),
            "solver": object(),
            "q_path_metadata": path,
        }
        identity = supercell_band._field_record_identity(supercell_config, self.field, [2, 2])
        data = supercell_band._persistent_data(raw, identity)
        self.assertEqual(data["q_path_anchor_indices"], path["q_path_anchor_indices"])
        self.assertEqual(data["q_path_anchor_labels"], ["q0", "q1", "q2"])
        self.assertNotIn("field", data)
        self.assertNotIn("solver", data)

    def test_plot_uses_anchor_ticks_and_legacy_record_has_safe_fallback(self):
        path = supercell_config.build_q_path(subdivisions=(2, 2))
        distance = [0.0, 0.1, 0.2, 0.3, 0.4]
        common = {
            "q_points_cartesian": [[0.0, 0.0]] * 5,
            "reciprocal_path_distance": distance,
            "reciprocal_basis": [[1.0, 0.0], [0.0, 1.0]],
            "freqs": [[0.1]] * 5,
            "actual_freqs": [[1.0]] * 5,
        }
        record = {"geometry_id": "TEST", "data": dict(common)}
        record["data"].update({key: path[key] for key in ("q_path_anchor_indices", "q_path_anchor_labels", "q_path_anchors", "q_path_subdivisions")})
        with patch.object(supercell_band, "build_supercell_solver", side_effect=AssertionError("adapter called")):
            figure, axis, _ = supercell_band.plot_supercell_band_record(record, save=False, show=False)
        try:
            self.assertEqual([tick.get_text() for tick in axis.get_xticklabels()], ["q0", "q1", "q2"])
            self.assertEqual(axis.get_xlabel(), "Generic reciprocal-path distance")
            self.assertNotIn("Gamma", axis.get_title())
            self.assertNotIn("K", axis.get_title())
            self.assertNotIn("M", axis.get_title())
            self.assertNotIn("X", axis.get_title())
        finally:
            plt.close(figure)
        legacy = {"geometry_id": "TEST", "data": dict(common)}
        with patch.object(supercell_band, "build_supercell_solver", side_effect=AssertionError("adapter called")):
            figure, axis, _ = supercell_band.plot_supercell_band_record(legacy, save=False, show=False)
        try:
            self.assertNotEqual([tick.get_text() for tick in axis.get_xticklabels()], ["q0", "q1", "q2"])
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
