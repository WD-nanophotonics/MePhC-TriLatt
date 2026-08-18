import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mephc.deformation import AnalyticDeformationField, periodic_supercell_field
from mephc.records import load_record

import supercell_band
import supercell_config


class SupercellRecordTests(unittest.TestCase):
    def setUp(self):
        self.q_points = ((0.0, 0.0), (0.25, 0.0), (0.25, 0.25))
        self.field = supercell_config.make_verified_field(replication=(2, 2), amplitude=0.02)

    def test_identity_and_q_path_are_stable_and_order_sensitive(self):
        first = supercell_band._record_parameters(
            supercell_config, self.field, self.q_points, 16, 2
        )
        second = supercell_band._record_parameters(
            supercell_config,
            supercell_config.make_verified_field(replication=(2, 2), amplitude=0.02),
            self.q_points,
            16,
            2,
        )
        self.assertEqual(first, second)
        reordered = supercell_band._record_parameters(
            supercell_config, self.field, tuple(reversed(self.q_points)), 16, 2
        )
        self.assertNotEqual(first[1]["q_point_digest"], reordered[1]["q_point_digest"])
        self.assertNotEqual(first[1]["path"], reordered[1]["path"])

    def test_field_identity_and_replication_change_namespace(self):
        changed_field = supercell_config.make_verified_field(replication=(2, 2), amplitude=0.03)
        changed_replication = supercell_config.make_verified_field(replication=(2, 3), amplitude=0.02)
        first = supercell_band._record_parameters(supercell_config, self.field, self.q_points, 16, 2)
        changed = supercell_band._record_parameters(supercell_config, changed_field, self.q_points, 16, 2)
        replicated = supercell_band._record_parameters(supercell_config, changed_replication, self.q_points, 16, 2)
        self.assertNotEqual(first[0], changed[0])
        self.assertNotEqual(first[0], replicated[0])
        self.assertNotEqual(first[2]["field_identity_sha256"], changed[2]["field_identity_sha256"])

    def test_auto_cache_hit_and_compute_force_adapter(self):
        def fake_compute(config_module, *, field, q_points, resolution, num_bands):
            values = np.arange(len(q_points) * num_bands, dtype=float).reshape(len(q_points), num_bands) / 10.0
            return {
                "q_points": np.asarray(q_points),
                "q_point_coordinate": supercell_band.Q_POINT_COORDINATE,
                "sample_coordinate": np.arange(len(q_points), dtype=float),
                "freqs": values,
                "actual_freqs": values * 10.0,
                "replication": supercell_band._field_replication(field),
                "resolution": resolution,
                "num_bands": num_bands,
                "field": field,
                "field_metadata": field.metadata(),
                "solver": object(),
                "metadata": {},
            }

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(supercell_band, "project_root", Path(directory)):
                with patch.object(supercell_band, "compute_supercell_band", side_effect=fake_compute) as adapter:
                    first, path, _ = supercell_band.compute_supercell_band_record(
                        supercell_config,
                        field=self.field,
                        q_points=self.q_points,
                        resolution=16,
                        num_bands=2,
                        run_mode="compute",
                        save_tmp=False,
                    )
                    self.assertEqual(adapter.call_count, 1)
                with patch.object(supercell_band, "compute_supercell_band", side_effect=AssertionError("cache miss")):
                    cached, cached_path, latest = supercell_band.compute_supercell_band_record(
                        supercell_config,
                        field=self.field,
                        q_points=self.q_points,
                        resolution=16,
                        num_bands=2,
                        run_mode="auto",
                        save_tmp=False,
                    )
                self.assertEqual(cached, first)
                self.assertEqual(cached_path, path)
                self.assertIsNone(latest)
                with patch.object(supercell_band, "compute_supercell_band", side_effect=fake_compute) as forced:
                    supercell_band.compute_supercell_band_record(
                        supercell_config,
                        field=self.field,
                        q_points=self.q_points,
                        resolution=16,
                        num_bands=2,
                        run_mode="compute",
                        save_tmp=False,
                    )
                self.assertEqual(forced.call_count, 1)

    def test_resolution_change_is_strict_cache_miss(self):
        def fake_compute(config_module, *, field, q_points, resolution, num_bands):
            values = np.full((len(q_points), num_bands), float(resolution))
            return {
                "q_points": np.asarray(q_points),
                "q_point_coordinate": supercell_band.Q_POINT_COORDINATE,
                "sample_coordinate": np.arange(len(q_points), dtype=float),
                "freqs": values,
                "actual_freqs": values * 10.0,
                "replication": supercell_band._field_replication(field),
                "resolution": resolution,
                "num_bands": num_bands,
                "field": field,
                "field_metadata": field.metadata(),
                "solver": object(),
                "metadata": {},
            }

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(supercell_band, "project_root", Path(directory)):
                with patch.object(supercell_band, "compute_supercell_band", side_effect=fake_compute) as first_adapter:
                    supercell_band.compute_supercell_band_record(
                        supercell_config,
                        field=self.field,
                        q_points=self.q_points,
                        resolution=16,
                        num_bands=2,
                        run_mode="compute",
                        reuse_requires_compute_match=True,
                        save_tmp=False,
                    )
                self.assertEqual(first_adapter.call_count, 1)
                with patch.object(supercell_band, "compute_supercell_band", side_effect=fake_compute) as second_adapter:
                    record, _, _ = supercell_band.compute_supercell_band_record(
                        supercell_config,
                        field=self.field,
                        q_points=self.q_points,
                        resolution=17,
                        num_bands=2,
                        run_mode="auto",
                        reuse_requires_compute_match=True,
                        save_tmp=False,
                    )
                self.assertEqual(second_adapter.call_count, 1)
                self.assertEqual(record["compute_params"]["resolution"], 17)

    def test_plot_only_missing_fails_before_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(supercell_band, "project_root", Path(directory)):
                with patch.object(supercell_band, "compute_supercell_band") as adapter:
                    with self.assertRaises(FileNotFoundError):
                        supercell_band.compute_supercell_band_record(
                            supercell_config,
                            field=self.field,
                            q_points=self.q_points,
                            resolution=16,
                            num_bands=2,
                            run_mode="plot_only",
                            save_tmp=False,
                        )
                    adapter.assert_not_called()

    def test_persistent_data_round_trips_without_runtime_objects(self):
        raw = {
            "q_points": np.asarray(self.q_points),
            "q_point_coordinate": supercell_band.Q_POINT_COORDINATE,
            "sample_coordinate": np.arange(3, dtype=float),
            "freqs": np.ones((3, 2)),
            "actual_freqs": np.ones((3, 2)) * 10.0,
            "replication": [2, 2],
            "resolution": 16,
            "num_bands": 2,
            "field": self.field,
            "field_metadata": self.field.metadata(),
            "solver": object(),
        }
        identity = supercell_band._field_record_identity(supercell_config, self.field, [2, 2])
        data = supercell_band._persistent_data(raw, identity)
        self.assertNotIn("field", data)
        self.assertNotIn("solver", data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.pkl"
            with path.open("wb") as handle:
                pickle.dump({"data": data}, handle)
            loaded = load_record(path)
        self.assertEqual(loaded["data"]["q_point_coordinate"], supercell_band.Q_POINT_COORDINATE)
        self.assertEqual(loaded["data"]["replication"], [2, 2])
        self.assertIsInstance(loaded["data"]["field_record_identity"], dict)

    def test_plot_record_uses_only_saved_data(self):
        record = {
            "geometry_id": "TRILATT_SUPERCELL_TEST",
            "data": {
                "sample_coordinate": [0.0, 1.0],
                "q_points_cartesian": [[0.0, 0.0], [0.1, 0.0]],
                "reciprocal_path_distance": [0.0, 0.1],
                "reciprocal_basis": [[1.0, 0.0], [0.0, 1.0]],
                "freqs": [[0.1, 0.2], [0.3, 0.4]],
                "actual_freqs": [[1.0, 2.0], [3.0, 4.0]],
            },
        }
        with patch.object(supercell_band, "build_supercell_solver", side_effect=AssertionError("adapter called")):
            figure, axis, output = supercell_band.plot_supercell_band_record(
                record, save=False, show=False
            )
        try:
            self.assertIsNone(output)
            self.assertEqual(axis.get_xlabel(), "Generic reciprocal-path distance")
            self.assertNotIn("Gamma", axis.get_title())
            self.assertNotIn("K", axis.get_title())
            self.assertNotIn("M", axis.get_title())
            self.assertNotIn("X", axis.get_title())
        finally:
            plt.close(figure)

    def test_reciprocal_coordinates_and_cumulative_distance_use_field_metric(self):
        q_points = ((0.0, 0.0), (0.25, 0.0), (0.25, 0.5))
        derived = supercell_band._reciprocal_path_data(self.field, q_points)
        expected_cartesian = np.asarray(q_points) @ self.field.reciprocal_basis.T
        expected_increments = np.linalg.norm(np.diff(expected_cartesian, axis=0), axis=1)
        np.testing.assert_allclose(derived["q_points_cartesian"], expected_cartesian)
        np.testing.assert_allclose(derived["reciprocal_path_distance"], [0.0, *np.cumsum(expected_increments)])
        np.testing.assert_allclose(derived["reciprocal_basis"], self.field.reciprocal_basis)
        self.assertEqual(derived["reciprocal_path_distance"][0], 0.0)
        self.assertFalse(np.isclose(expected_increments[0], expected_increments[1]))
        self.assertTrue(np.all(np.diff(derived["reciprocal_path_distance"]) >= 0.0))

    def test_persistent_record_contains_reciprocal_path_provenance(self):
        raw = {
            "q_points": np.asarray(self.q_points),
            "sample_coordinate": np.arange(3, dtype=float),
            "freqs": np.ones((3, 2)),
            "actual_freqs": np.ones((3, 2)) * 10.0,
            "replication": [2, 2],
            "resolution": 16,
            "num_bands": 2,
            "field": self.field,
            "field_metadata": self.field.metadata(),
            "solver": object(),
        }
        identity = supercell_band._field_record_identity(supercell_config, self.field, [2, 2])
        data = supercell_band._persistent_data(raw, identity)
        self.assertEqual(np.asarray(data["q_points_cartesian"]).shape, (3, 2))
        self.assertEqual(len(data["reciprocal_path_distance"]), 3)
        self.assertEqual(np.asarray(data["reciprocal_basis"]).shape, (2, 2))
        self.assertNotIn("field", data)
        self.assertNotIn("solver", data)
    def test_unstable_field_fails_before_persistent_write(self):
        base = AnalyticDeformationField(lambda points: np.zeros((len(points), 2)))
        unstable = periodic_supercell_field(base, supercell_config.canonical_lattice(), (2, 2))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(supercell_band, "project_root", Path(directory)):
                with patch.object(supercell_band, "compute_supercell_band") as adapter:
                    with self.assertRaisesRegex(ValueError, "E_R5_UNSTABLE_CALLABLE"):
                        supercell_band.compute_supercell_band_record(
                            supercell_config,
                            field=unstable,
                            q_points=self.q_points,
                            resolution=16,
                            num_bands=2,
                            run_mode="compute",
                            save_tmp=False,
                        )
                    adapter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
