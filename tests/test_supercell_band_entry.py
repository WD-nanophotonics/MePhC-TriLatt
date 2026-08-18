"""Focused tests for the bounded R6 user-facing band entry point."""

import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mephc.bravais import BravaisLattice2D
from mephc.deformation import PeriodicSupercellField

import supercell_band


class FakeBand:
    def __init__(self):
        self.received = None

    def calculate_actual_freqs(self, values):
        self.received = np.asarray(values, dtype=float)
        return self.received * 10.0


class FakeConfig:
    def canonical_lattice(self):
        return BravaisLattice2D.triangular()


class FakeSolver:
    def __init__(self):
        self.freqs = np.asarray([0.30, 0.40], dtype=float)
        self.all_freqs = np.asarray([[0.10, 0.20], [0.30, 0.40]], dtype=float)


class SupercellBandEntryTests(unittest.TestCase):
    def setUp(self):
        self.config = FakeConfig()
        self.q_points = ((0.0, 0.0), (0.25, 0.125))

    def test_invalid_parameters_do_not_call_adapter(self):
        invalid_cases = (
            {"replication": (0, 2)},
            {"replication": (2, 2, 2)},
            {"q_points": ((0.0, 0.0),)},
            {"q_points": ((0.0, float("nan")), (0.25, 0.0))},
            {"resolution": 0},
            {"num_bands": 0},
        )
        base = {
            "replication": (2, 2),
            "q_points": self.q_points,
            "resolution": 16,
            "num_bands": 2,
        }
        for overrides in invalid_cases:
            with self.subTest(overrides=overrides):
                kwargs = dict(base)
                kwargs.update(overrides)
                with patch("supercell_band.build_supercell_solver") as adapter:
                    with self.assertRaises(ValueError):
                        supercell_band.compute_supercell_band(self.config, **kwargs)
                    adapter.assert_not_called()

    def test_constructs_verified_field_and_forwards_exact_r6_inputs(self):
        fake_band = FakeBand()
        fake_solver = FakeSolver()
        adapter_result = (fake_solver, {"band": fake_band, "field": "adapter-field"})
        with patch("supercell_band.build_supercell_solver", return_value=adapter_result) as adapter:
            result = supercell_band.compute_supercell_band(
                self.config,
                replication=(2, 3),
                q_points=self.q_points,
                resolution=19,
                num_bands=2,
                amplitude=0.01,
            )

        self.assertEqual(len(adapter.call_args.args), 2)
        self.assertIs(adapter.call_args.args[0], self.config)
        field = adapter.call_args.args[1]
        self.assertIsInstance(field, PeriodicSupercellField)
        self.assertTrue(field.verified)
        np.testing.assert_array_equal(field.supercell.matrix, [[2, 0], [0, 3]])
        self.assertEqual(adapter.call_args.kwargs["q_points"], self.q_points)
        self.assertEqual(adapter.call_args.kwargs["resolution"], 19)
        self.assertEqual(adapter.call_args.kwargs["num_bands"], 2)
        self.assertIs(result["solver"], fake_solver)
        self.assertIs(result["field"], field)

    def test_extracts_normalized_and_thz_arrays_with_generic_q_semantics(self):
        fake_band = FakeBand()
        with patch(
            "supercell_band.build_supercell_solver",
            return_value=(FakeSolver(), {"band": fake_band}),
        ):
            result = supercell_band.compute_supercell_band(
                self.config,
                replication=(2, 2),
                q_points=self.q_points,
                resolution=16,
                num_bands=2,
            )

        np.testing.assert_allclose(result["freqs"], [[0.1, 0.2], [0.3, 0.4]])
        np.testing.assert_allclose(result["actual_freqs"], [[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(result["q_points"], self.q_points)
        np.testing.assert_allclose(result["sample_coordinate"], [0.0, 1.0])
        self.assertEqual(result["q_point_coordinate"], "generic_fractional_supercell")
        self.assertIsNotNone(fake_band.received)

    def test_plot_uses_sample_index_not_high_symmetry_labels(self):
        values = {
            "sample_coordinate": np.asarray([0.0, 1.0]),
            "freqs": np.asarray([[0.1, 0.2], [0.3, 0.4]]),
            "actual_freqs": np.asarray([[10.0, 20.0], [30.0, 40.0]]),
        }
        figure, axis, output = supercell_band.plot_supercell_band(values, show=False, save=False)
        try:
            self.assertIsNone(output)
            self.assertEqual(axis.get_xlabel(), "Generic q-point sample index")
            self.assertEqual(axis.get_ylabel(), "Frequency (THz)")
            self.assertEqual([label.get_text() for label in axis.get_xticklabels()[:2]], ["0", "1"])
            self.assertNotIn("Gamma", axis.get_title())
            self.assertNotIn("K", axis.get_title())
            self.assertNotIn("M", axis.get_title())
            self.assertNotIn("X", axis.get_title())
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
