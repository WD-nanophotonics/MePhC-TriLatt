import json
import unittest
from unittest.mock import Mock, patch

import numpy as np

import supercell_berry_diagnostic as diagnostic


class FakeField:
    reciprocal_basis = np.asarray([[2.0, 0.5], [0.25, 3.0]])
    supercell = type("Supercell", (), {"matrix": np.diag([2, 2])})()

    def require_verified(self):
        return None


class FakeCalculator:
    def __init__(self, *, value=0.125):
        self.value = value
        self.cartesian_calls = []
        self.calculate_calls = []

    def cartesian_to_reciprocal(self, point):
        self.cartesian_calls.append(np.asarray(point, dtype=float).copy())
        return np.asarray(point, dtype=float) * np.array([2.0, 3.0])

    def calculate(self, point, *, step, band_index):
        self.calculate_calls.append(
            {"point": np.asarray(point, dtype=float).copy(), "step": step, "band_index": band_index}
        )
        return self.value


class FakeBand:
    def __init__(self, frequencies, calculator=None):
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.calculator = calculator or FakeCalculator()
        self.factory_calls = []
        self.run_calls = []
        self._prepare_supercell_geometry = Mock(side_effect=AssertionError("private helper called"))

    def build_supercell_berry_calculator(self, pattern, field, **kwargs):
        self.factory_calls.append({"pattern": pattern, "field": field, **kwargs})
        return self.calculator

    def run_supercell(self, pattern, field, **kwargs):
        self.run_calls.append({"pattern": pattern, "field": field, **kwargs})
        return type("FakeSolver", (), {"all_freqs": self.frequencies})()


class FakeConfig:
    def __init__(self, band):
        self.band = band

    def make_band(self, *, resolution):
        self.resolution = resolution
        return self.band


class SupercellBerryDiagnosticR66BTests(unittest.TestCase):
    def setUp(self):
        self.field = FakeField()
        self.pattern = object()

    def run_case(self, frequencies, **kwargs):
        band = FakeBand(frequencies)
        config = FakeConfig(band)
        threshold = kwargs.pop("min_isolation_gap", 0.2)
        with patch.object(diagnostic, "finite_patch_preview", return_value=self.pattern) as preview:
            result = diagnostic.run_supercell_berry_diagnostic(
                config,
                field=self.field,
                generic_fractional_supercell=(0.1, 0.2),
                cartesian_step=0.5,
                target_band_index=0,
                num_bands=2,
                resolution=4,
                min_isolation_gap=threshold,
                overlap_tol=1e-6,
                **kwargs,
            )
        return result, band, preview, config

    def test_fractional_lower_left_uses_column_vector_basis_convention(self):
        result, band, _, _ = self.run_case(np.tile([[1.0, 1.5]], (4, 1)))
        np.testing.assert_allclose(result["cartesian_corners"], [
            [0.3, 0.625], [0.8, 0.625], [0.8, 1.125], [0.3, 1.125]
        ])
        self.assertEqual(len(band.calculator.cartesian_calls), 4)
        self.assertEqual(len(band.run_calls[0]["q_points"]), 4)

    def test_fractional_corners_use_calculator_public_conversion(self):
        result, band, _, _ = self.run_case(np.tile([[1.0, 1.5]], (4, 1)))
        expected = np.asarray(result["cartesian_corners"]) * np.array([2.0, 3.0])
        np.testing.assert_allclose(result["fractional_corners"], expected)
        self.assertEqual(len(band.calculator.cartesian_calls), 4)

    def test_factory_and_public_run_path_share_pattern_and_field(self):
        result, band, preview, config = self.run_case(np.tile([[1.0, 1.5]], (4, 1)))
        self.assertEqual(result["status"], "accepted_isolated_single_band")
        preview.assert_called_once_with(config, self.field, replication=(2, 2))
        self.assertIs(band.factory_calls[0]["pattern"], self.pattern)
        self.assertIs(band.factory_calls[0]["field"], self.field)
        self.assertIs(band.run_calls[0]["pattern"], self.pattern)
        self.assertIs(band.run_calls[0]["field"], self.field)
        band._prepare_supercell_geometry.assert_not_called()
        self.assertEqual(band.factory_calls[0]["polarization"], "TE")
        self.assertEqual(band.run_calls[0]["polarization"], "TE")

    def test_invalid_inputs_fail_before_factory_or_solver(self):
        invalid = (
            {"generic_fractional_supercell": (0.1,)},
            {"cartesian_step": 0.0},
            {"target_band_index": -1},
            {"target_band_index": 1, "num_bands": 2},
            {"num_bands": 1},
            {"min_isolation_gap": 0.0},
            {"overlap_tol": float("nan")},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                band = FakeBand(np.tile([[1.0, 1.5]], (4, 1)))
                config = FakeConfig(band)
                with self.assertRaises(ValueError):
                    diagnostic.run_supercell_berry_diagnostic(config, field=self.field, **overrides)
                self.assertEqual(band.factory_calls, [])
                self.assertEqual(band.run_calls, [])

    def test_isolation_rejects_before_calculate(self):
        result, band, _, _ = self.run_case(np.tile([[1.0, 1.1]], (4, 1)))
        self.assertEqual(result["status"], "rejected_band_isolation")
        self.assertIsNone(result["berry_curvature"])
        self.assertEqual(band.calculator.calculate_calls, [])
        self.assertAlmostEqual(result["minimum_isolation_gap"], 0.1)

    def test_exact_threshold_is_accepted_and_calculate_receives_cartesian_api(self):
        result, band, _, _ = self.run_case(
            np.tile([[1.0, 1.25]], (4, 1)), min_isolation_gap=0.25
        )
        self.assertEqual(result["status"], "accepted_isolated_single_band")
        self.assertEqual(result["berry_curvature"], 0.125)
        self.assertEqual(len(band.calculator.calculate_calls), 1)
        call = band.calculator.calculate_calls[0]
        np.testing.assert_allclose(call["point"], [0.3, 0.625])
        self.assertEqual(call["step"], 0.5)
        self.assertEqual(call["band_index"], 0)

    def test_nonfinite_frequency_fails_closed(self):
        result, band, _, _ = self.run_case(np.asarray([[1.0, 1.5], [1.0, np.nan], [1.0, 1.5], [1.0, 1.5]]))
        self.assertEqual(result["status"], "rejected_nonfinite_frequency")
        self.assertEqual(band.calculator.calculate_calls, [])

    def test_calculator_runtime_failure_is_serialized_as_rejection(self):
        band = FakeBand(np.tile([[1.0, 1.5]], (4, 1)))
        band.calculator.calculate = Mock(side_effect=ValueError("field shape mismatch"))
        config = FakeConfig(band)
        with patch.object(diagnostic, "finite_patch_preview", return_value=self.pattern):
            result = diagnostic.run_supercell_berry_diagnostic(config, field=self.field)
        self.assertEqual(result["status"], "rejected_berry_calculation")
        self.assertIsNone(result["berry_curvature"])
        self.assertIn("field shape mismatch", result["berry_error"])

    def test_result_is_serialization_safe(self):
        result, _, _, _ = self.run_case(np.tile([[1.0, 1.5]], (4, 1)))
        encoded = json.dumps(result, sort_keys=True)
        self.assertIn("accepted_isolated_single_band", encoded)
        self.assertNotIn("FakeBand", encoded)
        self.assertNotIn("FakeCalculator", encoded)
        self.assertNotIn("FakeField", encoded)


if __name__ == "__main__":
    unittest.main()
