"""Regression tests for the bounded R6 periodic-supercell adapter."""

import unittest
from types import SimpleNamespace

import numpy as np

from mephc.bravais import BravaisLattice2D
from mephc.deformation import AnalyticDeformationField, PeriodicityError, PeriodicSupercellField

from r5_deformation import build_supercell_solver


class FakeBand:
    def __init__(self):
        self.calls = []

    def run_supercell(self, pattern, field, **kwargs):
        self.calls.append({"pattern": pattern, "field": field, **kwargs})
        return {"solver": "fake-r6"}


class FakeConfig:
    def __init__(self):
        self.lattice = BravaisLattice2D.triangular()
        self.band = None
        self.make_band_calls = []

    def canonical_lattice(self):
        return self.lattice

    def build_pattern(self):
        return SimpleNamespace(
            pattern=np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]], dtype=float)
        )

    def make_band(self, *, resolution):
        self.make_band_calls.append(resolution)
        self.band = FakeBand()
        return self.band


def zero_field():
    return AnalyticDeformationField(
        lambda points: np.zeros_like(points),
        stable_id="r6-regression-zero",
    )


def periodic_field(matrix=((2, 0), (0, 3)), *, verify=True):
    return PeriodicSupercellField(
        zero_field(),
        BravaisLattice2D.triangular(),
        replication_matrix=matrix,
        verify=verify,
    )


class TriLattR6Tests(unittest.TestCase):
    def test_rejects_non_periodic_supercell_field(self):
        config = FakeConfig()
        with self.assertRaises(TypeError):
            build_supercell_solver(
                config,
                zero_field(),
                q_points=((0.0, 0.0),),
                resolution=32,
            )
        self.assertEqual(config.make_band_calls, [])

    def test_rejects_unverified_periodic_field(self):
        config = FakeConfig()
        field = periodic_field(verify=False)
        with self.assertRaises(PeriodicityError):
            build_supercell_solver(
                config,
                field,
                q_points=((0.0, 0.0),),
                resolution=32,
            )
        self.assertEqual(config.make_band_calls, [])

    def test_rejects_non_diagonal_replication(self):
        config = FakeConfig()
        field = periodic_field(matrix=((2, 1), (0, 3)))
        with self.assertRaises(ValueError):
            build_supercell_solver(
                config,
                field,
                q_points=((0.0, 0.0),),
                resolution=32,
            )
        self.assertEqual(config.make_band_calls, [])

    def test_verified_diagonal_field_runs_exact_r6_contract(self):
        config = FakeConfig()
        field = periodic_field(matrix=((2, 0), (0, 3)))
        q_points = ((0.0, 0.0), (0.125, 0.25))

        solver, metadata = build_supercell_solver(
            config,
            field,
            q_points=q_points,
            resolution=37,
            num_bands=4,
        )

        self.assertEqual(solver, {"solver": "fake-r6"})
        self.assertEqual(config.make_band_calls, [37])
        self.assertIsNotNone(config.band)
        self.assertEqual(len(config.band.calls), 1)
        call = config.band.calls[0]
        self.assertIs(call["q_points"], q_points)
        self.assertIs(call["field"], field)
        self.assertEqual(call["num_bands"], 4)
        self.assertEqual(call["resolution"], 37)
        self.assertEqual(call["polarization"], "TE")
        self.assertIs(metadata["band"], config.band)
        self.assertIs(metadata["field"], field)
        self.assertIs(metadata["pattern"], call["pattern"])
        self.assertEqual(metadata["replication"], [2, 3])


if __name__ == "__main__":
    unittest.main()
