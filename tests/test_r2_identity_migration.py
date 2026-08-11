"""Cross-consumer identity checks for the TriLatt central lattice factory."""

from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = load_module(ROOT / "config.py", "r2_trilatt_config")
workflow = load_module(ROOT / "workflow.py", "r2_trilatt_workflow")


class TriLattIdentityMigrationTests(unittest.TestCase):
    def test_factory_is_consumed_by_band_pattern_and_kspace(self):
        canonical = config.canonical_lattice()
        band = config.make_band(resolution=1)
        self.assertTrue(np.allclose(band.lattice_model.direct_basis, canonical.direct_basis))
        pattern = config.build_pattern()
        self.assertTrue(np.allclose(pattern.outer_instance.lattice_model.direct_basis, canonical.direct_basis))
        kspace, mode, points = workflow.hbz_sampling(
            config, grid_n=4, shrinking=0.01, symmetry_mode="auto"
        )
        self.assertEqual(mode, "c3")
        self.assertTrue(np.allclose(kspace.lattice_model.direct_basis, canonical.direct_basis))
        self.assertGreater(len(points), 0)


if __name__ == "__main__":
    unittest.main()
