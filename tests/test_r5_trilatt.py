"""TriLatt consumes the shared MePhC field kernel for preview geometry."""

import unittest
from types import SimpleNamespace

import numpy as np

from mephc.bravais import BravaisLattice2D
from mephc.deformation import AnalyticDeformationField
from r5_deformation import finite_patch_preview, primitive_guard


class TriLattR5Tests(unittest.TestCase):
    def test_finite_patch_keeps_motif_rigid(self):
        config = SimpleNamespace(canonical_lattice=BravaisLattice2D.triangular)
        pattern = [np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])]
        field = AnalyticDeformationField(lambda p: np.column_stack((0.1 * p[:, 0], np.zeros(len(p)))), stable_id="tri-local")
        patch = finite_patch_preview(config, field, replication=(2, 1), pattern=pattern)
        self.assertEqual(len(patch), 2)
        np.testing.assert_allclose(patch[0][1] - patch[0][0], [0.1, 0.0])
        np.testing.assert_allclose(patch[1][1] - patch[1][0], [0.1, 0.0])

    def test_local_primitive_guard_is_shared(self):
        field = AnalyticDeformationField(lambda p: np.zeros_like(p), stable_id="tri-local")
        with self.assertRaises(RuntimeError):
            primitive_guard(field, "primitive EFS")


if __name__ == "__main__":
    unittest.main()
