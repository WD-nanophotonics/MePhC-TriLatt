from contextlib import contextmanager
from pathlib import Path
import importlib.util
import pickle
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = load_module("trilatt_config", PROJECT_ROOT / "config.py")
k_runner = load_module("trilatt_k_runner", PROJECT_ROOT / "frequency_at_k.py")
band_runner = load_module("trilatt_band_runner", PROJECT_ROOT / "band_structure.py")
workflow = load_module("trilatt_workflow", PROJECT_ROOT / "workflow.py")
berry_runner = load_module("trilatt_berry_runner", PROJECT_ROOT / "berry_curvature.py")
efs_runner = load_module("trilatt_efs_runner", PROJECT_ROOT / "efs.py")


@contextmanager
def geometry_values(**updates):
    original = {name: getattr(config, name) for name in updates}
    try:
        for name, value in updates.items():
            setattr(config, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(config, name, value)


class GeometryRuleTests(unittest.TestCase):
    def test_config_has_no_shape_or_lattice_switch(self):
        self.assertFalse(hasattr(config, "hole_shape"))
        self.assertFalse(hasattr(config, "lattice_type"))

    def test_r2_none_calls_single_polygon_signature(self):
        fake_band = Mock()
        fake_band.create_unitcell.return_value = object()
        with geometry_values(r2=None, n1=16, theta1=25, n2=7, theta2=44):
            with patch.object(config, "make_band", return_value=fake_band):
                result = config.build_pattern()
        self.assertIs(result, fake_band.create_unitcell.return_value)
        fake_band.create_unitcell.assert_called_once_with(16, 25, show=False)

    def test_r2_value_calls_honeycomb_signature(self):
        fake_band = Mock()
        fake_band.create_unitcell.return_value = object()
        with geometry_values(r2=110, n1=3, theta1=0, n2=16, theta2=60):
            with patch.object(config, "make_band", return_value=fake_band):
                result = config.build_pattern()
        self.assertIs(result, fake_band.create_unitcell.return_value)
        fake_band.create_unitcell.assert_called_once_with(3, 0, 16, 60, show=False)

    def test_real_patterns_follow_n_and_r2(self):
        from mephc.patterns import pattern_summary

        with geometry_values(r2=None, n1=16, theta1=12):
            single = pattern_summary(config.build_pattern())
        self.assertEqual(single, {"layers": 1, "polygons": 1, "vertices": 16})

        with geometry_values(r2=110, n1=3, theta1=0, n2=16, theta2=60):
            double = pattern_summary(config.build_pattern())
        self.assertEqual(double, {"layers": 2, "polygons": 2, "vertices": 19})

    def test_second_polygon_parameters_required_only_when_r2_has_value(self):
        with geometry_values(r2=None, n2=None, theta2=None):
            config.validate_geometry()
        with geometry_values(r2=110, n2=None, theta2=60):
            with self.assertRaisesRegex(ValueError, "n2 and theta2"):
                config.validate_geometry()
        with geometry_values(r2=110, n2=3, theta2=None):
            with self.assertRaisesRegex(ValueError, "n2 and theta2"):
                config.validate_geometry()

    def test_geometry_id_uses_only_active_parameters(self):
        with geometry_values(r1=120, r2=None, n1=3, theta1=0, n2=16, theta2=99):
            single = config.geometry_id()
        self.assertEqual(single, "a400_r120_n3_t0_neff2p7_h100")

        with geometry_values(r1=120, r2=110, n1=3, theta1=0, n2=3, theta2=60):
            double = config.geometry_id()
        self.assertEqual(double, "a400_r120-110_n3-3_t0-60_neff2p7_h100")
        self.assertNotIn("shape", double)
        self.assertNotIn("lattice", double)


class PreviewTests(unittest.TestCase):
    def test_numpy_preview_uses_pattern_outline_and_writes_no_record(self):
        expected_pattern = config.build_pattern()
        expected_outline = expected_pattern.outer_instance.outline
        before = list(PROJECT_ROOT.glob("data/**/*.pkl"))

        with patch.object(band_runner, "preview_pattern", return_value=("figure", "axes")) as preview:
            figures = band_runner.preview_unit_cell(
                config,
                resolution=4,
                numpy_preview=True,
                mpb_preview=False,
                show=False,
            )

        self.assertEqual(figures["numpy"], ("figure", "axes"))
        self.assertEqual(preview.call_args.kwargs["outline"], expected_outline)
        self.assertFalse(preview.call_args.kwargs["show"])
        self.assertEqual(before, list(PROJECT_ROOT.glob("data/**/*.pkl")))

    def test_low_resolution_mpb_preview_returns_raw_and_rectified_figures(self):
        before = list(PROJECT_ROOT.glob("data/**/*.pkl"))
        figures = band_runner.preview_unit_cell(
            config,
            resolution=4,
            numpy_preview=False,
            mpb_preview=True,
            show=False,
        )
        self.assertIn("mpb", figures)
        self.assertEqual(len(figures["mpb"]), 2)
        self.assertEqual(before, list(PROJECT_ROOT.glob("data/**/*.pkl")))

    def test_preview_only_main_never_looks_up_or_computes_band_record(self):
        with (
            patch.object(band_runner, "preview_numpy", True),
            patch.object(band_runner, "preview_mpb", False),
            patch.object(band_runner, "run_calculation", False),
            patch.object(band_runner, "preview_unit_cell") as preview,
            patch.object(band_runner, "compute_band_structure") as compute,
            patch.object(band_runner, "plot_band_record") as plot,
        ):
            result = band_runner.main()
        self.assertEqual(result, (None, None, None))
        preview.assert_called_once()
        compute.assert_not_called()
        plot.assert_not_called()


class RecordControlTests(unittest.TestCase):
    def _resolve(self, **overrides):
        parameters = {
            "geometry_id": config.geometry_id(),
            "task_params": {"num_bands": 3, "path": "gkm", "n_per_segment": 10},
            "compute_params": {
                "resolution": 64,
                "polarization": "TE",
                "geometry": config.geometry_parameters(),
            },
            "run_mode": "auto",
            "record_path": None,
            "reuse_requires_compute_match": True,
        }
        parameters.update(overrides)
        return band_runner._resolve_existing_record(**parameters)

    def test_auto_and_compute_modes_have_distinct_lookup_behavior(self):
        expected_record = {"kind": "band"}
        expected_path = Path("/tmp/existing.pkl")
        with patch.object(
            band_runner,
            "find_matching_record",
            return_value=(expected_record, expected_path),
        ) as find:
            self.assertEqual(self._resolve(run_mode="auto"), (expected_record, expected_path))
            find.assert_called_once()

        with patch.object(band_runner, "find_matching_record") as find:
            self.assertEqual(self._resolve(run_mode="compute"), (None, None))
            find.assert_not_called()

    def test_plot_only_missing_record_raises_without_computing(self):
        with patch.object(band_runner, "find_matching_record", return_value=(None, None)):
            with self.assertRaises(FileNotFoundError):
                self._resolve(run_mode="plot_only")

    def test_explicit_record_path_bypasses_automatic_matching(self):
        expected = {"kind": "historical", "data": 123}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical.pkl"
            with path.open("wb") as file:
                pickle.dump(expected, file)
            with patch.object(band_runner, "find_matching_record") as find:
                record, loaded_path = self._resolve(
                    run_mode="compute",
                    record_path=path,
                )
        self.assertEqual(record, expected)
        self.assertEqual(loaded_path, path)
        find.assert_not_called()

    def test_compute_match_switch_is_forwarded_to_record_lookup(self):
        with patch.object(band_runner, "find_matching_record", return_value=(None, None)) as find:
            self._resolve(reuse_requires_compute_match=False)
        self.assertFalse(find.call_args.kwargs["require_compute_match"])

    def test_archive_writes_timestamped_copy_in_addition_to_selected_outputs(self):
        fake_band = Mock()
        fake_band.compute_band_path_with_berry.return_value = {
            "freqs": np.zeros((4, 1)),
            "actual_freqs": np.zeros((4, 1)),
            "distances": np.arange(4),
            "labels": ("Gamma", "K", "M", "Gamma"),
        }
        with (
            patch.object(config, "make_band", return_value=fake_band),
            patch.object(config, "build_pattern", return_value=object()),
            patch.object(band_runner, "save_record") as save,
        ):
            band_runner.compute_band_structure(
                config,
                resolution=4,
                num_bands=1,
                n_per_segment=1,
                run_mode="compute",
                archive=True,
                save=False,
                save_tmp=False,
            )
        save.assert_called_once()
        self.assertRegex(save.call_args.args[1].name, r"^band_nb1_gkm_\d{8}-\d{6}\.pkl$")


class BerryBandTests(unittest.TestCase):
    def _fake_band_result(self, compute_bc):
        return {
            "k_points": np.zeros((4, 2)),
            "freqs": np.ones((4, 2)),
            "actual_freqs": np.ones((4, 2)) * 100,
            "bcs": np.ones((4, 2)) if compute_bc else None,
            "distances": np.arange(4),
            "tick_indices": np.arange(4),
            "tick_positions": np.arange(4),
            "labels": ("Gamma", "K", "M", "Gamma"),
        }

    def test_band_berry_switch_changes_task_identity_and_solver_call(self):
        fake_band = Mock()
        fake_band.compute_band_path_with_berry.side_effect = (
            lambda *args, **kwargs: self._fake_band_result(kwargs["compute_bc"])
        )
        with (
            patch.object(config, "make_band", return_value=fake_band),
            patch.object(config, "build_pattern", return_value=object()),
        ):
            ordinary, _, _ = band_runner.compute_band_structure(
                config,
                resolution=4,
                num_bands=2,
                n_per_segment=1,
                compute_bc=False,
                run_mode="compute",
                save=False,
                save_tmp=False,
            )
            colored, _, _ = band_runner.compute_band_structure(
                config,
                resolution=4,
                num_bands=2,
                n_per_segment=1,
                compute_bc=True,
                berry_step=0.002,
                run_mode="compute",
                save=False,
                save_tmp=False,
            )

        self.assertFalse(ordinary["task_params"]["compute_bc"])
        self.assertIsNone(ordinary["task_params"]["berry_step"])
        self.assertIsNone(ordinary["data"]["bcs"])
        self.assertTrue(colored["task_params"]["compute_bc"])
        self.assertEqual(colored["task_params"]["berry_step"], 0.002)
        self.assertEqual(colored["data"]["bcs"].shape, colored["data"]["freqs"].shape)
        self.assertFalse(fake_band.compute_band_path_with_berry.call_args_list[0].kwargs["compute_bc"])
        self.assertTrue(fake_band.compute_band_path_with_berry.call_args_list[1].kwargs["compute_bc"])

    def test_berry_coloring_rejects_frequency_only_record(self):
        record = {
            "geometry_id": config.geometry_id(),
            "data": self._fake_band_result(False),
        }
        with self.assertRaisesRegex(ValueError, "no Berry curvature"):
            band_runner.plot_band_record(
                record,
                save=False,
                plot_params={"color_by_berry": True},
            )


class TriangularHBZTests(unittest.TestCase):
    def test_auto_symmetry_uses_c3_only_for_exact_polygon_symmetry(self):
        with geometry_values(r2=None, n1=3):
            self.assertEqual(workflow.resolve_symmetry_mode(config, "auto"), "c3")
        with geometry_values(r2=72, n1=3, n2=6):
            self.assertEqual(workflow.resolve_symmetry_mode(config, "auto"), "c3")
        with geometry_values(r2=72, n1=16, n2=16):
            self.assertEqual(workflow.resolve_symmetry_mode(config, "auto"), "raw_hbz")
            with self.assertRaisesRegex(ValueError, "requires every active polygon"):
                workflow.resolve_symmetry_mode(config, "c3")

    def test_final_samples_stay_inside_k_centered_hbz(self):
        from shapely.geometry import Point, Polygon

        with geometry_values(r2=None, n1=3):
            kspace, mode, raw_points = workflow.hbz_sampling(
                config,
                grid_n=6,
                shrinking=0.01,
                symmetry_mode="auto",
            )
            final_points, _ = workflow.c3_expand_arrays(
                kspace,
                raw_points,
                np.arange(len(raw_points), dtype=float),
            )
        self.assertEqual(mode, "c3")
        hbz = Polygon(kspace.shrunken_hbz_poly).buffer(1e-9)
        self.assertTrue(all(hbz.contains(Point(point)) for point in final_points))

        with geometry_values(r2=72, n1=16, n2=16):
            kspace, mode, final_points = workflow.hbz_sampling(
                config,
                grid_n=6,
                shrinking=0.01,
                symmetry_mode="auto",
            )
        self.assertEqual(mode, "raw_hbz")
        hbz = Polygon(kspace.shrunken_hbz_poly).buffer(1e-9)
        self.assertTrue(all(hbz.contains(Point(point)) for point in final_points))

    def test_berry_record_keeps_raw_and_expanded_c3_arrays(self):
        fake_band = Mock()

        def fake_compute(_pattern, points, step, num_bands, band_index):
            values = np.arange(len(points) * num_bands, dtype=float).reshape(len(points), num_bands)
            return {"k_points": np.asarray(points), "bcs": values}

        fake_band.compute_berry_grid.side_effect = fake_compute
        with (
            geometry_values(r2=None, n1=3),
            patch.object(config, "make_band", return_value=fake_band),
            patch.object(config, "build_pattern", return_value=object()),
        ):
            record, _, _ = berry_runner.compute_berry_curvature(
                config,
                resolution=4,
                num_bands=2,
                grid_n=6,
                shrinking=0.01,
                step=0.01,
                symmetry_mode="auto",
                run_mode="compute",
                save=False,
                save_tmp=False,
            )
        self.assertEqual(record["task_params"]["domain"], "k_centered_hbz")
        self.assertEqual(record["task_params"]["symmetry_policy"], "auto")
        self.assertEqual(record["task_params"]["symmetry"], "c3")
        self.assertGreater(
            len(record["data"]["k_points"]),
            len(record["data"]["raw_k_points"]),
        )
        self.assertEqual(
            record["data"]["bcs"].shape[1],
            record["data"]["raw_bcs"].shape[1],
        )

    def test_efs_record_keeps_normalized_thz_and_raw_arrays(self):
        from mephc.efs import EFSResult

        fake_band = Mock()

        def fake_compute(_pattern, points, num_bands):
            freqs = np.full((len(points), num_bands), 0.25)
            return EFSResult(
                k_points=points,
                freqs=freqs,
                actual_freqs=freqs * 299792.458 / config.a,
                metadata={},
            )

        fake_band.compute_efs.side_effect = fake_compute
        with (
            geometry_values(r2=72, n1=16, n2=16),
            patch.object(config, "make_band", return_value=fake_band),
            patch.object(config, "build_pattern", return_value=object()),
        ):
            record, _, _ = efs_runner.compute_efs(
                config,
                resolution=4,
                num_bands=2,
                grid_n=6,
                shrinking=0.01,
                band_index=0,
                symmetry_mode="auto",
                run_mode="compute",
                save=False,
                save_tmp=False,
            )
        result = record["data"]
        self.assertEqual(record["task_params"]["symmetry"], "raw_hbz")
        self.assertEqual(result.metadata["symmetry"], "raw_hbz")
        self.assertEqual(result.freqs.shape, result.actual_freqs.shape)
        self.assertEqual(result.metadata["raw_freqs"].shape, result.freqs.shape)
        np.testing.assert_allclose(
            result.actual_freqs,
            result.freqs * 299792.458 / config.a,
        )


class MPBSmokeTests(unittest.TestCase):
    def test_k_frequency_and_band_path_agree(self):
        k_result = k_runner.compute_k_frequencies(config, resolution=4, num_bands=1)
        self.assertEqual(k_result["freqs"].shape, (1,))
        self.assertTrue(np.all(np.isfinite(k_result["freqs"])))
        self.assertTrue(np.all(k_result["freqs"] > 0))
        np.testing.assert_allclose(
            k_result["actual_freqs"],
            k_result["freqs"] * 299792.458 / config.a,
            rtol=1e-12,
            atol=0.0,
        )

        record, _, latest_path = band_runner.compute_band_structure(
            config,
            resolution=4,
            num_bands=1,
            n_per_segment=1,
            run_mode="compute",
            save=False,
            save_tmp=False,
            source_case="test",
        )
        self.assertIsNone(latest_path)
        self.assertEqual(record["data"]["freqs"].shape, (4, 1))
        self.assertEqual(record["data"]["labels"], ("Gamma", "K", "M", "Gamma"))
        self.assertEqual(record["compute_params"]["geometry"], config.geometry_parameters())
        np.testing.assert_allclose(
            record["data"]["freqs"][1],
            k_result["freqs"],
            rtol=1e-7,
            atol=1e-10,
        )

    def test_low_resolution_real_berry_and_efs_are_finite(self):
        berry_record, _, _ = berry_runner.compute_berry_curvature(
            config,
            resolution=4,
            num_bands=1,
            grid_n=2,
            shrinking=0.01,
            step=0.01,
            band_index=None,
            symmetry_mode="auto",
            run_mode="compute",
            save=False,
            save_tmp=False,
            source_case="test",
        )
        efs_record, _, _ = efs_runner.compute_efs(
            config,
            resolution=4,
            num_bands=1,
            grid_n=2,
            shrinking=0.01,
            band_index=0,
            symmetry_mode="auto",
            run_mode="compute",
            save=False,
            save_tmp=False,
            source_case="test",
        )
        self.assertEqual(berry_record["data"]["symmetry"], "c3")
        self.assertTrue(np.all(np.isfinite(berry_record["data"]["bcs"])))
        self.assertTrue(np.all(np.isfinite(efs_record["data"].freqs)))
        np.testing.assert_allclose(
            efs_record["data"].actual_freqs,
            efs_record["data"].freqs * 299792.458 / config.a,
            rtol=1e-12,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
