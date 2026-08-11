from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config

from mephc.plotting import plot_band_path
from mephc.preview import preview_mpb_dielectric, preview_pattern
from mephc.records import (
    canonical_record_path,
    data_dir,
    find_matching_record,
    load_record,
    make_image_path,
    make_record,
    make_record_name,
    save_record,
    tmp_dir,
    update_archive_manifest,
)


# MPB pixels per normalized lattice period. Higher resolution improves spatial
# accuracy but increases memory use and runtime. With strict cache matching,
# changing this value forces a new calculation.
resolution = 64
# Number of eigenfrequency bands to calculate. It is part of the record task
# identity, so records with a different number of bands are never reused.
num_bands = 3
# Number of intervals on each Gamma-K, K-M, and M-Gamma segment.
# The complete path contains 3*n_per_segment + 1 sampled k-points.
n_per_segment = 10

# Preview settings do not participate in geometry IDs, record matching, or
# formal band data. NumPy preview shows the polygons returned by
# create_unitcell(). MPB preview runs a separate inexpensive solve and shows
# both raw and rectified dielectric maps.
preview_numpy = 0
preview_mpb = True
# Resolution used only by the optional MPB dielectric preview.
preview_resolution = 32
# True continues to record lookup/formal calculation after preview. Set False
# to inspect geometry only: no band record is read or written and no band PNG
# is generated.
run_calculation = True

# True calculates Berry curvature at every Gamma-K-M-Gamma path point and
# colors the band markers with those values. This is substantially slower than
# an ordinary band solve because every path point also needs Berry plaquettes.
# Set False for a frequency-only band structure.
color_by_berry = True
# Side length of each Berry plaquette in Cartesian reciprocal coordinates.
# Smaller values probe a more local curvature but can become less numerically
# stable; changing this value creates a different Berry-band record.
berry_step = 0.0005
# Number of bands requested when Berry coloring is enabled. Keep this at least
# as large as every band that should appear in the plot.
berry_num_bands = num_bands

# Record-selection modes:
#   "auto"      -> normal use: reuse a matching canonical record, otherwise
#                  calculate and write one.
#   "compute"   -> force a fresh solve and overwrite the canonical record.
#                  Use after changing solver code or when refreshing results.
#   "plot_only" -> never run MPB; require an existing matching record and only
#                  regenerate the plot. Missing records raise FileNotFoundError.
run_mode = "auto"
# False keeps only the stable canonical working record. True also writes a
# timestamped copy, useful when you need to preserve calculation history.
archive_record = False
# None uses run_mode and automatic matching. Set this to a specific existing
# .pkl file to load that exact record before any run_mode matching. This is
# useful for inspecting or replotting historical data; the explicit file is
# trusted and is not validated against the current config.
record_path = None
# True (recommended) requires resolution, polarization, and full geometry
# metadata to match before auto reuse. False ignores compute metadata and may
# reuse a canonical record produced at a different resolution; use only when
# intentionally accepting the older record's numerical accuracy.
reuse_requires_compute_match = True

# Everything below affects only rendering. Plot changes never trigger MPB and
# never affect record matching.
plot_params = {
    # save=True writes/overwrites the PNG derived from the record filename.
    "save": True,
    # show=True opens the band figure interactively after it is rendered.
    "show": False,
    # True plots THz; False plots normalized MPB frequency.
    "use_actual": True,
    "figsize": (6.0, 5.0),
    # Output raster resolution; affects PNG quality and size, not MPB accuracy.
    "dpi": 140,
    "title": "TriLatt band structure",
    # None preserves only the Gamma/K/M high-symmetry tick labels.
    "xlabel": None,
    "ylabel": "Frequency (THz)",
    "font_size": 11,
    "tick_size": 10,
    "grid": True,
    "grid_kwargs": {"axis": "y", "linestyle": ":", "linewidth": 0.5, "alpha": 0.75},
    "legend": True,
    "legend_kwargs": {"fontsize": 9, "loc": "best"},
    # line and scatter are independent; True/True overlays markers on lines.
    "line": True,
    "scatter": True,
    "linewidth": 1.5,
    # Matplotlib scatter marker area in points squared.
    "markersize": 18,
    "scatter_edgecolor": "black",
    "scatter_linewidth": 0.5,
    # A record must contain bcs before it can be colored. Keeping this switch
    # tied to the simulation switch above is the normal workflow.
    "color_by_berry": color_by_berry,
    "bc_cmap": "RdBu_r",
    # None lets Matplotlib choose limits from the record. Set both values when
    # comparing multiple geometries with one fixed Berry color scale.
    "bc_vmin": -20,
    "bc_vmax": 20,
    "bc_label": "Berry curvature",
    "colorbar": color_by_berry,
    "colorbar_kwargs": {"fraction": 0.046, "pad": 0.04},
}


def preview_unit_cell(
    config_module=config,
    *,
    resolution,
    numpy_preview=True,
    mpb_preview=True,
    show=True,
    preview_num_bands=1,
):
    """Preview geometry without reading or writing band records.

    The NumPy outline is taken from the MePhC pattern object's own lattice
    construction data; TriLatt does not duplicate unit-cell or sublattice
    coordinates. The MPB preview shows what the solver actually discretizes.
    """
    band = config_module.make_band(resolution=resolution)
    pattern = config_module.build_pattern()
    figures = {}
    if numpy_preview:
        outer = getattr(pattern, "outer_instance", None)
        outline = getattr(outer, "outline", None)
        figures["numpy"] = preview_pattern(
            pattern,
            outline=outline,
            show=show,
        )
    if mpb_preview:
        figures["mpb"] = preview_mpb_dielectric(
            band,
            pattern,
            num_bands=preview_num_bands,
            k_point=(0.0, 0.0),
            show=show,
        )
    return figures


def _resolve_existing_record(
    *,
    geometry_id,
    task_params,
    compute_params,
    run_mode,
    record_path,
    reuse_requires_compute_match,
):
    if record_path is not None:
        path = Path(record_path)
        return load_record(path), path
    if run_mode not in {"auto", "compute", "plot_only"}:
        raise ValueError("run_mode must be 'auto', 'compute', or 'plot_only'.")
    if run_mode in {"auto", "plot_only"}:
        record, path = find_matching_record(
            project_root,
            geometry_id,
            "band",
            task_params=task_params,
            compute_params=compute_params,
            require_compute_match=reuse_requires_compute_match,
        )
        if record is not None:
            return record, path
        if run_mode == "plot_only":
            expected = canonical_record_path(project_root, geometry_id, "band", task_params)
            raise FileNotFoundError(f"No matching band record found. Expected canonical path: {expected}")
    return None, None


def compute_band_structure(
    config_module=config,
    *,
    resolution,
    num_bands,
    n_per_segment,
    compute_bc=False,
    berry_step=0.0005,
    run_mode="auto",
    archive=False,
    reuse_requires_compute_match=True,
    record_path=None,
    save=True,
    save_tmp=True,
    source_case=None,
):
    """Load or compute a Gamma-K-M-Gamma band record."""
    if record_path is not None:
        path = Path(record_path)
        return load_record(path), path, None
    if resolution < 1:
        raise ValueError("resolution must be >= 1.")
    if num_bands < 1:
        raise ValueError("num_bands must be >= 1.")
    if n_per_segment < 1:
        raise ValueError("n_per_segment must be >= 1.")
    if compute_bc and berry_step <= 0:
        raise ValueError("berry_step must be positive when compute_bc=True.")

    geometry_id = config_module.geometry_id()
    task_params = {
        "num_bands": int(num_bands),
        "path": "gkm",
        "n_per_segment": int(n_per_segment),
        "compute_bc": bool(compute_bc),
        "berry_step": float(berry_step) if compute_bc else None,
    }
    compute_params = {
        "resolution": int(resolution),
        "polarization": "TE",
        "geometry": config_module.geometry_parameters(),
    }
    record, path = _resolve_existing_record(
        geometry_id=geometry_id,
        task_params=task_params,
        compute_params=compute_params,
        run_mode=run_mode,
        record_path=None,
        reuse_requires_compute_match=reuse_requires_compute_match,
    )
    if record is not None:
        return record, path, None

    band = config_module.make_band(resolution=resolution)
    result = band.compute_band_path_with_berry(
        config_module.build_pattern(),
        path=config_module.band_path(),
        n_per_segment=n_per_segment,
        step=berry_step,
        num_bands=num_bands,
        compute_bc=compute_bc,
    )
    record = make_record(
        "band",
        geometry_id,
        task_params=task_params,
        compute_params=compute_params,
        data=result,
        source_case=source_case,
    )
    canonical_path = canonical_record_path(project_root, geometry_id, "band", task_params)
    latest_path = tmp_dir(project_root) / "band_latest.pkl"
    if save:
        save_record(record, canonical_path)
        update_archive_manifest(PROJECT_ROOT, canonical_path, record)
    if archive:
        archive_name = make_record_name(
            "band",
            num_bands=num_bands,
            path="gkm",
            step=berry_step if compute_bc else None,
            created_at=record["created_at"],
        )
        archive_path = data_dir(project_root, geometry_id) / archive_name
        save_record(record, archive_path)
        update_archive_manifest(project_root, archive_path, record)
    if save_tmp:
        save_record(record, latest_path)
    return record, canonical_path, latest_path if save_tmp else None


def plot_band_record(
    record_or_path,
    *,
    show=False,
    save=True,
    use_actual=True,
    image_path=None,
    plot_params=None,
):
    """Render an existing band record without running MPB."""
    record_path_value = None
    if isinstance(record_or_path, (str, Path)):
        record_path_value = Path(record_or_path)
        record = load_record(record_path_value)
    else:
        record = record_or_path

    params = dict(plot_params or {})
    show = params.pop("show", show)
    save = params.pop("save", save)
    use_actual = params.pop("use_actual", use_actual)
    color_by_berry_value = params.pop("color_by_berry", False)
    if color_by_berry_value:
        bcs = record["data"].get("bcs")
        if bcs is None:
            raise ValueError(
                "This band record has no Berry curvature data. "
                "Recompute with color_by_berry=True."
            )
        params.setdefault("bc_values", bcs)
    if image_path is None and save:
        if record_path_value is None:
            raise ValueError("image_path is required when plotting an in-memory record with save=True.")
        image_path = make_image_path(project_root, record_path_value, record["geometry_id"])
    fig, ax = plot_band_path(
        record["data"],
        use_actual=use_actual,
        save_path=image_path if save else None,
        show=show,
        **params,
    )
    return fig, ax, image_path


def main():
    """Preview geometry, then optionally obtain and plot a band record."""
    if preview_numpy or preview_mpb:
        preview_unit_cell(
            config,
            resolution=preview_resolution,
            numpy_preview=preview_numpy,
            mpb_preview=preview_mpb,
            show=True,
        )

    if not run_calculation:
        print("preview complete; run_calculation is False")
        return None, None, None

    compute_num_bands = berry_num_bands if color_by_berry else num_bands
    record, output_record_path, latest_path = compute_band_structure(
        config,
        resolution=resolution,
        num_bands=compute_num_bands,
        n_per_segment=n_per_segment,
        compute_bc=color_by_berry,
        berry_step=berry_step,
        run_mode=run_mode,
        archive=archive_record,
        reuse_requires_compute_match=reuse_requires_compute_match,
        record_path=record_path,
        source_case=str(project_root),
    )
    image_path = None
    if plot_params.get("save", True) or plot_params.get("show", False):
        _, _, image_path = plot_band_record(output_record_path, plot_params=plot_params)
    print("geometry:", config.geometry_id())
    print("record:", output_record_path)
    print("tmp record:", latest_path)
    print("image:", image_path)
    print("normalized frequency shape:", record["data"]["freqs"].shape)
    print("actual frequency shape:", record["data"]["actual_freqs"].shape)
    if record["data"].get("bcs") is not None:
        print("berry curvature shape:", record["data"]["bcs"].shape)
    return record, output_record_path, image_path


if __name__ == "__main__":
    main()
