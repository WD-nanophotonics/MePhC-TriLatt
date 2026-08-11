from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np

import config
from workflow import (
    HBZ_DOMAIN,
    c3_expand_arrays,
    compute_parameters,
    hbz_sampling,
    resolve_existing_record,
)

from mephc.efs import EFSResult, plot_efs
from mephc.kspace import TriangularKSpace
from mephc.records import (
    canonical_record_path,
    data_dir,
    load_record,
    make_image_path,
    make_record,
    make_record_name,
    save_record,
    tmp_dir,
    update_archive_manifest,
)


# MPB spatial resolution for every HBZ k-point. Increasing it improves the
# dielectric discretization and raises runtime/memory cost.
resolution = 64
# Number of frequency bands calculated at each reciprocal-space point.
num_bands = 3
# Triangular reciprocal-grid density. At N=24 the C3 mini-space has about 128
# points and the directly sampled shrunken HBZ about 409 points.
grid_n = 24
# Inward HBZ boundary offset used to avoid numerically delicate edge points.
shrinking = 0.01
# Python 0-based band that defines this contour task and is plotted by default.
band_index = 0
# "auto" uses C3 reduction only for exact C3 polygon geometries. Current
# n1=n2=16 therefore selects raw_hbz. "c3" rejects non-C3 side counts;
# "raw_hbz" always calculates the complete K-centered HBZ directly.
symmetry_mode = "auto"

# "auto": reuse a matching record or calculate when absent.
# "compute": force recalculation and overwrite the canonical working record.
# "plot_only": never run MPB and fail clearly when no matching record exists.
run_mode = "auto"
# True writes a timestamped historical copy in addition to the canonical file.
archive_record = False
# Optional exact .pkl to load before run_mode processing. It is trusted as-is.
record_path = None
# True requires resolution, TE polarization, and all geometry metadata to
# match. False intentionally accepts a canonical record from another accuracy.
reuse_requires_compute_match = True

# Plot-only settings below never affect cache matching or calculated arrays.
plot_params = {
    "save": True,
    "show": False,
    # True contours THz values; False contours normalized MPB frequency.
    "use_actual": True,
    "band_index": band_index,
    "figsize": (5.4, 4.8),
    "dpi": 140,
    "title": "TriLatt EFS",
    "xlabel": "kx",
    "ylabel": "ky",
    "font_size": 11,
    "tick_size": 10,
    "grid": True,
    "grid_kwargs": {"linestyle": ":", "linewidth": 0.45, "alpha": 0.7},
    # Interpolated contour mesh resolution and SciPy interpolation method.
    "mesh_size": 120,
    "interpolation": "linear",
    # An integer requests that many automatically spaced contours. A list or
    # NumPy array requests exact normalized/THz levels.
    "levels": 8,
    "cmap": "viridis",
    "linewidth": 1.2,
    "colorbar": True,
    "colorbar_kwargs": {"fraction": 0.046, "pad": 0.04},
}


def compute_efs(
    config_module=config,
    *,
    resolution,
    num_bands,
    grid_n,
    shrinking,
    band_index,
    symmetry_mode="auto",
    run_mode="auto",
    archive=False,
    reuse_requires_compute_match=True,
    record_path=None,
    save=True,
    save_tmp=True,
    source_case=None,
):
    """Load or calculate normalized and THz frequencies on the K-centered HBZ."""
    if record_path is not None:
        path = Path(record_path)
        return load_record(path), path, None
    if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution < 1:
        raise ValueError("resolution must be an integer >= 1.")
    if not isinstance(num_bands, int) or isinstance(num_bands, bool) or num_bands < 1:
        raise ValueError("num_bands must be an integer >= 1.")
    if (
        not isinstance(band_index, int)
        or isinstance(band_index, bool)
        or band_index < 0
        or band_index >= num_bands
    ):
        raise ValueError(f"band_index must be between 0 and {num_bands - 1}.")

    kspace, actual_mode, raw_k_points = hbz_sampling(
        config_module,
        grid_n=grid_n,
        shrinking=shrinking,
        symmetry_mode=symmetry_mode,
    )
    task_params = {
        "num_bands": int(num_bands),
        "grid_n": int(grid_n),
        "shrinking": float(shrinking),
        "band_index": int(band_index),
        "domain": HBZ_DOMAIN,
        "symmetry_policy": symmetry_mode,
        "symmetry": actual_mode,
    }
    compute_params = compute_parameters(config_module, resolution)
    record, path = resolve_existing_record(
        config_module,
        kind="efs",
        task_params=task_params,
        compute_params=compute_params,
        run_mode=run_mode,
        record_path=None,
        reuse_requires_compute_match=reuse_requires_compute_match,
    )
    if record is not None:
        return record, path, None

    band = config_module.make_band(resolution=resolution)
    raw_result = band.compute_efs(
        config_module.build_pattern(),
        raw_k_points,
        num_bands=num_bands,
    )
    raw_freqs = np.asarray(raw_result.freqs, dtype=float)
    raw_actual_freqs = np.asarray(raw_result.actual_freqs, dtype=float)
    if actual_mode == "c3":
        final_k_points, (final_freqs, final_actual_freqs) = c3_expand_arrays(
            kspace,
            raw_k_points,
            raw_freqs,
            raw_actual_freqs,
        )
    else:
        final_k_points = np.asarray(raw_k_points, dtype=float)
        final_freqs = raw_freqs
        final_actual_freqs = raw_actual_freqs

    metadata = dict(raw_result.metadata)
    metadata.update(
        {
            "raw_k_points": np.asarray(raw_k_points, dtype=float),
            "raw_freqs": raw_freqs,
            "raw_actual_freqs": raw_actual_freqs,
            "grid_n": int(grid_n),
            "shrinking": float(shrinking),
            "domain": HBZ_DOMAIN,
            "symmetry_policy": symmetry_mode,
            "symmetry": actual_mode,
        }
    )
    result = EFSResult(
        k_points=final_k_points,
        freqs=final_freqs,
        actual_freqs=final_actual_freqs,
        metadata=metadata,
    )
    geometry_id = config_module.geometry_id()
    record = make_record(
        "efs",
        geometry_id,
        task_params=task_params,
        compute_params=compute_params,
        data=result,
        source_case=source_case,
    )
    canonical_path = canonical_record_path(project_root, geometry_id, "efs", task_params)
    latest_path = tmp_dir(project_root) / "efs_latest.pkl"
    if save:
        save_record(record, canonical_path)
        update_archive_manifest(PROJECT_ROOT, canonical_path, record)
    if archive:
        archive_name = make_record_name(
            "efs",
            band_index=band_index,
            grid_n=grid_n,
            symmetry=actual_mode,
            created_at=record["created_at"],
        )
        archive_path = data_dir(project_root, geometry_id) / archive_name
        save_record(record, archive_path)
        update_archive_manifest(PROJECT_ROOT, archive_path, record)
    if save_tmp:
        save_record(record, latest_path)
    return record, canonical_path, latest_path if save_tmp else None


def plot_efs_record(
    record_or_path,
    *,
    show=False,
    save=True,
    use_actual=True,
    band_index=None,
    image_path=None,
    plot_params=None,
):
    """Render an EFS contour from an existing record without running MPB."""
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
    band_index = params.pop("band_index", band_index)
    if band_index is None:
        band_index = int(record["task_params"]["band_index"])
    if image_path is None and save:
        if record_path_value is None:
            raise ValueError("image_path is required when saving an in-memory record.")
        image_path = make_image_path(
            project_root,
            record_path_value,
            record["geometry_id"],
        )

    fig, ax = plot_efs(
        record["data"],
        band_index=band_index,
        use_actual=use_actual,
        save_path=None,
        show=False,
        **params,
    )
    kspace = TriangularKSpace(
        N=int(record["task_params"]["grid_n"]),
        shrinking=float(record["task_params"]["shrinking"]),
    )
    outline = np.asarray(kspace.shrunken_hbz_poly, dtype=float)
    outline = np.vstack([outline, outline[0]])
    ax.plot(outline[:, 0], outline[:, 1], color="black", linewidth=1.0)
    fig.tight_layout()
    if save:
        image_path = Path(image_path)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(image_path, dpi=params.get("dpi", 100), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax, image_path


def main():
    """Obtain an EFS record and render its selected band."""
    record, output_record_path, latest_path = compute_efs(
        config,
        resolution=resolution,
        num_bands=num_bands,
        grid_n=grid_n,
        shrinking=shrinking,
        band_index=band_index,
        symmetry_mode=symmetry_mode,
        run_mode=run_mode,
        archive=archive_record,
        reuse_requires_compute_match=reuse_requires_compute_match,
        record_path=record_path,
        source_case=str(project_root),
    )
    image_path = None
    if plot_params.get("save", True) or plot_params.get("show", False):
        _, _, image_path = plot_efs_record(
            output_record_path,
            plot_params=plot_params,
        )
    print("geometry:", config.geometry_id())
    print("symmetry used:", record["data"].metadata["symmetry"])
    print("record:", output_record_path)
    print("tmp record:", latest_path)
    print("image:", image_path)
    print("normalized frequency shape:", record["data"].freqs.shape)
    print("actual frequency shape:", record["data"].actual_freqs.shape)
    return record, output_record_path, image_path


if __name__ == "__main__":
    main()
