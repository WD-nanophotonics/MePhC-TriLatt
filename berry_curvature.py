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

from mephc.kspace import TriangularKSpace
from mephc.workflows import save_record_outputs
from mephc.plotting import plot_scalar_field
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


# MPB spatial discretization used for every Berry plaquette. Higher values are
# more accurate and more expensive. With strict compute matching, changing it
# prevents reuse of a record calculated at another resolution.
resolution = 64
# Number of eigenmodes computed at every k-point.
num_bands = 3
# Triangular reciprocal-grid density. At N=24 the C3 mini-space has about 128
# points and the directly sampled shrunken HBZ about 409 points.
grid_n = 24
# Moves the HBZ boundary slightly inward to avoid delicate boundary plaquettes.
# It changes the sampled data and therefore belongs to the task identity.
shrinking = 0.01
# Berry plaquette side in Cartesian reciprocal coordinates. Smaller is more
# local but may be less numerically stable.
step = 0.0005
# None calculates all num_bands in one record. A Python 0-based integer
# calculates only that one band.
band_index = None
# Python 0-based band selected only for plotting a multi-band record.
plot_band_index = 2
# "auto" uses C3 reduction only when every active n is divisible by 3;
# otherwise it calculates the full K-centered HBZ directly. "c3" requires
# exact C3 geometry. "raw_hbz" always calculates every HBZ sample.
symmetry_mode = "auto"

# "auto": reuse a matching record or compute when missing.
# "compute": force a fresh solve and overwrite the canonical working record.
# "plot_only": never run MPB and require a matching record.
run_mode = "auto"
# True keeps a timestamped historical copy in addition to the canonical file.
archive_record = False
# Optional exact .pkl to load before automatic matching. It is trusted as-is.
record_path = None
# True requires resolution, TE polarization, and complete geometry metadata to
# match. False deliberately permits reuse of another resolution's result.
reuse_requires_compute_match = True

# Plot-only settings never participate in record identity or MPB calculation.
plot_params = {
    "save": True,
    "show": False,
    "figsize": (5.2, 4.8),
    "dpi": 140,
    "title": None,
    "xlabel": "kx",
    "ylabel": "ky",
    "font_size": 11,
    "tick_size": 10,
    "grid": True,
    "grid_kwargs": {"linestyle": ":", "linewidth": 0.45, "alpha": 0.7},
    # Scattered-data interpolation resolution and method.
    "mesh_size": 120,
    "interpolation": "linear",
    "cmap": "RdBu_r",
    # Fixed limits make different calculations visually comparable. Set both
    # to None to normalize colors independently for each record.
    "vmin": -1,
    "vmax": 1,
    "colorbar": True,
    "colorbar_label": "Berry curvature",
    "colorbar_kwargs": {"fraction": 0.046, "pad": 0.04},
}


def compute_berry_curvature(
    config_module=config,
    *,
    resolution,
    num_bands,
    grid_n,
    shrinking,
    step,
    band_index=None,
    symmetry_mode="auto",
    run_mode="auto",
    archive=False,
    reuse_requires_compute_match=True,
    record_path=None,
    save=True,
    save_tmp=True,
    source_case=None,
):
    """Load or calculate Berry curvature on the K-centered triangular HBZ."""
    if record_path is not None:
        path = Path(record_path)
        return load_record(path), path, None
    if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution < 1:
        raise ValueError("resolution must be an integer >= 1.")
    if not isinstance(num_bands, int) or isinstance(num_bands, bool) or num_bands < 1:
        raise ValueError("num_bands must be an integer >= 1.")
    if step <= 0:
        raise ValueError("step must be positive.")
    if band_index is not None and (
        not isinstance(band_index, int)
        or isinstance(band_index, bool)
        or band_index < 0
        or band_index >= num_bands
    ):
        raise ValueError(f"band_index must be None or between 0 and {num_bands - 1}.")

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
        "step": float(step),
        "band_index": band_index,
        "domain": HBZ_DOMAIN,
        "symmetry_policy": symmetry_mode,
        "symmetry": actual_mode,
    }
    compute_params = compute_parameters(config_module, resolution)
    record, path = resolve_existing_record(
        config_module,
        kind="bc",
        task_params=task_params,
        compute_params=compute_params,
        run_mode=run_mode,
        record_path=None,
        reuse_requires_compute_match=reuse_requires_compute_match,
    )
    if record is not None:
        return record, path, None

    band = config_module.make_band(resolution=resolution)
    raw_result = band.compute_berry_grid(
        config_module.build_pattern(),
        raw_k_points,
        step=step,
        num_bands=num_bands,
        band_index=band_index,
    )
    raw_bcs = np.asarray(raw_result["bcs"], dtype=float)
    if actual_mode == "c3":
        final_k_points, (final_bcs,) = c3_expand_arrays(
            kspace,
            raw_k_points,
            raw_bcs,
        )
    else:
        final_k_points = np.asarray(raw_k_points, dtype=float)
        final_bcs = raw_bcs

    result = dict(raw_result)
    result.update(
        {
            "raw_k_points": np.asarray(raw_k_points, dtype=float),
            "raw_bcs": raw_bcs,
            "k_points": final_k_points,
            "bcs": final_bcs,
            "grid_n": int(grid_n),
            "shrinking": float(shrinking),
            "domain": HBZ_DOMAIN,
            "symmetry_policy": symmetry_mode,
            "symmetry": actual_mode,
        }
    )
    geometry_id = config_module.geometry_id()
    record = make_record(
        "bc",
        geometry_id,
        task_params=task_params,
        compute_params=compute_params,
        data=result,
        source_case=source_case,
    )
    canonical_path, latest_path = save_record_outputs(
        project_root,
        geometry_id,
        "bc",
        task_params,
        record,
        archive=archive,
        archive_params={
            "num_bands": num_bands,
            "band_index": band_index,
            "grid_n": grid_n,
            "symmetry": actual_mode,
            "step": step,
        },
        save=save,
        save_tmp=save_tmp,
        tmp_name="bc_latest.pkl",
    )
    return record, canonical_path, latest_path


def _berry_image_path(record_path_value, geometry_id, band_index_value, multi_band):
    base = make_image_path(project_root, record_path_value, geometry_id)
    if multi_band:
        return base.with_name(f"{base.stem}_b{band_index_value + 1}{base.suffix}")
    return base


def plot_berry_record(
    record_or_path,
    *,
    band_index=0,
    show=False,
    save=True,
    image_path=None,
    plot_params=None,
):
    """Plot one Python 0-based band from an existing HBZ Berry record."""
    record_path_value = None
    if isinstance(record_or_path, (str, Path)):
        record_path_value = Path(record_or_path)
        record = load_record(record_path_value)
    else:
        record = record_or_path

    params = dict(plot_params or {})
    show = params.pop("show", show)
    save = params.pop("save", save)
    values = np.asarray(record["data"]["bcs"], dtype=float)
    multi_band = values.ndim == 2
    if multi_band:
        if band_index < 0 or band_index >= values.shape[1]:
            raise ValueError(f"band_index must be between 0 and {values.shape[1] - 1}.")
        values = values[:, band_index]
    elif band_index not in (0, None):
        raise ValueError("A single-band Berry record can only use band_index=0.")

    if image_path is None and save:
        if record_path_value is None:
            raise ValueError("image_path is required when saving an in-memory record.")
        image_path = _berry_image_path(
            record_path_value,
            record["geometry_id"],
            int(band_index or 0),
            multi_band,
        )
    params.setdefault("title", f"TriLatt Berry curvature (Band {int(band_index or 0) + 1})")
    params.setdefault("colorbar_label", "Berry curvature")
    fig, ax = plot_scalar_field(
        record["data"]["k_points"],
        values,
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
    """Obtain a Berry record and render the selected band."""
    record, output_record_path, latest_path = compute_berry_curvature(
        config,
        resolution=resolution,
        num_bands=num_bands,
        grid_n=grid_n,
        shrinking=shrinking,
        step=step,
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
        _, _, image_path = plot_berry_record(
            output_record_path,
            band_index=plot_band_index,
            plot_params=plot_params,
        )
    print("geometry:", config.geometry_id())
    print("symmetry used:", record["data"]["symmetry"])
    print("record:", output_record_path)
    print("tmp record:", latest_path)
    print("image:", image_path)
    print("berry curvature shape:", np.asarray(record["data"]["bcs"]).shape)
    return record, output_record_path, image_path


if __name__ == "__main__":
    main()
