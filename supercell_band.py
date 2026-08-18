"""Bounded user-facing R6/R6.2/R6.3 periodic-supercell band workflow."""

from numbers import Integral
from pathlib import Path
import hashlib
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import supercell_config
from mephc.deformation import PeriodicSupercellField
from mephc.r5 import record_identity
from mephc.records import load_record, make_image_path, make_record
from mephc.workflows import resolve_record, save_record_outputs
from r5_deformation import build_supercell_solver


Q_POINT_COORDINATE = "generic_fractional_supercell"
RECORD_KIND = "band"


def _positive_integer(value, name):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1.")
    return int(value)


def _validate_q_points(value):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 2:
        raise ValueError("q_points must contain at least two finite 2D points.")
    if not np.all(np.isfinite(array)):
        raise ValueError("q_points must contain only finite values.")
    return tuple(tuple(float(component) for component in point) for point in array)


def _field_replication(field):
    if not isinstance(field, PeriodicSupercellField):
        raise TypeError("R6 runner requires a PeriodicSupercellField.")
    field.require_verified()
    matrix = np.asarray(field.supercell.matrix, dtype=int)
    if matrix.shape != (2, 2) or not np.array_equal(matrix, np.diag(np.diag(matrix))):
        raise ValueError("R6 runner currently requires diagonal replication.")
    values = tuple(int(value) for value in np.diag(matrix))
    if any(value < 1 for value in values):
        raise ValueError("field supercell replication must be positive.")
    return list(values)


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _q_point_digest(q_points):
    payload = [list(point) for point in q_points]
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _field_record_identity(config_module, field, replication):
    field.require_verified()
    if not getattr(field, "stable_identity", False) or not getattr(field.field, "stable_identity", False):
        raise ValueError("E_R5_UNSTABLE_CALLABLE: field needs explicit stable_id before persistent record writes")
    return record_identity(
        field,
        reference_lattice=config_module.canonical_lattice(),
        replication=replication,
    )


def _record_namespace(config_module, identity):
    base_geometry_id = config_module.geometry_id()
    identity_digest = hashlib.sha256(
        _canonical_json(identity).encode("utf-8")
    ).hexdigest()
    return (
        f"TRILATT_SUPERCELL_{base_geometry_id}_"
        f"FIELD_{identity_digest[:16]}"
    ), identity_digest


def _record_parameters(config_module, field, q_points, resolution, num_bands):
    replication = _field_replication(field)
    identity = _field_record_identity(config_module, field, replication)
    identity_digest = hashlib.sha256(
        _canonical_json(identity).encode("utf-8")
    ).hexdigest()
    q_digest = _q_point_digest(q_points)
    geometry_id, namespace_digest = _record_namespace(config_module, identity)
    if namespace_digest != identity_digest:
        raise AssertionError("record namespace digest is not the field identity digest")
    task_params = {
        "num_bands": int(num_bands),
        "q_points": [list(point) for point in q_points],
        "q_point_coordinate": Q_POINT_COORDINATE,
        "q_point_digest": q_digest,
        "path": f"generic_q_{q_digest[:16]}",
    }
    compute_params = {
        "resolution": int(resolution),
        "polarization": "TE",
        "geometry": config_module.geometry_parameters(),
        "replication": list(replication),
        "field_record_identity": identity,
        "field_identity_sha256": identity_digest,
    }
    metadata = {
        "schema": "trilatt.supercell_band_record.v1",
        "base_geometry_id": config_module.geometry_id(),
        "supercell_geometry_namespace": geometry_id,
        "field_record_identity": identity,
        "field_identity_sha256": identity_digest,
        "replication": list(replication),
        "q_point_digest": q_digest,
        "q_point_coordinate": Q_POINT_COORDINATE,
    }
    return geometry_id, task_params, compute_params, metadata


def _frequency_arrays(solver, band, *, point_count, requested_bands):
    raw_frequencies = getattr(solver, "all_freqs", None)
    if raw_frequencies is None:
        raw_frequencies = solver.freqs
    normalized = np.asarray(raw_frequencies, dtype=float)
    expected_shape = (point_count, requested_bands)
    if normalized.shape != expected_shape:
        raise ValueError(
            f"R6 solver frequencies must have shape {expected_shape}, got {normalized.shape}."
        )
    if not np.all(np.isfinite(normalized)):
        raise ValueError("R6 solver returned non-finite normalized frequencies.")
    actual = np.asarray(band.calculate_actual_freqs(normalized), dtype=float)
    if actual.shape != expected_shape:
        raise ValueError(
            f"R6 THz frequencies must have shape {expected_shape}, got {actual.shape}."
        )
    if not np.all(np.isfinite(actual)):
        raise ValueError("R6 solver returned non-finite THz frequencies.")
    return normalized, actual


def compute_supercell_band(
    config_module=supercell_config,
    *,
    field=None,
    q_points=None,
    resolution=None,
    num_bands=None,
):
    """Run the R6 adapter with a verified caller field and generic q-points."""
    q_points = _validate_q_points(config_module.q_points if q_points is None else q_points)
    resolution = _positive_integer(
        config_module.resolution if resolution is None else resolution,
        "resolution",
    )
    num_bands = _positive_integer(
        config_module.num_bands if num_bands is None else num_bands,
        "num_bands",
    )
    if field is None:
        field = config_module.make_verified_field()
    replication = _field_replication(field)
    solver, metadata = build_supercell_solver(
        config_module,
        field,
        q_points=q_points,
        resolution=resolution,
        num_bands=num_bands,
    )
    band = metadata.get("band")
    if band is None or not hasattr(band, "calculate_actual_freqs"):
        raise ValueError("R6 adapter metadata must contain the band unit converter.")
    normalized, actual = _frequency_arrays(
        solver,
        band,
        point_count=len(q_points),
        requested_bands=num_bands,
    )
    return {
        "q_points": np.asarray(q_points, dtype=float),
        "q_point_coordinate": Q_POINT_COORDINATE,
        "sample_coordinate": np.arange(len(q_points), dtype=float),
        "freqs": normalized,
        "actual_freqs": actual,
        "replication": replication,
        "resolution": resolution,
        "num_bands": num_bands,
        "field": field,
        "field_metadata": field.metadata(),
        "solver": solver,
        "metadata": metadata,
    }


def _persistent_data(result, identity):
    """Project an R6 result to scientific/provenance data only."""
    return {
        "q_points": np.asarray(result["q_points"], dtype=float).tolist(),
        "q_point_coordinate": Q_POINT_COORDINATE,
        "sample_coordinate": np.asarray(result["sample_coordinate"], dtype=float).tolist(),
        "freqs": np.asarray(result["freqs"], dtype=float).tolist(),
        "actual_freqs": np.asarray(result["actual_freqs"], dtype=float).tolist(),
        "replication": list(result["replication"]),
        "resolution": int(result["resolution"]),
        "num_bands": int(result["num_bands"]),
        "field_metadata": result["field_metadata"],
        "field_record_identity": identity,
    }


def compute_supercell_band_record(
    config_module=supercell_config,
    *,
    field=None,
    q_points=None,
    resolution=None,
    num_bands=None,
    run_mode="auto",
    archive=False,
    reuse_requires_compute_match=True,
    record_path=None,
    save=True,
    save_tmp=True,
    source_case=None,
):
    """Resolve, compute, save, or load a persistent R6.3 band record."""
    if record_path is not None:
        path = Path(record_path)
        return load_record(path), path, None
    q_points = _validate_q_points(config_module.q_points if q_points is None else q_points)
    resolution = _positive_integer(
        config_module.resolution if resolution is None else resolution,
        "resolution",
    )
    num_bands = _positive_integer(
        config_module.num_bands if num_bands is None else num_bands,
        "num_bands",
    )
    if run_mode not in {"auto", "compute", "plot_only"}:
        raise ValueError("run_mode must be 'auto', 'compute', or 'plot_only'.")
    if field is None:
        field = config_module.make_verified_field()
    geometry_id, task_params, compute_params, metadata = _record_parameters(
        config_module, field, q_points, resolution, num_bands
    )
    record, path = resolve_record(
        project_root,
        geometry_id,
        RECORD_KIND,
        task_params=task_params,
        compute_params=compute_params,
        run_mode=run_mode,
        record_path=None,
        reuse_requires_compute_match=reuse_requires_compute_match,
    )
    if record is not None:
        return record, path, None

    result = compute_supercell_band(
        config_module,
        field=field,
        q_points=q_points,
        resolution=resolution,
        num_bands=num_bands,
    )
    record = make_record(
        RECORD_KIND,
        geometry_id,
        task_params=task_params,
        compute_params=compute_params,
        data=_persistent_data(result, metadata["field_record_identity"]),
        source_case=source_case,
    )
    record["metadata"] = metadata
    canonical_path, latest_path = save_record_outputs(
        project_root,
        geometry_id,
        RECORD_KIND,
        task_params,
        record,
        archive=archive,
        archive_params={
            "num_bands": num_bands,
            "path": task_params["path"],
        },
        save=save,
        save_tmp=save_tmp,
        tmp_name="supercell_band_latest.pkl",
    )
    return record, canonical_path, latest_path


def plot_supercell_band(result, *, use_actual=True, save=False, show=False, image_path=None):
    """Plot frequency versus generic q-point sample index."""
    values = np.asarray(result["actual_freqs"] if use_actual else result["freqs"], dtype=float)
    x_values = np.asarray(result["sample_coordinate"], dtype=float)
    fig, axis = plt.subplots(figsize=(6.0, 4.5))
    for band_index in range(values.shape[1]):
        axis.plot(
            x_values,
            values[:, band_index],
            marker="o",
            linewidth=1.4,
            label=f"band {band_index + 1}",
        )
    axis.set_xlabel("Generic q-point sample index")
    axis.set_ylabel("Frequency (THz)" if use_actual else "Normalized frequency")
    axis.set_title("TriLatt periodic-supercell band record")
    axis.set_xticks(x_values)
    axis.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.75)
    axis.legend()
    output_path = None
    if save:
        output_path = Path(image_path or supercell_config.plot_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axis, output_path


def plot_supercell_band_record(
    record_or_path, *, use_actual=True, save=False, show=False, image_path=None
):
    """Render a saved record without constructing or running the MPB adapter."""
    record_path_value = None
    if isinstance(record_or_path, (str, Path)):
        record_path_value = Path(record_or_path)
        record = load_record(record_path_value)
    else:
        record = record_or_path
    if save and image_path is None:
        if record_path_value is None:
            raise ValueError("image_path is required when saving an in-memory record.")
        image_path = make_image_path(
            project_root, record_path_value, record["geometry_id"]
        )
    return plot_supercell_band(
        record["data"],
        use_actual=use_actual,
        save=save,
        show=show,
        image_path=image_path,
    )


def main():
    record, output_record_path, latest_path = compute_supercell_band_record(
        supercell_config,
        run_mode=supercell_config.run_mode,
        archive=supercell_config.archive_record,
        reuse_requires_compute_match=supercell_config.reuse_requires_compute_match,
        record_path=supercell_config.record_path,
        save_tmp=supercell_config.save_tmp,
        source_case=str(project_root),
    )
    image_path = None
    if supercell_config.save_plot or supercell_config.show_plot:
        _, _, image_path = plot_supercell_band_record(
            output_record_path,
            use_actual=supercell_config.use_actual,
            save=supercell_config.save_plot,
            show=supercell_config.show_plot,
        )
    data = record["data"]
    print("record:", output_record_path)
    print("record mode:", supercell_config.run_mode)
    print("cache hit:", latest_path is None and supercell_config.record_path is None)
    print("replication:", data["replication"])
    print("q-point coordinate:", data["q_point_coordinate"])
    print("normalized frequency shape:", np.asarray(data["freqs"]).shape)
    print("actual frequency shape:", np.asarray(data["actual_freqs"]).shape)
    print("finite normalized frequencies:", bool(np.all(np.isfinite(data["freqs"]))))
    print("finite actual frequencies:", bool(np.all(np.isfinite(data["actual_freqs"]))))
    print("image:", image_path)
    return record, output_record_path, image_path


if __name__ == "__main__":
    main()
