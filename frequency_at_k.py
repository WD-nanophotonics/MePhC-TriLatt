from pathlib import Path
import sys

import numpy as np

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config


# MPB pixels per normalized lattice period. Higher values improve spatial
# accuracy but increase memory use and runtime.
resolution = 64
# Number of eigenfrequencies returned at K. Raising this computes more bands
# and normally makes the eigensolve slower.
num_bands = 3


def compute_k_frequencies(config_module=config, *, resolution, num_bands):
    """Compute frequencies at identity K or non-identity ``tracked_K1``."""
    if resolution < 1:
        raise ValueError("resolution must be >= 1.")
    if num_bands < 1:
        raise ValueError("num_bands must be >= 1.")

    landmark = config_module.reciprocal_landmark()
    k = np.asarray(landmark["cartesian"], dtype=float)
    if k.shape != (2,):
        raise ValueError("config.k_point() must return a 2D Cartesian k-point.")

    band = config_module.make_band(resolution=resolution)
    result = band.compute_efs(
        config_module.build_pattern(),
        k_points=[tuple(k)],
        num_bands=num_bands,
    )
    return {
        "k_point": k,
        "freqs": np.asarray(result.freqs[0], dtype=float),
        "actual_freqs": np.asarray(result.actual_freqs[0], dtype=float),
        "num_bands": int(num_bands),
        "resolution": int(resolution),
        "polarization": band.polarization,
        "k_coordinate": "cartesian_reciprocal",
        "landmark_kind": landmark["landmark_kind"],
        "display_label": landmark["display_label"],
        "landmark": landmark,
    }


def print_k_frequencies(result):
    kx, ky = result["k_point"]
    print(f"{result['display_label']} (Cartesian reciprocal): ({kx:.12g}, {ky:.12g})")
    print(f"polarization: {result['polarization']}")
    print(f"resolution: {result['resolution']}")
    print("band  normalized_frequency  actual_frequency_THz")
    for band_number, (normalized, actual) in enumerate(
        zip(result["freqs"], result["actual_freqs"]),
        start=1,
    ):
        print(f"{band_number:>4d}  {normalized:>20.12g}  {actual:>20.12g}")


def main():
    result = compute_k_frequencies(
        config,
        resolution=resolution,
        num_bands=num_bands,
    )
    print("geometry:", config.geometry_id())
    print_k_frequencies(result)
    return result


if __name__ == "__main__":
    main()
