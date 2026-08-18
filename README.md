# TriLatt

TriLatt is a parameter-driven triangular/honeycomb MPB workflow built on the
public `WD-nanophotonics/MePhC` package. Geometry is controlled by the active
polygon parameters and the optional global affine parameters in `config.py`;
there is no hole-shape or lattice-type setting.

Install the pinned public MePhC release in the `mp` environment:

```bash
/home/icy/miniconda3/envs/mp/bin/python -m pip install \\
  "mephc @ git+https://github.com/WD-nanophotonics/MePhC.git@v0.1.1"
```

## Local VS Code one-click run

When the repositories are checked out as sibling directories:

<pre>
/home/icy/MePhC
/home/icy/TriLatt
</pre>

Open the TriLatt folder in VS Code using the WSL window. The workspace
configuration selects /home/icy/miniconda3/envs/mp/bin/python and adds the
adjacent local MePhC checkout to both Python analysis and the integrated
terminal PYTHONPATH. No pip installation or network access is needed for
this local workflow.

In Run and Debug, press F5 to run the default band_structure.py
configuration. The same menu provides one-click entries for
frequency_at_k.py, berry_curvature.py, and efs.py.

## Geometry parameters

```python
a = 400
r1 = 74
r2 = 72

n1 = 16
theta1 = 0
n2 = 16
theta2 = 60

n_eff = 2.7
height = 100

# Optional global affine deformation of the periodic direct lattice.
stretch_factor = 1.0
stretch_angle_degrees = 0.0
```

- `r2 is None` calls `create_unitcell(n1, theta1)`. `n2/theta2` are ignored.
- A numeric `r2` calls `create_unitcell(n1, theta1, n2, theta2)`.
- `n` is the side count: `3` is a triangle and `16` approximates a circle.
- `theta` is passed directly to MePhC in degrees.
- `stretch_factor` and `stretch_angle_degrees` define the global affine map
  for the periodic lattice. They are geometry parameters, not solver
  resolution or plotting parameters.

## Run

```bash
cd /home/icy/TriLatt
/home/icy/miniconda3/envs/mp/bin/python frequency_at_k.py
/home/icy/miniconda3/envs/mp/bin/python band_structure.py
/home/icy/miniconda3/envs/mp/bin/python berry_curvature.py
/home/icy/miniconda3/envs/mp/bin/python efs.py
```

The K script prints normalized and THz frequencies at the configured reciprocal
landmark. Identity uses the legacy Cartesian K=`(2/3,0)`. Under non-identity
deformation the label is explicitly `tracked_K1`, selected from the current
Wigner-Seitz first BZ; it is not an assertion of unbroken C3 symmetry. The
fixed Gamma-K-M-Gamma path, fixed K, and K-centered HBZ descriptions below are
therefore identity-only compatibility behavior. The band script uses a
current-BZ generic path after deformation. Berry and EFS use the current full
BZ domain after deformation and disable invalid C3/HBZ reduction. Records are
saved under `data/<parameter_id>/` and plots under `image/<parameter_id>/`.

## Global affine deformation

`config.py` exposes one global periodic deformation:

```python
stretch_factor = 1.0
stretch_angle_degrees = 0.0
```

The current direct basis is `A = F @ A0`, and the reciprocal basis is derived
from that current basis. Every sublattice center, including the second
honeycomb sublattice, follows `F @ center_reference`. Polygon-local vertex
offsets remain unchanged, so each feature stays rigid: side lengths,
orientation, area, side count, and material are preserved. At factor `1.0`
the angle is canonicalized and the legacy geometry IDs and numerical path are
retained.

For non-identity deformation, `auto` mode does not reuse C3 reduction,
K-centered HBZ assumptions, or fixed reference K coordinates. The current BZ
and current reciprocal coordinates are stored in metadata. This entry-point
workflow is limited to global affine periodic deformation; local/non-affine
deformation, automatic symmetry discovery, and the SqrLatt downstream project
remain out of scope. A bounded, programmatic R6 periodic-supercell adapter is
available in `r5_deformation.py` through `build_supercell_solver(...)`. It
requires a verified `PeriodicSupercellField`, diagonal replication, and
explicit `q_points`, `resolution`, and `num_bands`, and runs the TE
adapter. This is not transparent periodic-supercell support in all user entry
scripts and does not provide nonperiodic finite/local deformation solver
support, transport, far-field, automatic symmetry discovery, or R7+
production capability. Record data and rendering metadata are separate: solver
parameters identify and cache the `.pkl` record, while plot settings only
control regeneration of derived images.

## Periodic-supercell band entry

`supercell_config.py` is the user-editable R6.2 configuration and field hook.
It defines diagonal `replication`, generic fractional supercell `q_points`,
`resolution`, `num_bands`, and plot controls. Its default deterministic
periodic field is explicitly an example rather than a physical material
preset; researchers can replace `make_verified_field` with another verified
analytic or sampled field.

`supercell_band.py` consumes the verified `PeriodicSupercellField` returned by
that hook and takes the field's own supercell metadata as the replication
authority. It calls the existing `r5_deformation.build_supercell_solver`,
returns normalized and THz frequency arrays, and plots against the
cumulative generic reciprocal-path distance derived from the verified field's
reciprocal basis. Cartesian q-points, path distances, and the basis provenance
are stored in the record so plot-only mode remains adapter-free. The q-points
are never labeled Gamma, K, M, or X.
R6.3/R6.4 add persistent records without changing MePhC: the record namespace
includes the base TriLatt geometry and a SHA-256 digest of the complete R5
field identity, while the task identity includes the ordered q-points and a
q-point digest. Saved payloads contain scientific arrays and provenance only;
solver, Band, and field runtime objects are never persisted.

The same entry point supports the bounded record modes below through
`supercell_config.py`:

```python
run_mode = "auto"       # reuse exact record, otherwise compute and save
run_mode = "compute"    # force MPB and refresh the canonical record
run_mode = "plot_only"  # load a matching record; never run MPB
archive_record = False
record_path = None       # optional explicit historical record, trusted as-is
reuse_requires_compute_match = True
```

`plot_only` and an exact cache hit do not instantiate the R6 MPB adapter.
Changing the verified field, field identity, replication, ordered q-points,
number of bands, resolution, or geometry prevents incompatible reuse. Plot
settings only regenerate a derived image.

Run it noninteractively with:

```bash
cd /home/icy/TriLatt
/home/icy/miniconda3/envs/mp/bin/python supercell_band.py
```

The default is a small demonstration solve with no interactive window and no
saved image. Edit `supercell_config.py` to change the field or plot controls.

## Band Berry coloring

The band entry calculates and colors Berry curvature by default:

```python
color_by_berry = True
berry_step = 0.0005
berry_num_bands = num_bands
```

Set `color_by_berry=False` for a faster frequency-only band calculation.
Ordinary and Berry-colored bands have different record identities, so one is
never silently reused as the other. `berry_step` is the plaquette side in
Cartesian reciprocal coordinates and changing it requests new simulation
data.

## Standalone Berry curvature

Edit the documented values at the top of `berry_curvature.py`, then run it.
The defaults calculate three bands and plot Python band index `2` (the third
band):

```python
resolution = 64
num_bands = 3
grid_n = 24
shrinking = 0.01
step = 0.0005
band_index = None
plot_band_index = 2
symmetry_mode = "auto"
```

`band_index=None` stores all requested bands; an integer calculates only that
0-based band. In `auto` mode, C3 reduction is used only when every active
polygon side count is divisible by 3. Thus `n=3` is reduced and expanded
exactly, while the current `n=16` geometry is calculated directly over the
complete HBZ. Explicit `symmetry_mode="c3"` rejects non-C3 geometry;
`"raw_hbz"` always uses direct sampling.

## EFS

For identity deformation, `efs.py` uses the legacy K-centered HBZ and its
safe symmetry rule. For non-identity deformation it uses the current full BZ
domain and does not assume a fixed K-centered HBZ:

```python
resolution = 64
num_bands = 3
grid_n = 24
shrinking = 0.01
band_index = 0
symmetry_mode = "auto"
```

`plot_params["use_actual"]=True` contours THz; `False` uses normalized MPB
frequency. `levels=8` requests eight automatic contours, while a list such as
`levels=[210, 220, 230]` requests exact values in the currently selected unit.
`mesh_size` and `interpolation` affect only the contour rendering.

## Geometry preview

The preview controls are at the top of `band_structure.py`.

```python
preview_numpy = True
preview_mpb = False
preview_resolution = 32
run_calculation = True
```

- NumPy preview displays the polygons produced by `create_unitcell()`.
- MPB preview displays raw and rectified dielectric maps from a separate,
  low-resolution solver run.
- To inspect geometry without calculating bands, set
  `run_calculation=False`. No band record is read/written and no band PNG is
  produced.
- Preview settings never affect record matching or formal band data.

## Common record operations

The following controls appear, with inline explanations, in band, Berry, and
EFS entry scripts.

Normal use—reuse the matching result or calculate it when missing:

```python
run_mode = "auto"
record_path = None
reuse_requires_compute_match = True
archive_record = False
```

Force a fresh MPB solve and overwrite the canonical working record:

```python
run_mode = "compute"
```

Only regenerate a plot from an already matching record; never run MPB:

```python
run_mode = "plot_only"
```

Keep a timestamped historical copy in addition to the canonical record:

```python
archive_record = True
```

Load and replot one exact historical file before applying `run_mode`:

```python
record_path = "/home/icy/TriLatt/data/.../bc_nb3_n24_raw_hbz_step0p0005_....pkl"
```

An explicit `record_path` is trusted as-is and is not checked against the
current config. Reset it to `None` to return to automatic matching.

Permit reuse of a canonical record generated with different compute metadata,
such as another resolution:

```python
reuse_requires_compute_match = False
```

This does not improve or convert the old data—it deliberately accepts the
old record's accuracy. Keep the default `True` unless that is intentional.

All entries inside `plot_params` affect only rendering. Changing `save`,
`show`, `use_actual`, `dpi`, line/scatter style, colormap, interpolation, or
contour levels never triggers a new MPB calculation.

Record task identities include every data-defining option. Band records
include the Berry switch and plaquette step. Berry/EFS records include grid
density, HBZ domain, requested symmetry policy, actual `c3`/`raw_hbz` mode,
selected band task, and Berry step where applicable. Resolution,
polarization, and complete geometry parameters are kept in compute metadata.

Binary records and generated images are local archive artifacts and are ignored
by Git. The tracked `archive_manifest.json` contains their relative names,
parameters, timestamps, and SHA-256 hashes for audit purposes.
