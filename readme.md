# pascal_modular

The core PASCAL model: a class-based re-implementation of PASCAL (Pan-Arctic
Behavioural and Life-history Simulator for Calanus), developed under the
Norwegian Research Council funded Migratory Crossroads project (2024-2027).

This repository is just the installable `pascal` Python package — the
copepod individual-based model (IBM) and its coupling to
[OpenDrift](https://opendrift.github.io/) for advection. It has no run
entrypoint or HPC/container setup of its own; for that, see the sibling
repositories:

- [`opendrift`](../opendrift) — the forked particle tracker this package
  couples to for advection scenarios (a small change exposing per-step
  control of the tracker's timestepping to the coupler; unmodified
  otherwise).
- [`pascal_run`](../pascal_run) — installs this package (and `opendrift`)
  into a container and runs it, locally or on an HPC, from a YAML config.
- [`pascal_benchmark`](../pascal_benchmark) — performance
  benchmarking/profiling harness for this package.

## Package layout

- `pascal.coupler` / `pascal.coupler_parallel` — `Pascal1D`/`PascalAdvection`
  simulation classes (sequential and multiprocess-parallel).
- `pascal.individual` — the `SuperIndividual` state and life-history logic.
- `pascal.biology.*` — growth, survival, and vertical-migration submodels.
- `pascal.data_logger` — spatial/temporal output aggregation and netCDF/CSV
  writing.
- `pascal.utils` — shared helpers (e.g. `load_locations_csv`).
- `pascal.scenarios` — scenario builders that turn either synthetic
  in-memory data or a CMEMS netCDF file into ready-to-run
  `Pascal1D`/`PascalAdvection` kwargs. Used directly by `pascal_benchmark`,
  and by `pascal_run`'s YAML-config glue (which overlays a run config's
  `biology`/`population`/`time` sections on top of these).

## Development setup

```bash
mamba env create -f environment.yml
conda activate pascal_modular

# Advection scenarios (PascalAdvection, coupler_parallel) depend on the
# opendrift fork - install it as an editable sibling checkout too.
pip install -e ../opendrift
pip install -e .

pytest
```

Values for genuinely calibrated constants in `pascal.scenarios`'s synthetic
builders (critical molting mass thresholds, developmental coefficients) are
illustrative, not validated against field data — fine for benchmarking, not
for scientific runs. For a real run, use `pascal_run`'s YAML config instead,
which lets every biology parameter be set explicitly.

## Status

This is the fourth iteration of PASCAL. It has been debugged and tested for
correct functionality but not yet validated against field data (expected
once Migratory Crossroads Work Package 2 field data is available). Model
outputs must be interpreted with this in mind.
