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
- [`pascal_docs`](../pascal_docs) — full documentation: a code-structure
  tour of this package
  ([`pascal_modular_structure.md`](../pascal_docs/pascal_modular_structure.md)),
  the biological model
  ([`biology_logic.md`](../pascal_docs/biology_logic.md)), worked examples,
  an API reference, and dev setup
  ([`CONTRIBUTING.md`](../pascal_docs/CONTRIBUTING.md)) covering all 5
  repos.

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
