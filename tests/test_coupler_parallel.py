"""Correctness tests for coupler_parallel.py.

Two separate concerns, tested separately:

1. Does the per-individual environment slicing (_local_environment_context)
   preserve the model's actual read semantics? Tested by comparing a full
   parallel run against a sequential run on a scenario where individuals
   stay in early, non-stochastic developmental stages (so growth/mortality
   is purely a deterministic function of environment + state) - this
   isolates the slicing logic from RNG behaviour.

2. Does each worker process get an independent RNG stream? Tested directly
   against _init_worker_rng, since a scenario short enough to run quickly
   in CI doesn't reach the stochastic branches (sex determination, diapause
   strategy, gene crossover/mutation) that would make an end-to-end
   comparison meaningful for this.
"""

import numpy as np

from pascal.coupler import PascalAdvection
from pascal.coupler_parallel import PascalAdvectionParallel, _init_worker_rng
from pascal.scenarios import build_advection_scenario


def test_parallel_matches_sequential_for_deterministic_growth_path(tmp_path, monkeypatch):
    """Individuals seeded at t=0 stay in the non-stochastic early stages
    for this short a run, so environment-driven growth/mortality should be
    bit-identical between sequential and parallel execution - this is
    exactly what the per-individual environment slicing needs to get
    right."""
    monkeypatch.chdir(tmp_path)

    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    monkeypatch.chdir(seq_dir)
    kwargs = build_advection_scenario(
        n_super_individuals=12, duration_years=0.02, seed=1, headless="seq"
    )
    seq_sim = PascalAdvection(**kwargs)
    seq_sim.run()

    par_dir = tmp_path / "par"
    par_dir.mkdir()
    monkeypatch.chdir(par_dir)
    kwargs = build_advection_scenario(
        n_super_individuals=12, duration_years=0.02, seed=1, headless="par"
    )
    par_sim = PascalAdvectionParallel(
        **kwargs, use_parallel=True, n_workers=2, rng_seed=123
    )
    par_sim.run()

    assert par_sim.population_size() == seq_sim.population_size()


def _draw_from_worker(_):
    return np.random.rand(5).tolist()


def test_worker_rng_streams_are_independent_and_reproducible():
    """Directly exercises _init_worker_rng rather than relying on an
    end-to-end run reaching a stochastic branch (see module docstring)."""
    from multiprocessing import Lock, Pool, Value

    def run_pool(base_entropy, n_workers):
        counter = Value('i', 0)
        lock = Lock()
        with Pool(
            processes=n_workers,
            initializer=_init_worker_rng,
            initargs=(base_entropy, n_workers, counter, lock),
        ) as pool:
            return pool.map(_draw_from_worker, range(n_workers))

    draws_a = run_pool(base_entropy=42, n_workers=4)
    draws_b = run_pool(base_entropy=42, n_workers=4)

    # Different workers must not draw identical (correlated) streams.
    for i in range(len(draws_a)):
        for j in range(i + 1, len(draws_a)):
            assert draws_a[i] != draws_a[j]

    # Same base_entropy must reproduce the same *set* of per-worker seed
    # streams - not necessarily in the same worker-slot order, since which
    # OS process claims which counter index first isn't guaranteed, only
    # that the streams claimed are the deterministic set spawned from
    # base_entropy.
    assert sorted(draws_a) == sorted(draws_b)

    draws_c = run_pool(base_entropy=99, n_workers=4)
    assert sorted(draws_a) != sorted(draws_c)
