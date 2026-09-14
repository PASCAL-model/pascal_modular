"""Parallelized version of PASCAL coupler.

This module provides drop-in replacement classes that use multiprocessing
to parallelize update_lifestage(), the per-super-individual update loop
that dominates runtime once coupler.py's log_spatial()/resolve_spatial()
bottleneck is fixed (see BENCHMARKING.md: 79% of a 200-super-individual/
0.5-year sequential run).

Why this isn't just "pool.map(update_lifestage, supindividuals)"
------------------------------------------------------------------
Every SuperIndividual holds a direct reference to the *entire* shared
per-timestep environment arrays (`self.environment`, `self.environment_profiles`
- see individual.py's __init__ and coupler.py's seed()), not just its own
column. In the advection case those arrays have shape
(n_depth, n_super_individuals): naively pickling each SuperIndividual
independently for Pool.map would serialize that whole array once per
individual, per timestep, in both directions - the wasteful behavior an
earlier, never-benchmarked version of this file had.

Instead, update_lifestage() here extracts each individual's own small
slice of the (fixed, known - see individual.py's PROFILE_ENVIRONMENT_VARIABLES
and SCALAR_ENVIRONMENT_VARIABLES) environment variables before dispatch,
ships that instead of the full array, and restores the individual's real
environment_index/environment/environment_profiles references (needed by
coupler.py's log_spatial/gene_hunt/respawn, which index into the tracker's
element arrays by environment_index) once results come back. Nothing about
individual.py's own read pattern (get_profile/get_zi) changes.

Usage:
    from coupler_parallel import PascalAdvectionParallel

    sim = PascalAdvectionParallel(
        ...,
        use_parallel=True,
        n_workers=8
    )
    sim.run()
"""

from multiprocessing import Lock, Pool, Value, cpu_count

import numpy as np

from .coupler import PascalSimulation, Pascal1D, PascalAdvection
from .individual import PROFILE_ENVIRONMENT_VARIABLES, SCALAR_ENVIRONMENT_VARIABLES
from .utils import dotdict


def _local_environment_context(individual):
    """Extract the small per-individual slice of environment data that
    update_lifestage()'s call tree actually reads, in place of the full
    shared arrays. Returned dicts use environment_index=0 (single column)
    since that's the only column they contain.
    """
    idx = individual.environment_index
    profiles = individual.environment_profiles
    env = individual.environment

    sliced_profiles = {"z": profiles["z"]}
    for var in PROFILE_ENVIRONMENT_VARIABLES:
        # Keep the depth axis, take only this individual's column, but stay
        # 2D - get_profile() indexes as [:, environment_index].
        sliced_profiles[var] = np.asarray(profiles[var])[:, idx:idx + 1]

    sliced_env = {}
    for var in SCALAR_ENVIRONMENT_VARIABLES:
        sliced_env[var] = np.asarray([env[var][idx]])

    return dotdict(sliced_env), dotdict(sliced_profiles)


def _update_individual_worker(task):
    """Worker function for parallel individual update.

    Must be at module level for pickle serialization.

    Args:
        task: None, or (individual, sliced_env, sliced_profiles)

    Returns:
        Updated SuperIndividual instance or None. Its environment_index is
        left at 0 and environment/environment_profiles point at the small
        slice passed in - the caller (update_lifestage below) is
        responsible for restoring the real values.
    """
    if task is None:
        return None
    individual, sliced_env, sliced_profiles = task
    individual.environment = sliced_env
    individual.environment_profiles = sliced_profiles
    individual.environment_index = 0
    individual.update_lifestage()
    return individual


def _init_worker_rng(base_entropy, n_workers, counter, counter_lock):
    """Pool initializer: give each worker process its own independent
    numpy global RNG stream.

    Without this, worker processes inherit whatever RNG state existed in
    the parent at Pool creation and never reseed - since individual.py
    draws from numpy's legacy global RNG (np.random.rand/choice/etc, used
    throughout for sex determination, diapause strategy, gene crossover/
    mutation), that would mean multiple workers drawing statistically
    correlated random streams for the life of the pool, silently biasing
    a stochastic IBM's results.

    Each worker atomically claims the next index from a counter shared via
    `counter`/`counter_lock` (fresh multiprocessing.Value/Lock created once
    per Pool in __init__), then derives its seed via SeedSequence.spawn -
    numpy's documented pattern for exactly this. Deliberately NOT using
    multiprocessing.current_process()._identity for this: that counter is
    global to the parent process's entire lifetime, not reset per Pool, so
    a second PascalSimulationParallel created later in the same process
    would get out-of-range worker "slot numbers" from the first Pool's
    workers still counted against it.
    """
    with counter_lock:
        worker_id = counter.value
        counter.value += 1
    seed_seq = np.random.SeedSequence(base_entropy).spawn(n_workers)[worker_id]
    np.random.seed(seed_seq.generate_state(4))


class PascalSimulationParallel(PascalSimulation):
    """Parallelized base simulation class.

    Adds multiprocessing support to the base PascalSimulation class.
    The update_lifestage() method runs in parallel across multiple cores.
    """

    def __init__(self, *args, use_parallel=True, n_workers=None,
                 rng_seed=None, **kwargs):
        """Initialize parallel simulation.

        Args:
            *args: Positional arguments for PascalSimulation
            use_parallel: If True, use multiprocessing (default: True)
            n_workers: Number of worker processes. If None, uses CPU count - 1
            rng_seed: Optional int/SeedSequence entropy for reproducible
                per-worker RNG streams (see _init_worker_rng). If None, a
                fresh, non-reproducible seed is drawn.
            **kwargs: Keyword arguments for PascalSimulation
        """
        super().__init__(*args, **kwargs)

        self.use_parallel = use_parallel

        if self.use_parallel:
            if n_workers is None:
                self.n_workers = max(1, cpu_count() - 1)
            else:
                self.n_workers = max(1, min(n_workers, cpu_count()))

            base_entropy = (
                rng_seed if rng_seed is not None
                else np.random.SeedSequence().entropy
            )
            worker_id_counter = Value('i', 0)
            worker_id_lock = Lock()
            self.pool = Pool(
                processes=self.n_workers,
                initializer=_init_worker_rng,
                initargs=(base_entropy, self.n_workers, worker_id_counter, worker_id_lock),
            )

            print(f"Parallel mode enabled: {self.n_workers} workers "
                  f"(CPU count: {cpu_count()})")
        else:
            self.pool = None
            self.n_workers = 1
            print("Sequential mode (no parallelization)")

    def __del__(self):
        """Clean up process pool on deletion."""
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()

    def update_lifestage(self):
        """Update all super-individuals (parallel or sequential).

        If use_parallel is True and there are enough individuals,
        updates are performed in parallel using multiprocessing.Pool.
        Otherwise, falls back to sequential execution.
        """
        n_active = len([si for si in self.supindividuals if si is not None])

        # Only parallelize if worthwhile (overhead vs benefit)
        if (self.use_parallel and self.pool is not None
                and n_active >= self.n_workers * 2):
            original_indices = [
                si.environment_index if si is not None else None
                for si in self.supindividuals
            ]
            tasks = [
                None if si is None else (si, *_local_environment_context(si))
                for si in self.supindividuals
            ]

            results = self.pool.map(_update_individual_worker, tasks)

            # Restore the real environment_index/environment references -
            # coupler.py indexes self.tracker.elements.lon/lat by
            # environment_index in log_spatial/gene_hunt/respawn, and the
            # next timestep's update_lifestage() needs the live (just
            # refreshed by update_environment()) shared arrays, not the
            # small slice each worker used.
            for si, orig_idx in zip(results, original_indices):
                if si is not None:
                    si.environment_index = orig_idx
                    si.environment = self.tracker.environment
                    si.environment_profiles = self.tracker.environment_profiles

            self.supindividuals = results
        else:
            # Sequential fallback (population too small to be worth
            # dispatching to workers) - uses the same batched mortality/
            # death-check pass as the base sequential path (see
            # coupler.py::apply_mortality_and_deathcheck_batch()).
            for this_individual in self.supindividuals:
                if this_individual is not None:
                    this_individual.run_stage_transition()
            self.apply_mortality_and_deathcheck_batch()

    def finish_run(self):
        """Override to ensure pool cleanup before parent finish_run."""
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

        super().finish_run()


class Pascal1DParallel(PascalSimulationParallel, Pascal1D):
    """Parallelized 1D simulation."""
    pass


class PascalAdvectionParallel(PascalSimulationParallel, PascalAdvection):
    """Parallelized advection simulation with OpenDrift.

    Example:
        sim = PascalAdvectionParallel(
            nsup_individ=1000,
            nindivid_per_sup=10000,
            global_settings=settings,
            reader=reader,
            timestep=timestep,
            start_date=start_date,
            duration=3,
            seeding_rate=10,
            outputgrid=outputgrid,
            use_parallel=True,
            n_workers=8
        )
        sim.run()
    """
    pass


def get_optimal_workers(n_individuals=None):
    """Determine optimal number of workers.

    Args:
        n_individuals: Number of super-individuals (optional)

    Returns:
        int: Recommended number of workers
    """
    n_cpu = cpu_count()

    if n_individuals is None:
        return max(1, n_cpu - 1)

    # Rule of thumb: at least 10 individuals per worker
    optimal = min(n_cpu - 1, n_individuals // 10)

    return max(1, optimal)
