"""Smoke test for the sequential simulation path.

This is deliberately minimal: it exists to catch "the model no longer runs
at all" before Phase 2 starts modifying coupler_parallel.py. It is not a
scientific correctness test.
"""

from pascal.coupler import Pascal1D
from pascal.scenarios import build_1d_scenario


def test_sequential_1d_run_completes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    kwargs = build_1d_scenario(
        n_super_individuals=10,
        n_virtual_per_super=10000,
        duration_years=0.02,
        seed=1,
        headless="smoke_run",
    )
    sim = Pascal1D(**kwargs)
    sim.run()

    output_dir = tmp_path / "smoke_run"
    assert (output_dir / "output_ps.nc").exists()
    assert (output_dir / "lifestats.csv").exists()
    assert sim.population_size() > 0


def test_sequential_1d_run_is_deterministic_when_not_stochastic(tmp_path, monkeypatch):
    """With stochastic=False, two runs with the same inputs should produce
    the same population trajectory. This is the baseline Phase 3 will use
    to check the parallel implementation doesn't diverge from sequential."""
    monkeypatch.chdir(tmp_path)

    results = []
    for i in range(2):
        run_dir = tmp_path / f"det_run_{i}"
        run_dir.mkdir()
        monkeypatch.chdir(run_dir)

        kwargs = build_1d_scenario(
            n_super_individuals=10,
            n_virtual_per_super=10000,
            duration_years=0.02,
            stochastic=False,
            seed=1,
            headless="run",
        )
        sim = Pascal1D(**kwargs)
        sim.run()
        results.append(sim.population_size())

    assert results[0] == results[1]
