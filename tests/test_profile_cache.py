"""Regression test for SuperIndividual.get_profile()'s per-timestep cache.

Found while investigating the advection (3D) scenario specifically: unlike
Pascal1D's synthetic benchmark data (deliberately built with z-levels
matching global_settings['depthrange'] exactly), OpenDrift's own profile
depth levels commonly don't match PASCAL's configured depthrange, which
triggers get_profile()'s np.interp() fallback - and that was being re-run
2-3x per individual per timestep for the same variable (once via
apply_dsc1_verticalmigration's get_profile() calls, again via get_zi() in
growth, again in mortality).
"""

from unittest.mock import patch

import numpy as np

from pascal.coupler import PascalAdvection
from pascal.scenarios import build_advection_scenario


def test_advection_scenario_actually_needs_interpolation(tmp_path, monkeypatch):
    """Sanity check the scenario exercises the code path this test is
    about - if OpenDrift's default profile levels ever changed to match
    depthrange, this whole cache would (correctly) become a no-op and this
    test would need to build its own mismatched z-levels instead."""
    monkeypatch.chdir(tmp_path)
    kwargs = build_advection_scenario(n_super_individuals=20, duration_years=0.02, seeding_rate=5, seed=1)
    sim = PascalAdvection(**kwargs)
    sim.run()
    si = next(s for s in sim.supindividuals if s is not None)
    assert si.depth_interpolate is True


def test_get_profile_cache_returns_consistent_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kwargs = build_advection_scenario(n_super_individuals=20, duration_years=0.02, seeding_rate=5, seed=1)
    sim = PascalAdvection(**kwargs)
    sim.run()

    si = next(s for s in sim.supindividuals if s is not None)
    cached = si.get_profile("temperature")

    # Bypass the cache to compute the same thing directly, and confirm
    # the cached value matches.
    direct = np.interp(
        si.global_settings["depthrange"],
        -si.environment_profiles["z"],
        si.environment_profiles["temperature"][:, si.environment_index],
    )
    assert np.array_equal(cached, direct)


def test_get_profile_only_interpolates_once_per_variable_per_timestep(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kwargs = build_advection_scenario(n_super_individuals=20, duration_years=0.02, seeding_rate=5, seed=1)
    sim = PascalAdvection(**kwargs)
    sim.run()

    si = next(s for s in sim.supindividuals if s is not None)
    sim.sync_environment_references()  # start from a known-clean cache
    assert si._profile_cache == {}

    with patch("pascal.individual.np.interp", wraps=np.interp) as mock_interp:
        si.get_profile("temperature")
        si.get_profile("temperature")
        si.get_zi("temperature")
        si.get_profile("food1concentration")

    # 2 distinct variables requested (temperature, food1concentration);
    # each should be interpolated exactly once regardless of how many
    # times it was requested.
    assert mock_interp.call_count == 2


def test_sync_environment_references_clears_the_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kwargs = build_advection_scenario(n_super_individuals=20, duration_years=0.02, seeding_rate=5, seed=1)
    sim = PascalAdvection(**kwargs)
    sim.run()

    si = next(s for s in sim.supindividuals if s is not None)
    si.get_profile("temperature")
    assert "temperature" in si._profile_cache

    sim.sync_environment_references()
    assert si._profile_cache == {}
