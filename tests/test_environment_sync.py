"""Regression test for the frozen-environment bug.

update_environment() (Pascal1D/PascalAdvection) creates brand new
environment/environment_profiles objects every timestep. SuperIndividual
only captures those references once, in seed() at construction time.
Without coupler.py's sync_environment_references() (called from run()
right after update_environment(), before update_lifestage()), an
already-active individual would keep reading whatever environment existed
at the timestep it was seeded - frozen for its entire lifespan - rather
than the current one.

Found while investigating why to vectorize update_lifestage: confirmed
independently in both Pascal1D and PascalAdvection using a synthetic
temperature ramp before this fix was written.
"""

import numpy as np

from pascal.coupler import Pascal1D
from pascal.scenarios import build_1d_scenario


def test_individual_environment_reference_tracks_current_timestep(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    kwargs = build_1d_scenario(
        n_super_individuals=20, n_virtual_per_super=10000,
        duration_years=0.05, seeding_rate=5, seed=1,
    )
    # Realistic-range temperature ramp: an individual reading a frozen,
    # seed-time value would report something well below the current ramp
    # value once enough timesteps have passed.
    temp = kwargs["reader"]["temperature"]
    ramp = np.linspace(2, 12, temp.shape[0])
    kwargs["reader"]["temperature"][:] = ramp[:, None, None]

    sim = Pascal1D(**kwargs)
    sim.run()

    active = [si for si in sim.supindividuals if si is not None]
    assert len(active) > 0

    current_ramp_value = ramp[sim.time_ind]
    for si in active:
        assert si.environment_profiles is sim.tracker.environment_profiles
        assert si.get_zi("temperature") == current_ramp_value
