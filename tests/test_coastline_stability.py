"""Regression test for the HPC IndexError in respawn()/set_tracker(): see
hpc_test/results/pascal_advection_1736647.log ("IndexError: index 9970 is
out of bounds for axis 0 with size 9970").

Root cause: OpenDrift's default general:coastline_action ('stranding')
deactivates - and physically removes, via elements.move_elements()'s
boolean-mask compaction - any element that touches land. PASCAL's
environment_index (coupler.py) is a raw position into
self.tracker.elements.lon/lat, handed out once at seed() and kept for a
super-individual's whole life; any deactivation shrinks that array and
shifts every later element's position, so environment_index either goes
out of bounds (the crash above) or silently starts pointing at the wrong
particle. PascalAdvection.prep_environment() now defaults
general:coastline_action to 'previous' (bounce back to the last wet
position) instead, which never removes elements.
"""

import numpy as np

from pascal.coupler import PascalAdvection
from pascal.scenarios import build_advection_scenario


def test_coastline_action_defaults_to_previous(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    kwargs = build_advection_scenario(
        n_super_individuals=20, n_virtual_per_super=10000,
        duration_years=0.01, seeding_rate=0, headless="coastline_run",
    )
    sim = PascalAdvection(**kwargs)

    assert sim.tracker.get_config('general:coastline_action') == 'previous'


def test_elements_on_land_are_not_removed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    kwargs = build_advection_scenario(
        n_super_individuals=20, n_virtual_per_super=10000,
        duration_years=0.01, seeding_rate=0, headless="coastline_run",
    )
    sim = PascalAdvection(**kwargs)
    sim.seed(5, [kwargs["start_locations"][0]] * 5)

    n_before = sim.tracker.num_elements_active()
    assert n_before == sim.nsup

    # Force every element onto land at the current timestep - the same
    # state OpenDrift itself would produce after a real coastline check -
    # and confirm 'previous' bounces them back to their last wet position
    # rather than deactivating (and removing) them, since environment_index
    # must stay a stable position into this array for the run's lifetime.
    sim.tracker.store_present_positions()
    sim.tracker.environment.land_binary_mask = np.ones(n_before)
    sim.tracker.interact_with_coastline()
    # Deactivation is schedule-then-compact: interact_with_coastline() only
    # flags elements.status; the real per-step loop always follows it with
    # remove_deactivated_elements(), which is what actually shrinks
    # self.elements via move_elements()'s boolean-mask compaction.
    sim.tracker.remove_deactivated_elements()

    assert sim.tracker.num_elements_active() == n_before
    assert len(sim.tracker.elements.lon) == n_before
    # environment_index values (handed out at seed time, 0..nsup-1) must
    # all still be valid positions into the (unshrunk) elements arrays.
    env_indices = sim.environment_indices()
    assert max(env_indices) < len(sim.tracker.elements.lon)
