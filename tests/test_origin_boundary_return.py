"""Tests for return-to-origin on lateral boundary exit (prompt_improvements.txt
feature 3): PascalDrift.remove_deactivated_elements() rescues elements
about to be permanently removed for 'missing_data'/'outside' deactivation by
teleporting them back to their recorded origin, instead of letting OpenDrift
physically remove them (which would invalidate environment_index the same
way coastline stranding did - see tests/test_coastline_stability.py).
"""

import numpy as np
import pytest

from pascal.coupler import PascalAdvection
from pascal.scenarios import build_advection_scenario


def test_seed_sets_tracker_origin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kwargs = build_advection_scenario(
        n_super_individuals=5, n_virtual_per_super=10000,
        duration_years=0.01, seeding_rate=0, headless="origin_seed_run",
    )
    sim = PascalAdvection(**kwargs)
    origins = [[1.0, 60.0], [2.0, 61.0], [3.0, 62.0]]
    sim.seed(3, origins)

    for si, expected in zip(sim.active_supindividuals(), origins):
        assert si.origin_lon == pytest.approx(expected[0])
        assert si.origin_lat == pytest.approx(expected[1])
        idx = si.environment_index
        assert sim.tracker.elements.origin_lon[idx] == pytest.approx(expected[0])
        assert sim.tracker.elements.origin_lat[idx] == pytest.approx(expected[1])


def test_missing_data_deactivation_bounces_back_to_origin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kwargs = build_advection_scenario(
        n_super_individuals=5, n_virtual_per_super=10000,
        duration_years=0.01, seeding_rate=0, headless="origin_rescue_run",
    )
    sim = PascalAdvection(**kwargs)
    origins = [[1.0, 60.0], [2.0, 61.0], [3.0, 62.0], [4.0, 63.0], [5.0, 64.0]]
    sim.seed(5, origins)

    n_before = sim.tracker.num_elements_active()
    assert n_before == sim.nsup

    # Move every element away from its origin, then force them all
    # 'missing_data'-deactivated the way report_missing_variables() would
    # for elements that left the reader's spatial/temporal domain.
    sim.tracker.store_present_positions()
    sim.tracker.elements.lon[:] = 99.0
    sim.tracker.elements.lat[:] = 89.0
    missing_mask = np.ones(n_before, dtype=bool)
    sim.tracker.deactivate_elements(missing_mask, reason='missing_data')
    # Deactivation is schedule-then-compact (see test_coastline_stability.py) -
    # remove_deactivated_elements() is what would normally shrink the
    # array; PascalDrift's override should rescue these first instead.
    sim.tracker.remove_deactivated_elements()

    assert sim.tracker.num_elements_active() == n_before
    np.testing.assert_allclose(sim.tracker.elements.lon, [o[0] for o in origins])
    np.testing.assert_allclose(sim.tracker.elements.lat, [o[1] for o in origins])
    active_idx = sim.tracker.status_categories.index('active')
    assert np.all(sim.tracker.elements.status == active_idx)


def test_deactivation_without_recorded_origin_is_still_removed(tmp_path, monkeypatch):
    """Elements PASCAL never seed()'ed (still the bulk seed_elements() pool
    from prep_environment(), origin_lon/lat still NaN) have nowhere to
    bounce back to, so they fall through to normal removal."""
    monkeypatch.chdir(tmp_path)
    kwargs = build_advection_scenario(
        n_super_individuals=5, n_virtual_per_super=10000,
        duration_years=0.01, seeding_rate=0, headless="origin_norescue_run",
    )
    sim = PascalAdvection(**kwargs)
    n_before = sim.tracker.num_elements_active()

    sim.tracker.store_present_positions()
    mask = np.ones(n_before, dtype=bool)
    sim.tracker.deactivate_elements(mask, reason='missing_data')
    sim.tracker.remove_deactivated_elements()

    assert sim.tracker.num_elements_active() == 0


def test_mating_spawn_origin_comes_from_a_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kwargs = build_advection_scenario(
        n_super_individuals=10, n_virtual_per_super=10000,
        duration_years=0.01, seeding_rate=0, headless="origin_mating_run",
    )
    sim = PascalAdvection(**kwargs)
    sim.seed(2, [[10.0, 70.0], [20.0, 71.0]])
    mother, father = sim.active_supindividuals()
    mother.malegenome = father.genome
    mother.male_origin = (father.origin_lon, father.origin_lat)
    mother.potentialfecundity = 1

    # Move the mother's *current* tracker position away from her origin -
    # the child's origin must track a parent's origin, not this (that's
    # inherited_locations's job - see test_start_locations.py).
    sim.tracker.elements.lon[mother.environment_index] = 55.0
    sim.tracker.elements.lat[mother.environment_index] = 55.0

    sim.respawn()

    child = sim.active_supindividuals()[-1]
    child_origin = (child.origin_lon, child.origin_lat)
    assert child_origin in {
        (mother.origin_lon, mother.origin_lat),
        (father.origin_lon, father.origin_lat),
    }
    assert child_origin != (55.0, 55.0)
