"""Tests for start-location control (prompt_improvements.txt feature 2):
loading candidate locations from a file, choosing one randomly (or cycling
in order when stochastic=False) at seed/respawn time, and mating-spawned
individuals starting at their mother's current position.
"""

import numpy as np
import pytest

from pascal.coupler import Pascal1D, PascalAdvection
from pascal.utils import load_locations_csv
from pascal.scenarios import build_1d_scenario, build_advection_scenario


def test_load_locations_csv_round_trips(tmp_path):
    path = tmp_path / "locations.csv"
    path.write_text("lon,lat\n14.25,69.8\n12.0,70.5\n16.3,71.2\n")

    locations = load_locations_csv(path)

    assert locations == [[14.25, 69.8], [12.0, 70.5], [16.3, 71.2]]


def test_select_start_locations_cycles_in_order_when_not_stochastic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kwargs = build_1d_scenario(
        n_super_individuals=5, seeding_rate=0, duration_years=0.001,
        stochastic=False, headless="cycle_run",
    )
    sim = Pascal1D(**kwargs, start_locations=[[0, 0], [1, 1], [2, 2]])

    first = sim.select_start_locations(5)
    assert first == [[0, 0], [1, 1], [2, 2], [0, 0], [1, 1]]

    # Cursor carries over across calls (initial seeding, then respawns).
    second = sim.select_start_locations(2)
    assert second == [[2, 2], [0, 0]]


def test_select_start_locations_is_random_when_stochastic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kwargs = build_1d_scenario(
        n_super_individuals=5, seeding_rate=0, duration_years=0.001,
        stochastic=True, headless="random_run",
    )
    sim = Pascal1D(**kwargs, start_locations=[[0, 0], [1, 1], [2, 2]])

    picked = sim.select_start_locations(50)
    assert len(picked) == 50
    assert all(loc in [[0, 0], [1, 1], [2, 2]] for loc in picked)
    # Not exhaustively deterministic, but with 50 draws from 3 options the
    # odds of only ever hitting one are astronomically small - a cheap
    # sanity check that this isn't secretly cycling.
    assert len({tuple(loc) for loc in picked}) > 1


def test_mating_spawn_uses_mothers_current_location(tmp_path, monkeypatch):
    """Regression test: respawn()'s inherited_locations used to be
    double-flattened by flatten_list() into a flat list of bare scalars
    (see coupler.py), which would raise a TypeError in set_tracker() the
    first time any mating-driven spawn actually happened.
    """
    monkeypatch.chdir(tmp_path)
    kwargs = build_advection_scenario(
        n_super_individuals=10, n_virtual_per_super=10000,
        duration_years=0.01, seeding_rate=0, headless="mating_location_run",
    )
    sim = PascalAdvection(**kwargs)
    sim.seed(2, [kwargs["start_locations"][0]] * 2)

    mother, father = sim.active_supindividuals()
    mother.malegenome = father.genome
    mother.male_origin = (father.origin_lon, father.origin_lat)
    mother.potentialfecundity = 1

    # Move the mother away from her origin so "current location" and
    # "origin" are distinguishable.
    sim.tracker.elements.lon[mother.environment_index] = 5.0
    sim.tracker.elements.lat[mother.environment_index] = 60.0

    sim.respawn()

    child = sim.active_supindividuals()[-1]
    assert sim.tracker.elements.lon[child.environment_index] == pytest.approx(5.0)
    assert sim.tracker.elements.lat[child.environment_index] == pytest.approx(60.0)
