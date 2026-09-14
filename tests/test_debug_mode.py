"""Tests for debug mode (prompt_improvements.txt feature 1): step-by-step
logging of the first super-individual's state, with a default variable set
(individual.py::DEFAULT_DEBUG_VARIABLES) when debug=True is passed instead
of an explicit list.
"""

from pascal.coupler import Pascal1D
from pascal.individual import DEFAULT_DEBUG_VARIABLES
from pascal.scenarios import build_1d_scenario


def test_debug_true_uses_default_variable_set(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kwargs = build_1d_scenario(
        n_super_individuals=5, seeding_rate=2, duration_years=0.02,
        headless="debug_default_run",
    )
    sim = Pascal1D(**kwargs, debug=True)

    assert sim.debug == list(DEFAULT_DEBUG_VARIABLES)
    assert set(sim.debug_output.keys()) == set(DEFAULT_DEBUG_VARIABLES)
    assert all(v == [] for v in sim.debug_output.values())


def test_debug_explicit_list_still_works(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kwargs = build_1d_scenario(
        n_super_individuals=5, seeding_rate=2, duration_years=0.02,
        headless="debug_explicit_run",
    )
    sim = Pascal1D(**kwargs, debug=["structuralmass", "age"])

    assert sim.debug == ["structuralmass", "age"]
    assert set(sim.debug_output.keys()) == {"structuralmass", "age"}


def test_debug_records_first_individual_over_time(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kwargs = build_1d_scenario(
        n_super_individuals=5, seeding_rate=2, duration_years=0.02,
        headless="debug_run",
    )
    sim = Pascal1D(**kwargs, debug=["structuralmass", "age"])
    sim.run()

    assert len(sim.debug_output["structuralmass"]) == len(sim.all_steps)
    assert len(sim.debug_output["age"]) == len(sim.all_steps)
