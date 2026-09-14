"""Regression test for coupler.py::apply_mortality_and_deathcheck_batch().

Compares the default (batched) PascalSimulation.update_lifestage() against
the pure per-individual reference implementation (each active individual's
own SuperIndividual.update_lifestage(), called one at a time - unchanged
by the split into run_stage_transition()/apply_mortality_and_deathcheck())
on the same seed, over a long-enough run that individuals actually reach
developmentalstage > 2 (where apply_dsc2_mortality applies) - a short run
would only exercise the dsc0 path and wouldn't catch a batching bug in the
dsc2 code specifically.
"""

from pascal.coupler import Pascal1D
from pascal.scenarios import build_1d_scenario


def _per_individual_update_lifestage(sim):
    for si in sim.supindividuals:
        if si is not None:
            si.update_lifestage()


def test_batched_mortality_matches_per_individual_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    batched_dir = tmp_path / "batched"
    batched_dir.mkdir()
    monkeypatch.chdir(batched_dir)
    kwargs = build_1d_scenario(
        n_super_individuals=30, n_virtual_per_super=10000,
        duration_years=0.3, seeding_rate=5, seed=2, headless="run",
    )
    batched_sim = Pascal1D(**kwargs)
    batched_sim.run()

    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    monkeypatch.chdir(reference_dir)
    kwargs = build_1d_scenario(
        n_super_individuals=30, n_virtual_per_super=10000,
        duration_years=0.3, seeding_rate=5, seed=2, headless="run",
    )
    reference_sim = Pascal1D(**kwargs)
    reference_sim.update_lifestage = lambda: _per_individual_update_lifestage(reference_sim)
    reference_sim.run()

    # Sanity check the scenario actually exercises stage > 2 (dsc2
    # mortality) individuals, not just the dsc0 path - otherwise this
    # test wouldn't be able to catch a dsc2-specific batching bug.
    active = [si for si in batched_sim.supindividuals if si is not None]
    assert any(si.developmentalstage > 2 for si in active)

    assert batched_sim.population_size() == reference_sim.population_size()

    # Per-individual state, not just the aggregate population sum.
    batched_active = [si for si in batched_sim.supindividuals if si is not None]
    reference_active = [si for si in reference_sim.supindividuals if si is not None]
    assert len(batched_active) == len(reference_active)
    for a, b in zip(batched_active, reference_active):
        assert a.nvindividuals == b.nvindividuals
        assert a.lifestatus == b.lifestatus
        assert a.developmentalstage == b.developmentalstage
