"""Regression coverage for the diapause/direct-development event-logging
path (SuperIndividual.get_log_data -> OutputLogger.add_ddev/add_den/
add_dex -> OutputLogger.write_events), fixed after being found dead end
to end: seed() never passed datalogger to new SuperIndividuals (so the
`if self.datalogger is not None` guards in diapause0/diapause1 were
always False), get_log_data() referenced a nonexistent attribute
(self.tnvindividuals) and used self.structuralmass for reservemass, and
den/dex were allocated 2D but indexed with 3 subscripts. See pascal_docs'
open_issues.md for the full history.
"""

import numpy as np

from pascal.coupler import Pascal1D
from pascal.scenarios import build_1d_scenario


def test_seed_passes_datalogger_to_superindividuals():
    kwargs = build_1d_scenario(
        n_super_individuals=5, n_virtual_per_super=1000,
        duration_years=0.01, seed=0, headless="test_seed_datalogger",
    )
    sim = Pascal1D(**kwargs)
    locations = sim.select_start_locations(3)
    sim.seed(3, locations, genome=None)

    seeded = [si for si in sim.supindividuals if si is not None]
    assert len(seeded) == 3
    for si in seeded:
        assert si.datalogger is sim.datalogger


def test_get_log_data_uses_nvindividuals_and_reservemass():
    kwargs = build_1d_scenario(
        n_super_individuals=1, n_virtual_per_super=1000,
        duration_years=0.01, seed=0, headless="test_get_log_data",
    )
    sim = Pascal1D(**kwargs)
    locations = sim.select_start_locations(1)
    sim.seed(1, locations, genome=None)
    si = next(s for s in sim.supindividuals if s is not None)

    si.nvindividuals = 100.0
    si.structuralmass = 10.0
    si.reservemass = 4.0

    data = si.get_log_data()

    assert data["individuals"] == 100.0
    assert data["structuralmass"] == 10.0 * 100.0 * 1e-6
    # reservemass must come from self.reservemass, not self.structuralmass
    assert data["reservemass"] == 4.0 * 100.0 * 1e-6
    assert data["reservemass"] != data["structuralmass"]


def test_full_run_logs_real_diapause_and_direct_development_events(tmp_path, monkeypatch):
    """End-to-end: a long enough 1D run should produce non-zero ddev/den/
    dex counts in the written output, not just avoid crashing - the bug
    this guards against (datalogger never wired to SuperIndividual) would
    otherwise silently produce an all-zero, seemingly-fine output file."""
    monkeypatch.chdir(tmp_path)
    kwargs = build_1d_scenario(
        n_super_individuals=50, n_virtual_per_super=10000,
        duration_years=1.0, seed=0, headless="test_full_run_events",
    )
    sim = Pascal1D(**kwargs)
    sim.run()

    import netCDF4 as nc
    with nc.Dataset(f"{sim.outputfolder}/output_ps.nc", "r") as ds:
        assert ds["ddev_individuals"][:].sum() > 0
        assert ds["den_individuals"][:].sum() > 0
        assert ds["dex_individuals"][:].sum() > 0
        assert list(ds["devstage"][:]) == list(range(13))
