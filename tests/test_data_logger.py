"""Tests for OutputLogger.resolve_spatial's vectorized scatter-add.

These pin down the *intended* single-pass-per-individual accumulation
semantics (each individual's contribution counted exactly once) as a
regression test, and separately document a bug the vectorization fixed:
the original implementation had a
    for d in range(devstages):
        if any individual is in stage d:
            <accumulate ALL individuals, not just stage-d ones>
loop, so an individual's contribution was added once per *distinct*
developmental stage present in the population that timestep, not once.
With super-individuals typically spanning most of the 13 stages
simultaneously, spatial output was inflated by roughly that factor.
"""

import numpy as np
import pytest

from pascal.data_logger import OutputLogger


def make_logger(tmp_path, devstages=13, lon=(0, 1, 2), lat=(0, 1), depth=(0, 1, 2, 3)):
    output_grid = {
        "lon": list(lon),
        "lat": list(lat),
        "depth": list(depth),
        "time": np.arange(5),
    }
    return OutputLogger(
        outputfolder=str(tmp_path / "logger_test"),
        total_timesteps=5,
        output_grid=output_grid,
        devstages=devstages,
    )


def naive_single_pass_resolve(logger, cxyz, data_dict):
    """Reference implementation: the original inner loop body, run exactly
    once over every row (i.e. with the buggy outer per-stage repeat loop
    removed) -- this is the behaviour resolve_spatial should have."""
    shape = [
        logger.devstages,
        len(logger.output_grid["lon"]),
        len(logger.output_grid["lat"]),
        len(logger.output_grid["depth"]),
    ]
    gridded = {var: np.zeros(shape) for var in data_dict}

    cxyz = cxyz.copy()
    cxyz[cxyz[:, 1] > logger.max_lon, 1] = logger.max_lon
    cxyz[cxyz[:, 1] < logger.min_lon, 1] = logger.min_lon
    cxyz[cxyz[:, 2] > logger.max_lat, 2] = logger.max_lat
    cxyz[cxyz[:, 2] < logger.min_lat, 2] = logger.min_lat

    for i, row in enumerate(cxyz):
        lon_ind = int(np.floor((row[1] - logger.min_lon) / logger.lon_res))
        lat_ind = int(np.floor((row[2] - logger.min_lat) / logger.lat_res))
        for var, data in data_dict.items():
            gridded[var][int(row[0]) - 1, lon_ind, lat_ind, int(row[3])] += data[i]

    return gridded


def random_scenario(rng, n=200, devstages=13, lon=(0, 1, 2), lat=(0, 1), depth=(0, 1, 2, 3)):
    # Span multiple distinct developmental stages within one call - this is
    # exactly the condition that triggered the old overcounting bug.
    col = rng.integers(0, devstages, size=n)
    lon_vals = rng.choice(lon, size=n).astype(float)
    lat_vals = rng.choice(lat, size=n).astype(float)
    zidx = rng.integers(0, len(depth), size=n).astype(float)
    cxyz = np.stack([col.astype(float), lon_vals, lat_vals, zidx]).T

    data_dict = {
        "nvindividuals": rng.uniform(0, 1000, size=n),
        "structuralmass": rng.uniform(0, 50, size=n),
    }
    return cxyz, data_dict


def test_resolve_spatial_matches_reference_single_pass(tmp_path):
    logger = make_logger(tmp_path)
    rng = np.random.default_rng(42)
    cxyz, data_dict = random_scenario(rng)

    expected = naive_single_pass_resolve(logger, cxyz, data_dict)
    actual = logger.resolve_spatial(cxyz, data_dict)

    for var in data_dict:
        assert np.allclose(actual[var], expected[var])


def test_resolve_spatial_conserves_total_mass_across_multiple_stages(tmp_path):
    """Regression test for the overcounting bug: with individuals spread
    across many distinct stages, the grid total must equal the sum of
    inputs exactly once, not multiplied by the number of distinct stages
    present."""
    logger = make_logger(tmp_path)
    rng = np.random.default_rng(7)
    cxyz, data_dict = random_scenario(rng, n=500)

    n_distinct_stages = len(np.unique(cxyz[:, 0]))
    assert n_distinct_stages > 5  # sanity check the scenario is meaningful

    result = logger.resolve_spatial(cxyz, data_dict)
    for var, data in data_dict.items():
        assert result[var].sum() == pytest.approx(data.sum())


def test_resolve_spatial_handles_empty_input(tmp_path):
    logger = make_logger(tmp_path)
    cxyz = np.zeros((0, 4))
    data_dict = {"nvindividuals": np.zeros((0,))}
    result = logger.resolve_spatial(cxyz, data_dict)
    assert result["nvindividuals"].sum() == 0


# --- add_ddev/add_den/add_dex/write_events: regression coverage for a
# previously entirely-dead code path (SuperIndividual never actually had a
# datalogger reference passed to it via coupler.py's seed(), so none of
# this had ever been exercised in a real run - see pascal_docs'
# open_issues.md) ---


def test_add_ddev_accumulates_at_current_timestep(tmp_path):
    logger = make_logger(tmp_path)
    logger.current_timestep = 2
    logger.add_ddev({"individuals": 5.0, "structuralmass": 1.5}, col=1)

    assert logger.ddev["individuals"][2, 1] == 5.0
    assert logger.ddev["structuralmass"][2, 1] == 1.5
    # untouched elsewhere
    assert logger.ddev["individuals"].sum() == 5.0


def test_add_den_and_add_dex_use_three_indices(tmp_path):
    """den/dex are indexed (timestep, stage, mode) - a 3-index write into
    what used to be 2D arrays (see open_issues.md); this pins the fixed
    shape down as a regression test."""
    logger = make_logger(tmp_path)
    logger.current_timestep = 0

    logger.add_den({"individuals": 3.0}, col1=0, col2=1)
    assert logger.den["individuals"][0, 0, 1] == 3.0
    assert logger.den["individuals"].shape == (5, 2, 2)

    logger.add_dex({"individuals": 4.0}, col1=1, col2=0)
    assert logger.dex["individuals"][0, 1, 0] == 4.0
    assert logger.dex["individuals"].shape == (5, 2, 2)


def _log_empty_spatial_timesteps(logger, n):
    """write_spatial() expects one log_spatial() call per timestep (as a
    real run.py's log_spatial()-every-outer-iteration would produce) -
    empty per-timestep contributions are enough to exercise write_spatial()
    itself without needing a full simulation."""
    empty_cxyz = np.zeros((0, 4))
    empty_data = {var: np.zeros((0,)) for var in logger.spatial_var_list}
    for _ in range(n):
        logger.log_spatial(empty_cxyz, empty_data)


def test_write_events_writes_accumulated_data(tmp_path):
    logger = make_logger(tmp_path)
    _log_empty_spatial_timesteps(logger, logger.total_timesteps)
    logger.current_timestep = 1
    logger.add_ddev({"individuals": 10.0, "structuralmass": 2.0, "reservemass": 1.0}, col=0)
    logger.add_den({"individuals": 7.0, "structuralmass": 3.0, "reservemass": 2.0}, col1=1, col2=0)
    logger.add_dex({"individuals": 6.0, "structuralmass": 1.0, "reservemass": 0.5}, col1=0, col2=0)

    # write_events() reopens the file write_spatial() creates, so both
    # must run for a real end-of-run output file.
    logger.write_spatial()
    logger.write_events()

    import netCDF4 as nc
    with nc.Dataset(logger.outputfile, "r") as ds:
        assert ds["ddev_individuals"][1, 0] == 10.0
        assert ds["den_structuralmass"][1, 1, 0] == 3.0
        assert ds["dex_reservemass"][1, 0, 0] == 0.5
        # genome_log is written even though nothing in this codebase ever
        # calls log_evolvable() - documented as all-zero, not an error.
        assert ds["genome_log"][:].sum() == 0


def test_write_spatial_devstage_coordinate_is_populated(tmp_path):
    logger = make_logger(tmp_path, devstages=13)
    _log_empty_spatial_timesteps(logger, logger.total_timesteps)
    logger.write_spatial()

    import netCDF4 as nc
    with nc.Dataset(logger.outputfile, "r") as ds:
        assert list(ds["devstage"][:]) == list(range(13))
