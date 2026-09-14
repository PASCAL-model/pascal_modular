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
