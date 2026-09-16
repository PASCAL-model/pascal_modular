"""Offline test for build_a20_advection_scenario().

Builds a tiny but structurally-realistic set of A20 ROMS/ECOSMO-style
netCDF files (one variable per file, split across the same history (his)/
diagnostic (dia)/quicksave (qck) groups the real A20 output uses, and
missing the same grid metadata (mask_rho/Vtransform) the real "reduced"
extract lacks - see pascal_modular's pascal.scenarios::build_a20_readers()
and its _inject_a20_grid_placeholders()/_build_a20_food_reader()/
_build_a20_swrad_reader() helpers, ported from a20_test/roms_readers.py).
Exercises the actual reader construction/aliasing and a full tiny
simulation run against it, not just that the function returns without
raising.
"""

import datetime as dt

import numpy as np
import xarray as xr

from pascal.coupler import PascalAdvection
from pascal.scenarios import (
    build_a20_advection_scenario,
    build_a20_readers,
    build_pred1dens_readers,
)

N_ETA, N_XI, N_S = 3, 3, 3
LAT_RHO = np.array([[69.0, 69.0, 69.0], [69.1, 69.1, 69.1], [69.2, 69.2, 69.2]])
LON_RHO = np.array([[14.0, 14.1, 14.2]] * 3)
S_RHO = np.linspace(-0.9, -0.1, N_S)
CS_R = np.linspace(-1.0, 0.0, N_S)
HC = 100.0
H = np.full((N_ETA, N_XI), 200.0)

TIME_UNITS = "seconds since 1995-01-24 00:00:00"
# 5 daily steps on the his/dia grid, more than enough to cover the short
# test run below.
HIS_TIMES = np.arange(5) * 86400.0
DIA_TIMES = HIS_TIMES + 43200.0  # noon-offset, like the real diagnostic group
QCK_TIMES = np.arange(5 * 24) * 3600.0  # hourly, full coverage (unlike the
# real reduced extract's truncated swrad file - not what this test checks)


def _grid_coords(ocean_time):
    return {
        "ocean_time": ("ocean_time", ocean_time, {"units": TIME_UNITS}),
        "lat_rho": (("eta_rho", "xi_rho"), LAT_RHO),
        "lon_rho": (("eta_rho", "xi_rho"), LON_RHO),
    }


def _write_his_var(path, var_name, values, rng, include_hc=True):
    """One history-group variable file: 3D profile variables (temp, salt,
    u_eastward, v_northward, w, AKs) carry s_rho/Cs_r/h directly (needed
    by every merged member for xr.merge's join="exact"), and hc too
    unless include_hc=False (some real extracts carry their own hc, some
    don't - see pascal.scenarios::A20_HC); zeta is 2D (surface only, no
    depth dim), matching the real files."""
    data_vars = {var_name: (("ocean_time", "s_rho", "eta_rho", "xi_rho"), values)}
    coords = {
        **_grid_coords(HIS_TIMES),
        "s_rho": ("s_rho", S_RHO),
        "Cs_r": ("s_rho", CS_R),
        "h": (("eta_rho", "xi_rho"), H),
    }
    if include_hc:
        coords["hc"] = HC
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.to_netcdf(path)


def _write_his_zeta(path, include_hc=True):
    zeta = np.zeros((len(HIS_TIMES), N_ETA, N_XI))
    coords = {
        **_grid_coords(HIS_TIMES),
        "s_rho": ("s_rho", S_RHO),
        "Cs_r": ("s_rho", CS_R),
        "h": (("eta_rho", "xi_rho"), H),
    }
    if include_hc:
        coords["hc"] = HC
    ds = xr.Dataset(
        data_vars={"zeta": (("ocean_time", "eta_rho", "xi_rho"), zeta)},
        coords=coords,
    )
    ds.to_netcdf(path)


def _write_dia_food(path, rng):
    """Diagnostic-group food1concentration file: has its own s_rho (a
    depth profile variable) but, like the real reduced extract, no
    Cs_r/hc/h/zeta of its own - build_a20_readers() borrows those from the
    his group."""
    chl = rng.uniform(0.01, 0.4, size=(len(DIA_TIMES), N_S, N_ETA, N_XI))
    ds = xr.Dataset(
        data_vars={"Chl_bc": (("ocean_time", "s_rho", "eta_rho", "xi_rho"), chl)},
        coords={**_grid_coords(DIA_TIMES), "s_rho": ("s_rho", S_RHO)},
    )
    ds.to_netcdf(path)


def _write_qck_swrad(path, rng):
    """Quicksave-group irradiance file: genuinely 2D (no s_rho at all,
    unlike the profile variables above), matching the real extract."""
    swrad = rng.uniform(0.0, 0.3, size=(len(QCK_TIMES), N_ETA, N_XI))
    ds = xr.Dataset(
        data_vars={"swrad": (("ocean_time", "eta_rho", "xi_rho"), swrad)},
        coords=_grid_coords(QCK_TIMES),
    )
    ds.to_netcdf(path)


def _write_a20_test_data(data_dir, rng, include_hc=True):
    shape_3d = (len(HIS_TIMES), N_S, N_ETA, N_XI)
    _write_his_var(data_dir / "reduced_temp.nc", "temp",
                    rng.uniform(2, 8, size=shape_3d), rng, include_hc=include_hc)
    _write_his_var(data_dir / "reduced_salt.nc", "salt",
                    rng.uniform(30, 35, size=shape_3d), rng, include_hc=include_hc)
    # File named reduced_u.nc/reduced_v.nc (not reduced_u_eastward.nc/
    # reduced_v_northward.nc) - matches the real extract's convention
    # (see pascal.scenarios::A20_HIS_FILE_NAMES); the variable *inside*
    # is still named u_eastward/v_northward.
    _write_his_var(data_dir / "reduced_u.nc", "u_eastward",
                    rng.uniform(-0.2, 0.2, size=shape_3d), rng, include_hc=include_hc)
    _write_his_var(data_dir / "reduced_v.nc", "v_northward",
                    rng.uniform(-0.2, 0.2, size=shape_3d), rng, include_hc=include_hc)
    _write_his_var(data_dir / "reduced_w.nc", "w",
                    rng.uniform(-0.001, 0.001, size=shape_3d), rng, include_hc=include_hc)
    _write_his_var(data_dir / "reduced_AKs.nc", "AKs",
                    rng.uniform(0.0001, 0.01, size=shape_3d), rng, include_hc=include_hc)
    _write_his_zeta(data_dir / "reduced_zeta.nc", include_hc=include_hc)
    _write_dia_food(data_dir / "reduced_Chl_bc.nc", rng)
    _write_qck_swrad(data_dir / "reduced_swrad.nc", rng)


def test_a20_readers_alias_and_supply_real_data(tmp_path):
    rng = np.random.default_rng(0)
    _write_a20_test_data(tmp_path, rng)

    kwargs = build_a20_advection_scenario(
        tmp_path, n_super_individuals=5, start_location=(14.1, 69.1),
        start_date=dt.datetime(1995, 1, 24, 6),
    )
    physical, food, swrad, constants = kwargs["reader"]

    # temp/Chl_bc/swrad don't carry standard_names PASCAL matches directly -
    # if the standard_name_mapping override wasn't wired through, these
    # would be absent and the model would silently fall back to its
    # constant defaults instead.
    assert "temperature" in physical.variables
    assert "food1concentration" in food.variables
    assert "irradiance" in swrad.variables

    env, profiles = physical.get_variables_interpolated(
        ["temperature"], time=dt.datetime(1995, 1, 24, 6),
        lon=np.array([14.1]), lat=np.array([69.1]), z=np.array([-5.0]),
        profiles=["temperature"], profiles_depth=50,
    )
    # Real, file-derived values (uniform(2, 8) above) - not PascalDrift's
    # fallback default (10).
    assert 2.0 <= env["temperature"][0] <= 8.0

    food_env, _ = food.get_variables_interpolated(
        ["food1concentration"], time=dt.datetime(1995, 1, 24, 12),
        lon=np.array([14.1]), lat=np.array([69.1]), z=np.array([-5.0]),
        profiles=["food1concentration"], profiles_depth=50,
    )
    assert 0.01 <= food_env["food1concentration"][0] <= 0.4


def test_full_run_against_a20_test_data_completes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "a20_data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    _write_a20_test_data(data_dir, rng)

    kwargs = build_a20_advection_scenario(
        data_dir,
        n_super_individuals=5,
        n_virtual_per_super=1000,
        duration_years=0.005,
        seeding_rate=5,
        start_location=(14.1, 69.1),
        start_date=dt.datetime(1995, 1, 24, 6),
        headless="a20_scenario_run",
    )
    sim = PascalAdvection(**kwargs)
    sim.run()

    output_dir = tmp_path / "a20_scenario_run"
    assert (output_dir / "output_ps.nc").exists()
    assert (output_dir / "lifestats.csv").exists()
    assert sim.population_size() > 0


# Deliberately a wider, plain regular lat/lon/depth grid than the A20 ROMS
# fixture above (mirrors the real a20_test/pred_data files: no curvilinear
# ROMS grid or missing-metadata workarounds needed here) - covers the same
# (14.1, 69.1) test point used throughout this file.
PRED_LON = np.array([13.0, 14.0, 15.0])
PRED_LAT = np.array([68.0, 69.0, 70.0])
PRED_DEPTH = np.array([0.0, 10.0, 50.0, 100.0])


def _write_pred_year_file(pred_data_dir, year, value_range, rng, days=30):
    """One a20_test/pred_data-style year file: vpdens on a plain
    (time, depth, lat, lon) grid, no standard_name (aliased by raw
    variable name, same as build_pred1dens_readers() expects),
    six-hourly timesteps on its own "hours since <year>-01-01" axis -
    `days` worth by default (not a full year, unlike the real files),
    enough to cover the short test runs below (which start ~24 days in,
    matching the A20 ROMS fixture's own start_date)."""
    year_dir = pred_data_dir / str(year)
    year_dir.mkdir(parents=True)
    times = np.arange(0, days * 24, 6, dtype="float64")
    lo, hi = value_range
    vpdens = rng.uniform(lo, hi, size=(len(times), len(PRED_DEPTH),
                                        len(PRED_LAT), len(PRED_LON)))
    ds = xr.Dataset(
        data_vars={
            "vpdens": (("time", "depth", "lat", "lon"), vpdens.astype("float32"),
                       {"long_name": "synthetic visual predator density",
                        "units": "1"}),
        },
        coords={
            "time": ("time", times,
                      {"units": f"hours since {year}-01-01 00:00:00",
                       "standard_name": "time"}),
            "depth": ("depth", PRED_DEPTH,
                      {"units": "m", "standard_name": "depth", "positive": "down"}),
            "lat": ("lat", PRED_LAT,
                    {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": ("lon", PRED_LON,
                    {"units": "degrees_east", "standard_name": "longitude"}),
        },
    )
    ds.to_netcdf(year_dir / "vpdens.nc")


def test_pred1dens_readers_alias_and_cover_each_year(tmp_path):
    rng = np.random.default_rng(0)
    pred_data_dir = tmp_path / "pred_data"
    _write_pred_year_file(pred_data_dir, 1995, (0.1, 0.2), rng)
    _write_pred_year_file(pred_data_dir, 1996, (0.5, 0.6), rng)

    readers = build_pred1dens_readers(pred_data_dir)
    assert len(readers) == 2
    for reader in readers:
        # vpdens carries no standard_name PASCAL matches directly - if the
        # standard_name_mapping override wasn't wired through, pred1dens
        # would be absent here.
        assert "pred1dens" in reader.variables

    readers_by_year = {r.name: r for r in readers}
    r1995 = readers_by_year["pred1dens_1995"]
    r1996 = readers_by_year["pred1dens_1996"]

    env_1995, _ = r1995.get_variables_interpolated(
        ["pred1dens"], time=dt.datetime(1995, 1, 1, 6),
        lon=np.array([14.0]), lat=np.array([69.0]), z=np.array([-5.0]),
        profiles=["pred1dens"], profiles_depth=50,
    )
    assert 0.1 <= env_1995["pred1dens"][0] <= 0.2

    env_1996, _ = r1996.get_variables_interpolated(
        ["pred1dens"], time=dt.datetime(1996, 1, 1, 6),
        lon=np.array([14.0]), lat=np.array([69.0]), z=np.array([-5.0]),
        profiles=["pred1dens"], profiles_depth=50,
    )
    assert 0.5 <= env_1996["pred1dens"][0] <= 0.6


def test_full_a20_run_reads_real_pred1dens_instead_of_fallback_constant(
    tmp_path, monkeypatch,
):
    """The whole point of wiring pred_data_dir in: env_pred1dens in a real
    run should reflect the real per-year field (0.1-0.2 here), not
    build_a20_readers()'s ConstantReader fallback (0.00001)."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)

    data_dir = tmp_path / "a20_data"
    data_dir.mkdir()
    _write_a20_test_data(data_dir, rng)

    pred_data_dir = tmp_path / "pred_data"
    _write_pred_year_file(pred_data_dir, 1995, (0.1, 0.2), rng)

    kwargs = build_a20_advection_scenario(
        data_dir,
        n_super_individuals=5,
        n_virtual_per_super=1000,
        duration_years=0.005,
        seeding_rate=5,
        start_location=(14.1, 69.1),
        start_date=dt.datetime(1995, 1, 24, 6),
        pred_data_dir=pred_data_dir,
        headless="a20_pred_run",
    )
    sim = PascalAdvection(**kwargs, debug=["env_pred1dens"], verbose=True)
    sim.run()

    assert sim.population_size() > 0
    pred_series = np.asarray(sim.debug_output["env_pred1dens"], dtype=float)
    assert len(pred_series) > 0
    assert np.all((pred_series >= 0.1) & (pred_series <= 0.2))


def _write_deterministic_temp_salt_zeta(data_dir, land_idx=(1, 1)):
    """Overwrites reduced_temp.nc/reduced_salt.nc/reduced_zeta.nc (already
    written with random noise by _write_a20_test_data()) with a
    deterministic setup for testing _a20_mld_from_density()/the real land
    mask (see build_a20_readers()'s/_inject_a20_grid_placeholders()'s
    2026-09-16 updates): one column permanently NaN at every timestep
    (land_idx - land, matching real ROMS _FillValue-masked land cells,
    where every variable is NaN, not just zeta); a "pycnocline" column
    (0, 0) with the two shallowest sigma layers identical and the deepest
    one colder/saltier (denser); and a "well-mixed" column (2, 2), left
    uniform top-to-bottom by the fill below - never crosses
    mld_threshold()'s density criterion, so it should report the
    "mixed to the bed" branch instead of a shallow mld.
    """
    shape_3d = (len(HIS_TIMES), N_S, N_ETA, N_XI)
    temp = np.full(shape_3d, 5.0)
    salt = np.full(shape_3d, 34.0)

    # Pycnocline column: index order is bottom-up (see S_RHO/CS_R above,
    # and z_r's own derivation) - so [1:] (indices 1, 2) is the two
    # shallowest layers, [0] the deepest.
    temp[:, 1:, 0, 0] = 8.0
    salt[:, 1:, 0, 0] = 34.0
    temp[:, 0, 0, 0] = 2.0
    salt[:, 0, 0, 0] = 35.0

    zeta = np.zeros((len(HIS_TIMES), N_ETA, N_XI))
    ei, xi = land_idx
    temp[:, :, ei, xi] = np.nan
    salt[:, :, ei, xi] = np.nan
    zeta[:, ei, xi] = np.nan

    grid_coords = {
        **_grid_coords(HIS_TIMES),
        "s_rho": ("s_rho", S_RHO),
        "Cs_r": ("s_rho", CS_R),
        "hc": HC,
        "h": (("eta_rho", "xi_rho"), H),
    }
    xr.Dataset(
        data_vars={"temp": (("ocean_time", "s_rho", "eta_rho", "xi_rho"), temp)},
        coords=grid_coords,
    ).to_netcdf(data_dir / "reduced_temp.nc")
    xr.Dataset(
        data_vars={"salt": (("ocean_time", "s_rho", "eta_rho", "xi_rho"), salt)},
        coords=grid_coords,
    ).to_netcdf(data_dir / "reduced_salt.nc")
    xr.Dataset(
        data_vars={"zeta": (("ocean_time", "eta_rho", "xi_rho"), zeta)},
        coords=grid_coords,
    ).to_netcdf(data_dir / "reduced_zeta.nc")


def test_mld_computed_from_density_reflects_stratification(tmp_path):
    """mld should come from PASCAL's own density calculation now (see
    pascal.scenarios::_a20_mld_from_density()), not the flat 30 m
    constant build_a20_readers() used to fall back to for every cell/time
    - a column with a real pycnocline should report a shallower mld than
    a well-mixed column with no pycnocline at all."""
    rng = np.random.default_rng(0)
    _write_a20_test_data(tmp_path, rng)
    _write_deterministic_temp_salt_zeta(tmp_path)

    readers = build_a20_readers(tmp_path)
    physical = readers[0]
    assert "mld" in physical.variables

    env, _ = physical.get_variables_interpolated(
        ["mld"], time=dt.datetime(1995, 1, 24, 6),
        lon=np.array([LON_RHO[0, 0], LON_RHO[2, 2]]),
        lat=np.array([LAT_RHO[0, 0], LAT_RHO[2, 2]]),
        z=np.array([-5.0, -5.0]),
    )
    mld_pycnocline, mld_wellmixed = env["mld"]

    assert np.isfinite(mld_pycnocline) and np.isfinite(mld_wellmixed)
    assert 0 < mld_pycnocline < H[0, 0]
    # The well-mixed column never crosses the density threshold, so it
    # should fall back to mld_threshold()'s "mixed to the bed" branch -
    # close to the full ~193 m sigma-layer stack depth (H=200 minus the
    # bottom sigma layer's own offset from the true seabed).
    assert mld_wellmixed > 150
    assert mld_pycnocline < 150
    assert mld_pycnocline < mld_wellmixed


def test_real_land_mask_from_nan_zeta(tmp_path):
    """_inject_a20_grid_placeholders() used to set mask_rho all-ones
    (every cell "water") regardless of the data; it's now derived from
    zeta's own NaN pattern (see its 2026-09-16 update) - a cell that's
    NaN at every timestep should report land_binary_mask == 1, not 0."""
    rng = np.random.default_rng(0)
    _write_a20_test_data(tmp_path, rng)
    _write_deterministic_temp_salt_zeta(tmp_path, land_idx=(1, 1))

    readers = build_a20_readers(tmp_path)
    physical = readers[0]

    env, _ = physical.get_variables_interpolated(
        ["land_binary_mask"], time=dt.datetime(1995, 1, 24, 6),
        lon=np.array([LON_RHO[1, 1], LON_RHO[0, 0]]),
        lat=np.array([LAT_RHO[1, 1], LAT_RHO[0, 0]]),
        z=np.array([0.0, 0.0]),
    )
    land_flag, water_flag = env["land_binary_mask"]
    assert land_flag == 1.0
    assert water_flag == 0.0


def test_a20_readers_work_when_source_files_have_no_hc(tmp_path):
    """Some real A20 extracts carry their own `hc` variable in every
    reduced_*.nc file (e.g. a20_test/a20_data's original 1995 extract),
    others don't (e.g. the 2026-09-16 2019-2020 subset) - both
    reader_ROMS_native's own depth calculation and
    _a20_mld_from_density() need hc regardless, so
    _inject_a20_grid_placeholders() injects A20_HC unconditionally
    rather than only filling a gap. Confirms build_a20_readers() still
    works, and mld still computes a real value, when no source file has
    hc at all."""
    rng = np.random.default_rng(0)
    _write_a20_test_data(tmp_path, rng, include_hc=False)

    readers = build_a20_readers(tmp_path)
    physical = readers[0]

    env, _ = physical.get_variables_interpolated(
        ["mld", "temperature"], time=dt.datetime(1995, 1, 24, 6),
        lon=np.array([14.1]), lat=np.array([69.1]), z=np.array([-5.0]),
    )
    assert np.isfinite(env["mld"][0])
    assert 0 < env["mld"][0] < H[0, 0]
    assert 2.0 <= env["temperature"][0] <= 8.0
