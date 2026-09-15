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
from pascal.scenarios import build_a20_advection_scenario

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


def _write_his_var(path, var_name, values, rng):
    """One history-group variable file: 3D profile variables (temp, salt,
    u_eastward, v_northward, w, AKs) carry s_rho/Cs_r/hc/h directly (needed
    by every merged member for xr.merge's join="exact"); zeta is 2D
    (surface only, no depth dim), matching the real files."""
    data_vars = {var_name: (("ocean_time", "s_rho", "eta_rho", "xi_rho"), values)}
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            **_grid_coords(HIS_TIMES),
            "s_rho": ("s_rho", S_RHO),
            "Cs_r": ("s_rho", CS_R),
            "hc": HC,
            "h": (("eta_rho", "xi_rho"), H),
        },
    )
    ds.to_netcdf(path)


def _write_his_zeta(path):
    zeta = np.zeros((len(HIS_TIMES), N_ETA, N_XI))
    ds = xr.Dataset(
        data_vars={"zeta": (("ocean_time", "eta_rho", "xi_rho"), zeta)},
        coords={
            **_grid_coords(HIS_TIMES),
            "s_rho": ("s_rho", S_RHO),
            "Cs_r": ("s_rho", CS_R),
            "hc": HC,
            "h": (("eta_rho", "xi_rho"), H),
        },
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


def _write_a20_test_data(data_dir, rng):
    shape_3d = (len(HIS_TIMES), N_S, N_ETA, N_XI)
    _write_his_var(data_dir / "reduced_temp.nc", "temp",
                    rng.uniform(2, 8, size=shape_3d), rng)
    _write_his_var(data_dir / "reduced_salt.nc", "salt",
                    rng.uniform(30, 35, size=shape_3d), rng)
    _write_his_var(data_dir / "reduced_u_eastward.nc", "u_eastward",
                    rng.uniform(-0.2, 0.2, size=shape_3d), rng)
    _write_his_var(data_dir / "reduced_v_northward.nc", "v_northward",
                    rng.uniform(-0.2, 0.2, size=shape_3d), rng)
    _write_his_var(data_dir / "reduced_w.nc", "w",
                    rng.uniform(-0.001, 0.001, size=shape_3d), rng)
    _write_his_var(data_dir / "reduced_AKs.nc", "AKs",
                    rng.uniform(0.0001, 0.01, size=shape_3d), rng)
    _write_his_zeta(data_dir / "reduced_zeta.nc")
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
