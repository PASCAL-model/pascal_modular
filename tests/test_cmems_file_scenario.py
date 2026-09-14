"""Offline test for build_cmems_advection_scenario_from_file().

Unlike build_cmems_advection_scenario() (the live-reader version, not in
this suite - needs network + real Copernicus Marine credentials), the
file-based version only needs a local netCDF, so it's fully testable here:
this builds a small but properly CF-tagged synthetic file (same
dims/variable names/standard_names as a real hpc/download_cmems_data.py
download - see that script's module docstring for the mapping) rather than
downloading real data, and checks the actual reader construction, variable
aliasing and a full tiny simulation run against it - not just that the
function returns without raising.
"""

import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr

from pascal.coupler import PascalAdvection
from pascal.scenarios import build_cmems_advection_scenario_from_file


def _write_synthetic_cmems_file(path):
    """A minimal file with the same shape/attrs contract as a real
    hpc/download_cmems_data.py download: thetao/mlotst without a
    standard_name PASCAL recognizes directly (relying on
    build_cmems_advection_scenario_from_file's explicit
    standard_name_mapping override, exactly as the real download's
    thetao/mlotst need), vxo/vyo carrying the real CF standard_names
    (eastward_/northward_sea_water_velocity) that OpenDrift's own synonym
    table resolves to x_/y_sea_water_velocity without any override -
    confirmed against the real dataset 2026-08-04, see
    download_cmems_data.py.
    """
    times = pd.date_range("2022-01-01", periods=4, freq="D")
    depths = np.array([0.0, 10.0, 50.0], dtype="float32")
    lats = np.linspace(69.0, 70.0, 6)
    lons = np.linspace(13.5, 14.5, 6)

    rng = np.random.default_rng(0)
    shape_3d = (len(times), len(depths), len(lats), len(lons))
    shape_2d = (len(times), len(lats), len(lons))

    ds = xr.Dataset(
        data_vars={
            "thetao": (
                ("time", "depth", "latitude", "longitude"),
                rng.uniform(2, 8, size=shape_3d).astype("float32"),
                {"standard_name": "sea_water_potential_temperature",
                 "units": "degC"},
            ),
            "mlotst": (
                ("time", "latitude", "longitude"),
                rng.uniform(20, 60, size=shape_2d).astype("float32"),
                {"standard_name":
                     "ocean_mixed_layer_thickness_defined_by_sigma_theta",
                 "units": "m"},
            ),
            "vxo": (
                ("time", "depth", "latitude", "longitude"),
                rng.uniform(-0.2, 0.2, size=shape_3d).astype("float32"),
                {"standard_name": "eastward_sea_water_velocity",
                 "units": "m s-1"},
            ),
            "vyo": (
                ("time", "depth", "latitude", "longitude"),
                rng.uniform(-0.2, 0.2, size=shape_3d).astype("float32"),
                {"standard_name": "northward_sea_water_velocity",
                 "units": "m s-1"},
            ),
        },
        coords={
            "time": times,
            "depth": ("depth", depths, {"standard_name": "depth", "axis": "Z",
                                         "positive": "down"}),
            "latitude": ("latitude", lats,
                         {"standard_name": "latitude", "units": "degrees_north"}),
            "longitude": ("longitude", lons,
                          {"standard_name": "longitude", "units": "degrees_east"}),
        },
    )
    ds.to_netcdf(path)


def test_temperature_and_mld_alias_resolve_from_raw_short_names(tmp_path):
    cmems_file = tmp_path / "synthetic_cmems.nc"
    _write_synthetic_cmems_file(cmems_file)

    kwargs = build_cmems_advection_scenario_from_file(
        cmems_file,
        n_super_individuals=5,
        start_location=(14.0, 69.5),
        start_date=dt.datetime(2022, 1, 2),
    )
    physical = kwargs["reader"][0]

    # thetao/mlotst don't carry standard_names PASCAL matches directly
    # (see _write_synthetic_cmems_file docstring) - if the override wasn't
    # wired through to the Reader constructor, these would be absent and
    # the model would silently fall back to its constant defaults instead.
    assert "temperature" in physical.variables
    assert "mld" in physical.variables
    # vxo/vyo's real CF standard_names should resolve via OpenDrift's own
    # synonym table, with no override needed - confirmed against the real
    # dataset in download_cmems_data.py.
    assert "x_sea_water_velocity" in physical.variables
    assert "y_sea_water_velocity" in physical.variables

    env, profiles = physical.get_variables_interpolated(
        ["temperature", "mld", "x_sea_water_velocity", "y_sea_water_velocity"],
        time=dt.datetime(2022, 1, 2),
        lon=np.array([14.0]), lat=np.array([69.5]), z=np.array([-5.0]),
        profiles=["temperature"], profiles_depth=50,
    )
    # Real, file-derived values (uniform(2, 8) / uniform(20, 60) above) -
    # not PascalDrift's fallback defaults (10 for temperature, 50 for mld
    # happens to overlap the range here, so check temperature specifically,
    # which can't be confused with its fallback).
    assert 2.0 <= env["temperature"][0] <= 8.0
    assert not np.isnan(env["mld"][0])


def test_full_run_against_local_file_completes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cmems_file = tmp_path / "synthetic_cmems.nc"
    _write_synthetic_cmems_file(cmems_file)

    kwargs = build_cmems_advection_scenario_from_file(
        cmems_file,
        n_super_individuals=10,
        n_virtual_per_super=1000,
        duration_years=0.005,
        seeding_rate=5,
        start_location=(14.0, 69.5),
        start_date=dt.datetime(2022, 1, 1),
        headless="cmems_file_run",
    )
    sim = PascalAdvection(**kwargs)
    sim.run()

    output_dir = tmp_path / "cmems_file_run"
    assert (output_dir / "output_ps.nc").exists()
    assert (output_dir / "lifestats.csv").exists()
    assert sim.population_size() > 0
