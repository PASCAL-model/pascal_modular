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

# Reused by the bgc/era5 tests below - a smaller version of
# _write_synthetic_cmems_file's own grid/time axis, since those files
# don't need velocity/temperature at all.
_BGC_ERA5_LATS = np.linspace(69.0, 70.0, 6)
_BGC_ERA5_LONS = np.linspace(13.5, 14.5, 6)


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


def _write_synthetic_bgc_file(path):
    """A minimal file with the same shape/naming contract as a real
    hpc/download_cmems_data.py BGC download (--dataset-id
    cmems_mod_arc_bgc_anfc_ecosmo_P1D-m --variables chl): a depth-varying
    `chl` variable with no CF standard_name (same convention as
    thetao/mlotst above - build_cmems_advection_scenario_from_file()'s
    bgc_variable relies on its own explicit standard_name_mapping, not
    the file carrying one already)."""
    times = pd.date_range("2022-01-01", periods=4, freq="D")
    depths = np.array([0.0, 10.0, 50.0], dtype="float32")

    rng = np.random.default_rng(1)
    shape_3d = (len(times), len(depths), len(_BGC_ERA5_LATS), len(_BGC_ERA5_LONS))

    ds = xr.Dataset(
        data_vars={
            "chl": (
                ("time", "depth", "latitude", "longitude"),
                rng.uniform(0.1, 3.0, size=shape_3d).astype("float32"),
                {"units": "mg m-3"},
            ),
        },
        coords={
            "time": times,
            "depth": ("depth", depths, {"standard_name": "depth", "axis": "Z",
                                         "positive": "down"}),
            "latitude": ("latitude", _BGC_ERA5_LATS,
                         {"standard_name": "latitude", "units": "degrees_north"}),
            "longitude": ("longitude", _BGC_ERA5_LONS,
                          {"standard_name": "longitude", "units": "degrees_east"}),
        },
    )
    ds.to_netcdf(path)


def _write_synthetic_era5_file(path):
    """A minimal file matching a real hpc/download_era5_data.py download:
    genuinely 2D (surface-only, no depth dimension at all - ERA5 single-
    level fields have none), `avg_sdswrf` with no CF standard_name (ERA5's
    own CDS netCDF output doesn't reliably set one either, hence
    era5_variable's explicit standard_name_mapping rather than relying
    on discovery). Variable name confirmed 2026-09-17 against a real CDS
    download - see pascal_docs' CMEMS+ERA5 example and Open Issues; the
    older `msdwswrf` name this fixture used to use was never actually
    correct."""
    times = pd.date_range("2022-01-01", periods=16, freq="6h")

    rng = np.random.default_rng(2)
    shape_2d = (len(times), len(_BGC_ERA5_LATS), len(_BGC_ERA5_LONS))

    ds = xr.Dataset(
        data_vars={
            "avg_sdswrf": (
                ("time", "latitude", "longitude"),
                rng.uniform(0.0, 500.0, size=shape_2d).astype("float32"),
                {"units": "W m**-2"},
            ),
        },
        coords={
            "time": times,
            "latitude": ("latitude", _BGC_ERA5_LATS,
                         {"standard_name": "latitude", "units": "degrees_north"}),
            "longitude": ("longitude", _BGC_ERA5_LONS,
                          {"standard_name": "longitude", "units": "degrees_east"}),
        },
    )
    ds.to_netcdf(path)


def test_bgc_food1concentration_alias_resolves_from_raw_short_name(tmp_path):
    cmems_file = tmp_path / "synthetic_cmems.nc"
    bgc_file = tmp_path / "synthetic_bgc.nc"
    _write_synthetic_cmems_file(cmems_file)
    _write_synthetic_bgc_file(bgc_file)

    kwargs = build_cmems_advection_scenario_from_file(
        cmems_file, bgc_file=bgc_file,
        n_super_individuals=5, start_location=(14.0, 69.5),
        start_date=dt.datetime(2022, 1, 2),
    )
    physical, bgc, constants = kwargs["reader"]
    assert "food1concentration" in bgc.variables

    env, _ = bgc.get_variables_interpolated(
        ["food1concentration"], time=dt.datetime(2022, 1, 2),
        lon=np.array([14.0]), lat=np.array([69.5]), z=np.array([-5.0]),
        profiles=["food1concentration"], profiles_depth=50,
    )
    # Real, file-derived value (uniform(0.1, 3.0) above) - not PascalDrift's
    # fallback default (0).
    assert 0.1 <= env["food1concentration"][0] <= 3.0


def test_era5_irradiance_alias_resolves_from_raw_short_name(tmp_path):
    cmems_file = tmp_path / "synthetic_cmems.nc"
    era5_file = tmp_path / "synthetic_era5.nc"
    _write_synthetic_cmems_file(cmems_file)
    _write_synthetic_era5_file(era5_file)

    kwargs = build_cmems_advection_scenario_from_file(
        cmems_file, era5_file=era5_file,
        n_super_individuals=5, start_location=(14.0, 69.5),
        start_date=dt.datetime(2022, 1, 2),
    )
    physical, era5, constants = kwargs["reader"]
    assert "irradiance" in era5.variables

    env, _ = era5.get_variables_interpolated(
        ["irradiance"], time=dt.datetime(2022, 1, 2, 6),
        lon=np.array([14.0]), lat=np.array([69.5]), z=np.array([-5.0]),
        profiles=["irradiance"], profiles_depth=50,
    )
    # Real, file-derived value (uniform(0.0, 500.0) above) - the fallback
    # (0) is inside that range, so require strictly positive instead of
    # just in-range to actually distinguish real data from the fallback.
    assert env["irradiance"][0] > 0.0


def test_full_run_with_bgc_and_era5_files_completes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cmems_file = tmp_path / "synthetic_cmems.nc"
    bgc_file = tmp_path / "synthetic_bgc.nc"
    era5_file = tmp_path / "synthetic_era5.nc"
    _write_synthetic_cmems_file(cmems_file)
    _write_synthetic_bgc_file(bgc_file)
    _write_synthetic_era5_file(era5_file)

    kwargs = build_cmems_advection_scenario_from_file(
        cmems_file, bgc_file=bgc_file, era5_file=era5_file,
        n_super_individuals=10,
        n_virtual_per_super=1000,
        duration_years=0.005,
        seeding_rate=5,
        start_location=(14.0, 69.5),
        start_date=dt.datetime(2022, 1, 1),
        headless="cmems_bgc_era5_run",
    )
    sim = PascalAdvection(**kwargs)
    sim.run()

    output_dir = tmp_path / "cmems_bgc_era5_run"
    assert (output_dir / "output_ps.nc").exists()
    assert sim.population_size() > 0
