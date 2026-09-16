"""Scenario builders: turn explicit kwargs into the kwargs PascalSimulation
subclasses expect.

Originally written for synthetic scenario benchmarking/profiling (no
dependency on real netCDF input data or the (currently missing)
aux_funcs.py helper that the old testcase/ scripts relied on, so runs stay
reproducible on any machine with the conda environment installed) but also
used directly by real 3D/CMEMS-backed runs (build_cmems_advection_scenario*).

Values for genuinely calibrated constants (critical molting mass thresholds,
developmental coefficients) in build_global_settings() are illustrative,
not validated against field data -- fine for timing/scaling/profiling work,
not for scientific runs; real runs should come from a YAML config instead
(see pascal_run's src/scenarios.py::build_scenario_from_config, which
overlays config['biology'] via build_global_settings_from_config).
"""

import datetime as dt
import os
from pathlib import Path

import numpy as np

from pascal.coupler import DEFAULT_DEPTHRANGE

PROFILE_VARS = ["temperature", "food1concentration", "irradiance", "pred1dens"]

# A20 ROMS/ECOSMO output variable -> file name, one variable per file (see
# build_a20_advection_scenario()). u_eastward/v_northward already carry
# standard_names identifying them as earth-relative (not grid-relative)
# velocity, so reader_ROMS_native skips rotation automatically - no grid
# "angle" needed, unlike native ROMS u/v.
A20_HIS_VARS = ["temp", "salt", "u_eastward", "v_northward", "w", "AKs", "zeta"]


def compute_total_tsteps(start_date, duration_years, timestep, isplit=1):
    """Mirror PascalSimulation's timestep-count logic so synthetic arrays
    are sized exactly as long as the run will need them to be."""
    end_time = start_date + dt.timedelta(days=365 * duration_years)
    total_tsteps = int(np.ceil((end_time - start_date) / timestep))
    rem = total_tsteps % isplit
    if rem != 0:
        total_tsteps -= (isplit - rem)
    return total_tsteps


def build_global_settings(stochastic=True, rng=None):
    """Synthetic but structurally-plausible global settings.

    cmm_lower/cmm_upper are monotonically increasing per developmental stage
    (0=egg .. 12=adult), matching the shape/ordering the real model expects.
    """
    cmm_mid = np.array(
        [0.3, 0.5, 0.8, 1.5, 3, 6, 12, 25, 45, 70, 110, 160, 230]
    )
    return {
        "developmentalcoefficient": np.array([595.00, 388.00, 581.00]),
        "maxirradiance": 0.3,
        "minirradiance": 0.001,
        "cmm_lower": cmm_mid * 0.9,
        "cmm_upper": cmm_mid * 1.1,
        "ageceiling": 2180,
        "fecundityceiling": 10,
        "nonvisualpredatorreldensity": 0.00075,
        "backgroundmortalityrisk": 0.00075,
        "virtualindividualthrehold": 100,
        "diapausemetabolicrateadj0": 0.2500,
        "diapausemetabolicrateadj1": 0.5000,
        "energyallocthreshold1": 38.00,
        "energyallocthreshold2": 159.00,
        "stochastic": stochastic,
        "diapausedepththreshold0": 50,
        "diapausedepththreshold1": 500,
        "diapausedepththreshold2": 1200,
        "maxmatingdistance": 1000,
    }


def build_1d_reader(time_len, depthrange=DEFAULT_DEPTHRANGE, rng=None):
    """Synthetic single-water-column environment for Pascal1D.

    Shapes follow Pascal1D.update_environment's expectations:
      - profile variables: [time, depth, n_index]
      - 'mld' (not depth-resolved): [time, n_index]
      - 'z': depth levels, static (no leading time dim)
    n_index is 1 because Pascal1D pins every super-individual to the same
    (only) spatial column.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_depth = len(depthrange)
    n_index = 1

    surface_irradiance = 0.25
    attenuation = 0.05  # per depth-index, loose stand-in for a real k_d profile
    irradiance_profile = surface_irradiance * np.exp(
        -attenuation * np.arange(n_depth)
    )

    data = {
        "z": -np.asarray(depthrange, dtype=float),
        "temperature": rng.uniform(2, 10, size=(time_len, n_depth, n_index)),
        "food1concentration": rng.uniform(
            0.01, 0.4, size=(time_len, n_depth, n_index)
        ),
        "irradiance": np.tile(
            irradiance_profile[None, :, None], (time_len, 1, n_index)
        ),
        "pred1dens": rng.uniform(0, 0.001, size=(time_len, n_depth, n_index)),
        "mld": np.full((time_len, n_index), 50.0),
    }
    return data


def build_1d_scenario(
    n_super_individuals=50,
    n_virtual_per_super=10000,
    duration_years=0.05,
    timestep_seconds=21600,
    seeding_rate=10,
    stochastic=True,
    seed=0,
    headless="bench_run",
):
    """Return kwargs ready to pass to coupler.Pascal1D(**kwargs)."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # the model itself uses the legacy global RNG

    timestep = dt.timedelta(seconds=timestep_seconds)
    start_date = dt.datetime(2010, 1, 1)

    total_tsteps = compute_total_tsteps(start_date, duration_years, timestep)
    # +5 timesteps of headroom: update_environment() is called once more
    # than the outer step count during prep_environment().
    reader = build_1d_reader(total_tsteps + 5, rng=rng)

    return {
        "nsupindividuals": n_super_individuals,
        "nvindividualspersupindividual": n_virtual_per_super,
        "global_settings": build_global_settings(stochastic=stochastic, rng=rng),
        "reader": reader,
        "timestep": timestep,
        "start_date": start_date,
        "duration": duration_years,
        "seeding_rate": seeding_rate,
        "headless": headless,
    }


def build_advection_scenario(
    n_super_individuals=50,
    n_virtual_per_super=10000,
    duration_years=0.05,
    timestep_seconds=21600,
    seeding_rate=10,
    stochastic=True,
    seed=0,
    headless="bench_run",
):
    """Return kwargs ready to pass to coupler.PascalAdvection(**kwargs)
    (or coupler_parallel.PascalAdvectionParallel(**kwargs)).

    Unlike Pascal1D (where every super-individual shares a single
    environment_index=0, so environment_profiles arrays only ever have one
    column), this gives each super-individual its own OpenDrift tracker
    element/column via a spatially-uniform ConstantReader. That's what
    actually exercises the thing coupler_parallel.py's per-individual
    slicing exists for: in this scenario environment_profiles arrays have
    shape (n_depth, n_super_individuals), and every SuperIndividual holds a
    reference to the *whole* array, not just its own column.
    """
    from opendrift.readers.reader_constant import Reader as ConstantReader

    np.random.seed(seed)

    timestep = dt.timedelta(seconds=timestep_seconds)
    start_date = dt.datetime(2010, 1, 1)

    reader = ConstantReader({
        "x_sea_water_velocity": 0.1,
        "y_sea_water_velocity": 0.05,
        "x_wind": 0,
        "y_wind": 0,
        "temperature": 8,
        "sea_water_salinity": 35,
        "land_binary_mask": 0,
        "ocean_vertical_diffusivity": 0.02,
        "food1concentration": 0.05,
        "irradiance": 0.1,
        "pred1dens": 0.00001,
        "mld": 100,
    })

    outputgrid = {"lon": [-1, 0, 1], "lat": [69, 70, 71]}

    return {
        "nsupindividuals": n_super_individuals,
        "nvindividualspersupindividual": n_virtual_per_super,
        "global_settings": build_global_settings(stochastic=stochastic),
        "reader": reader,
        "timestep": timestep,
        "start_date": start_date,
        "duration": duration_years,
        "seeding_rate": seeding_rate,
        "start_locations": [[0.0, 70.0]],
        "outputgrid": outputgrid,
        "headless": headless,
    }


def _alias_reader_variable(reader, new_name, existing_name):
    """Register `new_name` as an alias for `existing_name` on an
    already-constructed CF-based reader (e.g. reader_copernicusmarine.Reader).

    reader.variable_mapping (built once in __init__) maps standard_name ->
    raw file variable name; reader.variables is a *snapshot* list taken
    from variable_mapping.keys() at construction time, not a live view of
    it - so both need updating for the reader to actually offer the alias
    to OpenDrift's environment system.
    """
    reader.variable_mapping[new_name] = reader.variable_mapping[existing_name]
    if new_name not in reader.variables:
        reader.variables.append(new_name)


def build_cmems_advection_scenario(
    n_super_individuals=50,
    n_virtual_per_super=10000,
    duration_years=0.05,
    timestep_seconds=21600,
    seeding_rate=10,
    stochastic=True,
    seed=0,
    start_date=None,
    start_location=(14.25, 69.8),
    food1concentration_constant=0.05,
    pred1dens_constant=0.00001,
    irradiance_constant=0.1,
    headless="bench_run",
):
    """Return kwargs ready to pass to coupler.PascalAdvection(**kwargs),
    backed by live Copernicus Marine (CMEMS) physical data instead of a
    synthetic reader - for checking the realistic 3D deployment path
    specifically, not for routine/repeatable benchmarking (this hits a
    real, rate-limited external API and needs network + credentials - see
    pascal_benchmark's BENCHMARKING.md "live-CMEMS reader configuration
    feeds PASCAL zero real data" section for why this function exists at
    all).

    PASCAL's required_variables (pascal_drift.py) use short internal names
    that don't match CMEMS's real CF standard names, so nothing gets
    matched and everything silently falls back to constant defaults unless
    aliased. Confirmed mapping (2026-08-04, reviewed):
        temperature -> sea_water_temperature       (physical reader)
        mld         -> ocean_mixed_layer_thickness (physical reader)
    x/y_sea_water_velocity need no alias - OpenDrift's reader machinery
    already rotates eastward/northward -> x/y internally.
    ocean_vertical_diffusivity/land_binary_mask need no alias either -
    both handled via tracker_config below (a parameterized diffusivity
    model and OpenDrift's auto-landmask), not reader lookup.

    Only the physical CMEMS product is fetched here - food1concentration
    is given as a constant (not sourced from the biogeochemistry reader),
    per 2026-08-05 decision: the food1concentration alias to
    mass_concentration_of_chlorophyll_a_in_sea_water was confirmed
    correctly *registered*, but returned 0.0 in practice via OpenDrift's
    interpolated fetch despite the raw underlying variable having real
    data nearby - at the time, an unresolved OpenDrift/reader
    interpolation issue (see pascal_benchmark's BENCHMARKING.md for the
    full writeup).

    UPDATE 2026-09-16, root cause found and fixed upstream: this and the
    next paragraph's symptom were the same bug, not two separate ones -
    `Environment.get_environment()` only initialized `env_profiles` from
    whichever reader group happened to run first, so any profile variable
    *only* supplied by a later reader group (the bgc reader here; a
    ConstantReader after a physical reader in build_advection_scenario()-
    style setups) was silently dropped and replaced by PASCAL's fallback
    constant instead of raising - see a20_test/README.md's "Bug 3" for the
    full mechanism and the opendrift fork's fix (with regression tests).
    Not yet re-verified here specifically: whether re-enabling the bgc
    reader now correctly returns real chlorophyll data (the bug that
    silently zeroed it is fixed), and separately, whether
    mass_concentration_of_chlorophyll_a_in_sea_water needs a unit/scale
    conversion to become PASCAL's food1concentration - that domain
    question was never answered and still needs the model owner, fix or
    no fix. food1concentration therefore stays a constant here for now;
    re-enabling the bgc reader is a reasonable follow-up, not done as
    part of this fix.

    Previously observed here too (now explained by the same root cause):
    even a plain ConstantReader-supplied food1concentration read back as
    0.0 when the live `physical` CMEMS reader was also in the reader list
    (works correctly when ConstantReader is the *only* reader) - a second
    reader group's profile variable, same bug. Whether population still
    collapses in a live CMEMS run now that this is fixed hasn't been
    re-tested (would need re-running a real CMEMS scenario, out of scope
    for this pass) - flagged as follow-up.

    irradiance/pred1dens have no CMEMS equivalent at all. pred1dens now
    has a real (non-CMEMS) source - see pascal.scenarios::
    build_pred1dens_readers(), real visual-predator-density fields
    covering 1995-1997 (a20_test/pred_data) - already wired into
    build_a20_advection_scenario() but not into this CMEMS scenario;
    doing so is a reasonable follow-up given the two share a domain
    (a20_test/pred_data's grid covers this scenario's default
    start_location) but hasn't been done. irradiance needs a real
    derivation from a surface radiation product (e.g. ERA5 via the
    Copernicus Climate Data Store), not yet integrated. Both (plus
    food1concentration) are given as constants here as an explicit
    stand-in, using the same values already proven not to cause the
    population collapse a food1concentration=0 fallback does (see
    build_advection_scenario()).

    pred1lightdep (formerly one of these constants) has been removed:
    it never appeared in any actual PASCAL model code, only in
    required_variables/reader defaults, so it was dead weight rather
    than an open data gap.
    """
    from netrc import netrc

    from opendrift.readers.reader_constant import Reader as ConstantReader
    from opendrift.readers.reader_copernicusmarine import Reader as CMEMSReader

    np.random.seed(seed)

    if "COPERNICUSMARINE_SERVICE_USERNAME" not in os.environ:
        credentials = netrc()
        os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = credentials.hosts["copernicusmarine"][0]
        os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = credentials.hosts["copernicusmarine"][1]

    physical = CMEMSReader("cmems_mod_arc_phy_anfc_6km_detided_P1D-m")
    _alias_reader_variable(physical, "temperature", "sea_water_temperature")
    _alias_reader_variable(physical, "mld", "ocean_mixed_layer_thickness")

    constants = ConstantReader({
        "food1concentration": food1concentration_constant,
        "pred1dens": pred1dens_constant,
        "irradiance": irradiance_constant,
    })

    timestep = dt.timedelta(seconds=timestep_seconds)
    if start_date is None:
        start_date = dt.datetime(2024, 6, 1)

    lon, lat = start_location
    outputgrid = {
        "lon": [lon - 1, lon, lon + 1, lon + 2],
        "lat": [lat - 1, lat, lat + 1],
    }

    return {
        "nsupindividuals": n_super_individuals,
        "nvindividualspersupindividual": n_virtual_per_super,
        "global_settings": build_global_settings(stochastic=stochastic),
        "reader": [physical, constants],
        "timestep": timestep,
        "start_date": start_date,
        "duration": duration_years,
        "seeding_rate": seeding_rate,
        "start_locations": [[lon, lat]],
        "outputgrid": outputgrid,
        "tracker_config": {
            "general:use_auto_landmask": True,
            "vertical_mixing:diffusivitymodel": "windspeed_Sundby1983",
        },
        "headless": headless,
    }


def build_cmems_advection_scenario_from_file(
    cmems_file,
    n_super_individuals=50,
    n_virtual_per_super=10000,
    duration_years=0.05,
    timestep_seconds=21600,
    seeding_rate=10,
    stochastic=True,
    seed=0,
    start_date=None,
    start_location=(14.25, 69.8),
    food1concentration_constant=0.05,
    pred1dens_constant=0.00001,
    irradiance_constant=0.1,
    headless="bench_run",
):
    """Same scientific setup as build_cmems_advection_scenario(), but reads
    a local netCDF file (as produced by pascal_run's hpc/download_cmems_data.py)
    via reader_netCDF_CF_generic instead of streaming live data via
    reader_copernicusmarine. Intended for actual HPC runs, where compute
    nodes typically have no internet access and a multi-year run can't
    afford per-timestep calls to a rate-limited external API - see
    pascal_run's hpc/download_cmems_data.py module docstring and README.

    Unlike the live reader (reader_copernicusmarine.Reader), which doesn't
    forward a standard_name_mapping kwarg to its parent class - hence
    _alias_reader_variable()'s post-construction monkeypatch above -
    reader_netCDF_CF_generic.Reader takes standard_name_mapping directly,
    so the thetao/mlotst -> temperature/mld rename happens cleanly at
    construction time. vxo/vyo need no mapping: confirmed 2026-08-04 they
    already carry CF standard_names eastward_/northward_sea_water_velocity,
    which OpenDrift matches to x_/y_sea_water_velocity automatically.

    food1concentration/irradiance/pred1dens are constants here for the
    same reasons as build_cmems_advection_scenario() (see
    its docstring for the full detail, including the 2026-09-16 update on
    the multi-reader-group bug that used to zero food1concentration/
    pred1dens and is now fixed upstream, and pred1dens's new real - but
    not yet wired in here - data source).
    """
    from opendrift.readers.reader_constant import Reader as ConstantReader
    from opendrift.readers.reader_netCDF_CF_generic import Reader as CFReader

    np.random.seed(seed)

    physical = CFReader(
        str(cmems_file),
        standard_name_mapping={"thetao": "temperature", "mlotst": "mld"},
    )

    constants = ConstantReader({
        "food1concentration": food1concentration_constant,
        "pred1dens": pred1dens_constant,
        "irradiance": irradiance_constant,
    })

    timestep = dt.timedelta(seconds=timestep_seconds)
    if start_date is None:
        start_date = dt.datetime(2024, 6, 1)

    lon, lat = start_location
    outputgrid = {
        "lon": [lon - 1, lon, lon + 1, lon + 2],
        "lat": [lat - 1, lat, lat + 1],
    }

    return {
        "nsupindividuals": n_super_individuals,
        "nvindividualspersupindividual": n_virtual_per_super,
        "global_settings": build_global_settings(stochastic=stochastic),
        "reader": [physical, constants],
        "timestep": timestep,
        "start_date": start_date,
        "duration": duration_years,
        "seeding_rate": seeding_rate,
        "start_locations": [[lon, lat]],
        "outputgrid": outputgrid,
        "tracker_config": {
            "general:use_auto_landmask": True,
            "vertical_mixing:diffusivitymodel": "windspeed_Sundby1983",
        },
        "headless": headless,
    }


def _inject_a20_grid_placeholders(ds):
    """The A20 ROMS/ECOSMO output ships one variable per file, none of
    which carry the grid metadata (mask_rho, Vtransform, ...) that
    normally lives in a separate ROMS grid file (HPC-only, not shipped
    alongside the netCDF output - see reduced_temp.nc's grd_file global
    attribute in a real extract). Synthesize the minimum
    reader_ROMS_native.Reader needs to run at all:

    - mask_rho = all-ones (all water). Real simplification, not just a
      stand-in for a missing attribute: land cells still come back NaN in
      the actual variable data (ROMS's own _FillValue masking, decoded by
      xarray automatically), so a particle sitting on real land still gets
      missing environment data. But OpenDrift's own coastline detection
      (general:coastline_action, land_binary_mask) will never trigger,
      since it only looks at this synthetic mask.
    - Vtransform = 2, matching s_rho's declared standard_name
      "ocean_s_coordinate_g2" (CF convention: _g1 <-> Vtransform=1, _g2 <->
      Vtransform=2). Without this reader_ROMS_native defaults silently to
      Vtransform=1, which would compute the wrong physical depth for every
      sigma layer. Confirmed against a real A20 ROMS config (a20_v3y.in:
      Vtransform == 2, Vstretching == 2, THETA_S == 6.0, THETA_B == 0.1,
      TCLINE/hc == 100.0, N == 35).
    """
    import xarray as xr

    ds = ds.copy()
    ds["mask_rho"] = xr.DataArray(
        np.ones(ds["lat_rho"].shape, dtype=np.float64), dims=ds["lat_rho"].dims
    )
    ds["Vtransform"] = xr.DataArray(2)
    return ds


def _build_a20_food_reader(data_dir, food_variable):
    """food1concentration as its own reader, on the diagnostic (dia) A20
    output group's own native daily-noon axis - see
    build_a20_advection_scenario() for why this is a separate reader
    rather than being merged into the physical (his) one."""
    import xarray as xr

    from opendrift.readers.reader_ROMS_native import Reader as ROMSReader

    dia = xr.open_dataset(data_dir / f"reduced_{food_variable}.nc", decode_times=False,
                          chunks={"ocean_time": 1})
    # dia has no zeta of its own - Reader.zeta then falls back to a static
    # 2D all-zero array with no time dimension, and any non-surface (z !=
    # 0) request raises IndexError. Borrow zeta from the his group's file,
    # reindexed onto dia's own (noon) ocean_time axis. dia also lacks
    # hc/Cs_r/h entirely (only s_rho, time-invariant so copied as-is) -
    # Reader.get_variables() needs all four for any non-surface request.
    temp_ds = xr.open_dataset(data_dir / "reduced_temp.nc", decode_times=False,
                              chunks={"ocean_time": 1})
    zeta_ds = xr.open_dataset(data_dir / "reduced_zeta.nc", decode_times=False,
                              chunks={"ocean_time": 1})
    dia["zeta"] = zeta_ds["zeta"].reindex(ocean_time=dia["ocean_time"], method="nearest")
    dia["hc"] = temp_ds["hc"]
    dia["Cs_r"] = temp_ds["Cs_r"]
    dia["h"] = temp_ds["h"]
    dia = _inject_a20_grid_placeholders(dia)
    return ROMSReader(
        filename=dia, name="a20_food",
        standard_name_mapping={food_variable: "food1concentration"},
    )


def _build_a20_swrad_reader(data_dir):
    """irradiance as its own reader, on the quicksave (qck) A20 output
    group's own native hourly axis - recovers the real diurnal cycle. This
    file typically only covers part of a longer run's total duration (the
    real 25-year archive's qck files may have fuller coverage than a
    reduced test extract); discard_reader_if_not_relevant() drops this
    reader once its end_time is passed, and irradiance correctly reverts to
    the environment:fallback:irradiance constant for the rest of the run."""
    import xarray as xr

    from opendrift.readers.reader_ROMS_native import Reader as ROMSReader

    qck = xr.open_dataset(data_dir / "reduced_swrad.nc", decode_times=False,
                           chunks={"ocean_time": 1})
    zeta_ds = xr.open_dataset(data_dir / "reduced_zeta.nc", decode_times=False,
                               chunks={"ocean_time": 1})
    h_ds = xr.open_dataset(data_dir / "reduced_temp.nc", decode_times=False,
                            chunks={"ocean_time": 1})
    # swrad is genuinely 2D (surface-only, no s_rho) and carries neither
    # zeta nor h of its own - both needed by reader_ROMS_native's depth
    # machinery for any non-surface (z != 0) request, which is what PASCAL
    # always issues since irradiance is a depth profile variable.
    qck["zeta"] = zeta_ds["zeta"].reindex(ocean_time=qck["ocean_time"], method="nearest")
    qck["h"] = h_ds["h"]
    qck = _inject_a20_grid_placeholders(qck)
    qck_reader = ROMSReader(filename=qck, name="a20_swrad",
                             standard_name_mapping={"swrad": "irradiance"})
    # Monkeypatch a single sigma layer at the surface (hc=0, Cs_r=[0]) so
    # get_variables() takes its normal 3D code path instead of crashing on
    # a missing self.hc for any z != 0 request - makes irradiance depth
    # constant (ROMS quicksave output has no attenuation profile to give it
    # real depth structure, same simplification as the CMEMS scenarios'
    # irradiance constant above).
    qck_reader.hc = np.array(0.0)
    qck_reader.Cs_r = np.array([0.0])
    qck_reader.sigma = np.array([0.0])
    qck_reader.num_layers = 1
    return qck_reader


def build_pred1dens_readers(pred_data_dir, variable="vpdens"):
    """One reader per year subfolder's netCDF file under pred_data_dir
    (pred_data_dir/<year>/*.nc - the layout of a20_test/pred_data),
    aliasing `variable` (a raw netCDF variable name, not a CF
    standard_name - e.g. "vpdens", same convention as
    build_cmems_advection_scenario_from_file's thetao/mlotst aliasing)
    directly to PASCAL's pred1dens.

    Each file covers exactly one calendar year on its own "hours since
    <year>-01-01" time axis, on a plain regular lat/lon/depth grid (unlike
    the A20 ROMS output, no grid-metadata workarounds are needed here), so
    a plain reader_netCDF_CF_generic.Reader per file is enough - one
    reader per file lets OpenDrift's normal multi-reader priority
    mechanism (Environment.get_reader_groups(); a reader whose own
    start_time/end_time doesn't cover the requested timestep is skipped in
    favour of the next one in priority order) pick whichever year's file
    actually covers a given timestep, rather than needing any manual
    per-timestep file-switching logic here.

    Returned in year order (oldest first) purely for readability - reader
    priority only matters relative to whatever reader list position the
    caller inserts this at (see build_a20_readers()), not the order
    within this list, since these files' time coverages don't overlap
    each other.
    """
    from opendrift.readers.reader_netCDF_CF_generic import Reader as CFReader

    pred_data_dir = Path(pred_data_dir)
    files = sorted(pred_data_dir.glob("*/*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No netCDF files found under {pred_data_dir}/*/*.nc "
            "(expected one per year subfolder, e.g. <pred_data_dir>/1995/*.nc)"
        )
    return [
        CFReader(str(f), name=f"pred1dens_{f.parent.name}",
                 standard_name_mapping={variable: "pred1dens"})
        for f in files
    ]


def build_a20_readers(data_dir, food_variable="Chl_bc", pred_data_dir=None,
                       pred_variable="vpdens"):
    """The reader stack for a real A20 ROMS/ECOSMO run: physical variables
    (temperature/salinity/velocities/vertical diffusivity/depth/synthetic
    land mask) from the history (his) output group; food1concentration
    from its own diagnostic (dia) reader, on its own native noon-offset
    axis; irradiance from its own quicksave (qck) reader, on its own native
    hourly axis; visual predator density from pred_data_dir if given (see
    build_pred1dens_readers()); and a ConstantReader for whatever's left
    with no data source (mld always; pred1dens too when pred_data_dir is
    None - same gap the CMEMS scenarios above already have). The real
    pred1dens readers, when present, are placed *before*
    the constant in the list, so they take priority for whichever years
    they cover and the constant only kicks in outside that range.

    food1concentration/irradiance each being their own reader (rather than
    being reindexed onto his's axis and merged into one reader) only works
    correctly with the opendrift fork's fix for a multi-reader-group
    profile-variable bug (a later reader group's new profile variable used
    to be silently dropped and replaced by PASCAL's fallback constant - see
    build_cmems_advection_scenario()'s docstring for the same bug's effect
    on pred1dens there).

    data_dir must contain one ROMS output variable per file, named
    reduced_<var>.nc (temp, salt, u_eastward, v_northward, w, AKs, zeta,
    <food_variable>, swrad) - the convention used by the A20 test extract.
    """
    from opendrift.readers.reader_constant import Reader as ConstantReader
    from opendrift.readers.reader_ROMS_native import Reader as ROMSReader
    import xarray as xr

    data_dir = Path(data_dir)

    his = xr.merge(
        [xr.open_dataset(data_dir / f"reduced_{v}.nc", decode_times=False,
                          chunks={"ocean_time": 1}) for v in A20_HIS_VARS],
        compat="override", join="exact",
    )
    his = _inject_a20_grid_placeholders(his)
    physical_reader = ROMSReader(
        filename=his, name="a20_physical",
        standard_name_mapping={"temp": "temperature"},
    )

    pred_readers = (
        build_pred1dens_readers(pred_data_dir, variable=pred_variable)
        if pred_data_dir is not None else []
    )

    return ([physical_reader, _build_a20_food_reader(data_dir, food_variable),
             _build_a20_swrad_reader(data_dir)] + pred_readers +
            [ConstantReader({
                "pred1dens": 0.00001,
                "mld": 30,
            })])


def build_a20_advection_scenario(
    data_dir,
    n_super_individuals=50,
    n_virtual_per_super=10000,
    duration_years=0.05,
    timestep_seconds=21600,
    seeding_rate=10,
    stochastic=True,
    seed=0,
    start_date=None,
    start_location=(14.7, 69.53),
    food_variable="Chl_bc",
    pred_data_dir=None,
    pred_variable="vpdens",
    headless="bench_run",
):
    """Return kwargs ready to pass to coupler.PascalAdvection(**kwargs),
    backed by real A20 ROMS/ECOSMO output (data_dir - see build_a20_readers()
    for the expected file layout) instead of a synthetic reader.

    Promoted from a20_test/roms_readers.py + run_a20_test.py (a standalone
    script predating this repo split) into a real, reusable scenario
    builder alongside the CMEMS ones above - see that directory's
    README.md for the full investigation (three ROMS-reader bugs worked
    around here, plus the multi-reader-group profile bug shared with
    build_cmems_advection_scenario()) this was built from.

    start_location defaults to a point roughly mid-domain, matching the
    CMEMS scenarios' Lofoten/Vestfjorden test point - the A20 grid covers
    this area. start_date defaults to one day after the A20 test extract's
    own start time (1995-01-24), since asking for data before a reader's
    start_time raises.

    pred_data_dir, if given, points at real visual-predator-density fields
    (a20_test/pred_data - one netCDF per calendar-year subfolder; see
    build_pred1dens_readers()) instead of the ConstantReader fallback used
    for pred1dens otherwise.
    """
    np.random.seed(seed)

    data_dir = Path(data_dir)
    readers = build_a20_readers(
        data_dir, food_variable=food_variable,
        pred_data_dir=pred_data_dir, pred_variable=pred_variable,
    )

    timestep = dt.timedelta(seconds=timestep_seconds)
    if start_date is None:
        start_date = dt.datetime(1995, 1, 25)

    lon, lat = start_location
    outputgrid = {
        "lon": [lon - 1, lon, lon + 1, lon + 2],
        "lat": [lat - 1, lat, lat + 1],
    }

    return {
        "nsupindividuals": n_super_individuals,
        "nvindividualspersupindividual": n_virtual_per_super,
        "global_settings": build_global_settings(stochastic=stochastic),
        "reader": readers,
        "timestep": timestep,
        "start_date": start_date,
        "duration": duration_years,
        "seeding_rate": seeding_rate,
        "start_locations": [[lon, lat]],
        "outputgrid": outputgrid,
        "tracker_config": {
            "general:use_auto_landmask": True,
            "vertical_mixing:diffusivitymodel": "windspeed_Sundby1983",
        },
        "headless": headless,
    }
