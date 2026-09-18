"""Output aggregation and netCDF writing for PASCAL runs.

Owns two kinds of output: gridded spatial/temporal population state
(:meth:`OutputLogger.log_spatial`, written by :meth:`OutputLogger.write_spatial`)
and scalar diapause/direct-development event counters
(:meth:`OutputLogger.add_ddev`/``add_den``/``add_dex``, not currently
written to disk anywhere - see :class:`OutputLogger`'s docstring).
"""

import numpy as np
import netCDF4 as nc
import os

DEFAULT_SAVE = {'nvindividuals':{"units":"no. of individuals", "longname":"estimated stage-, time- and space-specific population size of Calanus finmarchicus"},
                'structuralmass':{"units":"gC", "longname":"estimated stage-, time- and space-specific biomass of Calanus finmarchicus"},
                'reservemass':{"units":"gC", "longname":"estimated stage-, time- and space-specific biomass of Calanus finmarchicus"},
                'feedingrate':{"units":"gC", "longname":"estimated stage-, time- and space-specific biomass of Calanus finmarchicus"},
                'egestionrate':{"units":"gC", "longname":"estimated stage-, time- and space-specific biomass of Calanus finmarchicus"},
                'metabolicrate':{"units":"gC", "longname":"estimated stage-, time- and space-specific biomass of Calanus finmarchicus"}}

class OutputLogger(object):
    """Accumulates and writes one PASCAL run's output. Constructed once per
    run by :class:`pascal.coupler.PascalSimulation` and passed down to
    every :class:`~pascal.individual.SuperIndividual` so diapause/
    direct-development events can be logged as they happen.

    Notes
    -----
    ``self.ddev``/``self.den``/``self.dex`` (direct-development and
    diapause entry/exit counters, filled by :meth:`add_ddev`/:meth:`add_den`/
    :meth:`add_dex`) and ``self.genome_log`` (filled by
    :meth:`log_evolvable`, which is itself never called anywhere in this
    codebase) are accumulated in memory but have no corresponding
    ``write_*`` method - only :meth:`write_spatial` actually writes to
    disk. ``write_evolvable`` exists as a stub (``pass``, "to be
    implemented"). Whether this is an intentional work-in-progress or an
    oversight is unclear from the code alone.
    """

    def __init__(self, outputfolder, total_timesteps, output_grid, devstages = 13, no_evolvable=8, save_spatial=DEFAULT_SAVE):
        """
        Parameters
        ----------
        outputfolder : str
            Directory to write output into; created if missing.
        total_timesteps : int
            Number of model timesteps this run will take - preallocates
            the ``ddev``/``den``/``dex``/``genome_log`` arrays' time axis.
        output_grid : dict
            Spatial/temporal output grid (``lon``/``lat``/``depth``/``time``)
            as built by ``PascalSimulation.prep_outputgrid()``.
        devstages : int, optional
            Number of developmental-stage bins in gridded output.
        no_evolvable : int, optional
            Number of evolvable genes tracked by ``genome_log``.
        save_spatial : dict, optional
            ``{variable_name: {"units", "longname"}}`` for each spatial
            variable to log/write; defaults to ``DEFAULT_SAVE``.
        """
        # These are split by developmental stages (up to 13 with 12 (13) being female (male) at stage zz)
        self.currentsubpopulation = 'all' # At the moment not logged by subpopulation
        self.total_mass = [] 
        self.spatial_atts = save_spatial
        self.spatial_var_list = list(save_spatial.keys())
        self.spatial_output = {}
        for save_var in save_spatial.keys():
            self.spatial_output[save_var] = []
        self.devstages = devstages
        self.total_timesteps = total_timesteps
        self.no_evolvable = no_evolvable
        self.output_grid = output_grid
        self.prep_grid()

        #this logs the numbers, structural masses and reserve masses of the direct-developing super-individuals without diapause entry (decided by gene value or shallow depth)
        #dimensions: <time> <genetic, environmental>
        self.ddev = {'individuals':np.zeros([self.total_timesteps, 2]), 'structuralmass':np.zeros([self.total_timesteps, 2]), 'reservemass':np.zeros([self.total_timesteps, 2])}
        #this logs the numbers, structural masses and reserve masses of diapause entries (civ, cv stages including true and active diapausing individuals)
        #dimensions:<time> <civ, cv> <true, active>
        self.den = {'individuals':np.zeros([self.total_timesteps, 2]), 'structuralmass':np.zeros([self.total_timesteps, 2]), 'reservemass':np.zeros([self.total_timesteps, 2])}
        #this logs the numbers, structural masses and reserve masses of diapause entries (civ, cv stages including true and active diapausing individuals)
        #dimensions:<time> <civ, cv> <true, active>
        self.dex = {'individuals':np.zeros([self.total_timesteps, 2]), 'structuralmass':np.zeros([self.total_timesteps, 2]), 'reservemass':np.zeros([self.total_timesteps, 2])}

        # Log the changing genomes
        self.genome_log = np.zeros([self.total_timesteps, self.no_evolvable, 2])

        # Setup the output
        self.outputfolder = outputfolder
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)

        self.outputfile = f'./{outputfolder}/output_ps.nc'

    def prep_grid(self):
        """Derive ``min/max_lon``/``min/max_lat`` and grid cell resolution
        from ``self.output_grid``, for use by :meth:`resolve_spatial`.
        A single-point grid (min == max) gets a resolution of 1 (unused,
        since :meth:`resolve_spatial` clamps single-point grids to that
        one cell regardless)."""
        self.min_lon = np.min(self.output_grid['lon'])
        self.max_lon = np.max(self.output_grid['lon'])
        self.min_lat = np.min(self.output_grid['lat'])
        self.max_lat = np.max(self.output_grid['lat'])
        if self.min_lon == self.max_lon:
            self.lon_res = 1
        else:
            self.lon_res = self.output_grid['lon'][1] - self.output_grid['lon'][0]

        if self.min_lat == self.max_lat:
            self.lat_res = 1
        else:
            self.lat_res = self.output_grid['lat'][1] - self.output_grid['lat'][0]

    def add_ddev(self, data, col):
        """Accumulate one direct-development event (a non-diapausing
        individual reaching adulthood) into ``self.ddev``.

        Parameters
        ----------
        data : dict
            From :meth:`pascal.individual.SuperIndividual.get_log_data`.
        col : int
            0 for genetically-determined, 1 for environmentally-determined
            (bottom-depth-constrained) - see
            :meth:`~pascal.individual.SuperIndividual.diapause0`.

        Notes
        -----
        **Verified bug**: references ``self.current_timestep``, which is
        never set anywhere on this class - calling this raises
        ``AttributeError``. Not fixed here (see
        :meth:`~pascal.individual.SuperIndividual.get_log_data`'s
        docstring for the related bug in its caller).
        """
        for var, add_data in data.items():
            self.ddev[var][self.current_timestep, col] += add_data

    def add_den(self, data, col1, col2):
        """Accumulate one diapause-entry event into ``self.den``.

        Parameters
        ----------
        data : dict
            From :meth:`pascal.individual.SuperIndividual.get_log_data`.
        col1 : int
            0 for CIV, 1 for CV.
        col2 : int
            Diapause mode (0: "true" diapause, 1: "active" diapause - see
            :meth:`~pascal.individual.SuperIndividual.diapause0`).

        Notes
        -----
        Same ``self.current_timestep`` bug as :meth:`add_ddev`.
        """
        for var, add_data in data.items():
            self.den[var][self.current_timestep, col1, col2] += add_data

    def add_dex(self, data, col1, col2):
        """Accumulate one diapause-exit event into ``self.dex``.

        Parameters
        ----------
        data : dict
            From :meth:`pascal.individual.SuperIndividual.get_log_data`.
        col1 : int
            0 for CIV, 1 for CV.
        col2 : int
            Always 0 in current call sites (see
            :meth:`~pascal.individual.SuperIndividual.diapause1`) - a
            second mode value is never passed, unlike :meth:`add_den`.

        Notes
        -----
        Same ``self.current_timestep`` bug as :meth:`add_ddev`.
        """
        for var, add_data in data.items():
            self.dex[var][self.current_timestep, col1, col2] += add_data

    def log_spatial(self, cxyz, data_dict):
        """Grid one timestep's per-individual spatial contributions (via
        :meth:`resolve_spatial`) and append the result to
        ``self.spatial_output``, for later writing by
        :meth:`write_spatial`.

        Parameters
        ----------
        cxyz : numpy.ndarray
            Shape ``(n_individuals, 4)``: stage-column, lon, lat, depth-index
            per active individual (built by
            ``pascal.coupler.PascalSimulation.log_spatial``).
        data_dict : dict
            ``{variable: array of per-individual values}``, one entry per
            ``self.spatial_var_list`` variable.
        """
        # This is done at the coupler level as it varies based on the forcing (1-D or spatially resolved)
        # TODO - write at each timestep rather than dumping to a dict
        resolved_dict = self.resolve_spatial(cxyz, data_dict)
        for varname, data in self.spatial_output.items():
           data.append(resolved_dict[varname])

    def log_evolvable(self, genomes, timestep):
        """Record the population's per-gene mean/std into
        ``self.genome_log`` at ``timestep``.

        Notes
        -----
        Never called anywhere in this codebase, and ``genome_log`` has no
        corresponding write method either (see this class's docstring) -
        appears to be unused/incomplete rather than exercised.
        """
        self.genome_log[timestep, :, 0] = np.mean(genomes)
        self.genome_log[timestep, :, 1] = np.std(genomes)

    def resolve_spatial(self, cxyz, data_dict):
        """Bin one timestep's per-individual contributions onto the
        ``(devstage, lon, lat, depth)`` output grid via ``np.add.at``
        (correctly accumulating when multiple individuals fall in the same
        cell). Out-of-grid lon/lat are clamped to the grid's min/max rather
        than dropped.

        Parameters
        ----------
        cxyz : numpy.ndarray
            Shape ``(n, 4)``: stage-column, lon, lat, depth-index.
        data_dict : dict
            ``{variable: array of length n}``.

        Returns
        -------
        dict
            ``{variable: ndarray of shape (devstages, n_lon, n_lat, n_depth)}``.
        """
        # nb: vectorized replacement for what used to be a nested Python loop
        # over every individual, wrapped in a "for d in devstages: if any
        # individual is in stage d: <loop over ALL individuals again>" outer
        # loop. That outer loop didn't restrict the inner loop to stage-d
        # individuals, so it re-added every individual's contribution once
        # per *distinct* developmental stage present that timestep - with a
        # population spanning most of the 13 stages (the common case), this
        # inflated spatial output by up to ~13x. Fixed here as a byproduct
        # of vectorizing: each individual's contribution is now added
        # exactly once via np.add.at (which, unlike `arr[idx] += vals`,
        # correctly accumulates when multiple individuals map to the same
        # grid cell instead of silently dropping duplicates).
        shape = [
            self.devstages,
            len(self.output_grid['lon']),
            len(self.output_grid['lat']),
            len(self.output_grid['depth']),
        ]
        gridded_data = {var: np.zeros(shape) for var in data_dict.keys()}

        if len(cxyz) == 0:
            return gridded_data

        cxyz = cxyz.copy()
        cxyz[cxyz[:, 1] > self.max_lon, 1] = self.max_lon
        cxyz[cxyz[:, 1] < self.min_lon, 1] = self.min_lon
        cxyz[cxyz[:, 2] > self.max_lat, 2] = self.max_lat
        cxyz[cxyz[:, 2] < self.min_lat, 2] = self.min_lat

        # nb: stage index is (col - 1); col==0 (egg stage) therefore wraps
        # to the last stage slot via numpy's negative-index handling, same
        # as the scalar loop this replaces - preserved as-is since changing
        # it would be a modelling decision, not a performance one.
        stage_ind = cxyz[:, 0].astype(np.int64) - 1
        lon_ind = np.floor((cxyz[:, 1] - self.min_lon) / self.lon_res).astype(np.int64)
        lat_ind = np.floor((cxyz[:, 2] - self.min_lat) / self.lat_res).astype(np.int64)
        depth_ind = cxyz[:, 3].astype(np.int64)
        index = (stage_ind, lon_ind, lat_ind, depth_ind)

        for var, data in data_dict.items():
            np.add.at(gridded_data[var], index, np.asarray(data))

        return gridded_data

    def write_spatial(self):
        """Write accumulated gridded spatial output (``self.spatial_output``,
        filled over the run by :meth:`log_spatial`) to
        ``<outputfolder>/output_ps.nc`` (NETCDF4_CLASSIC), one variable per
        ``self.spatial_atts`` entry, dimensioned
        ``(time, devstage, lon, lat, depth)``."""
        #file1: space-, time-, and tage-specific population size (datatype = np.int32)
        #-----------------------------------------------------------------------------
        #nb: dimensions: <stage> <longitude> <latitude> <depth> <time>
        #datafile creation
        outputfile = self.outputfile
        populationsize_ds = nc.Dataset(outputfile, "w", format = "NETCDF4_CLASSIC")

        #writing datafile attributes (add as needed)
        populationsize_ds.title = "PASCALv4 output datafile: population size"
        populationsize_ds.subtitle = f"subpopulation ID: {self.currentsubpopulation}" 
        populationsize_ds.project = "NFR Migratory Crossroads"
        populationsize_ds.author = "Kanchana Bandra"
        populationsize_ds.warning = "evaluation output - do not use for analyses"

        #creating dataset dimensions
        stagedim = populationsize_ds.createDimension("devstage", self.devstages)
        londim = populationsize_ds.createDimension("lon", len(self.output_grid['lon']))
        latdim = populationsize_ds.createDimension("lat", len(self.output_grid['lat']))
        depthdim = populationsize_ds.createDimension("depth", len(self.output_grid['depth']))
        timedim = populationsize_ds.createDimension("time", self.total_timesteps)
        
        #creating dimensionality variabels & data variables
        stagevar = populationsize_ds.createVariable("devstage", np.int32, ("devstage", ))
        stagevar.units = "dim.less"
        stagevar.longname = "developmental stage"

        lonvar = populationsize_ds.createVariable("lon", np.float32, ("lon", ))
        lonvar.units = "degrees east"
        lonvar.longname = "longitude"
        lonvar[:] = self.output_grid['lon']

        latvar = populationsize_ds.createVariable("lat", np.float32, ("lat", ))
        latvar.units = "degrees north"
        latvar.longname = "latitude"
        latvar[:] = self.output_grid['lat']

        depthvar = populationsize_ds.createVariable("depth", np.int32, ("depth", ))
        depthvar.units = "m"
        depthvar.longname = "depth levels"
        depthvar[:] = self.output_grid['depth']

        timevar = populationsize_ds.createVariable("time", np.int32, ("time", ))
        timevar.units = "6 h"
        timevar.longname = "time of year in 6h intervals"
        timevar[:] = self.output_grid['time']

        for var, atts in self.spatial_atts.items():
            populationsize_ds = self._write_4d_var(populationsize_ds, var, self.spatial_output[var], attributes = atts)
        
        populationsize_ds.close()


    def _write_4d_var(self, ds, varname, data, attributes={}, dtype=np.int32):
        dv1 = ds.createVariable(varname, dtype, ("time", "devstage", "lon", "lat", "depth",))
        for att_name, att_val in attributes.items():
            setattr(dv1,att_name,att_val)
        dv1[:] = self.pad_data(np.asarray(data))
        return ds

    def write_evolvable(self):
        # to be implemented
        pass

    def pad_data(self, data):
        target_shape =  (self.total_timesteps, self.devstages, len(self.output_grid['lon']), len(self.output_grid['lat']),
                            len(self.output_grid['depth']))
        if data.shape != target_shape:
            padded = np.full(target_shape, np.nan, dtype=data.dtype)
            padded[:data.shape[0], ...] = data
            return padded
        else:
            return data
