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
    def __init__(self, outputfolder, total_timesteps, output_grid, devstages = 13, no_evolvable=8, save_spatial=DEFAULT_SAVE):
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
        for var, add_data in data.items():
            self.ddev[var][self.current_timestep, col] += add_data

    def add_den(self, data, col1, col2):
        for var, add_data in data.items():
            self.den[var][self.current_timestep, col1, col2] += add_data

    def add_dex(self, data, col1, col2):
        for var, add_data in data.items():
            self.dex[var][self.current_timestep, col1, col2] += add_data

    def log_spatial(self, cxyz, data_dict):
        # This is done at the coupler level as it varies based on the forcing (1-D or spatially resolved)
        # TODO - write at each timestep rather than dumping to a dict
        resolved_dict = self.resolve_spatial(cxyz, data_dict)
        for varname, data in self.spatial_output.items():
           data.append(resolved_dict[varname])

    def log_evolvable(self, genomes, timestep):
        self.genome_log[timestep, :, 0] = np.mean(genomes)
        self.genome_log[timestep, :, 1] = np.std(genomes)

    def resolve_spatial(self, cxyz, data_dict):
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
