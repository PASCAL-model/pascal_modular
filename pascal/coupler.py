from math import e
from .individual import SuperIndividual, DEFAULT_DEBUG_VARIABLES
from .data_logger import OutputLogger
from .pascal_drift import PascalDrift
from .utils import dotdict, flatten_list, flatten_dict, points_within_distance
from .biology import survival as sv

import datetime as dt
import logging
import numpy as np
import pandas as pd
import termcolor
from time import gmtime, strftime

DEFAULT_DEPTHRANGE = np.array([1, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 19, 22, 26, 30,
                               35, 41, 48, 56, 66, 78, 93, 110, 131, 156, 187, 223,
                               267, 319, 381, 454, 542, 644, 764, 903, 1063, 1246])

class PascalSimulation(object):
    def __init__(
        self, nsupindividuals, nvindividualspersupindividual,
        global_settings, reader, timestep, start_date, duration,
        seeding_rate, timestep_isplit=1, diapause_depth=500, outputgrid=None,
        debug=None, opendriftoutfile=None, verbose=False, start_locations=None,
        tracker_config=None, headless=None
    ):
        print("")
        termcolor.cprint(
            text="Pan-Arctic Behavioural and Life-history Simulator for Calanus, "
                 "PASCAL version 4.20",
            color="cyan"
        )
        termcolor.cprint(
            text="Kanchana Bandara et al. | NFR Migratory Crossroads 2024-2027",
            color="cyan"
        )
        termcolor.cprint(
            text="Evaluation execution for functionality testing and debugging",
            color="cyan"
        )
        termcolor.cprint(
            text="_" * 84,
            color="light_blue"
        )
        if headless is None:
            print("")
            termcolor.cprint(
                text="enter a unique identifier for the execution (e.g., pascalv42_r001):",
                color="light_red"
            )
            self.outputfolder = input("TYPE ID HERE AND PRESS ENTER: ")
            termcolor.cprint(text="_" * 84, color="light_blue")
            print("")
        else:
            print("")
            self.outputfolder = headless
            termcolor.cprint(
                text=f"Run output folder {headless}",
                color="light_red"
            )
            termcolor.cprint(text="_" * 84, color="light_blue")
            print("")


        execstarttime_prt = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        termcolor.cprint(
            text=f"\nexecution started at: {execstarttime_prt} GMT",
            color="light_blue"
        )
        termcolor.cprint(text="_" * 84, color="light_blue")

        self.nsup = nsupindividuals
        self.supindividuals = [None for j in np.arange(0, nsupindividuals)]
        self.ni_per_sup = nvindividualspersupindividual
        self.global_settings = global_settings
        if 'depthrange' not in global_settings.keys():
            self.global_settings['depthrange'] = DEFAULT_DEPTHRANGE
        self.diapause_depth = diapause_depth

        self.start_time = start_date
        # Guess this doesn't really deal with leap years...
        self.end_time = start_date + dt.timedelta(days=365 * duration)
        self.current_time = start_date
        self.timestep = timestep
        self.isplit = timestep_isplit

        total_tsteps = int(np.ceil((self.end_time - self.start_time) / self.timestep))
        # Ensure total_tsteps is a multiple of self.isplit; if not, reduce it
        rem = total_tsteps % self.isplit
        if rem != 0:
            total_tsteps -= (self.isplit - rem)

        self.all_steps = np.arange(0,total_tsteps,self.isplit)

        self.seeding_rate = seeding_rate

        # Setup the environment, logger, and the super individuals
        if start_locations is None:
            start_locations = [[0,0]]
        elif isinstance(start_locations, np.ndarray):
            start_locations = [tuple(loc) for loc in start_locations]

        self.start_locations = start_locations
        # Cursor for select_start_locations()'s deterministic cycling mode
        # (global_settings["stochastic"] is False) - persists across calls
        # (initial seeding in run(), then every respawn()) so locations are
        # cycled through in order across the whole run rather than reset.
        self._start_location_cursor = 0
        self.opendriftout = opendriftoutfile
        self.tracker_config = tracker_config
        self.prep_environment(reader)

        self.prep_outputgrid(outputgrid)
        self.datalogger = OutputLogger(
            self.outputfolder, len(self.all_steps), self.outputgrid
        )

        # debug=True means "log the default variable set" rather than an
        # explicit list - see individual.py::DEFAULT_DEBUG_VARIABLES.
        self.debug = list(DEFAULT_DEBUG_VARIABLES) if debug is True else debug
        if self.debug is not None:
            self.debug_output = {}
            for this_var in self.debug:
                self.debug_output[this_var] = []

        self.individual_stats = {}
        self.next_unique_id = 0

        self.verbose = verbose

    def select_start_locations(self, n):
        """Pick n locations from self.start_locations for a fresh seed/
        reseed batch.

        Random with replacement by default (matches the original,
        always-random behavior). If global_settings["stochastic"] is
        False, cycles through self.start_locations in order instead - a
        persistent cursor (self._start_location_cursor) carries the
        position across calls (initial seeding, then every respawn()) so
        the whole run cycles through the list once per full pass, rather
        than restarting from the first location every call.
        """
        n_locs = len(self.start_locations)
        if not self.global_settings.get("stochastic", True):
            idx = [(self._start_location_cursor + i) % n_locs for i in range(n)]
            self._start_location_cursor = (self._start_location_cursor + n) % n_locs
        else:
            idx = np.random.choice(n_locs, size=n, replace=True)
        return [self.start_locations[i] for i in idx]

    def run(self):
        termcolor.cprint(text = "[SIMULATION IN PROGRESS]", color = "light_red")

        # Setup initial individuals
        seed_locations = self.select_start_locations(self.seeding_rate)
        self.seed(self.seeding_rate, seed_locations, genome=None)

        for this_step in self.all_steps:
            self.update_environment()
            self.sync_environment_references()
            for isplit in np.arange(0, self.isplit):
                self.update_lifestage()

            self.log_spatial()
            self.gene_hunt()
            self.clean_dead()
            self.respawn()
            if self.current_time.day == 1 and self.current_time.hour == 0:
                self.report()

            if self.debug is not None:
                self.debug_out()

            # Increment datetime
            self.current_time += self.timestep

        # Tidy up
        self.finish_run()

    def sync_environment_references(self):
        """Re-point every active individual at the current environment/
        environment_profiles objects, and refresh their maxdepth/mindepth.

        update_environment() (Pascal1D/PascalAdvection) creates brand new
        environment/environment_profiles objects every timestep, but
        SuperIndividual only captures those references once, in seed()
        at construction time. Without this, an individual would keep
        reading whatever environment existed at the timestep it was
        seeded - frozen for its entire lifespan - rather than the
        current one, since nothing else re-syncs an already-existing
        individual's reference.

        maxdepth/mindepth are computed once here rather than once per
        individual in SuperIndividual.update_vert(): depth ('z') has no
        per-individual axis - every individual sees the same water column
        grid - so recomputing max/min of it per individual per timestep
        was pure redundant work (measured at ~17% of total runtime; see
        BENCHMARKING.md).

        Also clears each individual's get_profile() cache: it must not
        survive into a new timestep's (possibly different) environment
        data. This is the only place that cache is cleared, so it's safe
        for get_profile() to cache unconditionally - see BENCHMARKING.md
        for why that mattered (in advection mode, get_profile()'s
        np.interp() path was being redundantly re-run 2-3x per individual
        per timestep for the same variable).
        """
        z = self.tracker.environment_profiles["z"]
        maxdepth = np.max(-z)
        mindepth = np.min(-z)
        for this_individual in self.supindividuals:
            if this_individual is not None:
                this_individual.environment = self.tracker.environment
                this_individual.environment_profiles = self.tracker.environment_profiles
                this_individual.maxdepth = maxdepth
                this_individual.mindepth = mindepth
                this_individual._profile_cache = {}

    def update_lifestage(self):
        for this_individual in self.supindividuals:
            if this_individual is not None:
                this_individual.run_stage_transition()
        self.apply_mortality_and_deathcheck_batch()

    def apply_mortality_and_deathcheck_batch(self):
        """Vectorized replacement for calling each individual's own
        apply_mortality_and_deathcheck() one at a time.

        Safe to run as a separate pass *after* every individual's
        run_stage_transition() above (rather than interleaved,
        per-individual, as SuperIndividual.update_lifestage() does it):
        neither the dsc2 mortality calculation nor the death check reads
        any other individual's state, and neither draws random numbers -
        all the stochastic behavior (sex determination, diapause strategy,
        gene crossover/mutation) lives in run_stage_transition(), whose
        iteration order over self.supindividuals is unchanged above. So
        this batching cannot change the sequence of random draws, and
        produces bit-identical results to the pure per-individual path
        (checked in tests/test_coupler_batched_mortality.py).
        """
        active = self.active_supindividuals()
        if len(active) == 0:
            return

        stage = np.array([si.developmentalstage for si in active])
        mortality_mask = stage > 2
        if np.any(mortality_mask):
            subset = active[mortality_mask]
            strmass = np.array([si.structuralmass for si in subset])
            maxstrmass = np.array([si.maxstructuralmass for si in subset])
            resmass = np.array([si.reservemass for si in subset])
            vpreldensity = np.array([si.get_zi("pred1dens") for si in subset])
            irradiance = np.array([si.get_zi("irradiance") for si in subset])
            nvindividuals = np.array([si.nvindividuals for si in subset])

            risk = sv.mortalityrisk_dsc2_vectorized(
                strmass=strmass,
                maxstrmass=maxstrmass,
                resmass=resmass,
                vpreldensity=vpreldensity,
                irradiance=irradiance,
                maxirradiance=self.global_settings["maxirradiance"],
                minirradiance=self.global_settings["minirradiance"],
                nvpreldensity=self.global_settings["nonvisualpredatorreldensity"],
                bgmrisk=self.global_settings["backgroundmortalityrisk"],
            )
            # nb: apply_dsc2_mortality() does NOT truncate to int (unlike
            # apply_dsc0_mortality/apply_dsc1_mortality, which do) - kept
            # as float here to match exactly.
            new_nvindividuals = nvindividuals * (1.00 - risk)
            for si, nv in zip(subset, new_nvindividuals):
                si.nvindividuals = nv

        # Universal death check (applies regardless of stage, matching
        # SuperIndividual.apply_mortality_and_deathcheck()).
        nvindividuals_all = np.array([si.nvindividuals for si in active])
        age_all = np.array([si.age for si in active])
        totalfecundity_all = np.array([si.totalfecundity for si in active])
        dead_mask = (
            (nvindividuals_all <= self.global_settings["virtualindividualthrehold"])
            | (age_all >= self.global_settings["ageceiling"])
            | (totalfecundity_all >= self.global_settings["fecundityceiling"])
        )
        for si, is_dead in zip(active, dead_mask):
            if is_dead:
                si.lifestatus = 0

    def log_spatial(self):
        varlist = self.datalogger.spatial_var_list
        data_dict = {}
        for this_var in varlist:
            data_dict[this_var] = []

        c = []
        z = []
        env_indices = []

        # Computed once and reused below (rather than calling
        # active_supindividuals()/environment_indices() again) - safe
        # because nothing mutates self.supindividuals within this method.
        for si in self.active_supindividuals():
            c_add, z_add, d_add = si.get_spatial_log_data(
                self.datalogger.spatial_var_list
            )
            c.append(c_add)
            z.append(z_add)
            env_indices.append(si.environment_index)
            for k,v in data_dict.items():
                v.append(d_add[k])

        for k,v in data_dict.items():
            data_dict[k] = np.asarray(v)

        cxyz = np.stack([
            np.asarray(c),
            self.tracker.elements.lon[env_indices],
            self.tracker.elements.lat[env_indices],
            np.asarray(z)
        ]).T

        self.datalogger.log_spatial(cxyz, data_dict)

    def respawn(self):
        # Computed once and reused throughout (including in the final reset
        # loop below) instead of calling active_supindividuals() repeatedly -
        # safe because self.seed() only ever fills previously-None slots, so
        # this snapshot stays valid; newly-seeded individuals already start
        # with potentialfecundity=0, so they don't need to be in the final
        # reset loop either.
        active = self.active_supindividuals()

        nspaces = np.sum(np.asarray(self.supindividuals) == None)
        if nspaces > 0:  # Skip if there ain't no space
            if self.current_time.year == self.start_time.year:
                nseeds = self.seeding_rate
                seed_locations = self.select_start_locations(nseeds)
            else:
                nseeds = 0
            # The blendrn/threshold process is individual based
            inherited_genome = flatten_list([
                [si.get_child_genome() for k in np.arange(0, si.potentialfecundity)]
                for si in active
            ])
            # nb: NOT flatten_list()'d, unlike inherited_genome above - the
            # comprehension below already combines the si/k loops into one
            # flat list of [lon, lat] pairs (one per offspring), so
            # flatten_list() (previously applied here) was flattening it a
            # second time, into a flat list of bare scalars - locations[i]
            # in seed()/set_tracker() would then get a lone float instead
            # of a [lon, lat] pair the first time any mating-driven spawn
            # actually happened.
            inherited_locations = [
                [self.tracker.elements.lon[si.environment_index],
                 self.tracker.elements.lat[si.environment_index]]
                for si in active
                for k in np.arange(0, si.potentialfecundity)
            ]
            # Mating-spawned origin: inherited from a parent (get_child_origin()),
            # not the mother's current position - see individual.py.
            inherited_origins = flatten_list([
                [si.get_child_origin() for k in np.arange(0, si.potentialfecundity)]
                for si in active
            ])

            nspawns = len(inherited_genome)

            # This writes out the logic from the decision tree, could probably
            # be simplified but might reduce readibility
            if nseeds > 0 and nspawns > 0:
                if nspawns + nseeds <= nspaces:
                    self.seed(nseeds, seed_locations, genome=None)
                    self.seed(nspawns, inherited_locations, genome=inherited_genome, origins=inherited_origins)
                    if self.verbose:
                        print(f'__respawn__ Seeding {nseeds} and spawning {nspawns}')
                else:
                    if nspaces > nspawns:
                        self.seed(nspawns, inherited_locations, genome=inherited_genome, origins=inherited_origins)
                        if self.verbose:
                            print(f'__respawn__ Spawning {nspawns}')
                    else:
                        adjusted_genome, adjusted_locations, adjusted_origins = self.fecundity_proportional_selection(
                            nspawns, active
                        )
                        self.seed(nspaces, adjusted_locations, genome=adjusted_genome, origins=adjusted_origins)
                        if self.verbose:
                            print(
                                f'__respawn__ Spawning {len(adjusted_genome)} '
                                'through fecundity proportional selection'
                            )

            elif nspawns > 0 and nseeds == 0:
                if nspaces > nspawns:
                    self.seed(nspawns, inherited_locations, genome=inherited_genome, origins=inherited_origins)
                else:
                    adjusted_genome, adjusted_locations, adjusted_origins = self.fecundity_proportional_selection(
                        nspawns, active
                    )
                    if self.verbose:
                        print(
                            f'__respawn__ Spawning {len(adjusted_genome)} '
                            'through fecundity proportional selection'
                        )
                    self.seed(nspaces, adjusted_locations, genome=adjusted_genome, origins=adjusted_origins)

            elif nseeds > 0 and nspawns == 0:
                if nspaces > nseeds:
                    # Not sure why we don't just seed all available spaces?
                    self.seed(nseeds, seed_locations, genome=None)
                    if self.verbose:
                        print(f'__respawn__ Seeding {nseeds}')

        # Reset potential fecundity in individuals
        for si in active:
            si.potentialfecundity = 0

    def seed(self, nseeds, locations, genome=None, origins=None):
        # Empty spaces are always shuffled to the end of the array
        # so just start from the first None
        firstNone = np.min(np.where(np.isin(self.supindividuals, None)))

        if genome is None:
            genome = [None for i in np.arange(0,nseeds)]

        # origins default to the spawn locations themselves - correct for
        # fresh seeds (the individual's origin is wherever it started).
        # Mating-spawned individuals pass their own (parent-inherited)
        # origins explicitly - see respawn()/get_child_origin().
        if origins is None:
            origins = locations

        for i in np.arange(0, nseeds):
            environment_index = self.free_env_indices.pop(0)
            self.set_tracker(environment_index, locations[i])
            self.set_tracker_origin(environment_index, origins[i])

            # Should diapause depth be random?
            self.supindividuals[i + firstNone] = SuperIndividual(
                self.global_settings,
                self.diapause_depth,
                self.tracker.environment,
                self.tracker.environment_profiles,
                environment_index,
                nindividuals=self.ni_per_sup,
                genes=genome[i],
                unique_id=self.next_unique_id,
                origin=origins[i]
            )
            self.individual_stats[self.next_unique_id] = {
                'start_step': self.current_time
            }
            self.next_unique_id += 1

    def fecundity_proportional_selection(self, nspaces, active=None):
        if active is None:
            active = self.active_supindividuals()

        potentialfecundity = [
            si.potentialfecundity for si in active
        ]
        nspawns = np.sum(potentialfecundity)
        realizedfecundity = np.round(
            potentialfecundity / nspawns * nspaces, decimals=0
        ).astype(np.int32)
        diff = nspaces - np.sum(realizedfecundity)

        indices = np.argsort(potentialfecundity)
        if diff > 0:
            for i in range(diff):
                realizedfecundity[indices[-(i + 1)]] += 1
        elif diff < 0:
            for i in range(abs(diff)):
                # Remove fecundity from the bottom up
                realizedfecundity[indices[-(i + 1)]] -= 1

        adjusted_genome = flatten_list([
            [si.get_child_genome() for k in np.arange(0, rf)]
            for si, rf in zip(active, realizedfecundity)
        ])

        # nb: NOT flatten_list()'d - see respawn()::inherited_locations for
        # why (this comprehension is already a flat list of [lon, lat]
        # pairs; flatten_list()'ing it again turned it into a flat list of
        # bare scalars).
        adjusted_locations = [
            [self.tracker.elements.lon[si.environment_index],
             self.tracker.elements.lat[si.environment_index]]
            for si, rf in zip(active, realizedfecundity)
            for k in np.arange(0, rf)
        ]

        adjusted_origins = flatten_list([
            [si.get_child_origin() for k in np.arange(0, rf)]
            for si, rf in zip(active, realizedfecundity)
        ])

        return adjusted_genome, adjusted_locations, adjusted_origins

    def clean_dead(self):
        # We can use active individuals because Nones should always be
        # at the end of the array
        # Computed once and reused below instead of calling
        # active_supindividuals() again per removed individual per loop.
        active = self.active_supindividuals()
        remove = [
            j for j, si in enumerate(active)
            if si.lifestatus == 0
        ]
        for i in remove:
            self.record_lifestats(active[i])

        # Keep track of free slots in the particle tracker
        for i in remove:
            self.free_env_indices.append(active[i].environment_index)

        if len(remove) > 0:
            [self.supindividuals.pop(i - j) for j, i in enumerate(remove)]
            self.supindividuals = flatten_list([
                self.supindividuals, [None for i in remove]
            ])
            if self.verbose:
                print(
                    f'__clean_dead__ removed {len(remove)} - {remove} si, '
                    f'len array {len(self.supindividuals)}'
                )

    def finish_run(self):
        if self.debug is not None:
            np.save(f'{self.outputfolder}/debug_output.npy', self.debug_output)
        self.write_lifestats()
        self.datalogger.write_spatial()

    def report(self):
        progress = f'{self.progress():.0f}'
        month = self.current_time.strftime('%b')[0].capitalize()
        year = self.current_time.year
        pop_size = self.population_size()
        n_si = len(self.active_supindividuals())
        termcolor.cprint(
            text=f"[PROG:{progress:>8}%] [MO: {month}] [YR: {year}] "
                 f"[ESTIMATED POPULATION SIZE: {pop_size}] No si = {n_si}"
        )
        if self.current_time.month == 12 and self.current_time.hour == 0:
            print("")

    def record_lifestats(self, individ):
        self.individual_stats[individ.unique_id].update({
            'end_age': individ.age,
            'end_individuals': individ.nvindividuals,
            'sex': individ.sex,
            'end_step': self.current_time,
            'total_fecundity': individ.totalfecundity,
            'end_stage': individ.developmentalstage,
            'genes': individ.genome,
            'end_structmass': individ.structuralmass,
            'end_cmm': individ.get_currentcmm(),
            'end_diapause_state': individ.diapausestate,
            'end_diapause_strategy': individ.diapausestrategy
        })

    def write_lifestats(self):
        df = pd.DataFrame(flatten_dict(self.individual_stats))
        # Death-related columns only exist once at least one super-individual
        # has died (record_lifestats() is what adds them) - a short run can
        # legitimately end with everyone still alive.
        if 'total_fecundity' in df.columns:
            death_cause = np.zeros(len(df))
            fecundity_mask = (
                df['total_fecundity'] >= self.global_settings['fecundityceiling']
            )
            death_cause[fecundity_mask] = 3
            age_mask = df['end_age'] >= self.global_settings['ageceiling']
            death_cause[age_mask] = 2
            threshold_mask = (
                df['end_individuals'] <=
                self.global_settings['virtualindividualthrehold']
            )
            death_cause[threshold_mask] = 1
            df['death_cause'] = death_cause
        df.to_csv(f'{self.outputfolder}/lifestats.csv')

    def progress(self):
        return (self.time_ind/len(self.all_steps))*100

    def population_size(self):
        return np.sum([si.nvindividuals for si in self.active_supindividuals()])

    def active_supindividuals(self):
        mask = ~np.isin(self.supindividuals, None)
        return np.asarray(self.supindividuals)[mask]

    def environment_indices(self):
        return [si.environment_index for si in self.active_supindividuals()]

    def debug_out(self):
        if self.supindividuals[0] is not None:
            for this_var in self.debug:
                if this_var.split('_')[0] == 'env':
                    # maxsplit=1: env variable names can contain
                    # underscores themselves (e.g. "env_ocean_vertical_diffusivity"
                    # -> "ocean_vertical_diffusivity"), so only the "env"
                    # prefix itself should be split off.
                    var_name = this_var.split('_', 1)[1]
                    self.debug_output[this_var].append(
                        self.supindividuals[0].get_zi(var_name)
                    )
                else:
                    self.debug_output[this_var].append(
                        getattr(self.supindividuals[0], this_var)
                    )

class Pascal1D(PascalSimulation):

    def prep_environment(self, reader):
        self.free_env_indices = list(np.zeros(self.nsup, dtype=int))
        self.all_data = reader
        self.time_ind = -1
        self.update_environment()

    def prep_outputgrid(self, outputgrid):
        self.outputgrid = {
            'lon': [0],
            'lat': [0],
            'depth': self.global_settings['depthrange'],
            'time': self.all_steps
        }

    def update_environment(self):
        self.time_ind+= 1
        init_dict = {}
        for k,v in self.all_data.items():
            if k == 'z':
                init_dict['z'] = v
            else:
                init_dict[k] = v[self.time_ind,...]
        init_dict['lon'] = np.asarray([0])
        init_dict['lat'] = np.asarray([0])
        self.tracker = dotdict({
            'environment': dotdict(init_dict),
            'environment_profiles': dotdict(init_dict),
            'elements': dotdict(init_dict)
        })
    
    def set_tracker(self, environment_index, location):
        pass

    def set_tracker_origin(self, environment_index, origin):
        pass

    def gene_hunt(self):
        # In 1-D all the animals are near to each other so all females are
        # considered near to all males, therefore just pick a random male
        active = self.active_supindividuals()
        noninseminated_females = [
            i for i in active
            if i.sex == 'F' and i.inseminationstate == 0
        ]
        males = [i for i in active if i.sex == 'M']

        if len(males) > 0:
            for this_f in noninseminated_females:
                selected_male = np.random.choice(males)
                this_f.malegenome = selected_male.genome
                this_f.male_origin = (selected_male.origin_lon, selected_male.origin_lat)
                this_f.inseminationstate = 1
        

class PascalAdvection(PascalSimulation):

    def prep_environment(self, reader):
        self.free_env_indices = list(np.arange(0, self.nsup))
        self.time_ind = 0
        # loglevel used to be 100 (above CRITICAL - total silence from
        # OpenDrift, including its own warnings about missing forcing
        # data silently falling back to a configured constant; see
        # opendrift's environment.py Environment.get_environment() -
        # 2026-09-16 update promoted those specific messages from
        # debug to warning, but that only reaches stdout/stderr if this
        # tracker's own loglevel lets WARNING through). WARNING now
        # surfaces those (and OpenDrift's other genuine warnings, e.g.
        # reader setup notices) without the volume of INFO/DEBUG.
        self.tracker = PascalDrift(loglevel=logging.WARNING)
        if isinstance(reader, (list, tuple)):
            for r in reader:
                self.tracker.add_reader(r)
        else:
            self.tracker.add_reader(reader)

        self.tracker.set_config('general:use_auto_landmask', False)
        self.tracker.set_config('general:seafloor_action', 'lift_to_seafloor')
        # Default 'stranding' deactivates (and physically removes) any
        # element that touches land, which permanently shrinks and
        # reindexes self.tracker.elements - environment_index (a raw
        # array position, handed out once at seed() and kept for an
        # individual's whole life) then silently points at the wrong
        # particle, or goes out of bounds during respawn(). 'previous'
        # instead bounces a grounded element back to its last wet
        # position, keeping the elements array (and therefore
        # environment_index) stable for the run's lifetime.
        self.tracker.set_config('general:coastline_action', 'previous')

        if self.tracker_config is not None:
            for k,v in self.tracker_config.items():
                self.tracker.set_config(k, v)
       
        start_loc = self.start_locations[0]
        self.tracker.seed_elements(
                lon=start_loc[0], lat=start_loc[1], z=-10, number=self.nsup, radius=10,
                time=self.start_time - self.timestep
            )
        self.tracker.run_prep(
            time_step=self.timestep.seconds*self.isplit,
            steps=None,
            time_step_output=None,
            duration=None,
            end_time=self.end_time + dt.timedelta(days=365),
            stop_on_error=True,
            outfile=self.opendriftout,
            export_variables=['x', 'y', 'temperature']
        )
        # Need this to populate the environment
        self.tracker.run_1step()

    def prep_outputgrid(self, outputgrid):
        outputgrid['time'] = self.all_steps
        outputgrid['depth'] = self.global_settings['depthrange']
        self.outputgrid = outputgrid

    def set_tracker(self, environment_index, location):
        self.tracker.elements.lon[environment_index] = location[0]
        self.tracker.elements.lat[environment_index] = location[1]

    def set_tracker_origin(self, environment_index, origin):
        self.tracker.elements.origin_lon[environment_index] = origin[0]
        self.tracker.elements.origin_lat[environment_index] = origin[1]

    def update_environment(self):
        self.time_ind+= 1
        self.tracker.run_1step()

    def gene_hunt(self):
        # Get genes from nearby males
        active = self.active_supindividuals()
        noninseminated_females = [
            i for i in active
            if i.sex == 'F' and i.inseminationstate == 0
        ]

        # nb: must be an ndarray (not a plain list) - all_males[near_males]
        # below indexes it with a boolean numpy array, which raises
        # TypeError on a plain list.
        all_males = np.array(
            [i for i in active if i.sex == 'M'], dtype=object
        )

        all_males_ll = np.stack([
            self.tracker.elements.lon[
                [j.environment_index for j in all_males]
            ],
            self.tracker.elements.lat[
                [j.environment_index for j in all_males] ]
        ]).T

        for this_f in noninseminated_females:
            this_f_ll = [self.tracker.elements.lon[
                this_f.environment_index],
                self.tracker.elements.lat[this_f.environment_index]]
            near_males = points_within_distance(
                this_f_ll, all_males_ll,
                self.global_settings['maxmatingdistance'])
            if np.sum(near_males) > 0:
                selected_male = np.random.choice(all_males[near_males])
                this_f.malegenome = selected_male.genome
                this_f.male_origin = (selected_male.origin_lon, selected_male.origin_lat)
                this_f.inseminationstate = 1

    def finish_run(self):
        self.tracker.run_end()
    
        super(PascalAdvection, self).finish_run()



