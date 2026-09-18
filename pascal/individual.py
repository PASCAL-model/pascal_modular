"""Pan-Arctic Behavioural and Life-history Simulator for Calanus (PASCAL).

Version 4.00 :: Python development
Super-individual-based model for simulating behavioural and life-history
strategies of the North Atlantic copepod, Calanus finmarchicus.
"""

# Standard library imports
import math
from copy import deepcopy

# Third-party imports
import numpy as np

# Local imports
from .biology import growth as gdm
from .biology import survival as sv
from .biology import vertical_migration as vm
from .utils import dotdict

EXPECTED_GENES = [
    "a1_bodysize",
    "a2_irradiancesensitivity",  # defines the spectral sensitivity of a given super individual
    "a3_pred1sensitivity",  # defines the visual predator sensitivity (i.e., the ability of a super individual to percieve a visual predator in its environment)
    "a4_pred1reactivity",  # defines the reactivity to visual predators
    "a5_energyallocation",  # defines the energy allocation pattern of a given super individual
    "a6_diapauseprobability",  # defines the probability of diapause entry of a given super individual (higher: likely to diapause, lower: less likely to diapause and more likely to develop directly to adulthood)
    "a7_diapauseentry",  # defines the timing of diapause entry of a given super individual
    "a8_diapauseexit",
]

# The complete set of environment variables update_lifestage()'s call tree
# reads, via get_profile()/get_zi() (depth-resolved, environment_profiles)
# and direct access (environment). Kept explicit here - alongside the code
# that actually reads them - rather than inferred elsewhere, so anything
# parallelizing/distributing SuperIndividual state (e.g. coupler_parallel.py,
# which must ship only each individual's own slice of the environment
# rather than the full shared array) has one place to check that stays in
# sync if a new variable is added to the read methods below.
PROFILE_ENVIRONMENT_VARIABLES = (
    "temperature",
    "food1concentration",
    "irradiance",
    "pred1dens",
)
SCALAR_ENVIRONMENT_VARIABLES = ("mld",)

# Default set of per-timestep scalar state logged by debug mode
# (coupler.py::PascalSimulation, when debug=True rather than an explicit
# variable list) - the state that actually changes over a super-individual's
# lifespan, excluding constants (eggmass, cxthreshold...), back-references
# (environment, environment_profiles, global_settings, datalogger), and
# non-scalar fields (genome, malegenome) that debug_out()'s plain getattr()
# path isn't set up to serialize.
DEFAULT_DEBUG_VARIABLES = (
    "developmentalstage",
    "structuralmass",
    "reservemass",
    "maxstructuralmass",
    "age",
    "sex",
    "nvindividuals",
    "zpos",
    "zidx",
    "diapausestate",
    "diapausestrategy",
    "thermalavg",
    "feedingrate",
    "growthrate",
    "egestionrate",
    "metabolicrate",
    "reproductiveallocation",
    "totalfecundity",
    "potentialfecundity",
    "inseminationstate",
    "timeofdiapauseentry",
    "timeofdiapauseexit",
    "lifestatus",
)

class SuperIndividual(object):
    """A single PASCAL super-individual: one simulated *Calanus* copepod
    representing ``nindividuals`` real (virtual) individuals that share an
    identical life history.

    Holds all per-individual biological state (developmental stage, body
    mass, diapause state, genome, fecundity, ...) and the methods that
    advance it one timestep at a time (:meth:`update_lifestage` and its
    stage-dispatch tree). A super-individual is tied 1:1 to one element/slot
    in the owning :class:`~pascal.coupler.PascalSimulation`'s particle
    tracker via ``environment_index`` - that slot supplies its position and
    the environment data it reads each timestep (see
    :meth:`get_profile`/:meth:`get_zi`).

    Instances are created and destroyed by
    :class:`~pascal.coupler.PascalSimulation` (``seed()``/``respawn()``/
    ``clean_dead()``), not typically constructed directly by user code.
    """

    def __init__(
        self,
        global_settings,
        diapausedepth,
        environment,
        environment_profiles,
        environment_index,
        eggmass=0.23,
        nindividuals=10000,
        genes=None,
        cxthreshold=0.7,
        muthreshold=0.2,
        datalogger=None,
        unique_id=None,
        origin=None,
    ):
        """
        Parameters
        ----------
        global_settings : dict
            Shared run-wide parameters (biological rate constants, the
            vertical ``depthrange`` grid, mortality/fecundity ceilings,
            ``stochastic`` flag, ...) - the same object is shared by
            reference across every individual in a run.
        diapausedepth : float
            Initial value for ``self.diapausedepth`` (later overwritten by
            :func:`pascal.biology.vertical_migration.diapausedepthselection`
            once an individual actually enters diapause).
        environment, environment_profiles : dict-like
            Current-timestep environment data (scalar and depth-profile
            variables respectively), as produced by the owning
            simulation's ``update_environment()``. Re-pointed to the
            current timestep's objects every step by
            :meth:`pascal.coupler.PascalSimulation.sync_environment_references`,
            not re-fetched here.
        environment_index : int
            This individual's slot index into the particle tracker's
            element arrays (position, origin, ...).
        eggmass : float, optional
            Starting structural mass (also the egg mass used for fecundity
            accounting elsewhere).
        nindividuals : int, optional
            Initial virtual-individual count this super-individual
            represents.
        genes : dict-like, optional
            Explicit genome (one value in [0, 1] per entry of
            ``EXPECTED_GENES``). If ``None``, a fresh genome is drawn
            (random if ``global_settings["stochastic"]``, else a fixed
            0.2 for every gene).
        cxthreshold, muthreshold : float, optional
            Per-gene crossover/mutation probability thresholds used by
            :meth:`get_child_genome` when this individual mates.
        datalogger : pascal.data_logger.OutputLogger, optional
            If given, diapause-entry/exit and direct-development events are
            logged to it (see :meth:`diapause0`/:meth:`diapause1`).
        unique_id : int, optional
            Identifier used by the owning simulation's lifetime-stats
            bookkeeping; not used internally by this class.
        origin : tuple of (float, float), optional
            ``(lon, lat)`` this individual is considered to have started
            from, used to rescue it back to safety if its tracker element
            would otherwise leave the model domain (see
            ``pascal_drift.py::PascalDrift.remove_deactivated_elements``).
            Defaults to ``(None, None)``.
        """
        # these are reflective of individual states and vary during the lifespan of super individuals depending on the individual-environment interactions and internal processes (e.g., hardcoded strategies)
        self.global_settings = global_settings
        if "stochastic" not in self.global_settings.keys():
            self.global_settings["stochastic"] = True

        self.eggmass = deepcopy(eggmass)

        # defines the living (1) or dead (0) state of super individuals
        self.lifestatus = 1
        # defines the no. of virtual individuals contained in a super individual - this no. is defined by the constant, 'nvindividualspersupindividual' above
        self.nvindividuals = nindividuals
        # defines the developmental stage of super individuals:
        # 0:Egg, 1:NI, 2:NII, 3:NIII, 4:NIV, 5:NV, 6:NVI, 7:CI, 8:CII, 9:CIII, 10:CIV, 11:CV, 12:CVI-F, 13:CVI-M
        self.developmentalstage = 0
        # defines the mean temperature trajectory encountered during the early lifestages
        # nb:this is obsolete beyond non-feeding stages, whose development is estimated as a function of growth
        self.thermalavg = 0.00
        # defines the structural body mass of the super individual (min = 0.23 ugC at embryonic stage), initializes with 0.00
        self.structuralmass = eggmass
        # defines the maximum lifetime structural mass of a super individual (used for starvation risk estimation)
        self.maxstructuralmass = eggmass
        # defines the energy reserve mass of the super individual (max = 0.70 x structural mass), initializes with 0.00
        self.reservemass = 0.00
        # defines the age of the super individual
        self.age = 0
        # defines the sex of the super individual (M:male, F:female, U:undefined)
        self.sex = "U"
        # defines the time of diapause entry of the super individual
        self.timeofdiapauseentry = 0
        # defines the time of diapause exit of the super individual
        self.timeofdiapauseexit = 0
        # defines the structural body mass at diapause entry
        self.structuralmassatdiapauseentry = 0.00
        # defines the energy reserve mass at diapause entry
        self.reservemassatdiapauseentry = 0.00
        # defines the developmental stage at diapause entry
        self.developmentalstageatdiapauseentry = 0
        # defines the depth of diapause
        self.diapausedepth = diapausedepth
        # defines the state of diapause of CIV and CV individuals 0, 1, or 2 ...unsure what each means
        self.diapausestate = 0
        # defines the energy reserve mass at diapause exit
        # defines whether the super individual will enter diapause and then potentially molt to the adult or potentilly develop directly to adulthood without diapause
        self.diapausestrategy = -1

        self.reservemassatdiapauseexit = 0.00
        # defines the insemination state of females (0: not inseminated, 1: inseminated)
        self.inseminationstate = 0
        # defines the energy allocated to reproductive output
        self.reproductiveallocation = 0.00
        # defines the male genome copied to a female after mating
        self.malegenome = None
        # defines the (lon, lat) origin of the male copied to a female
        # after mating - see get_child_origin()
        self.male_origin = None
        # defines the total no. of eggs produced by a female during its lifespan
        # nb:not all of these eggs are spawned into the super individual pool - only the reward-based ones (see below)
        self.totalfecundity = 0
        # defines the no. of eggs procuced by a female at each timepoint
        self.potentialfecundity = 0

        # evolvable attributes ('genes')
        # ______________________________

        # these are attributes whose values freely evolve across time and space as the model is iteratively computed
        # no artificial forcing is applied to optimize the free attribute combination - it is dependent on the 'simulated natural selection' that happens within the model
        # all evolvable attributes range from 0 - 1 in floating point designation
        # defines the body size trajectory that a super individual follows during its lifespan
        # nb: pascalv4 does not support p2sensitivity or p2reactivity attributes - these can be included in future developments
        if genes is not None:
            self.genome = genes
        else:
            self.genome = dotdict(
                {
                    g: (
                        np.random.rand(1)[0]
                        if self.global_settings["stochastic"]
                        else 0.2
                    )
                    for g in EXPECTED_GENES
                }
            )

        # This is determined only by the genes
        self.adultsize = (
            self.global_settings["cmm_lower"][12]
            + (
                self.global_settings["cmm_upper"][12]
                - self.global_settings["cmm_lower"][12]
            )
            * self.genome.a1_bodysize
        )

        # The thresholds for mutation/crossover of genes during reproduction
        self.cxthreshold = cxthreshold
        self.muthreshold = muthreshold

        # Its alive to start with
        self.alive = True

        # Environment variables
        self.environment = environment
        self.environment_profiles = environment_profiles
        self.environment_index = environment_index
        # this estimates the normalized and range-scaled 0.1-0.9) ambient shortwave irradiance for the calculation of light dependence of the visual predation risk
        self.zidx = 0
        self.zpos = 1
        # Cache for get_profile(); cleared every timestep by
        # coupler.py::sync_environment_references(). Initialized here too
        # (not just there) so direct use of SuperIndividual outside the
        # normal coupler.py run() loop doesn't hit an AttributeError.
        self._profile_cache = {}

        if len(self.environment_profiles["z"]) != len(
            self.global_settings["depthrange"]
        ):
            self.depth_interpolate = True
        else:
            self.depth_interpolate = ~(
                -self.environment_profiles["z"] == self.global_settings["depthrange"]
            ).all()

        self.time = 0  # For logging

        # Run variables
        self.temporalres = 6

        # Data logging only happens if a logger object is passed to the individual
        self.datalogger = datalogger

        # Diagnostic variables for logging
        self.feedingrate = 0
        self.growthrate = 0
        self.egestionrate = 0
        self.metabolicrate = 0

        # Unique id for outputing individual based data
        self.unique_id = unique_id

        # The (lon, lat) location this super-individual (or, for a
        # mating-spawned one, the parent it inherited from - see
        # get_child_origin()) originally started from. Used by
        # PascalAdvection to move a super-individual back to safety if its
        # tracker element would otherwise be deactivated for leaving the
        # domain (see pascal_drift.py::PascalDrift.remove_deactivated_elements()).
        self.origin_lon, self.origin_lat = origin if origin is not None else (None, None)

    def update_vert(self):
        """Recompute ``self.zidx`` (the index into ``depthrange`` nearest
        ``self.zpos``) after a vertical-position update."""
        self.zidx = np.argmin(abs(self.global_settings["depthrange"] - self.zpos))
        # maxdepth/mindepth are set externally by
        # coupler.py::sync_environment_references(), once per timestep for
        # every individual, rather than recomputed here per-individual:
        # depth ('z') has no per-individual axis (it's the same water
        # column grid for everyone), so redoing this max/min per individual
        # per timestep was pure redundant work - it was ~17% of total
        # runtime by itself (see BENCHMARKING.md).

    def get_profile(self, var):
        """Return this individual's full depth profile for environment
        variable ``var`` at its current environment slot, interpolated onto
        ``global_settings["depthrange"]`` if the reader's own depth levels
        differ.

        Parameters
        ----------
        var : str
            Name of a profile environment variable (e.g. ``"temperature"``,
            ``"food1concentration"``, ``"irradiance"``, ``"pred1dens"`` -
            see ``PROFILE_ENVIRONMENT_VARIABLES``).

        Returns
        -------
        numpy.ndarray
            One value per depth level of ``global_settings["depthrange"]``.

        Notes
        -----
        Results are cached per individual per timestep (cleared by
        :meth:`pascal.coupler.PascalSimulation.sync_environment_references`)
        since the same variable is commonly re-requested more than once
        per timestep.
        """
        # Cached per individual per timestep (cleared by
        # coupler.py::sync_environment_references()) since the same var is
        # commonly re-requested more than once per timestep (e.g.
        # "temperature" via get_profile() in vertical migration, then again
        # via get_zi() in growth, then again in mortality) - when
        # depth_interpolate is True (the common case in advection mode,
        # where a reader's own profile depth levels rarely match PASCAL's
        # configured depthrange exactly - see BENCHMARKING.md) each of
        # those was independently re-running np.interp() over the same
        # data. Harmless to cache the non-interpolating branch too, since
        # the dict lookup is cheap either way.
        if var in self._profile_cache:
            return self._profile_cache[var]

        if self.depth_interpolate:
            result = np.interp(
                self.global_settings["depthrange"],
                -self.environment_profiles["z"],
                self.environment_profiles[var][:, self.environment_index],
            )
        else:
            result = self.environment_profiles[var][:, self.environment_index]

        self._profile_cache[var] = result
        return result

    def get_zi(self, var):
        """Return this individual's environment variable ``var`` at its
        current vertical position (``self.zidx``) only, via
        :meth:`get_profile`.

        Parameters
        ----------
        var : str
            Name of a profile environment variable.

        Returns
        -------
        float
        """
        return self.get_profile(var)[self.zidx]

    def update_lifestage(self):
        """Advance this individual by one timestep: run its stage-specific
        growth/development/vertical-migration logic
        (:meth:`run_stage_transition`), then its mortality/death check
        (:meth:`apply_mortality_and_deathcheck`).

        Not used by :class:`~pascal.coupler.PascalSimulation` directly -
        the coupler calls the two steps separately so the mortality step
        can be batched (vectorized) across all individuals at once; see
        :meth:`pascal.coupler.PascalSimulation.apply_mortality_and_deathcheck_batch`.
        """
        self.run_stage_transition()
        self.apply_mortality_and_deathcheck()

    def run_stage_transition(self):
        """Dispatch to this individual's stage-specific growth/development/
        vertical-migration method based on ``self.developmentalstage``:
        :meth:`stage_lt_2` (egg, NI, NII), :meth:`stage_3_9`
        (NIII-CIII), :meth:`stage_10_11` (CIV/CV, which further dispatch to
        :meth:`diapause0`/:meth:`diapause1`/:meth:`diapause2`), or
        :meth:`stage_12` (adult female/male). A dead or not-yet-seeded
        individual (``developmentalstage`` outside these ranges) is a
        no-op."""
        # the growth & development, survival and reproductive simulation happens within this if() condition based on developmental stage
        # no else() condition is written, as the loop skips if a super indivdual is dead or unseeded/uninitialized

        # this structures the simulation into following developmental stage categories:
        # 1. non-feeding egg, NI and NII stages (index: 0, 1, 2)
        # 2. feeding but non-energy-storing NIII-NVI,CI-CIII stages (index: 3, 4, 5, 6, 7, 8, 9)
        # 3. feeding and energy-storing CIV and CV (diapausing) stages (index: 10, 11)
        # 4. adult females (index: 12)
        # 5. adult males (index: 13)
        # nb:these stage groupings are for C. finmarchicus only - for C.glacialis and C.hyperboreus, stage compositions of some categories may vary

        # simulation of de-growth, development and survival of non-feeding stages (egg, NI, NII): dsc-I
        # ______________________________________________________________________________________________

        if self.developmentalstage <= 2:
            self.stage_lt_2()

        elif self.developmentalstage >= 3 and self.developmentalstage < 10:
            self.stage_3_9()

        elif self.developmentalstage == 10 or self.developmentalstage == 11:
            self.stage_10_11()

        elif self.developmentalstage == 12:
            self.stage_12()

    def apply_mortality_and_deathcheck(self):
        """Apply stage->2 predation/background mortality
        (:meth:`apply_dsc2_mortality`) to shrink ``nvindividuals``, then
        mark ``self.lifestatus = 0`` if this individual has hit any of the
        three death conditions: ``nvindividuals`` at or below
        ``global_settings["virtualindividualthrehold"]``, ``age`` at or
        past ``ageceiling``, or ``totalfecundity`` at or past
        ``fecundityceiling``. A dead flag here does not remove the
        individual - see
        :meth:`pascal.coupler.PascalSimulation.clean_dead`.
        """
        # Split out from update_lifestage() so coupler.py can batch this
        # part (uniform, branch-simple math shared by every stage>2
        # individual) across many individuals in one vectorized pass
        # instead of calling it once per individual - see
        # coupler.py::apply_mortality_and_deathcheck_batch() and
        # BENCHMARKING.md. run_stage_transition() (the stage-specific
        # growth/vertical-migration dispatch above) is NOT batched: its
        # vertical migration functions have genuinely per-individual
        # variable-length search logic that doesn't reduce to simple
        # array masking - see BENCHMARKING.md for why that's out of scope.
        if self.developmentalstage > 2:
            self.apply_dsc2_mortality()

        # post-developmental-stage processing
        # ___________________________________

        # simulation of death of super individuals (i.e., when all virtual individual dies, a super individual also dies)
        # all state variables, gene values, loggers etc. are reset for a new super individual to take its place (these do not need to be re-initialized at seeding/spawning; only the 'gene' values do)

        if (
            self.nvindividuals <= self.global_settings["virtualindividualthrehold"]
            or self.age >= self.global_settings["ageceiling"]
            or self.totalfecundity >= self.global_settings["fecundityceiling"]
        ):
            self.lifestatus = 0

    def stage_lt_2(self):
        """Advance one timestep for non-feeding early stages (Egg, NI, NII:
        ``developmentalstage`` 0-2): passive vertical positioning
        (:func:`~pascal.biology.vertical_migration.verticalmigration_dsc0`),
        temperature-only growth/development
        (:func:`~pascal.biology.growth.growthdevelopmentmetabolism_dsc0`),
        stage advancement once age reaches the stage's developmental time,
        and mortality (:meth:`apply_dsc0_mortality`)."""
        # eggs and non-feeding naupliar stages (Egg-NII)
        self.update_vert()
        self.zpos, self.zidx = vm.verticalmigration_dsc0(
            smld=self.environment["mld"][self.environment_index],
            maxdepth=self.maxdepth,
            mindepth=self.mindepth,
            depthrange=self.global_settings["depthrange"],
            continuousdepth=False,
        )

        # update the current thermal history (this is an arithmetic mean)
        self.thermalavg = (
            self.thermalavg + self.get_zi("temperature")
        ) / 2.00  #!!!!!!!!!!!!!!

        # this is the parameter "a" in Belehrádek’s (1935) temperature function, adopted from Campbel et al. (2001), see: https://doi.org/10.3354/meps221161
        currentdevelopmentalcoefficient = self.global_settings[
            "developmentalcoefficient"
        ][self.developmentalstage]

        (
            currentdevelopmentaltime,
            self.feedingrate,
            currentgrowthrate,
            self.metabolicrate,
            self.egestionrate,
        ) = gdm.growthdevelopmentmetabolism_dsc0(
            temperature=self.get_zi("temperature"),
            devcoef=currentdevelopmentalcoefficient,
            thist=self.thermalavg,
            strmass=self.structuralmass,
            tempres=self.temporalres,
        )
        self.structuralmass = self.structuralmass + currentgrowthrate
        self.age += 1

        if self.age >= currentdevelopmentaltime:
            self.developmentalstage += 1
        # end if

        # mortality risk and survival
        self.apply_dsc0_mortality()

    def stage_3_9(self):
        """Advance one timestep for feeding, non-energy-storing stages
        (NIII-CIII: ``developmentalstage`` 3-9): active vertical migration
        (:meth:`apply_dsc1_verticalmigration`), food/temperature-dependent
        growth (:func:`~pascal.biology.growth.growthanddevelopment_dsc1`),
        stage advancement once structural mass reaches the current
        critical molting mass (:meth:`get_currentcmm`), and predation/
        background mortality
        (:func:`~pascal.biology.survival.mortalityrisk_dsc1`, applied
        directly here rather than via :meth:`apply_dsc1_mortality` - see
        that method's docstring)."""
        # feeding but non-energy-storing stages (NIII-CIII)
        self.update_vert()
        maxdistance, traveldistance = self.apply_dsc1_verticalmigration()

        self.feedingrate, currentgrowthrate, self.egestionrate, self.metabolicrate = (
            gdm.growthanddevelopment_dsc1(
                temperature=self.get_zi("temperature"),
                foodcon=self.get_zi("food1concentration"),
                strmass=self.structuralmass,
                maxzd=maxdistance,
                actzd=traveldistance,
                temporalres=self.temporalres,
            )
        )

        self.structuralmass += currentgrowthrate

        if self.structuralmass > self.maxstructuralmass:
            self.maxstructuralmass = deepcopy(self.structuralmass)
        # end if

        # stage advancement and agein
        self.age += 1

        if self.structuralmass >= self.get_currentcmm():
            self.developmentalstage += 1
        # end if

        # mortality risk and survival
        currentmortalityrisk = sv.mortalityrisk_dsc1(
            strmass=self.structuralmass,
            maxstrmass=self.maxstructuralmass,
            vpreldensity=self.get_zi("pred1dens"),
            irradiance=self.get_zi("irradiance"),
            maxirradiance=self.global_settings["maxirradiance"],
            minirradiance=self.global_settings["minirradiance"],
            nvpreldensity=self.global_settings["nonvisualpredatorreldensity"],
            bgmrisk=self.global_settings["backgroundmortalityrisk"],
        )

        self.nvindividuals = int(self.nvindividuals * (1.00 - currentmortalityrisk))

    def stage_10_11(self):
        """Advance one timestep for CIV/CV (``developmentalstage`` 10 or
        11): on first arrival, randomly commits this individual to a
        diapause strategy (``diapausestrategy``: 1 with probability
        ``genome.a6_diapauseprobability``, else 0 for direct development),
        then dispatches on ``diapausestate`` to :meth:`diapause0`
        (pre-diapause), :meth:`diapause1` (in diapause) or :meth:`diapause2`
        (post-diapause)."""
        # if the diapause strategy is undefined (-1: typical for newly seeded/spawned super individual arriving at civ/cv for the first time), define the diapause strategy (0, 1)
        # nb:the diapause strategy is linked to the diapause probability 'gene'
        if self.diapausestrategy == -1:
            # random number for diapause strategy determination
            dsdet = np.random.rand(1).squeeze()
            # falls into 0 (direct development, no diapause) or 1 (diapause) depending on the diapause probability 'gene' value
            self.diapausestrategy = (
                1 if dsdet < self.genome.a6_diapauseprobability else 0
            )
        # end if
        self.update_vert()

        if self.diapausestate == 0:
            self.diapause0()
        elif self.diapausestate == 1:
            self.diapause1()
        elif self.diapausestate == 2:
            self.diapause2()
        else:
            print("Warning undefined diapause stage!!!")

    def stage_12(self):
        """Advance one timestep for adults (``developmentalstage`` 12):
        determines sex on first arrival (50/50), does vertical migration
        (:meth:`apply_dsc2_verticalmigration`), then sex-specific
        growth/reproduction - males degrow with no feeding
        (:meth:`apply_dsc3_male_growthdevel`); females feed and channel
        growth into reproductive allocation once inseminated
        (:meth:`apply_dsc2_growthdevel`), accumulating
        ``potentialfecundity`` once enough reserve has been allocated to
        cover at least one ``eggmass``. Note that unlike the other
        ``stage_*``/``diapause*`` methods, this one does not itself call
        a mortality method - mortality for adults is applied only via
        :meth:`apply_mortality_and_deathcheck`
        (:meth:`apply_dsc2_mortality`)."""
        # adult stages (male and female)

        # sex determination (if not pre-determined)
        if self.sex == "U":
            # drawing a uniform random number to compare with the threshold of 0.50
            sexdet = np.random.rand(1).squeeze()
            self.sex = "M" if sexdet < 0.5 else "F"
        # end if

        # Update vertical position
        self.update_vert()

        # Do vertical migration
        maxdistance, traveldistance = self.apply_dsc2_verticalmigration()

        # sex-specific processing of growth and reproduction
        if self.sex == "M":
            # males: no feeding, and therfore sustains degrowth
            # growth, development and metabolism
            currentgrowthrate = self.apply_dsc3_male_growthdevel(
                maxdistance=maxdistance, traveldistance=traveldistance
            )

            # reserve exhaustion
            if self.reservemass >= abs(currentgrowthrate):
                self.reservemass += currentgrowthrate
            else:
                self.structuralmass += currentgrowthrate
            # end if

        elif self.sex == "F":
            # femaless: feed, but no somatic growth - all energy accumulation is channelled to egg production
            eggproductionstate = 0
            # growth, development and metabolism
            currentgrowthrate = self.apply_dsc2_growthdevel(
                maxdistance=maxdistance, traveldistance=traveldistance
            )

            # reserve allocation
            if currentgrowthrate >= 0.00:
                if self.structuralmass < self.adultsize:
                    if self.inseminationstate == 1:
                        self.structuralmass += currentgrowthrate * (
                            1.00 - self.genome.a5_energyallocation
                        )
                        self.reproductiveallocation = (
                            currentgrowthrate * self.genome.a5_energyallocation
                        )
                        eggproductionstate = 1

                    else:
                        self.structuralmass += currentgrowthrate
                    # end if

                    if self.structuralmass > self.maxstructuralmass:
                        self.maxstructuralmass = deepcopy(self.structuralmass)
                    # end if

                else:
                    if self.inseminationstate == 1:
                        self.reproductiveallocation = currentgrowthrate
                        eggproductionstate = 1
                    # end if
                # end if
            else:

                if self.reservemass >= abs(currentgrowthrate):
                    # no update to structural mass
                    self.reservemass += currentgrowthrate
                else:
                    # no update to reserve mass
                    self.structuralmass += currentgrowthrate
                # end if

            # end if

            # reproduction and spawning - in the coupler/individual vesrion the inseminationstate == 0 part is handled in the coupler
            if self.inseminationstate == 1:
                if (
                    eggproductionstate == 1
                    and self.reproductiveallocation >= self.eggmass
                ):
                    self.potentialfecundity = int(
                        self.reproductiveallocation // self.eggmass
                    )
                    self.totalfecundity += (
                        self.potentialfecundity
                    )  # Used for killing off if fecundity ceiling reached
                # end if
            # end if

            # reset egg-production status
            eggproductionstate = 0

        # end if

        self.age += 1

    def diapause0(self):
        """Advance one timestep for a CIV/CV individual at pre-diapause
        (``diapausestate == 0``): growth/reserve allocation
        (:func:`~pascal.biology.growth.growthanddevelopment_dsc2`, split
        between structural and reserve mass depending on
        ``diapausestrategy`` and the two ``energyallocthreshold*``
        settings), then either stage advancement (direct-development path,
        ``diapausestrategy == 0``) or an entry into "true" vs. "active"
        diapause (``diapausestrategy != 0``) once the reserve/structural
        mass ratio passes ``genome.a7_diapauseentry``, chosen by
        ``maxdepth`` against ``diapausedepththreshold0/1``
        (:func:`~pascal.biology.vertical_migration.diapausedepthselection`).
        Diapause entry/direct-development events are logged via
        ``self.datalogger`` when set.

        """
        # feeding and energy storing (diapause) stages at pre-diapause
        maxdistance, traveldistance = self.apply_dsc2_verticalmigration()

        # growth, development and metabolism
        currentgrowthrate = self.apply_dsc2_growthdevel(
            maxdistance=maxdistance, traveldistance=traveldistance
        )

        # reserve allocation
        if currentgrowthrate > 0:

            if self.diapausestrategy == 0:

                if self.reservemass / self.structuralmass >= 1.00:
                    self.structuralmass = self.structuralmass + currentgrowthrate
                    # Moved max structural mass to end of if clause as occured in both cases
                else:
                    # update both structural mass and reserve mass
                    if (
                        self.structuralmass
                        < self.global_settings["energyallocthreshold1"]
                    ):
                        currentenergyallocation = 0.00
                    elif (
                        self.structuralmass
                        >= self.global_settings["energyallocthreshold1"]
                        and self.structuralmass
                        <= self.global_settings["energyallocthreshold2"]
                    ):
                        currentenergyallocation = (
                            currentgrowthrate
                            * self.genome.a5_energyallocation
                            * 1.00
                            / (1.00 + math.exp((60.00 - self.structuralmass) / 20.00))
                        )
                    else:
                        currentenergyallocation = (
                            currentgrowthrate * self.genome.a5_energyallocation
                        )
                    # end if

                    self.reservemass = self.reservemass + currentenergyallocation
                    self.structuralmass = (
                        self.structuralmass
                        + currentgrowthrate
                        - currentenergyallocation
                    )

                # end if

                if self.structuralmass > self.maxstructuralmass:
                    self.maxstructuralmass = self.structuralmass

                # end if

            else:

                if self.structuralmass >= self.adultsize:

                    # no update to structural mass
                    if (
                        self.structuralmass
                        < self.global_settings["energyallocthreshold1"]
                    ):
                        currentenergyallocation = 0.00
                    elif (
                        self.structuralmass
                        >= self.global_settings["energyallocthreshold1"]
                        and self.structuralmass
                        <= self.global_settings["energyallocthreshold2"]
                    ):
                        currentenergyallocation = (
                            currentgrowthrate
                            * 1.00
                            / (1.00 + math.exp((60.00 - self.structuralmass) / 20.00))
                        )
                    else:
                        currentenergyallocation = currentgrowthrate
                    # end if

                    self.reservemass += currentenergyallocation

                else:

                    # update both structural mass and reserve mass
                    if (
                        self.structuralmass
                        < self.global_settings["energyallocthreshold1"]
                    ):
                        currentenergyallocation = 0.00
                    elif (
                        self.structuralmass
                        >= self.global_settings["energyallocthreshold1"]
                        and self.structuralmass
                        <= self.global_settings["energyallocthreshold2"]
                    ):
                        currentenergyallocation = (
                            currentgrowthrate
                            * self.genome.a5_energyallocation
                            * 1.00
                            / (1.00 + math.exp((60.00 - self.structuralmass) / 20.00))
                        )
                    else:
                        currentenergyallocation = (
                            currentgrowthrate * self.genome.a5_energyallocation
                        )
                    # end if

                    self.structuralmass = (
                        self.structuralmass
                        + currentgrowthrate
                        - currentenergyallocation
                    )
                    self.reservemass = self.reservemass + currentenergyallocation

                    if self.structuralmass > self.maxstructuralmass:
                        self.maxstructuralmass = deepcopy(self.structuralmass)

                    # end if
                # end if
            # end if

        else:
            self.update_mass(currentgrowthrate)

        # end if

        # stage advancement and ageing
        self.age += 1

        currentcmm = self.get_currentcmm()
        # molting from developmental stage 'd'to 'd + 1' occurs only if the current structural mass exceeds the stage-specific critical molting mass

        if self.diapausestrategy == 0:
            if self.structuralmass >= currentcmm:
                self.developmentalstage += 1

                # nb:here, the data logging occurs only when the super-individual reaches the adult
                # this makes sure that logging for each direct-dev. happens only once in its lifespan!

                if self.datalogger is not None and self.developmentalstage == 12:
                    # obs:this population size is an overestimate, as it is reduced (potentially) in the mortality risk function below!
                    # this logs the no. of direct-developing (non-diapausing) genetically-determined entries
                    # index:<nddev, cwddev, swddev> <time> <genetic, environmental>
                    self.datalogger.add_ddev(self.get_log_data(), 0)

                # end if

            # end if

        else:

            if self.reservemass / self.structuralmass >= self.genome.a7_diapauseentry:

                if self.maxdepth >= self.global_settings["diapausedepththreshold1"]:
                    # the individual enters into a "true" diapause (basal metabolism drops by ca. 75%)
                    self.diapausestate = 1
                    # set the diapause mode to "true diapause" (= 0)
                    self.diapausemode = 0

                    self.diapausedepth, self.diapausedepthidx = (
                        vm.diapausedepthselection(
                            maxdepth=self.maxdepth,
                            ddt0=self.global_settings["diapausedepththreshold0"],
                            ddt1=self.global_settings["diapausedepththreshold1"],
                            ddt2=self.global_settings["diapausedepththreshold2"],
                            depthrange=self.global_settings["depthrange"],
                            continuousdepth=False,
                        )
                    )

                    self.reservesatdiapauseentry = deepcopy(self.reservemass)

                    # datalogging (related to diapause entry)
                    if self.datalogger is not None:
                        # diapause log is stage-specific: this defines the stage-specific index for the datalog
                        diapauselogidx = 0 if self.developmentalstage == 10 else 1
                        # this logs diapause entry data of civ stages of true diapause
                        self.datalogger.add_den(
                            self.get_log_data(), diapauselogidx, self.diapausemode
                        )
                    # end if

                elif (
                    self.maxdepth < self.global_settings["diapausedepththreshold1"]
                    and self.maxdepth >= self.global_settings["diapausedepththreshold0"]
                ):

                    # the individual enters into an "active" diapause (basal metabolism drops by ca. 50%)
                    self.diapausestate = 1

                    # set the diapause mode to "true diapause" (= 1)
                    self.diapausemode = 1

                    self.diapausedepth, self.diapausedepthidx = (
                        vm.diapausedepthselection(
                            maxdepth=self.maxdepth,
                            ddt0=self.global_settings["diapausedepththreshold0"],
                            ddt1=self.global_settings["diapausedepththreshold1"],
                            ddt2=self.global_settings["diapausedepththreshold2"],
                            depthrange=self.global_settings["depthrange"],
                            continuousdepth=False,
                        )
                    )

                    self.reservesatdiapauseentry = deepcopy(self.reservemass)

                    # datalogging (related to diapause entry)
                    if self.datalogger is not None:
                        # diapause log is stage-specific: this defines the stage-specific index for the datalog
                        diapauselogidx = 0 if self.developmentalstage == 10 else 1
                        # this logs diapause entry data of civ stages of active diapause
                        self.datalogger.add_den(
                            self.get_log_data(), diapauselogidx, self.diapausemode
                        )

                    # end if
                else:

                    # although they meet the diapause entry criterion, these individuals inhabit in waters with bottom depth < 50 m
                    # therefore, they cannot enter diapause (they may if they are swept off to waters with deeper bottom depths)
                    # unless a potential diapause entry occurs, they are kept at "diapausestate" = 0, feed, grow, develops and develops into adulthood & complete the life cycle

                    if self.structuralmass >= currentcmm:
                        self.developmentalstage += 1

                    # nb:here, the data logging occurs only when the super-individual reaches the adult
                    # this makes sure that logging for each direct-dev. happens only once in its lifespan!
                    if self.datalogger is not None and self.developmentalstage == 12:
                        # obs:this population size is an overestimate, as it is reduced (potentially) in the mortality risk function below!
                        # this logs the no. of direct-developing (non-diapausing) environmentally-determined entries (bottom-depth-constrained)
                        self.datalogger.add_ddev(self.get_log_data(), 1)
                    # end if
                # end if
            # end if

            else:

                if self.structuralmass >= currentcmm and self.developmentalstage == 10:
                    self.developmentalstage += 1
                # end if
            # end if

    def diapause1(self):
        """Advance one timestep for a CIV/CV individual in diapause
        (``diapausestate == 1``): reduced-metabolism degrowth
        (:func:`~pascal.biology.growth.growthanddevelopment_dsc2_diapause`,
        using ``diapausemetabolicrateadj0``/``1`` depending on
        ``diapausemode`` set at diapause entry in :meth:`diapause0`), then
        exit to ``diapausestate = 2`` once the fraction of reserve mass
        consumed since diapause entry reaches ``genome.a8_diapauseexit``.
        Logs the exit event via ``self.datalogger`` when set."""
        # feeding and energy storing (diapause) stages at diapause
        # estimation of potential degrowth and diapause metabolism
        if self.diapausemode == 0:
            currentdiapausemetabolicrateadj = self.global_settings[
                "diapausemetabolicrateadj0"
            ]
        else:
            currentdiapausemetabolicrateadj = self.global_settings[
                "diapausemetabolicrateadj1"
            ]

        self.feedingrate, currentgrowthrate, self.egestionrate, self.metabolicrate = (
            gdm.growthanddevelopment_dsc2_diapause(
                temperature=self.get_zi("temperature"),
                strmass=self.structuralmass,
                resmass=self.reservemass,
                dmrate=currentdiapausemetabolicrateadj,
                temporalres=self.temporalres,
            )
        )
        # reserve exhaustion
        if self.reservemass >= abs(currentgrowthrate):
            self.reservemass = self.reservemass + currentgrowthrate
        else:
            self.structuralmass = self.structuralmass + currentgrowthrate
        # end if

        # ageing and exit from diapause
        self.age += 1

        if (1.00 - self.reservemass / self.reservesatdiapauseentry) >= self.genome[
            "a8_diapauseexit"
        ]:
            self.diapausestate = 2

            # datalogging(diapause-exit related)
            if self.datalogger is not None:
                # diapause log is stage-specific: this defines the stage-specific index for the datalog
                diapauselogidx = 0 if self.developmentalstage == 10 else 1
                # this logs diapause exit data of civ stages of type-r and -e
                self.datalogger.add_dex(self.get_log_data(), diapauselogidx, 0)
            # end if

        # end if

    def diapause2(self):
        """Advance one timestep for a CIV/CV individual at post-diapause
        (``diapausestate == 2``): resumed growth
        (:meth:`apply_dsc2_growthdevel`), then advances to the next
        developmental stage once structural mass reaches the current
        critical molting mass (:meth:`get_currentcmm`)."""
        # feeding and energy storing (diapause) stages at post-diapause
        # determine new vertical position
        maxdistance, traveldistance = self.apply_dsc2_verticalmigration()

        # growth, development and metabolism
        currentgrowthrate = self.apply_dsc2_growthdevel(
            maxdistance=maxdistance, traveldistance=traveldistance
        )

        # reserve allocation
        if currentgrowthrate >= 0.00:
            self.structuralmass += currentgrowthrate

            if self.structuralmass > self.maxstructuralmass:
                self.maxstructuralmass = deepcopy(self.structuralmass)
            # end if

        else:

            if self.reservemass >= abs(currentgrowthrate):
                # no update to structural mass
                self.reservemass += currentgrowthrate

            else:
                # no update to reserve mass
                self.structuralmass += currentgrowthrate
            # end if
        # end if

        # stage advancement and ageing
        self.age += 1

        currentcmm = self.get_currentcmm()

        if self.structuralmass >= currentcmm:
            self.developmentalstage += 1
        # end if

    def apply_dsc1_verticalmigration(self):
        """Run active (dsc1) vertical migration for stage_3_9 individuals via
        :func:`~pascal.biology.vertical_migration.verticalmigration_dsc1`,
        updating ``self.zpos``/``self.zidx`` in place.

        Returns
        -------
        maxdistance, traveldistance : float
            Passed straight through to :func:`~pascal.biology.growth.growthanddevelopment_dsc1`.
        """
        self.zpos, self.zidx, maxdistance, traveldistance = vm.verticalmigration_dsc1(
            temprange=self.get_profile("temperature"),
            fconrange=self.get_profile("food1concentration"),
            iradrange=self.get_profile("irradiance"),
            maxirad=self.global_settings["maxirradiance"],
            visualpredatordensityrange=self.get_profile("pred1dens"),
            lghtsensitivity=self.genome.a2_irradiancesensitivity,
            predsensitivity=self.genome.a3_pred1sensitivity,
            predreactivity=self.genome.a4_pred1reactivity,
            pvp=self.zpos,
            pvi=self.zidx,
            strmass=self.structuralmass,
            mindepth=self.mindepth,
            maxdepth=self.maxdepth,
            depthrange=self.global_settings["depthrange"],
            continuousdepth=False,
            temporalres=self.temporalres,
        )
        return maxdistance, traveldistance

    def apply_dsc2_verticalmigration(self):
        """Run energy-storing-stage (dsc2) vertical migration for
        adult/diapause-eligible individuals via
        :func:`~pascal.biology.vertical_migration.verticalmigration_dsc2`
        (reserve-mass-aware, unlike dsc1), updating ``self.zpos``/
        ``self.zidx`` in place.

        Returns
        -------
        maxdistance, traveldistance : float
            Passed straight through to the matching ``growthanddevelopment_dsc2*``
            growth function.
        """
        self.zpos, self.zidx, maxdistance, traveldistance = vm.verticalmigration_dsc2(
            temprange=self.get_profile("temperature"),
            fconrange=self.get_profile("food1concentration"),
            iradrange=self.get_profile("irradiance"),
            maxirad=self.global_settings["maxirradiance"],
            visualpredatordensityrange=self.get_profile("pred1dens"),
            lghtsensitivity=self.genome.a2_irradiancesensitivity,
            predsensitivity=self.genome.a3_pred1sensitivity,
            predreactivity=self.genome.a4_pred1reactivity,
            pvp=self.zpos,
            pvi=self.zidx,
            strmass=self.structuralmass,
            resmass=self.reservemass,
            mindepth=self.mindepth,
            maxdepth=self.maxdepth,
            depthrange=self.global_settings["depthrange"],
            continuousdepth=False,
            temporalres=self.temporalres,
        )
        return maxdistance, traveldistance

    def apply_dsc2_growthdevel(self, maxdistance, traveldistance):
        """Compute feeding-stage growth via
        :func:`~pascal.biology.growth.growthanddevelopment_dsc2`, updating
        ``feedingrate``/``egestionrate``/``metabolicrate`` in place.

        Parameters
        ----------
        maxdistance, traveldistance : float
            From :meth:`apply_dsc2_verticalmigration`.

        Returns
        -------
        float
            Net structural/reserve growth rate for this timestep.
        """
        self.feedingrate, currentgrowthrate, self.egestionrate, self.metabolicrate = (
            gdm.growthanddevelopment_dsc2(
                temperature=self.get_zi("temperature"),
                foodcon=self.get_zi("food1concentration"),
                strmass=self.structuralmass,
                resmass=self.reservemass,
                maxzd=maxdistance,
                actzd=traveldistance,
                temporalres=self.temporalres,
            )
        )
        return currentgrowthrate

    def apply_dsc3_male_growthdevel(self, maxdistance, traveldistance):
        """Compute non-feeding adult-male degrowth via
        :func:`~pascal.biology.growth.growthanddevelopment_dsc3_male`,
        updating ``feedingrate``/``egestionrate``/``metabolicrate`` in
        place (``feedingrate`` is 0 for males - no feeding).

        Parameters
        ----------
        maxdistance, traveldistance : float
            From :meth:`apply_dsc2_verticalmigration`.

        Returns
        -------
        float
            Net (typically negative) growth rate for this timestep.
        """
        self.feedingrate, currentgrowthrate, self.egestionrate, self.metabolicrate = (
            gdm.growthanddevelopment_dsc3_male(
                temperature=self.get_zi("temperature"),
                strmass=self.structuralmass,
                resmass=self.reservemass,
                maxzd=maxdistance,
                actzd=traveldistance,
                temporalres=self.temporalres,
            )
        )
        return currentgrowthrate

    def apply_dsc0_mortality(self):
        """Apply non-feeding-stage (dsc0) predation/background mortality
        via :func:`~pascal.biology.survival.mortalityrisk_dsc0`, shrinking
        ``nvindividuals`` in place (truncated to ``int``). Called from
        :meth:`stage_lt_2`."""
        currentmortalityrisk = sv.mortalityrisk_dsc0(
            strmass=self.structuralmass,
            maxstrmass=self.maxstructuralmass,
            devstage=self.developmentalstage,
            vpreldensity=self.get_zi("pred1dens"),
            irradiance=self.get_zi("irradiance"),
            maxirradiance=self.global_settings["maxirradiance"],
            minirradiance=self.global_settings["minirradiance"],
            nvpreldensity=self.global_settings["nonvisualpredatorreldensity"],
            bgmrisk=self.global_settings["backgroundmortalityrisk"],
        )

        self.nvindividuals = int(self.nvindividuals * (1.0 - currentmortalityrisk))

    def apply_dsc2_mortality(self):
        """Apply energy-storing/adult-stage (dsc2) predation/background
        mortality via :func:`~pascal.biology.survival.mortalityrisk_dsc2`,
        shrinking ``nvindividuals`` in place (kept as ``float``, unlike
        :meth:`apply_dsc0_mortality`/:meth:`apply_dsc1_mortality` which
        truncate to ``int``). Called from
        :meth:`apply_mortality_and_deathcheck` for every individual with
        ``developmentalstage > 2`` - the vectorized equivalent used by
        :class:`~pascal.coupler.PascalSimulation` is
        :func:`~pascal.biology.survival.mortalityrisk_dsc2_vectorized`."""
        currentmortalityrisk = sv.mortalityrisk_dsc2(
            strmass=self.structuralmass,
            maxstrmass=self.maxstructuralmass,
            resmass=self.reservemass,
            irradiance=self.get_zi("irradiance"),
            maxirradiance=self.global_settings["maxirradiance"],
            minirradiance=self.global_settings["minirradiance"],
            vpreldensity=self.get_zi("pred1dens"),
            nvpreldensity=self.global_settings["nonvisualpredatorreldensity"],
            bgmrisk=self.global_settings["backgroundmortalityrisk"],
        )

        self.nvindividuals = self.nvindividuals * (1 - currentmortalityrisk)

    def get_currentcmm(self):
        """Return this individual's current critical molting mass (CMM):
        the structural mass threshold at which it advances to the next
        developmental stage, interpolated between
        ``global_settings["cmm_lower"/"cmm_upper"][developmentalstage]``
        by its genome's ``a1_bodysize`` value (same formula as
        ``self.adultsize``, evaluated at stage 12, in ``__init__``).

        Returns
        -------
        float
        """
        return (
            self.global_settings["cmm_lower"][self.developmentalstage]
            + (
                self.global_settings["cmm_upper"][self.developmentalstage]
                - self.global_settings["cmm_lower"][self.developmentalstage]
            )
            * self.genome.a1_bodysize
        )

    def update_mass(self, currentgrowthrate):
        """Apply a computed growth rate to structural/reserve mass:
        positive growth always goes to structural mass; negative growth
        (degrowth) is drawn from reserve mass first if there is enough,
        otherwise from structural mass. Updates ``maxstructuralmass`` if
        structural mass reaches a new high.

        Parameters
        ----------
        currentgrowthrate : float
            Net growth rate for this timestep (may be negative).
        """
        if currentgrowthrate > 0:
            # if the net growth rate is positive, all the surplus assimilation is channeled to somatic growth
            # no changes in the reserve mass
            self.structuralmass = self.structuralmass + currentgrowthrate

            # this updates the maximum structural mass if necessary:
            if self.structuralmass >= self.maxstructuralmass:
                self.maxstructuralmass = deepcopy(self.structuralmass)

        elif self.reservemass >= abs(currentgrowthrate):
            # if the reserve mass is sufficient to balance the degrowth (i.e., metabolic demands of diapause entry):
            # reserves are proportionally mobilized and no change occurs in the structural mass (despite the "+" operator, it is effectively a subtraction as growt rate is negative)
            self.reservemass = self.reservemass + currentgrowthrate

        else:
            # if reserves are not sufficient to balance the degrowth:
            # the structural mass is proportionally catabolized (despite the + operator, it is effectively a subtraction as growth rate is negative)
            # no change to the reserve mass
            self.structuralmass = self.structuralmass + currentgrowthrate

    def get_child_genome(self):
        """Produce one offspring genome from this (female) individual's
        genome and its stored ``malegenome`` (set at mating time - see
        :meth:`pascal.coupler.PascalSimulation.gene_hunt`), via per-gene
        BLX-alpha crossover (Takahashi et al. 2001,
        10.1109/CEC.2001.934452) and random-replacement mutation, gated by
        ``cxthreshold``/``muthreshold``. Deterministic (no crossover/
        mutation, plain inheritance of the female's genome) when
        ``global_settings["stochastic"]`` is ``False``.

        Returns
        -------
        dotdict
            One offspring genome, same keys as ``EXPECTED_GENES``.
        """
        spawning_f = self.genome
        spawning_m = self.malegenome
        spawning_n = dotdict()
        for attr_name in spawning_f.keys():
            if self.global_settings["stochastic"]:
                # this is the crossover probability per-gene (if this value is lower than crossover threshold, then crossover occurs)
                cxprob = np.random.rand(1).squeeze()
                # this is mutation probability per-gene (if this value is lower than the mutation threshold, then mutation occurs)
                muprob = np.random.rand(1).squeeze()
                # this is the blend value per-gene in BLX-alpha algorithm
                # nb: see Tkahashi et al, (2001) 10.1109/CEC.2001.934452
                cxval = np.random.rand(1).squeeze()
            else:
                cxprob = 0.5
                muprob = 0.5
                cxval = 0.5

            # crossover algorithm (BLX-alpha)
            if cxprob < self.cxthreshold:
                # if the crossover threshold is met, then male and female genomes are blended with BLX-alpha crossover
                spawning_n[attr_name] = (cxval * spawning_f[attr_name]) + (
                    (1.00 - cxval) * spawning_m[attr_name]
                )
            else:
                # otherwise, the female genome is inherited without blending
                spawning_n[attr_name] = spawning_f[attr_name]
            # end if
            # mutation algorithm (random replacement)
            # nb:the <else> condition is not written because, no mutation does not change the genome
            if muprob < self.muthreshold and self.global_settings["stochastic"]:
                spawning_n[attr_name] = np.random.rand(1)[0]

        return spawning_n

    def get_child_origin(self):
        """Return the ``(lon, lat)`` origin an offspring of this individual
        should inherit - one of its two parents' origins, chosen at
        random when ``global_settings["stochastic"]`` (else always the
        mother's).

        Returns
        -------
        tuple of (float, float)
        """
        # A mating-spawned super-individual inherits its origin from one
        # of its parents (not its own current position) - see
        # prompt_improvements.txt feature 3. Chosen randomly between the
        # mother (self) and father (self.male_origin, set at mating time -
        # see coupler.py::gene_hunt()) when stochastic, otherwise always
        # the mother's, matching get_child_genome()'s stochastic/
        # deterministic split above.
        if self.global_settings["stochastic"] and np.random.rand(1).squeeze() < 0.5:
            return self.male_origin
        return (self.origin_lon, self.origin_lat)

    def get_spatial_log_data(self, varlist):
        """Return this individual's contribution to one timestep's spatial
        output grid, for
        :meth:`pascal.coupler.PascalSimulation.log_spatial`.

        Parameters
        ----------
        varlist : list of str
            Variable names to include (``pascal.data_logger.OutputLogger``'s
            ``spatial_var_list``).

        Returns
        -------
        col : int
            Developmental-stage index to bin into (13 for adult males,
            since ``developmentalstage`` alone doesn't distinguish adult
            sex - females stay at 12).
        zidx : int
            Current depth-bin index.
        out_dict : dict
            ``{var: value}`` for each requested variable; mass/rate
            variables (``structuralmass``, ``reservemass``,
            ``feedingrate``, ``egestionrate``, ``metabolicrate``) are
            scaled by ``nvindividuals`` and converted from per-individual
            µgC to total gC (``* 1e-6``) - other variables are returned
            as-is.
        """
        if self.developmentalstage == 12 and self.sex == "M":
            col = 13
        else:
            col = self.developmentalstage

        carbon_adj = [
            "structuralmass",
            "reservemass",
            "feedingrate",
            "egestionrate",
            "metabolicrate",
        ]

        out_dict = {}
        for var in varlist:
            if var in carbon_adj:
                out_dict[var] = getattr(self, var) * self.nvindividuals * 1e-6
            else:
                out_dict[var] = getattr(self, var)

        return col, self.zidx, out_dict  # Check this is the correct Z

    def get_log_data(self):
        """Return this individual's contribution to a diapause-entry/exit
        or direct-development log event, for
        ``pascal.data_logger.OutputLogger.add_ddev``/``add_den``/``add_dex``
        (called from :meth:`diapause0`/:meth:`diapause1`).

        Returns
        -------
        dict
            ``{"individuals", "structuralmass", "reservemass"}``.

        Notes
        -----
        **Verified bug**: ``self.tnvindividuals`` does not exist anywhere
        on this class (the attribute is ``nvindividuals``, no leading
        ``t``) - this raises ``AttributeError`` the first time this method
        is actually reached (i.e. the first diapause entry/exit or
        direct-development event in any run using a datalogger). Also note
        ``reservemass``'s value here is computed from ``structuralmass``,
        not ``self.reservemass`` - unclear from the code alone whether
        that is intentional or itself a copy-paste bug; not changed here
        since fixing behavior is out of scope for a documentation pass.
        Separately, ``pascal.data_logger.OutputLogger.add_ddev``/
        ``add_den``/``add_dex`` (the callers) also reference
        ``self.current_timestep``, which is never set anywhere on
        ``OutputLogger`` either - a second, independent bug in the same
        code path.
        """
        return {
            "individuals": self.nvindividuals,
            "structuralmass": (self.structuralmass * self.tnvindividuals) / 1e6,
            "reservemass": (self.structuralmass * self.nvindividuals) / 1e6,
        }
