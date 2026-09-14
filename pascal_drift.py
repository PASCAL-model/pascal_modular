#import sys
#sys.path.append(r'/home/michael/Projects/Migratory_crossroads/Models/pascal_modular/opendrift_pascal')
from opendrift.models.oceandrift import OceanDrift, Lagrangian3DArray

import numpy as np

class PascalEnv(Lagrangian3DArray):
    """Extending Lagrangian3DArray with specific properties for biofoulable plastic
    """

    variables = Lagrangian3DArray.add_variables([
        ('unfouled_diameter', {'dtype': np.float32,
                      'units': 'm',
                      'default': 0.0014}),  #
        ('unfouled_density', {'dtype': np.float32,
                                       'units':'kg/m^3',
                                       'default': 1028}),  #
        ('biofilm_no_attached_algae', {'dtype': np.float32,
                                       'units': '',
                                       'default': 0}),
        ('total_density', {'dtype': np.float32,
                     'units': 'kg/m^3',
                     'default': 1028.}),
        ('total_diameter', {'dtype': np.float32,
                      'units': 'm',
                      'default': 0.0014}),
        # The (lon, lat) location the owning PASCAL super-individual (or,
        # for a mating-spawned one, the parent it inherited from) started
        # from - see coupler.py::seed()/set_tracker_origin(). NaN default
        # marks elements PASCAL hasn't assigned yet (e.g. the bulk
        # seed_elements() pool in PascalAdvection.prep_environment(),
        # before individual seed() calls overwrite each slot in use) as
        # having no origin to rescue back to - see
        # PascalDrift.remove_deactivated_elements() below.
        ('origin_lon', {'dtype': np.float32,
                      'units': 'degrees_east',
                      'default': np.nan}),
        ('origin_lat', {'dtype': np.float32,
                      'units': 'degrees_north',
                      'default': np.nan})])

class PascalDrift(OceanDrift):
    ElementType = PascalEnv

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
        #'sea_surface_wave_significant_height': {'fallback': 0},
        #'sea_ice_area_fraction': {'fallback': 0},
        #'x_wind': {'fallback': 0},
        #'y_wind': {'fallback': 0},
        'land_binary_mask': {'fallback': None},
        'sea_floor_depth_below_sea_level': {'fallback': 1000},
        'ocean_vertical_diffusivity': {'fallback': 0.02, 'profiles': True},
        'mld': {'fallback': 50},
        'temperature': {'fallback': 10, 'profiles': True},
        #'sea_water_salinity': {'fallback': 34, 'profiles': True},
        #'surface_downward_x_stress': {'fallback': 0},
        #'surface_downward_y_stress': {'fallback': 0},
        #'turbulent_kinetic_energy': {'fallback': 0},
        #'turbulent_generic_length_scale': {'fallback': 0},
        'upward_sea_water_velocity': {'fallback': 0},
        'food1concentration':{'fallback':0, 'profiles': True},
        'irradiance':{'fallback':0, 'profiles': True},
        'pred1dens':{'fallback':0, 'profiles': True},
        'pred1lightdep':{'fallback':0, 'profiles': True},
      }

    # Default colors for plotting
    status_colors = {'initial': 'green', 'active': 'blue',
                     'hatched': 'red', 'eaten': 'yellow', 'died': 'magenta'}


    def __init__(self, *args, **kwargs):

        # Calling general constructor of parent class
        super(PascalDrift, self).__init__(*args, **kwargs)

    def remove_deactivated_elements(self):
        """Rescue elements about to be permanently removed for leaving the
        domain laterally, by teleporting them back to their origin and
        reactivating, before OpenDrift's own remove_deactivated_elements()
        physically deletes them.

        Two deactivation reasons count as "left the domain laterally":
        'missing_data' (report_missing_variables() - wandered outside the
        reader's spatial/temporal coverage, the live risk for a bounded
        reader like the CMEMS Barents file) and 'outside'
        (increase_age_and_retire()'s drift:deactivate_*_of validity-domain
        check). Both run, and schedule elements for removal via
        deactivate_elements(), earlier in the same run_1step() call, right
        before this method - same "physical removal shrinks/reindexes
        self.elements.* mid-run" hazard PascalAdvection.prep_environment()
        already works around for coastline stranding via
        general:coastline_action='previous', which has no equivalent
        built-in knob for these two reasons.

        Elements with no recorded origin (origin_lon/lat still NaN - e.g.
        the bulk seed_elements() pool in prep_environment() before PASCAL's
        own seed() has claimed a slot) are left for the base class to
        remove as normal; there's nowhere to bounce them back to.
        """
        # status is a scalar 0 until the first deactivation of any kind
        # ever happens (see basemodel's deactivate_elements(), which is
        # what expands it into a per-element array) - nothing to rescue in
        # that case, and treating it as a length-1 array would mismatch
        # origin_lon/lat's real per-element length below.
        if len(np.atleast_1d(self.elements.status)) == 1:
            super().remove_deactivated_elements()
            return

        rescue_reasons = ('missing_data', 'outside')
        rescue_mask = np.zeros(len(self.elements.status), dtype=bool)
        for reason in rescue_reasons:
            if reason in self.status_categories:
                rescue_mask |= (
                    self.elements.status == self.status_categories.index(reason)
                )
        rescue_mask &= (
            np.isfinite(self.elements.origin_lon)
            & np.isfinite(self.elements.origin_lat)
        )

        if np.any(rescue_mask):
            self.elements.lon[rescue_mask] = self.elements.origin_lon[rescue_mask]
            self.elements.lat[rescue_mask] = self.elements.origin_lat[rescue_mask]
            self.elements.status[rescue_mask] = self.status_categories.index('active')
            self.elements.moving[rescue_mask] = 1

        super().remove_deactivated_elements()

    def update(self):
        """Update positions and properties of plastic particles."""
        # Turbulent Mixing
        self.update_terminal_velocity()
        self.vertical_mixing()

        # Horizontal advection
        self.advect_ocean_current()
        
        # Vertical advection
        if self.get_config('drift:vertical_advection') is True:
            self.vertical_advection()

        # Reproduction and reseeding

