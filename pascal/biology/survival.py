def mortalityrisk_dsc0(strmass: float, maxstrmass: float, devstage: int, vpreldensity: float, irradiance: float, maxirradiance: float, minirradiance: float, nvpreldensity: float, bgmrisk: float) -> float:

    """
    functionality:
    this returns the estimated total mortality risk (as a probability of death) of a super individual in the developmental stage category 0 (dsc0: eggs, nauplius I and II)
    nb:for starvation risk, eggs are skipped in this estimation using an <if> condition
    nb:for non-energy-storing stages (E-CIII) the structural mass applies for allometric scaling; for the others, total body mass applies
    
    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    strmass                  : the somatic body mass (s, t)
    maxstrmass               : the maximum somatic body mass (s, t) reached during the lifespan by super individual s at time t
    devstage                 : the developmental stage (s, t)
    vpreldensity             : the relative visual predator density (x, y, t, z)
    nvpreldensity            : the relative non-visual predator density (x, y, t, z)
    irradiance               : the irradiance (PAR) of the current time location and depth (x, y, z, t)
    maxirradiance            : the expected theoreitical global maximum irradiance (PAR)
    minirradiance            : the expected theoreitical global minimum irradiance (PAR) 
    bgmrisk                  : background mortality risk expressed as probability of death
    
    return:
    totalmortlityrisk        : estimated total mortality risk (= probability of death by all possible sources of mortality)

    """

    import math

    #starvation risk
    #this is the catabolized structural mass expressed as a proportion of the maximum structural mass
    strcat = (maxstrmass - strmass) / maxstrmass
    #the starvation risk is estimated from a piecewice function, assuming there is starvation tolerance upto 10% structural mass catabolization
    #see Threlkeld (1976):  https://doi.org/10.1111/j.1365-2427.1976.tb01640.x
    #nb:eggs (s = 0) are skipped from starvation risk estimation (i.e., it returns 0.00)
    if strcat <= 0.10 or devstage == 0:

        starvationrisk = 0.00
    
    else:

        starvationrisk = 1.00 / (1.00 + math.exp((0.25 - strcat) / 0.05))
    
    #end if
    
    #light dependency of visual predation risk
    vplightdependence = 0.1 + (irradiance - minirradiance) * ((0.9 - 0.1) / (maxirradiance - minirradiance))

    #size-dependency of light-dependent visual predation risk
    visualpredationrisk = vpreldensity * vplightdependence * (1.00 / (1.00 + math.exp((350.00 - strmass) / 75.00)))

    #non-visual predation risk
    #this is kept simple for the time being; can be advanced depeding on the need
    nonvisualpredationrisk = nvpreldensity

    #total mortality risk
    totalmortalityrisk = starvationrisk + visualpredationrisk + nonvisualpredationrisk + bgmrisk
    
    if totalmortalityrisk > 1.00:

        totalmortalityrisk = 1.00
    
    #end if
    
    return totalmortalityrisk
    
#end def

def mortalityrisk_dsc1(strmass: float, maxstrmass: float, vpreldensity: float, irradiance: float, maxirradiance: float, minirradiance: float, nvpreldensity: float, bgmrisk: float) -> float:

    """
    functionality:
    this returns the estimated total mortality risk (as a probability of death) of a super individual in the developmental stage category 0 (dsc-I: NIII-CII)
    nb:for starvation risk, eggs are skipped in this estimation using an <if> condition
    nb:for non-energy-storing stages (E-CIII) the structural mass applies for allometric scaling; for the others, total body mass applies
    
    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    strmass                  : the somatic body mass (s, t)
    maxstrmass               : the maximum somatic body mass (s, t) reached during the lifespan by super individual s at time t
    vpreldensity             : the relative visual predator density (x, y, t, z)
    nvpreldensity            : the relative non-visual predator density (x, y, t, z)
    irradiance               : the irradiance (PAR) of the current time location and depth (x, y, z, t)
    maxirradiance            : the expected theoreitical global maximum irradiance (PAR)
    minirradiance            : the expected theoreitical global minimum irradiance (PAR) 
    bgmrisk                  : background mortality risk expressed as probability of death
    
    return:
    totalmortlityrisk        : estimated total mortality risk (= probability of death by all possible sources of mortality)

    """

    import math

    #starvation risk
    #this is the catabolized structural mass expressed as a proportion of the maximum structural mass
    if strmass < maxstrmass:
        
        #the starvation risk is estimated from a piecewice function, assuming there is starvation tolerance upto 10% structural mass catabolization
        #see Threlkeld (1976):  https://doi.org/10.1111/j.1365-2427.1976.tb01640.x
        strcat = (maxstrmass - strmass) / maxstrmass
        
        if strcat <= 0.10:

            starvationrisk = 0.00
    
        else:

            starvationrisk = 1.00 / (1.00 + math.exp((0.25 - strcat) / 0.05))
    
        #end if
        
    else:
        
        starvationrisk = 0.00
        
    #end if 
    
    #light dependency of visual predation risk
    vplightdependence = 0.1 + (irradiance - minirradiance) * ((0.9 - 0.1) / (maxirradiance - minirradiance))

    #size-dependency of light-dependent visual predation risk
    visualpredationrisk = vpreldensity * vplightdependence * (1.00 / (1.00 + math.exp((350.00 - strmass) / 75.00)))

    #non-visual predation risk
    #this is kept simple for the time being; can be advanced depeding on the need
    nonvisualpredationrisk = nvpreldensity

    #total mortality risk
    totalmortalityrisk = starvationrisk + visualpredationrisk + nonvisualpredationrisk + bgmrisk
    
    if totalmortalityrisk > 1.00:

        totalmortalityrisk = 1.00
    
    #end if
    
    return totalmortalityrisk
    
#end def

def mortalityrisk_dsc2(strmass: float, maxstrmass: float, resmass: float, vpreldensity: float, irradiance: float, maxirradiance: float, minirradiance: float, nvpreldensity: float, bgmrisk: float) -> float:

    """
    functionality:
    this returns the estimated total mortality risk (as a probability of death) of a super individual in the developmental stage category 0 (dsc-I: NIII-CII)
    nb:for starvation risk, eggs are skipped in this estimation using an <if> condition
    nb:for non-energy-storing stages (E-CIII) the structural mass applies for allometric scaling; for the others, total body mass applies
    
    references:
    x = longitude, y = latitude, t = time, z = depth, s = super individual, d = developmental stage 

    args:
    strmass                  : the somatic body mass (s, t)
    maxstrmass               : the maximum somatic body mass (s, t) reached during the lifespan by super individual s at time t
    strmass                  : the energy reserve mass (s, t)
    vpreldensity             : the relative visual predator density (x, y, t, z)
    nvpreldensity            : the relative non-visual predator density (x, y, t, z)
    irradiance               : the irradiance (PAR) of the current time location and depth (x, y, z, t)
    maxirradiance            : the expected theoreitical global maximum irradiance (PAR)
    minirradiance            : the expected theoreitical global minimum irradiance (PAR) 
    bgmrisk                  : background mortality risk expressed as probability of death
    
    return:
    totalmortlityrisk        : estimated total mortality risk (= probability of death by all possible sources of mortality)

    """

    import math

    #starvation risk
    #this is the catabolized structural mass expressed as a proportion of the maximum structural mass
    if strmass < maxstrmass:
        
        #the starvation risk is estimated from a piecewice function, assuming there is starvation tolerance upto 10% structural mass catabolization
        #see Threlkeld (1976):  https://doi.org/10.1111/j.1365-2427.1976.tb01640.x
        strcat = (maxstrmass - strmass) / maxstrmass
        
        if strcat <= 0.10:

            starvationrisk = 0.00
    
        else:

            starvationrisk = 1.00 / (1.00 + math.exp((0.25 - strcat) / 0.05))
    
        #end if
        
    else:
        
        starvationrisk = 0.00
        
    #end if 
    
    #total mass
    totalmass = strmass + resmass

    #light dependency of visual predation risk
    vplightdependence = 0.1 + (irradiance - minirradiance) * ((0.9 - 0.1) / (maxirradiance - minirradiance))

    #size-dependency of light-dependent visual predation risk
    visualpredationrisk = vpreldensity * vplightdependence * (1.00 / (1.00 + math.exp((350.00 - totalmass) / 75.00)))

    #non-visual predation risk
    #this is kept simple for the time being; can be advanced depeding on the need
    nonvisualpredationrisk = nvpreldensity

    #total mortality risk
    totalmortalityrisk = starvationrisk + visualpredationrisk + nonvisualpredationrisk + bgmrisk

    if totalmortalityrisk > 1.00:

        totalmortalityrisk = 1.00

    #end if

    return totalmortalityrisk

#end def


def mortalityrisk_dsc2_vectorized(strmass, maxstrmass, resmass, vpreldensity, irradiance, maxirradiance, minirradiance, nvpreldensity, bgmrisk):
    """Vectorized sibling of mortalityrisk_dsc2, for batching the mortality
    calculation across many super-individuals at once instead of calling
    mortalityrisk_dsc2() once per individual.

    strmass/maxstrmass/resmass/vpreldensity/irradiance are expected to be
    numpy arrays (one element per individual); maxirradiance/minirradiance/
    nvpreldensity/bgmrisk are scalars (global settings, shared by every
    individual) and broadcast automatically.

    Deliberately kept as a separate function alongside the scalar
    mortalityrisk_dsc2() (not a replacement) so the scalar version stays
    available/unchanged for anyone else using it. See
    coupler.py::apply_mortality_and_deathcheck_batch() for the caller and
    tests/test_survival_vectorized.py for the property-based check that
    this matches mortalityrisk_dsc2() exactly on many random inputs.
    """
    import numpy as np

    strcat = (maxstrmass - strmass) / maxstrmass
    starvationrisk = np.where(
        strmass < maxstrmass,
        np.where(
            strcat <= 0.10,
            0.00,
            1.00 / (1.00 + np.exp((0.25 - strcat) / 0.05)),
        ),
        0.00,
    )

    totalmass = strmass + resmass

    vplightdependence = 0.1 + (irradiance - minirradiance) * ((0.9 - 0.1) / (maxirradiance - minirradiance))
    visualpredationrisk = vpreldensity * vplightdependence * (1.00 / (1.00 + np.exp((350.00 - totalmass) / 75.00)))
    nonvisualpredationrisk = nvpreldensity

    totalmortalityrisk = starvationrisk + visualpredationrisk + nonvisualpredationrisk + bgmrisk

    return np.minimum(totalmortalityrisk, 1.00)

#end def