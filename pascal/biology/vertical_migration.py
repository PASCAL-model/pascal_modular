def verticalmigration_dsc0(smld: int, maxdepth: int, mindepth: int, depthrange: int, continuousdepth: bool, stochastic: bool=True) -> int:

    """
    functionality:
    this returns the absolute vertical position of a super individual in the developmental stage category 0 (dsc0: eggs, nauplius I and II)
    this is not diel or seasonal vertical migration - at these stages, super individuals are too small and do not have resources to perform notable active vertical movements
    
    args:
    smld                        : current surface mixed layer depth
    maxdepth                    : maximum depth of the model environment
    mindepth                    : minimum depth of the model environment
    depthrange                  : the range of depth bins used in the model (= the vertical resolution)
    continuousdepth             : a boolean specifying whether the depthgrade is continuous or discreet
    
    return:
    zpos                        : estimated current vertical position of the super individual
    zidx                        : index position corresponding to the zpos

    """
    
    import numpy as np
    
    #this estimates the absolute and position of the super individual as a function of mixed layer depth
    #nb:thsis routine needs an update possibly based on the w-component of seawater velocity
    if smld > maxdepth:
        
        zpos = maxdepth
        zidx = zpos - 1 if continuousdepth else np.argmin(np.abs(depthrange - zpos)).item()

    elif smld == 0:

        zpos = mindepth
        zidx = zpos - 1 if continuousdepth else np.argmin(np.abs(depthrange - zpos)).item()

    else:

        if stochastic:
            zsel = np.random.randint(low = mindepth, high = smld, size = 1).squeeze().item()
        else:
            zsel = np.floor(smld - mindepth)

        zidx = zsel - 1 if continuousdepth else np.argmin(np.abs(depthrange - zsel)).item()
        zpos = depthrange[zidx]

    #end if

    return zpos, zidx

#end def

def verticalmigration_dsc1(temprange: float, fconrange: float, iradrange: float, maxirad: float, visualpredatordensityrange: float, lghtsensitivity: float, predsensitivity: float, predreactivity: float, pvp: int, pvi: int, strmass: float, mindepth: int, maxdepth: int, depthrange: int, continuousdepth: bool, temporalres: int) -> int:

    """
    functionality:
    this returns the vertical position of a super individual in the developmental stage category I (dsc1: NIII-CIII)
    
    args:
    temprange                   : range of temperatures across all depth layers at a given longitude, latitude and time (just depth in 1D configuration)
    fconrange                   : range of food concentration category 1 (= phytoplankton) at a given longitude, latitude and time
    iradrange                   : range of shortwave solar irradiance at a given longitude, latitude and time
    visualpredatordensityrange  : range of category 1 predator density at a given longitude, latitude and time
    lghtsensitivity             : evolvable attribute for irradiance sensitivity of a given super individual
    predsensitivity             : evolvable attribute for predator sensitivity of a given super individual
    predreactivity              : evolvable attribute for predator reactivity of a given super individual
    pvp                         : absolute vertical position at the previous timepoint (the function calculates the current vertical position)
    pvi                         : relative vertical position (index) at the previous timepoint
    strmass                     : structural mass of a given super individual
    mindepth                    : minimum depth
    maxdepth                    : maximum depth
    depthrange                  : vertically segragated depth intervals (vertical resolution of the model)
    continuousdepth             : a boolean inidicating whether the depthgrade is continuous or discreet
    temporalres                 : the temporal resolution of the model
    
    return:
    zpos                        : estimated current vertical position of the super individual
    zidx                        : estimated depth index corresponding to the above vertical position
    maxdistance                 : estimated maximum vertical distance searchable by the super invididual
    traveldistance              : estimated actual vertical distance searched by the super individual

    """

    import numpy as np
    import math
    
    #these are the standard depths of the model environment (similar to variable "depthgrade" in the main program)
    # depthrange = np.array([1, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 41, 48, 56, 66, 78, 93, 110, 131, 156, 187, 223, 267, 319, 381, 454, 542, 644, 764, 903, 1063, 1246])
    #this estimates the theoreitical maximum distance that be vertically searched by a given super individual as a power function of somatic body mass
    #nb:the estimate is adjusted to the temporal resolution of the model
    #the function is based on Bandara et al. (2021), which follows a literature review of copepod swimming speeds in natural habitats, see: https://doi.org/10.1016/j.ecolmodel.2021.109739
    maxdistance = int(temporalres * 5.2287 * strmass ** 0.4862)
    #this estimates the minimum and maximum points of the 1D water column that the super individual can search
    searchceiling = int(pvp - maxdistance)
    searchfloor = int(pvp + maxdistance)

    #both above estimates must be trimmed so that they does not exceed minimum and maximum depths (else, in theory, animals get burried in the sediment like worms or shoot out into the air like goddamn flying fish!!!)
    if searchceiling < mindepth:

        searchceiling = mindepth
    
    #end if
    
    if searchfloor > maxdepth:

        searchfloor = maxdepth
    
    #end if
    
    #these are index positions related to the minimum and maximum points of the water column searchable by a given super individual
    searchceiling_idx = searchceiling - 1 if continuousdepth else np.argmin(np.abs(depthrange - searchceiling))
    searchfloor_idx = searchfloor - 1 if continuousdepth else np.argmin(np.abs(depthrange - searchfloor))

    #if the vertical search distance is not sufficient to move between model-resolvable adjacent depth layers, then the model assums no movement of the super individual
    #obs:this may get super-individuals stuck in deeper layers when they are small and lacks vertical swimming potential
    if searchceiling_idx == searchfloor_idx:
        
        zpos = pvp
        zidx = pvi
    
    #if not (i.e., search range spans across 2 or more model-resolved depth gradings), the vertical behavioural submodel applies
    else:
        
        #this is the dynamic evolvable attribute scaling function defined by Edirisinghe (2022), where vertical behavioural reactions of super individuals are modified by percieved visual predation risk
        #where behavioural reaction is by default elicited via ambient irradiance & it is altered by predator presence and the copepod's detection & reaction capabilities therein
        #the asymptotic exponantial scalar expresses the size-dependence of irradiance sensitivity by copepods (where, smaller ones tend to stay closer to the surface and vice versa)
        if visualpredatordensityrange[pvi] >= predsensitivity:
        
            thresholdirradiance = lghtsensitivity * predreactivity * maxirad * (1.00 - (1.00 / (1.00 + math.exp((350.00 - strmass) / 75.00))))
        
        else:

            thresholdirradiance = lghtsensitivity * maxirad * (1.00 - (1.00 / (1.00 + math.exp((350.00 - strmass) / 75.00))))
        
        #end if
        
        #this is the vertical search range of the super individual given as a numpy array
        searchrange = np.arange(searchceiling_idx, searchfloor_idx, 1).astype(np.int64)
        #this is the range of irradiance values corresponding to the search range of the super individual    
        iradrange_sr = iradrange[searchrange]
        #these are environmental variables corresponding to the search range of the super individual
        #nb:the celsius is converted to kelvin for growth potential estimation; see bellow
        temprange_sr = temprange[searchrange] + 273.15
        f1conrange_sr = fconrange[searchrange]

        #the preferred depth by the super individual is the depth that meets or closest to meeting the irraiance threshold with maximum growth potential
        #the growth potential is estimated as a product of temperature and foodconcentration as described in Edirisinghe (2022)
        growthpotential_sr = (temprange_sr * f1conrange_sr) ** 2.72

        #if at least one depth bin offers a refuge from light and light-dependent predation
        if np.any(iradrange_sr <= thresholdirradiance):
            
            potentialzpos = np.array(np.where(iradrange_sr <= thresholdirradiance)).squeeze()
            
            if potentialzpos.size == 1:
            
                zidx = searchrange[potentialzpos]
                zpos = depthrange[zidx]
            
            else:
                
                zidx = searchrange[potentialzpos[np.argmax(growthpotential_sr[potentialzpos])]]
                zpos = depthrange[zidx]
            
            #end if
            
        #if none of the searchable depths offer a refuge from light and light-dependent predation
        else:
            
            #super individuals select the deepest depth (as downwelling light is vertically extinct in the water column)
            zidx = searchrange[-1]
            zpos = depthrange[zidx]

        #end if
            
    #end if

    traveldistance = abs(pvp - zpos)

    return zpos, zidx, maxdistance, traveldistance

#end def

def verticalmigration_dsc2(temprange: float, fconrange: float, iradrange: float, maxirad: float, visualpredatordensityrange: float, lghtsensitivity: float, predsensitivity: float, predreactivity: float, pvp: int, pvi: int, strmass: float, resmass: float, mindepth: int, maxdepth: int, depthrange: int, continuousdepth: bool, temporalres: int) -> int:

    """
    functionality:
    this returns the vertical position of a super individual in the developmental stage category II and beyond (dsc2: CIV-CV)
    the main difference of this function and the that of dsc-I is that the totalmass is factored in here due to energy storage
    
    args:
    temprange                   : range of temperatures across all depth layers at a given longitude, latitude and time (just depth in 1D configuration)
    fconrange                   : range of food concentration category 1 (= phytoplankton) at a given longitude, latitude and time
    iradrange                   : range of shortwave solar irradiance at a given longitude, latitude and time
    visualpredatordensityrange  : range of category 1 predator density at a given longitude, latitude and time
    lghtsensitivity             : evolvable attribute for irradiance sensitivity of a given super individual
    predsensitivity             : evolvable attribute for predator sensitivity of a given super individual
    predreactivity              : evolvable attribute for predator reactivity of a given super individual
    pvp                         : absolute vertical position at the previous timepoint (the function calculates the current vertical position)
    pvi                         : relative vertical position (index) at the previous timepoint
    strmass                     : structural mass of a given super individual
    resmass                     : energy reserve mass of a given super individual
    mindepth                    : minimum depth
    maxdepth                    : maximum depth
    depthrange                  : vertically segragated depth intervals (vertical resolution of the model)
    continuousdepth             : a boolean inidicating whether the depthgrade is continuous or discreet
    temporalres                 : the temporal resolution of the model
    
    return:
    zpos                        : estimated current vertical position of the super individual
    zidx                        : estimated depth index corresponding to the above vertical position
    maxdistance                 : estimated maximum vertical distance searchable by the super invididual
    traveldistance              : estimated actual vertical distance searched by the super individual

    """

    import numpy as np
    import math
    
    #this calculates the total mass
    totalmass = strmass + resmass

    #these are the standard depths of the model environment (similar to variable "depthgrade" in the main program)
    # depthrange = np.array([1, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 41, 48, 56, 66, 78, 93, 110, 131, 156, 187, 223, 267, 319, 381, 454, 542, 644, 764, 903, 1063, 1246])
    #this estimates the theoreitical maximum distance that be vertically searched by a given super individual as a power function of somatic body mass
    #nb:the estimate is adjusted to the temporal resolution of the model
    #the function is based on Bandara et al. (2021), which follows a literature review of copepod swimming speeds in natural habitats, see: https://doi.org/10.1016/j.ecolmodel.2021.109739
    maxdistance = int(temporalres * 5.2287 * strmass ** 0.4862)
    #this estimates the minimum and maximum points of the 1D water column that the super individual can search
    searchceiling = int(pvp - maxdistance)
    searchfloor = int(pvp + maxdistance)

    #both above estimates must be trimmed so that they does not exceed minimum and maximum depths (else, in theory, animals get burried in the sediment like worms or shoot out into the air like goddamn flying fish!!!)
    if searchceiling < mindepth:

        searchceiling = mindepth
    
    #end if
    
    if searchfloor > maxdepth:

        searchfloor = maxdepth
    
    #end if
    
    #these are index positions related to the minimum and maximum points of the water column searchable by a given super individual
    searchceiling_idx = searchceiling - 1 if continuousdepth else np.argmin(np.abs(depthrange - searchceiling))
    searchfloor_idx = searchfloor - 1 if continuousdepth else np.argmin(np.abs(depthrange - searchfloor))

    #if the vertical search distance is not sufficient to move between model-resolvable adjacent depth layers, then the model assums no movement of the super individual
    #obs:this may get super-individuals stuck in deeper layers when they are small and lacks vertical swimming potential
    if searchceiling_idx == searchfloor_idx:
        
        zpos = pvp
        zidx = pvi
    
    #if not (i.e., search range spans across 2 or more model-resolved depth gradings), the vertical behavioural submodel applies
    else:
        
        #this is the dynamic evolvable attribute scaling function defined by Edirisinghe (2022), where vertical behavioural reactions of super individuals are modified by percieved visual predation risk
        #where behavioural reaction is by default elicited via ambient irradiance & it is altered by predator presence and the copepod's detection & reaction capabilities therein
        #the asymptotic exponantial scalar expresses the size-dependence of irradiance sensitivity by copepods (where, smaller ones tend to stay closer to the surface and vice versa)
        if visualpredatordensityrange[pvi] >= predsensitivity:
        
            thresholdirradiance = lghtsensitivity * predreactivity * maxirad * (1.00 - (1.00 / (1.00 + math.exp((350.00 - totalmass) / 75.00))))
        
        else:

            thresholdirradiance = lghtsensitivity * maxirad * (1.00 - (1.00 / (1.00 + math.exp((350.00 - totalmass) / 75.00))))
        
        #end if
        
        #this is the vertical search range of the super individual given as a numpy array
        searchrange = np.arange(searchceiling_idx, searchfloor_idx, 1).astype(np.int64)
        #this is the range of irradiance values corresponding to the search range of the super individual    
        iradrange_sr = iradrange[searchrange]
        #these are environmental variables corresponding to the search range of the super individual
        #nb:the celsius is converted to kelvin for growth potential estimation; see bellow
        temprange_sr = temprange[searchrange] + 273.15
        f1conrange_sr = fconrange[searchrange]

        #the preferred depth by the super individual is the depth that meets or closest to meeting the irraiance threshold with maximum growth potential
        #the growth potential is estimated as a product of temperature and foodconcentration as described in Edirisinghe (2022)
        growthpotential_sr = (temprange_sr * f1conrange_sr) ** 2.72

        #if at least one depth bin offers a refuge from light and light-dependent predation
        if np.any(iradrange_sr <= thresholdirradiance):
            
            potentialzpos = np.array(np.where(iradrange_sr <= thresholdirradiance)).squeeze()
            
            if potentialzpos.size == 1:
            
                zidx = searchrange[potentialzpos]
                zpos = depthrange[zidx]
            
            else:
                
                zidx = searchrange[potentialzpos[np.argmax(growthpotential_sr[potentialzpos])]]
                zpos = depthrange[zidx]
            
            #end if
            
        #if none of the searchable depths offer a refuge from light and light-dependent predation
        else:
            
            #super individuals select the deepest depth (as downwelling light is vertically extinct in the water column)
            zidx = searchrange[-1]
            zpos = depthrange[zidx]

        #end if
            
    #end if

    traveldistance = abs(pvp - zpos)
    
    return zpos, zidx, maxdistance, traveldistance

#end def

def diapausedepthselection(maxdepth: int, ddt0: int, ddt1: int, ddt2: int, depthrange: int, continuousdepth: bool, stochastic: bool=True) -> int:

    """
    functionality:
    this returns the absolute vertical position of a super individual in the developmental stage category 0 (dsc0: eggs, nauplius I and II)
    this is not diel or seasonal vertical migration - at these stages, super individuals are too small and do not have resources to perform notable active vertical movements
    
    args:
    maxdepth                    : maximum depth of the model environment
    ddt0                        : diapause depth threshold 0 (typically, 50 m): depth above which no diapause is possible
    ddt1                        : diapause depth threshold 1 (typically, 500 m): depth above which and below "ddt0", "active" diapause is possible
    ddt2                        : diapause depth threshold 0 (typically, 1200 m): depth below which C. finmarchicus do not typically venture into diapause
    depthrange                  : the range of depth bins used in the model (= the vertical resolution)
    continuousdepth             : a boolean specifying whether the depthgrade is continuous or discreet
    
    return:
    diapausedepth               : estimated current vertical position of the super individual
    diapausedepthidx            : index position corresponding to the zpos

    """
    
    import numpy as np

    if maxdepth >= ddt1:

        #depth ceiling and floor estimation

        #if the bottom depth is deepeer than ca. 500 m      
        if maxdepth >= ddt2:
            
            #if the bottom depth is deeper than 1200 m:
            #sets a ca. 600 m ceiling
            diapausedepthceiling = ddt1 + 100
            
            #sets a ca. 1000 m floor
            diapausedepthfloor = ddt2 - 200
        
            #randomizing diapause depth ceiling and floor values
            if stochastic:
                diapausedepthvar = np.random.randint(low = 0, high = 100, size = 1).squeeze().item()
                diapausedepthopr = np.random.randint(low = 0, high = 2, size = 1).squeeze().item()
            else:
                diapausedepthvar = 50
                diapausedepthopr = 1
            
            diapausedepthceiling = diapausedepthceiling - diapausedepthvar if diapausedepthopr == 0 else  diapausedepthceiling + diapausedepthvar
            
            if stochastic:
                diapausedepthvar = np.random.randint(low = 0, high = 100, size = 1).squeeze().item()
                diapausedepthopr = np.random.randint(low = 0, high = 2, size = 1).squeeze().item()

            diapausedepthfloor = diapausedepthfloor - diapausedepthvar if diapausedepthopr == 0 else diapausedepthfloor + diapausedepthvar

        else:

            #if the bottom depth is shallower than 1200 m:
            #sets a ca. 100 m off the bottom floor
            diapausedepthfloor = maxdepth - 100

            #sets a ceiling ca. 100 m above the floor depth
            diapausedepthceiling = diapausedepthfloor - 100

            #randomizing diapause depth ceiling and floor values
            if stochastic:
                diapausedepthvar = np.random.randint(low = 0, high = 50, size = 1).squeeze().item()
            else:
                diapausedepthvar = 25

            diapausedepthceiling -= diapausedepthvar
            
            if stochastic:
                diapausedepthvar = np.random.randint(low = 0, high = 50, size = 1).squeeze().item()

            diapausedepthfloor += diapausedepthvar
        
        #end if

        #selecting diapause depth from a Gaussian probability distribution, within the range, mean and sd defined by the depth floor and depth ceiling
        diapausedepthrange = np.arange(diapausedepthceiling, diapausedepthfloor, 1)
        diapausedepthmean = np.nanmean(diapausedepthrange)
        diapausedepthsd = np.nanstd(diapausedepthrange)

        diapausedepthselprob = np.exp(-0.5 * ((diapausedepthrange - diapausedepthmean) / diapausedepthsd)**2)
        diapausedepthselprob /= np.sum(diapausedepthselprob)
        
        if stochastic:
            selecteddiapausedepth = np.random.choice(a = diapausedepthrange, replace = False, size = 1, p = diapausedepthselprob).squeeze().item()
        else:
            selecteddiapausedepth = diapause_depth_range[0]

        diapausedepthidx = np.argmin(np.abs(depthrange - selecteddiapausedepth)).item()
        
        diapausedepth = depthrange[diapausedepthidx]
        diapausedepthidx = diapausedepth - 1 if continuousdepth else diapausedepthidx

    else:

        #if the maximum depth (seafloor depth) is shallower than diapause depth threshold 1 (< 500 m)
        #nb:although not explicitly mentioned, this function is called only when the bottom depth is > 50 m (diapause depth threshold)
        if maxdepth >= ddt0 + 100:

            #if seafloor depth is deeper than ca. 150 m:
            #sets a floor ca. 50 m above the seafloor
            diapausedepthfloor = maxdepth - 50

            #sets a ceiling ca. 50 m above the diapause depth floor
            diapausedepthceiling = diapausedepthfloor - 50

            #randomizing diapause depth ceiling and floor values
            if stochastic:
                diapausedepthvar = np.random.randint(low = 0, high = 50, size = 1).squeeze().item()
            else:
                diapausedepthvar = 25

            diapausedepthceiling -= diapausedepthvar
            
            if stochastic:
                diapausedepthvar = np.random.randint(low = 0, high = 50, size = 1).squeeze().item()

            diapausedepthfloor += diapausedepthvar

        else:

            #if seafloor depth is below 150 m and greater than 50 m (the latter is the basis of the function call, not explicitly defined):
            #sets a floor just above the seafloor depth
            diapausedepthfloor = maxdepth - 1

            #sets a ceiling 20 m above the diapause depth floor (20 is arbitary and a safe value to use esp. in case of a seafloor depth of 51 m!)
            diapausedepthceiling = diapausedepthfloor - 20

            #randomizing diapause depth ceiling and floor values is not performed due to potentially narrow margins between diapause depth floor and ceiling

        #end if

        #selecting diapause depth from a Gaussian probability distribution, within the range, mean and sd defined by the depth floor and depth ceiling
        diapausedepthrange = np.arange(diapausedepthceiling, diapausedepthfloor, 1)
        diapausedepthmean = np.nanmean(diapausedepthrange)
        diapausedepthsd = np.nanstd(diapausedepthrange)

        diapausedepthselprob = np.exp(-0.5 * ((diapausedepthrange - diapausedepthmean) / diapausedepthsd)**2)
        diapausedepthselprob /= np.sum(diapausedepthselprob)
    
        if stochastic:
            selecteddiapausedepth = np.random.choice(a = diapausedepthrange, replace = False, size = 1, p = diapausedepthselprob).squeeze().item()
        else:
            selecteddiapausedepth = diapause_depth_range[0]

        diapausedepthidx = np.argmin(np.abs(depthrange - selecteddiapausedepth)).item()

        diapausedepth = depthrange[diapausedepthidx]
        diapausedepthidx = diapausedepth - 1 if continuousdepth else diapausedepthidx

    #end if

    return diapausedepth, diapausedepthidx

#end def
