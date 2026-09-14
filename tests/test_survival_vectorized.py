"""Property-based check that mortalityrisk_dsc2_vectorized matches the
scalar mortalityrisk_dsc2 exactly, across many random inputs spanning both
branches of its piecewise starvation-risk logic (strmass < maxstrmass vs
not, strcat <= 0.10 vs not)."""

import numpy as np
import pytest

from pascal.biology import survival as sv


def random_cases(rng, n=500):
    strmass = rng.uniform(0.1, 300, size=n)
    # Deliberately let maxstrmass be sometimes below, sometimes above/equal
    # to strmass, and sometimes very close (small strcat) - exercises both
    # branches and the strcat<=0.10 boundary.
    maxstrmass = strmass * rng.uniform(0.7, 1.3, size=n)
    resmass = rng.uniform(0, 100, size=n)
    vpreldensity = rng.uniform(0, 0.01, size=n)
    irradiance = rng.uniform(0, 0.3, size=n)
    return strmass, maxstrmass, resmass, vpreldensity, irradiance


def test_vectorized_matches_scalar_elementwise():
    rng = np.random.default_rng(0)
    strmass, maxstrmass, resmass, vpreldensity, irradiance = random_cases(rng)

    maxirradiance = 0.3
    minirradiance = 0.001
    nvpreldensity = 0.00075
    bgmrisk = 0.00075

    vectorized = sv.mortalityrisk_dsc2_vectorized(
        strmass=strmass,
        maxstrmass=maxstrmass,
        resmass=resmass,
        vpreldensity=vpreldensity,
        irradiance=irradiance,
        maxirradiance=maxirradiance,
        minirradiance=minirradiance,
        nvpreldensity=nvpreldensity,
        bgmrisk=bgmrisk,
    )

    for i in range(len(strmass)):
        scalar = sv.mortalityrisk_dsc2(
            strmass=strmass[i],
            maxstrmass=maxstrmass[i],
            resmass=resmass[i],
            vpreldensity=vpreldensity[i],
            irradiance=irradiance[i],
            maxirradiance=maxirradiance,
            minirradiance=minirradiance,
            nvpreldensity=nvpreldensity,
            bgmrisk=bgmrisk,
        )
        assert vectorized[i] == pytest.approx(scalar, abs=1e-12), (
            f"mismatch at case {i}: vectorized={vectorized[i]} scalar={scalar}"
        )


def test_vectorized_handles_the_strcat_threshold_boundary_exactly():
    # strcat == 0.10 exactly: maxstrmass=100, strmass=90 -> strcat=0.10
    strmass = np.array([90.0])
    maxstrmass = np.array([100.0])
    resmass = np.array([0.0])
    vpreldensity = np.array([0.0])
    irradiance = np.array([0.0])

    vectorized = sv.mortalityrisk_dsc2_vectorized(
        strmass=strmass, maxstrmass=maxstrmass, resmass=resmass,
        vpreldensity=vpreldensity, irradiance=irradiance,
        maxirradiance=0.3, minirradiance=0.001,
        nvpreldensity=0.0, bgmrisk=0.0,
    )
    scalar = sv.mortalityrisk_dsc2(
        strmass=90.0, maxstrmass=100.0, resmass=0.0,
        vpreldensity=0.0, irradiance=0.0,
        maxirradiance=0.3, minirradiance=0.001,
        nvpreldensity=0.0, bgmrisk=0.0,
    )
    assert vectorized[0] == pytest.approx(scalar, abs=1e-12)
