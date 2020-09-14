#!/usr/bin/env python3
import numba
import math

@numba.jit(nopython=True)
def deltaR(lphi, leta, jphi, jeta):
    dphi = abs(lphi - jphi)
    if (dphi > math.pi):
        dphi = 2*math.pi - dphi
    return (leta - jeta)**2 + dphi**2

@numba.jit(nopython=True)
def jetRel(lpt, leta, lphi, jpt, jeta, jphi):
    p_jet = jpt*math.cosh(jeta)
    p_lep = lpt*math.cosh(leta)
    p_dot = jpt*lpt*(math.cosh(jeta - leta) - math.cos(jphi - lphi))
    return (p_dot*(2*p_jet*p_lep - p_dot)) / ((p_jet - p_lep)**2 + 2*p_dot)

@numba.jit(nopython=True)
def in_zmass(lpt, leta, lphi, Lpt, Leta, Lphi):
    delta = 15
    zMass = 91.188
    up = zMass + delta
    down = zMass - delta
    mass = 2*lpt*Lpt*(math.cosh(leta-Leta) - math.cos(lphi-Lphi))
    return mass < 12**2 or (mass > down**2 and mass < up**2)
