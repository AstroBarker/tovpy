'''
Grid routines
'''

import numpy as np
from astropy import constants as const

class GridType:
    """
    Holds our mesh. A Mesh object will hold:
    nElements : number of zones
    rmax : physical extent of domain (in cm)
    r    : actual radius array
    dr   : Rmax / (nElements - 1)
    """
    def __init__( self, nElements, Rmax ):
        self.rmax      = Rmax
        self.r         = np.zeros( nElements )
        self.dr        = Rmax / ( nElements - 1 )
        self.nElements = nElements

        self.r[0] = self.dr #0.0
        for i in range( 1, nElements ):
            self.r[i] = self.r[i-1] + self.dr


    def __str__(self):
        return f"Mesh of length {self.rmax} [units] with {self.nElements} elements."


class State(object):
    """
    Hold pressure, density, mass
    """

    def __init__( self, grid, rho_c, eos, N ):
        self.rho = np.zeros( N )
        self.P   = np.zeros( N )
        self.m   = np.zeros( N )

        r = grid.r[0]
        dr = grid.dr
        pi = np.pi
        c = const.c.cgs.value

        self.rho[0] = rho_c
        self.P[0]   = eos.ComputePressure( rho_c )
        eps         = eos.ComputeSpecificInternalEnergy( self.P[0], rho_c )
        self.m[0]   = 4./3.*pi*r**3 * rho_c*( 1.0 + eps/c**2 )

