'''
TriLinear interpolation routines.
Here we use numba to speed up computations by "just in time" compiling 
the functions. This speeds things up quite a lot.

Contains:
---------

Index1D - binary search (using numpy) to "locate ourselves" in the table 
TriLinear - TriLinear interpolation
Loginterpolate_Linear - Use above functions to interpolate on a 3D table
'''

import numpy as np
from numba import jit


@jit
def Index1D( val, array ):
    """
    Wrapper for search algorithm using numpy
    """

    ind = np.searchsorted( array, val ) - 1
    return max( ind, 0 )


@jit
def TriLinear( p000, p100, p010, p110, 
               p001, p101, p011, p111, 
               dX1,  dX2,  dX3 ):
    """
    Grandma's favorite TriLinear interpolation
    """

    xd = dX1
    yd = dX2
    zd = dX3

    # Interpolate along x
    c00 = p000 * (1 - xd) + p100 * xd
    c01 = p001 * (1 - xd) + p101 * xd
    c10 = p010 * (1 - xd) + p110 * xd
    c11 = p011 * (1 - xd) + p111 * xd

    #interpolate y
    c0  = c00 * (1 - yd) + c10 * yd
    c1  = c01 * (1 - yd) + c11 * yd

    # interpolate z
    val = c0 * (1 - zd) + c1 * zd
    
    return val


@jit
def LogInterpolate_Linear( D, T, Y, Ds, Ts, Ys, Table, os ):
    """
    Interpolation of log quantites. 
    Log interpolation is (probably) more accurate, 
    though we do have some undesireable log10 and 10** operations.

    Parameters:
    -----------
    D, T, Y : point we wish to interpolate
    Table   : self.log_pressure or self.log_energy - what to interpolate
    os      : offset value. self.energy_offset etc. 0.0 for pressure, mu.
    """

    # Definitions:
    Ds = 10**(Ds)
    Ts = 10**(Ts)
    Ys =     (Ys)

    # Locate ourselves in the table.
    # This is just a binary search algorithm
    iD = Index1D( D, Ds )
    iT = Index1D( T, Ts )
    iY = Index1D( Y, Ys )

    # Slopes for interpolation
    dD = np.log10( D / Ds[iD] ) / np.log10( Ds[iD+1] / Ds[iD] )
    dT = np.log10( T / Ts[iT] ) / np.log10( Ts[iT+1] / Ts[iT] )
    dY = ( Y - Ys[iY] ) / ( Ys[iY+1] - Ys[iY] )

    # Get the vertices surrounding our point
    p000 = Table[ iD  , iT  , iY   ]
    p100 = Table[ iD+1, iT  , iY   ]
    p010 = Table[ iD  , iT+1, iY   ]
    p110 = Table[ iD+1, iT+1, iY   ]
    p001 = Table[ iD  , iT  , iY+1 ]
    p101 = Table[ iD+1, iT  , iY+1 ]
    p011 = Table[ iD  , iT+1, iY+1 ]
    p111 = Table[ iD+1, iT+1, iY+1 ]

    # Interpolate!
    Interpolant = 10.0**( TriLinear( p000, p100, p010, p110, p001, p101, p011, p111, dD, dT, dY ) ) - os

    return Interpolant


@jit
def Interpolate_Linear( D, T, Y, Ds, Ts, Ys, Table, os ):
    """
    Interpolation of not-log quantites. (mu's) (but log spaced)
    Log interpolation is (probably) more accurate, 
    though we do have some undesireable log10 and 10** operations.

    Parameters:
    -----------
    D, T, Y : point we wish to interpolate
    Table   : self.log_pressure or self.log_energy - what to interpolate
    os      : offset value. self.energy_offset etc. 0.0 for pressure, mu.
    """

    # Definitions:
    Ds = 10**(Ds)
    Ts = 10**(Ts)
    Ys =     (Ys)

    # Locate ourselves in the table.
    # This is just a binary search algorithm
    iD = Index1D( D, Ds )
    iT = Index1D( T, Ts )
    iY = Index1D( Y, Ys )

    # Slopes for interpolation
    dD = np.log10( D / Ds[iD] ) / np.log10( Ds[iD+1] / Ds[iD] )
    dT = np.log10( T / Ts[iT] ) / np.log10( Ts[iT+1] / Ts[iT] )
    dY = ( Y - Ys[iY] ) / ( Ys[iY+1] - Ys[iY] )

    # Get the vertices surrounding our point
    p000 = Table[ iD  , iT  , iY   ]
    p100 = Table[ iD+1, iT  , iY   ]
    p010 = Table[ iD  , iT+1, iY   ]
    p110 = Table[ iD+1, iT+1, iY   ]
    p001 = Table[ iD  , iT  , iY+1 ]
    p101 = Table[ iD+1, iT  , iY+1 ]
    p011 = Table[ iD  , iT+1, iY+1 ]
    p111 = Table[ iD+1, iT+1, iY+1 ]

    # Interpolate!
    Interpolant = ( TriLinear( p000, p100, p010, p110, p001, p101, p011, p111, dD, dT, dY ) ) - os

    return Interpolant