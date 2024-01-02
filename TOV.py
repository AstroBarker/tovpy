'''
Main TOV solver
'''

import time

import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const

from Grid import GridType, State
from EoS import Initialize_EOS
from Integrator import rk4, forward_euler


def TOV_rhs( state, rho, eps, r ):
    """
    Return RHS of TOV equations for integrator.
    """

    pi = np.pi
    G = const.G.cgs.value
    c = const.c.cgs.value

    # This is not used, may not be correct.
    if( r == 0.0 ):
        dpdr = 0.0
        dmdr = 4.0*pi*r**2 * rho * ( 1.0 + eps / c**2 )
        return np.array([dpdr, dmdr])
    else:

        P = state[0]
        m = state[1]

        dpdr = - G*( rho * (1.0 + eps/c**2) + P/c**2 ) * \
            (m + 4.0*pi*r**3 * P/c**2)/( r*(r - 2.0*G*m/c**2) )
        dmdr = 4.0*pi*r**2 * rho * ( 1.0 + eps / c**2 )
    
        return np.array([dpdr, dmdr])

def Initialize_TOV( nElements, rMax, rho_c, eos, K, gamma ):
    """
    Initialize grid, eos, and state objects.
    """

    # Initialize grid, eos, state
    grid  = GridType( nElements, rMax )
    eos_  = Initialize_EOS( eos, K, gamma )
    state = State( grid, rho_c, eos_, nElements )

    return grid, eos_, state

def TOV_solve( rho_c, eos, nElements = 5000, K = 1.98183e-6, rmax = 5000000.0 ):

    K         = 1.98183e-6 # K in cgs
    rmax      = 5000000.0  # outer radius 50 kilometer
    nElements = 5000

    t0 = time.time()
    grid, eos, state = \
        Initialize_TOV( nElements, rmax, rho_c, eos, K, 2.75 )

    p_final, m_final = rk4( TOV_rhs, grid, eos, state ) 
    t1 = time.time()

    # print( f"Central density  : {rho_c/1e15}e15 g/cc. \nNeutron star mass: {m_final/1.988e33} Msun\n" )   
    # print( f"Time: {t1-t0}" )

    return grid, state


def TOV_Max_Mass( eos ):
    """
    Run TOV_solve with many central densities to determine
    a mass - radius curve
    """

    rho_cs = np.logspace(np.log10(0.3e15), np.log10(2.4e15), 29)
    # rho_cs = 10.**np.linspace(14.2,15.3,num=50)
    mass   = np.zeros_like( rho_cs )
    radius = np.zeros_like( rho_cs )

    for i in range( len(rho_cs) ):
        grid, state = TOV_solve( rho_cs[i], eos )
        i_max = np.max( np.where( state.P > 0 ) )
        mass[i]   = state.m[ i_max+0 ]
        radius[i] = grid.r[ i_max+0 ]
        
    return mass, radius


def Plot_State( state, grid ):
    """
    Plots density, mass, pressure.
    """

    fig, ax = plt.subplots( 3, 1, figsize=(8, 18) )

    rho = state.rho[ state.P > 0.0 ]
    P = state.P[ state.P > 0.0 ]
    m = state.m[ state.P > 0.0 ]
    r = grid.r[ state.P > 0.0 ]

    Msun = const.M_sun.cgs.value

    ax[0].loglog( r/1e5, rho )
    ax[1].loglog( r/1e5, P )
    ax[2].loglog( r/1e5, m/Msun )
    ax[0].set(ylabel = r"Density [g cm$^{-1}$]")
    ax[1].set(ylabel = r"Pressure [erg cm$^{-1}$]")
    ax[2].set(ylabel = r"Mass [M$_{\odot}$]", xlabel = r"Radius [km]")
    # plt.show()

    return fig, ax

if __name__ == "__main__":
    grid, state = TOV_solve(1.3e15, "Polytropic")
    eos = "Polytropic"

    max_mass, rad = TOV_Max_Mass( eos )
#print(test.m[test.P>0.0])
#Plot_State( state, grid )

