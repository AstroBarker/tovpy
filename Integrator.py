'''
ODE integrator routines
'''

import numpy as np

def rk4( f, grid, eos, state ):
    """
    4th order runge-kutta. Not the sleakest but works.

    Parameters:
    -----------
    f     : RHS function to use with the solver.
    grid  : grid object
    eos   : eos instance
    state : state object
    """

    h     = grid.dr
    sol   = np.array( [state.P[0], state.m[0]] )
    xvals = np.zeros( grid.nElements )
    yvals = np.zeros( grid.nElements )
    xvals[0] = sol[0]
    yvals[0] = sol[1]

    steps = 0
    # If pressure becomes negative, we're done.
    while( sol[0] > 0.0 and steps < 5000 ):

        i = steps
        # stages and such
        # f(rho, eps, P, m, r)
        # sol contains P, m
        
        rho = eos.ComputeDensity( sol[0] ) # sol[0] is pressure
        eps = eos.ComputeSpecificInternalEnergy( sol[0], rho )
        r   = grid.r[i]

        k1 = f( sol, rho, eps, r  )
        k2 = f( sol + 0.5*k1, rho, eps, r )
        k3 = f( sol + 0.5*k2, rho, eps, r )
        k4 = f( sol +     k3, rho, eps, r )
        sol += h * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

        xvals[i+1] = sol[0]
        yvals[i+1] = sol[1]
        state.P[i+1] = sol[0]
        state.m[i+1] = sol[1]
        state.rho[i] = rho
        rho_old = rho

        steps += 1
        

    return xvals[i+1], yvals[i+1]


def forward_euler( f, grid, eos, state ):
    """
    Standard forward Euler solver

    Parameters:
    -----------
    f        : RHS function to use with the solver.
    h        : step
    interval : [a,b]
    ival     : [x0, v0] -- initial conditions 
    *args    : anything needed by f 
    """

    h     = grid.dr
    sol   = np.array( [state.P[0], state.m[0]] )
    xvals = np.zeros( grid.nElements )
    yvals = np.zeros( grid.nElements )

    # step forward!
    for i in range(grid.nElements - 1 ):

        rho = eos.ComputeDensity( sol[0] ) # sol[0] is pressure
        eps = eos.ComputeSpecificInternalEnergy( sol[0], rho )
        r   = grid.r[i]

        # should've set up func() to take in time
        # for the sake of generality.
        # I'll fix later if needed.
        sol += h * f( sol, rho, eps, r ) 

        xvals[i+1] = sol[0]
        yvals[i+1] = sol[1]
        state.P[i+1] = sol[0]
        state.m[i+1] = sol[1]

        if( sol[0] < 0.0 ):
            break

    return xvals[i+1], yvals[i+1]